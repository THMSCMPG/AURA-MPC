"""PINN + RK4TRAN closed-loop integration utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from data import NumericNormalizer
from models import PINNSurrogate

_WEATHER_DIM = 7
_PANEL_DIM = 4
_LOCATION_DIM = 3
_BASE_INPUT_DIM = _WEATHER_DIM + _PANEL_DIM + _LOCATION_DIM


def _resolve_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("PINN checkpoint must deserialize to a state dict or wrapper dict")
    if "input_proj.0.weight" in checkpoint:
        return checkpoint
    state = checkpoint.get("model_state") or checkpoint.get("state_dict")
    if isinstance(state, dict):
        return state
    raise KeyError("PINN checkpoint does not contain model weights")


class RK4TRANValidator:
    """Wrapper around the single-state RK4TRAN evaluator binary."""

    def __init__(
        self,
        binary_path: Path | str,
        timeout_s: float = 10.0,
    ) -> None:
        self.binary_path = Path(binary_path)
        self.timeout_s = float(timeout_s)
        if not self.binary_path.exists():
            raise FileNotFoundError(f"RK4TRAN evaluator not found: {self.binary_path}")

    def predict(
        self,
        weather: dict[str, float],
        panel_state: dict[str, float],
        location: dict[str, float],
        time_components: dict[str, float],
    ) -> dict[str, float]:
        args = [
            str(self.binary_path),
            f"{location['lon']:.10f}",
            f"{location['lat']:.10f}",
            f"{location['elevation']:.10f}",
            f"{time_components['minute']:.10f}",
            f"{time_components['hour']:.10f}",
            f"{time_components['day_of_year']:.10f}",
            f"{time_components['month']:.10f}",
            f"{time_components['year']:.10f}",
            f"{weather['T_amb']:.10f}",
            f"{weather['wind_speed']:.10f}",
            f"{weather['wind_dir']:.10f}",
            f"{weather['humidity']:.10f}",
            f"{weather['irradiance']:.10f}",
            f"{weather['cloud_cover']:.10f}",
            f"{weather['pressure']:.10f}",
            f"{panel_state['pv_height']:.10f}",
            f"{panel_state['pitch']:.10f}",
            f"{panel_state['roll']:.10f}",
            f"{panel_state['yaw']:.10f}",
        ]
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "unknown RK4TRAN failure"
            raise RuntimeError(f"RK4TRAN evaluation failed: {stderr}")
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"RK4TRAN returned invalid JSON: {proc.stdout!r}") from exc
        if not isinstance(parsed, dict):
            raise TypeError("RK4TRAN output must decode to a JSON object")
        required = ("T_operating", "eta", "G_eff")
        missing = [key for key in required if key not in parsed]
        if missing:
            raise KeyError(f"RK4TRAN output missing fields: {missing}")
        return {
            "T_operating": float(parsed["T_operating"]),
            "eta": float(parsed["eta"]),
            "G_eff": float(parsed["G_eff"]),
            "runtime_ms": float(parsed.get("runtime_ms", 0.0)),
        }


class PINNValidator:
    """PINN model wrapper for closed-loop inference."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        normalizer_path: Optional[Path | str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PINN checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        state_dict = _resolve_state_dict(checkpoint)
        if "input_proj.0.weight" not in state_dict:
            raise KeyError("PINN checkpoint is missing input projection weights")
        input_dim = int(state_dict["input_proj.0.weight"].shape[1])
        time_dim = max(0, input_dim - _BASE_INPUT_DIM)

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.model = PINNSurrogate(
            input_dim=input_dim,
            hidden_dim=128,
            num_residual_blocks=4,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.normalizer: NumericNormalizer | None = None
        if normalizer_path is not None:
            normalizer_file = Path(normalizer_path)
            if normalizer_file.exists():
                self.normalizer = NumericNormalizer.load(normalizer_file)

    def _build_time_tensor(self, time_components: dict[str, float]) -> Tensor:
        raw = [
            float(time_components["minute"]),
            float(time_components["hour"]),
            float(time_components["day_of_year"]),
            float(time_components["month"]),
            float(time_components["year"]),
        ]
        if self.time_dim <= 0:
            return torch.zeros(0, dtype=torch.float32)
        if len(raw) >= self.time_dim:
            values = raw[: self.time_dim]
        else:
            values = raw + [0.0] * (self.time_dim - len(raw))
        return torch.tensor(values, dtype=torch.float32)

    def _normalize_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.normalizer is None:
            return sample
        return self.normalizer.normalize(sample)

    def predict(
        self,
        weather: dict[str, float],
        panel_state: dict[str, float],
        location: dict[str, float],
        time_components: dict[str, float],
    ) -> dict[str, float]:
        with torch.no_grad():
            sample = {
                "weather": torch.tensor(
                    [
                        weather["T_amb"],
                        weather["wind_speed"],
                        weather["wind_dir"],
                        weather["humidity"],
                        weather["irradiance"],
                        weather["cloud_cover"],
                        weather["pressure"],
                    ],
                    dtype=torch.float32,
                ),
                "panel_state": torch.tensor(
                    [
                        panel_state["pv_height"],
                        panel_state["pitch"],
                        panel_state["roll"],
                        panel_state["yaw"],
                    ],
                    dtype=torch.float32,
                ),
                "location": torch.tensor(
                    [
                        location["lat"],
                        location["lon"],
                        location["elevation"],
                    ],
                    dtype=torch.float32,
                ),
                "time": self._build_time_tensor(time_components),
            }
            sample = self._normalize_sample(sample)
            x = torch.cat(
                [sample["weather"], sample["panel_state"], sample["location"], sample["time"]],
                dim=0,
            ).to(self.device)
            if x.numel() != self.input_dim:
                raise ValueError(
                    f"PINN input width mismatch: built {x.numel()} features but checkpoint expects {self.input_dim}"
                )
            outputs = self.model(x.unsqueeze(0))
        return {
            "T_operating": float(outputs["T_operating"].squeeze().item()),
            "T_operating_sigma": float(outputs["T_operating_sigma"].squeeze().item()),
            "eta": float(outputs["eta"].squeeze().item()),
            "eta_sigma": float(outputs["eta_sigma"].squeeze().item()),
        }


class ComparisonMetrics:
    """Accumulator for PINN-vs-RK4TRAN drift statistics."""

    def __init__(self) -> None:
        self.T_errors: list[float] = []
        self.eta_errors: list[float] = []
        self.T_pinn: list[float] = []
        self.T_rk4: list[float] = []
        self.eta_pinn: list[float] = []
        self.eta_rk4: list[float] = []

    def update(self, pinn_pred: dict[str, float], rk4_pred: dict[str, float]) -> None:
        t_p = float(pinn_pred["T_operating"])
        t_r = float(rk4_pred["T_operating"])
        eta_p = float(pinn_pred["eta"])
        eta_r = float(rk4_pred["eta"])
        self.T_pinn.append(t_p)
        self.T_rk4.append(t_r)
        self.T_errors.append(abs(t_p - t_r))
        self.eta_pinn.append(eta_p)
        self.eta_rk4.append(eta_r)
        self.eta_errors.append(abs(eta_p - eta_r))

    def get_summary(self) -> dict[str, float]:
        if not self.T_errors:
            return {}
        t_errors = torch.tensor(self.T_errors, dtype=torch.float32)
        eta_errors = torch.tensor(self.eta_errors, dtype=torch.float32)
        t_pinn = torch.tensor(self.T_pinn, dtype=torch.float32)
        t_rk4 = torch.tensor(self.T_rk4, dtype=torch.float32)
        eta_pinn = torch.tensor(self.eta_pinn, dtype=torch.float32)
        eta_rk4 = torch.tensor(self.eta_rk4, dtype=torch.float32)
        return {
            "T_mae": float(t_errors.mean().item()),
            "T_rmse": float(torch.sqrt((t_errors**2).mean()).item()),
            "T_bias": float((t_pinn - t_rk4).mean().item()),
            "eta_mae": float(eta_errors.mean().item()),
            "eta_rmse": float(torch.sqrt((eta_errors**2).mean()).item()),
            "eta_bias": float((eta_pinn - eta_rk4).mean().item()),
        }

    def reset(self) -> None:
        self.T_errors.clear()
        self.eta_errors.clear()
        self.T_pinn.clear()
        self.T_rk4.clear()
        self.eta_pinn.clear()
        self.eta_rk4.clear()


class SandboxPINNAgent:
    """Closed-loop agent exposing PINN estimates and RK4TRAN validation."""

    def __init__(
        self,
        pinn_checkpoint: Path | str,
        rk4_binary: Optional[Path | str] = None,
        normalizer_path: Optional[Path | str] = None,
        device: str = "cpu",
        correction_alpha: float = 0.25,
    ) -> None:
        self.pinn = PINNValidator(
            checkpoint_path=pinn_checkpoint,
            normalizer_path=normalizer_path,
            device=device,
        )
        self.rk4 = RK4TRANValidator(rk4_binary) if rk4_binary else None
        self.metrics = ComparisonMetrics()
        self.correction_alpha = float(correction_alpha)
        self._bias_state = {"T_operating": 0.0, "eta": 0.0}

    def _update_bias(self, discrepancy: dict[str, float]) -> None:
        for key in self._bias_state:
            self._bias_state[key] = (
                (1.0 - self.correction_alpha) * self._bias_state[key]
                + self.correction_alpha * float(discrepancy[key])
            )

    def _bias_correct(self, pinn_out: dict[str, float]) -> dict[str, float]:
        corrected_eta = float(pinn_out["eta"] + self._bias_state["eta"])
        corrected = {
            "T_operating": float(pinn_out["T_operating"] + self._bias_state["T_operating"]),
            "T_operating_sigma": float(pinn_out["T_operating_sigma"]),
            "eta": float(max(0.0, min(1.0, corrected_eta))),
            "eta_sigma": float(pinn_out["eta_sigma"]),
        }
        return corrected

    def predict(
        self,
        weather: dict[str, float],
        panel_state: dict[str, float],
        location: dict[str, float],
        time_components: dict[str, float],
        include_rk4: bool = True,
    ) -> dict[str, dict[str, float] | None]:
        pinn_out = self.pinn.predict(weather, panel_state, location, time_components)
        rk4_out: dict[str, float] | None = None
        discrepancy = {"T_operating": 0.0, "eta": 0.0}

        if include_rk4:
            if self.rk4 is None:
                raise RuntimeError("RK4TRAN validation requested but no evaluator binary is configured")
            rk4_out = self.rk4.predict(weather, panel_state, location, time_components)
            discrepancy = {
                "T_operating": float(rk4_out["T_operating"] - pinn_out["T_operating"]),
                "eta": float(rk4_out["eta"] - pinn_out["eta"]),
            }
            self._update_bias(discrepancy)
            self.metrics.update(pinn_out, rk4_out)

        corrected = self._bias_correct(pinn_out)
        if rk4_out is not None and "G_eff" in rk4_out:
            corrected["G_eff"] = float(rk4_out["G_eff"])

        return {
            "pinn": pinn_out,
            "corrected": corrected,
            "rk4": rk4_out,
            "discrepancy": discrepancy,
            "bias_correction": dict(self._bias_state),
        }

    def get_metrics(self) -> dict[str, float]:
        summary = self.metrics.get_summary()
        summary.update(
            {
                "bias_T_operating": float(self._bias_state["T_operating"]),
                "bias_eta": float(self._bias_state["eta"]),
            }
        )
        return summary

    def reset_metrics(self) -> None:
        self.metrics.reset()
        self._bias_state = {"T_operating": 0.0, "eta": 0.0}
