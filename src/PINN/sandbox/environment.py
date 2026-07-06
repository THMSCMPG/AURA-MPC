"""Closed-loop sandbox environment for policy -> PINN -> RK4TRAN control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class EpisodeConditions:
    """Environmental conditions for one rollout."""

    lat: float
    lon: float
    alt: float
    day_of_year: int
    hour: float
    minute: float
    month: int
    year: int
    ambient_c: float
    wind_mps: float
    wind_dir: float
    humidity: float
    irradiance: float
    cloud_cover: float
    pressure: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EpisodeConditions":
        return cls(
            lat=float(values.get("lat", 36.5)),
            lon=float(values.get("lon", -87.3)),
            alt=float(values.get("alt", values.get("elevation", 100.0))),
            day_of_year=int(values.get("day_of_year", 172)),
            hour=float(values.get("hour", 12.0)),
            minute=float(values.get("minute", 0.0)),
            month=int(values.get("month", 6)),
            year=int(values.get("year", 2024)),
            ambient_c=float(values.get("ambient_c", 25.0)),
            wind_mps=float(values.get("wind_mps", 4.0)),
            wind_dir=float(values.get("wind_dir", 180.0)),
            humidity=float(values.get("humidity", 0.5)),
            irradiance=float(values.get("irradiance", 850.0)),
            cloud_cover=float(values.get("cloud_cover", 0.1)),
            pressure=float(values.get("pressure", 101325.0)),
        )


def _solar_declination_rad(day_of_year: int) -> float:
    return math.radians(23.44) * math.sin(2.0 * math.pi * (day_of_year - 81) / 365.0)


def _sun_vector(lat_deg: float, day_of_year: int, local_hour: float) -> FloatArray:
    lat = math.radians(lat_deg)
    dec = _solar_declination_rad(day_of_year)
    hour_angle = math.radians(15.0 * (local_hour - 12.0))
    sin_alt = (
        math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    )
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(
        -math.sin(hour_angle),
        math.tan(dec) * math.cos(lat) - math.sin(lat) * math.cos(hour_angle),
    )
    return np.array(
        [
            math.cos(alt) * math.sin(az),
            math.cos(alt) * math.cos(az),
            max(0.0, math.sin(alt)),
        ],
        dtype=np.float32,
    )


def _panel_normal(pitch_deg: float, yaw_deg: float, roll_deg: float) -> FloatArray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [math.cos(roll), 0.0, math.sin(roll)],
            [0.0, 1.0, 0.0],
            [-math.sin(roll), 0.0, math.cos(roll)],
        ],
        dtype=np.float32,
    )
    normal = rz @ rx @ ry @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (normal / norm).astype(np.float32)


class PanelEnv:
    """Closed-loop environment using RK4TRAN-validated reward shaping."""

    def __init__(
        self,
        pinn_agent,
        seed: int = 42,
        dt_s: float = 1.0,
        episode_steps: int = 64,
        pitch_bound: float = 45.0,
        yaw_bound: float = 180.0,
        roll_bound: float = 20.0,
        z_min: float = 0.5,
        z_max: float = 2.0,
        max_pitch_rate: float = 30.0,
        max_yaw_rate: float = 60.0,
        max_roll_rate: float = 20.0,
        max_z_rate: float = 0.5,
        t_ref_k: float = 298.15,
        temp_margin_k: float = 0.0,
        reward_w_capture: float = 1.0,
        reward_w_temp: float = 0.4,
        reward_w_correction: float = 0.05,
        capture_scale: float = 1.0e-3,
        temp_scale: float = 1.0,
        pose_change_penalty: float = 1.0e-3,
        correction_temp_scale_k: float = 15.0,
        correction_eta_scale: float = 0.05,
        validation_mode: str = "every_step",
        validation_period: int = 1,
        reward_source: str = "validated",
        condition_bounds: dict[str, float] | None = None,
    ) -> None:
        self.pinn_agent = pinn_agent
        self._rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.dt_s = float(dt_s)
        self.episode_steps = int(episode_steps)

        self.pitch_bound = float(pitch_bound)
        self.yaw_bound = float(yaw_bound)
        self.roll_bound = float(roll_bound)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.max_pitch_rate = float(max_pitch_rate)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_roll_rate = float(max_roll_rate)
        self.max_z_rate = float(max_z_rate)

        self.t_ref_k = float(t_ref_k)
        self.temp_margin_k = float(temp_margin_k)
        self.reward_w_capture = float(reward_w_capture)
        self.reward_w_temp = float(reward_w_temp)
        self.reward_w_correction = float(reward_w_correction)
        self.capture_scale = float(capture_scale)
        self.temp_scale = float(temp_scale)
        self.pose_change_penalty = float(pose_change_penalty)
        self.correction_temp_scale_k = float(correction_temp_scale_k)
        self.correction_eta_scale = float(correction_eta_scale)
        self.validation_mode = str(validation_mode)
        self.validation_period = max(1, int(validation_period))
        self.reward_source = str(reward_source)

        self.condition_bounds = {
            "lat_min": -65.0,
            "lat_max": 65.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
            "alt_min": 0.0,
            "alt_max": 2000.0,
            "day_min": 1,
            "day_max": 365,
            "hour_min": 5.0,
            "hour_max": 19.0,
            "ambient_c_min": -15.0,
            "ambient_c_max": 45.0,
            "wind_min": 0.0,
            "wind_max": 20.0,
            "cloud_min": 0.0,
            "cloud_max": 1.0,
            "g_peak_min": 500.0,
            "g_peak_max": 1100.0,
            "humidity_min": 0.0,
            "humidity_max": 1.0,
            "pressure_min": 80000.0,
            "pressure_max": 105000.0,
        }
        if condition_bounds:
            self.condition_bounds.update(condition_bounds)

        self._step_idx = 0
        self._pose = self._default_pose()
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_discrepancy = {"T_operating": 0.0, "eta": 0.0}
        self.conditions = self._sample_conditions()

    @property
    def observation_dim(self) -> int:
        return 18

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def pose(self) -> dict[str, float]:
        return dict(self._pose)

    def _default_pose(self) -> dict[str, float]:
        return {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "z": (self.z_min + self.z_max) * 0.5,
        }

    def _sample_conditions(self) -> EpisodeConditions:
        b = self.condition_bounds
        return EpisodeConditions(
            lat=float(self._rng.uniform(b["lat_min"], b["lat_max"])),
            lon=float(self._rng.uniform(b["lon_min"], b["lon_max"])),
            alt=float(self._rng.uniform(b["alt_min"], b["alt_max"])),
            day_of_year=int(self._rng.integers(int(b["day_min"]), int(b["day_max"]) + 1)),
            hour=float(self._rng.uniform(b["hour_min"], b["hour_max"])),
            minute=float(self._rng.uniform(0.0, 60.0)),
            month=int(self._rng.integers(1, 13)),
            year=2024,
            ambient_c=float(self._rng.uniform(b["ambient_c_min"], b["ambient_c_max"])),
            wind_mps=float(self._rng.uniform(b["wind_min"], b["wind_max"])),
            wind_dir=float(self._rng.uniform(0.0, 360.0)),
            humidity=float(self._rng.uniform(b["humidity_min"], b["humidity_max"])),
            irradiance=float(self._rng.uniform(b["g_peak_min"], b["g_peak_max"])),
            cloud_cover=float(self._rng.uniform(b["cloud_min"], b["cloud_max"])),
            pressure=float(self._rng.uniform(b["pressure_min"], b["pressure_max"])),
        )

    def _feature_dicts(self) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        c = self.conditions
        weather = {
            "T_amb": float(c.ambient_c + 273.15),
            "wind_speed": float(c.wind_mps),
            "wind_dir": float(c.wind_dir),
            "humidity": float(c.humidity),
            "irradiance": float(c.irradiance),
            "cloud_cover": float(c.cloud_cover),
            "pressure": float(c.pressure),
        }
        panel_state = {
            "pv_height": float(self._pose["z"]),
            "pitch": float(self._pose["pitch"]),
            "roll": float(self._pose["roll"]),
            "yaw": float(self._pose["yaw"]),
        }
        location = {
            "lat": float(c.lat),
            "lon": float(c.lon),
            "elevation": float(c.alt),
        }
        time_components = {
            "minute": float(c.minute),
            "hour": float(c.hour),
            "day_of_year": float(c.day_of_year),
            "month": float(c.month),
            "year": float(c.year),
        }
        return weather, panel_state, location, time_components

    def _effective_irradiance(self) -> tuple[float, float]:
        c = self.conditions
        sun = _sun_vector(c.lat, c.day_of_year, c.hour + c.minute / 60.0)
        normal = _panel_normal(self._pose["pitch"], self._pose["yaw"], self._pose["roll"])
        incidence = float(np.clip(float(np.dot(sun, normal)), 0.0, 1.0))
        g_eff = float(c.irradiance * incidence * (1.0 - c.cloud_cover))
        return g_eff, incidence

    def _build_observation(self) -> FloatArray:
        c = self.conditions
        sun = _sun_vector(c.lat, c.day_of_year, c.hour + c.minute / 60.0)
        obs = np.array(
            [
                c.lat / 90.0,
                c.lon / 180.0,
                c.day_of_year / 365.0,
                c.hour / 24.0,
                c.ambient_c / 50.0,
                c.wind_mps / 20.0,
                c.cloud_cover,
                c.humidity,
                c.irradiance / 1400.0,
                sun[0],
                sun[1],
                sun[2],
                self._pose["pitch"] / max(1.0, self.pitch_bound),
                self._pose["yaw"] / max(1.0, self.yaw_bound),
                self._pose["roll"] / max(1.0, self.roll_bound),
                self._pose["z"] / max(1e-6, self.z_max),
                self._last_discrepancy["T_operating"] / max(1.0, self.correction_temp_scale_k),
                self._last_discrepancy["eta"] / max(1e-6, self.correction_eta_scale),
            ],
            dtype=np.float32,
        )
        return obs

    def _coerce_pose(self, pose: dict[str, Any] | None) -> dict[str, float]:
        if pose is None:
            return self._default_pose()
        return {
            "pitch": float(np.clip(float(pose.get("pitch", 0.0)), -self.pitch_bound, self.pitch_bound)),
            "yaw": float(np.clip(float(pose.get("yaw", 0.0)), -self.yaw_bound, self.yaw_bound)),
            "roll": float(np.clip(float(pose.get("roll", 0.0)), -self.roll_bound, self.roll_bound)),
            "z": float(np.clip(float(pose.get("z", self._default_pose()["z"])), self.z_min, self.z_max)),
        }

    def reset(
        self,
        seed: int | None = None,
        conditions: dict[str, Any] | EpisodeConditions | None = None,
        pose: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if conditions is None:
            self.conditions = self._sample_conditions()
        elif isinstance(conditions, EpisodeConditions):
            self.conditions = conditions
        else:
            self.conditions = EpisodeConditions.from_mapping(conditions)
        self._step_idx = 0
        self._pose = self._coerce_pose(pose)
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_discrepancy = {"T_operating": 0.0, "eta": 0.0}
        obs = self._build_observation()
        evaluation = self._evaluate_state(validate_with_rk4=self._should_validate(self._step_idx))
        info = self._build_info(
            action=np.zeros(4, dtype=np.float32),
            evaluation=evaluation,
            reward_breakdown=None,
            policy_context={"mode": "reset"},
            decision_reason="Simulation reset; waiting for the next control decision.",
            terminated=False,
            truncated=False,
        )
        return obs, info

    def _clamp_and_slew(self, action: FloatArray) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(4)
        targets = {
            "pitch": float(np.clip(action[0], -1.0, 1.0) * self.pitch_bound),
            "yaw": float(np.clip(action[1], -1.0, 1.0) * self.yaw_bound),
            "roll": float(np.clip(action[2], -1.0, 1.0) * self.roll_bound),
            "z": float(self.z_min + 0.5 * (np.clip(action[3], -1.0, 1.0) + 1.0) * (self.z_max - self.z_min)),
        }
        rate_limits = {
            "pitch": self.max_pitch_rate * self.dt_s,
            "yaw": self.max_yaw_rate * self.dt_s,
            "roll": self.max_roll_rate * self.dt_s,
            "z": self.max_z_rate * self.dt_s,
        }
        ranges = {
            "pitch": (-self.pitch_bound, self.pitch_bound),
            "yaw": (-self.yaw_bound, self.yaw_bound),
            "roll": (-self.roll_bound, self.roll_bound),
            "z": (self.z_min, self.z_max),
        }
        next_pose: dict[str, float] = {}
        for key in ("pitch", "yaw", "roll", "z"):
            delta = targets[key] - self._pose[key]
            limited = float(np.clip(delta, -rate_limits[key], rate_limits[key]))
            lo, hi = ranges[key]
            next_pose[key] = float(np.clip(self._pose[key] + limited, lo, hi))
        return next_pose

    def _should_validate(self, step_idx: int) -> bool:
        mode = self.validation_mode.lower()
        if mode == "every_step":
            return True
        if mode == "periodic":
            return step_idx % self.validation_period == 0
        if mode == "episodic":
            return False
        raise ValueError(f"Unsupported validation_mode: {self.validation_mode}")

    def _evaluate_state(self, validate_with_rk4: bool) -> dict[str, Any]:
        weather, panel_state, location, time_components = self._feature_dicts()
        return self.pinn_agent.predict(
            weather=weather,
            panel_state=panel_state,
            location=location,
            time_components=time_components,
            include_rk4=validate_with_rk4,
        )

    def _reward_components(
        self,
        evaluation: dict[str, Any],
        pose_delta_norm: float,
    ) -> dict[str, float | str]:
        g_eff_geom, incidence = self._effective_irradiance()
        rk4 = evaluation.get("rk4")
        corrected = evaluation["corrected"]
        pinn = evaluation["pinn"]
        discrepancy = evaluation["discrepancy"]

        if rk4 is not None and self.reward_source == "validated":
            reward_outputs = rk4
            reward_origin = "rk4_validated"
        elif rk4 is not None and self.reward_source == "hybrid":
            reward_outputs = {
                "T_operating": float(rk4["T_operating"]),
                "eta": float(max(0.0, 0.5 * (rk4["eta"] + corrected["eta"]))),
                "G_eff": float(rk4.get("G_eff", g_eff_geom)),
            }
            reward_origin = "rk4_hybrid"
        else:
            reward_outputs = {
                "T_operating": float(corrected["T_operating"]),
                "eta": float(corrected["eta"]),
                "G_eff": float(corrected.get("G_eff", g_eff_geom)),
            }
            reward_origin = "bias_corrected_pinn"

        electrical_output = float(max(0.0, reward_outputs["G_eff"]) * max(0.0, reward_outputs["eta"]))
        capture_reward = self.reward_w_capture * self.capture_scale * electrical_output
        over_temp = max(
            0.0,
            float(reward_outputs["T_operating"]) - (self.t_ref_k + self.temp_margin_k),
        )
        temp_penalty = self.reward_w_temp * self.temp_scale * over_temp
        smoothness_penalty = self.pose_change_penalty * pose_delta_norm
        correction_penalty = 0.0
        if rk4 is not None:
            correction_penalty = self.reward_w_correction * (
                abs(float(discrepancy["T_operating"])) / max(1.0, self.correction_temp_scale_k)
                + abs(float(discrepancy["eta"])) / max(1.0e-6, self.correction_eta_scale)
            )

        total_reward = capture_reward - temp_penalty - smoothness_penalty - correction_penalty
        return {
            "reward_origin": reward_origin,
            "G_eff_geom": g_eff_geom,
            "G_eff_used": float(reward_outputs["G_eff"]),
            "incidence_cos": incidence,
            "electrical_output_proxy": electrical_output,
            "capture_reward": float(capture_reward),
            "temp_penalty": float(temp_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "correction_penalty": float(correction_penalty),
            "pinn_temperature_k": float(pinn["T_operating"]),
            "validated_temperature_k": float(reward_outputs["T_operating"]),
            "pinn_eta": float(pinn["eta"]),
            "validated_eta": float(reward_outputs["eta"]),
            "total_reward": float(total_reward),
        }

    def _decision_reason(
        self,
        previous_pose: dict[str, float],
        next_pose: dict[str, float],
        evaluation: dict[str, Any],
        reward_breakdown: dict[str, float | str],
    ) -> str:
        rk4 = evaluation.get("rk4")
        drift_t = float(evaluation["discrepancy"]["T_operating"])
        drift_eta = float(evaluation["discrepancy"]["eta"])
        parts: list[str] = []

        pitch_delta = next_pose["pitch"] - previous_pose["pitch"]
        yaw_delta = next_pose["yaw"] - previous_pose["yaw"]
        roll_delta = next_pose["roll"] - previous_pose["roll"]
        z_delta = next_pose["z"] - previous_pose["z"]

        if abs(pitch_delta) > 0.25:
            direction = "toward stronger capture" if pitch_delta > 0 else "away from peak irradiance"
            parts.append(f"Pitch moved {direction}.")
        if abs(yaw_delta) > 0.25:
            parts.append("Yaw adjusted to improve sun alignment.")
        if abs(roll_delta) > 0.25:
            parts.append("Roll changed to fine-tune incidence.")
        if abs(z_delta) > 0.01:
            parts.append("Height changed to trade convective cooling against exposure.")

        validated_temp = float(reward_breakdown["validated_temperature_k"])
        if validated_temp > self.t_ref_k + self.temp_margin_k:
            parts.append("Thermal protection is active because validated temperature is above target.")

        if rk4 is not None and (abs(drift_t) > 2.0 or abs(drift_eta) > 0.01):
            parts.append("RK4TRAN disagrees with the PINN enough to apply a correction penalty.")

        if not parts:
            parts.append("Pose was largely held steady to preserve smoothness while maintaining reward.")
        return " ".join(parts)

    def _advance_time(self) -> None:
        total_minutes = self.conditions.minute + self.dt_s / 60.0
        hour = self.conditions.hour
        day = self.conditions.day_of_year
        while total_minutes >= 60.0:
            total_minutes -= 60.0
            hour += 1.0
        while hour >= 24.0:
            hour -= 24.0
            day += 1
        if day > 365:
            day = 1
        self.conditions = EpisodeConditions(
            lat=self.conditions.lat,
            lon=self.conditions.lon,
            alt=self.conditions.alt,
            day_of_year=day,
            hour=hour,
            minute=total_minutes,
            month=self.conditions.month,
            year=self.conditions.year,
            ambient_c=self.conditions.ambient_c,
            wind_mps=self.conditions.wind_mps,
            wind_dir=self.conditions.wind_dir,
            humidity=self.conditions.humidity,
            irradiance=self.conditions.irradiance,
            cloud_cover=self.conditions.cloud_cover,
            pressure=self.conditions.pressure,
        )

    def _build_info(
        self,
        action: FloatArray,
        evaluation: dict[str, Any],
        reward_breakdown: dict[str, float | str] | None,
        policy_context: dict[str, Any],
        decision_reason: str,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        weather, panel_state, location, time_components = self._feature_dicts()
        info = {
            "step_index": int(self._step_idx),
            "conditions": self.conditions.to_dict(),
            "pose": dict(self._pose),
            "action": [float(v) for v in np.asarray(action, dtype=np.float32).reshape(4)],
            "inputs": {
                "weather": weather,
                "panel_state": panel_state,
                "location": location,
                "time": time_components,
            },
            "pinn_prediction": evaluation["pinn"],
            "corrected_prediction": evaluation["corrected"],
            "rk4_prediction": evaluation.get("rk4"),
            "discrepancy": evaluation["discrepancy"],
            "bias_correction": evaluation["bias_correction"],
            "reward_breakdown": reward_breakdown,
            "decision_reason": decision_reason,
            "policy_context": policy_context,
            "validation": {
                "mode": self.validation_mode,
                "period": self.validation_period,
                "reward_source": self.reward_source,
                "performed": evaluation.get("rk4") is not None,
            },
            "metrics": self.pinn_agent.get_metrics(),
            "terminated": terminated,
            "truncated": truncated,
        }
        return info

    def step(
        self,
        action: FloatArray,
        policy_context: dict[str, Any] | None = None,
        validate_with_rk4: bool | None = None,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(4)
        previous_pose = dict(self._pose)
        next_pose = self._clamp_and_slew(action_arr)
        self._pose = next_pose

        pose_delta_norm = float(
            np.linalg.norm(
                [
                    (next_pose["pitch"] - previous_pose["pitch"]) / max(1.0, self.pitch_bound),
                    (next_pose["yaw"] - previous_pose["yaw"]) / max(1.0, self.yaw_bound),
                    (next_pose["roll"] - previous_pose["roll"]) / max(1.0, self.roll_bound),
                    (next_pose["z"] - previous_pose["z"]) / max(1e-6, self.z_max - self.z_min),
                ]
            )
        )

        do_validate = self._should_validate(self._step_idx) if validate_with_rk4 is None else bool(validate_with_rk4)
        evaluation = self._evaluate_state(validate_with_rk4=do_validate)
        self._last_discrepancy = {
            "T_operating": float(evaluation["discrepancy"]["T_operating"]),
            "eta": float(evaluation["discrepancy"]["eta"]),
        }
        reward_breakdown = self._reward_components(evaluation=evaluation, pose_delta_norm=pose_delta_norm)
        reward = float(reward_breakdown["total_reward"])

        self._last_action = action_arr
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_steps

        decision_reason = self._decision_reason(
            previous_pose=previous_pose,
            next_pose=next_pose,
            evaluation=evaluation,
            reward_breakdown=reward_breakdown,
        )
        info = self._build_info(
            action=action_arr,
            evaluation=evaluation,
            reward_breakdown=reward_breakdown,
            policy_context=policy_context or {},
            decision_reason=decision_reason,
            terminated=terminated,
            truncated=truncated,
        )
        self._advance_time()
        obs = self._build_observation()
        return obs, reward, terminated, truncated, info
