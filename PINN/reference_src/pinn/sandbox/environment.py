"""Physics-based 3D sandbox environment for phase-2 policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ...config import AppConfig
from ..physics import (
    celsius_to_kelvin,
    diurnal_irradiance,
    effective_irradiance,
    faiman_steady_state,
    sandia_efficiency,
)

_POSE_KEYS: tuple[str, ...] = ("pitch", "yaw", "roll", "z")
FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class EpisodeConditions:
    """Domain-randomized environmental conditions for one episode."""

    lat: float
    lon: float
    day_of_year: int
    hour: float
    ambient_c: float
    wind_mps: float
    cloud_cover: float
    g_peak: float


def _solar_declination_rad(day_of_year: int) -> float:
    return math.radians(23.44) * math.sin(2.0 * math.pi * (day_of_year - 81) / 365.0)


def _sun_vector(lat_deg: float, day_of_year: int, local_hour: float) -> FloatArray:
    """Approximate sun direction in local ENU coordinates."""
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
        dtype=np.float64,
    )


def _panel_normal(pitch_deg: float, yaw_deg: float, roll_deg: float) -> FloatArray:
    """Return panel normal vector after yaw(z)-pitch(x)-roll(y) rotations."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [math.cos(roll), 0.0, math.sin(roll)],
            [0.0, 1.0, 0.0],
            [-math.sin(roll), 0.0, math.cos(roll)],
        ],
        dtype=np.float64,
    )
    normal = rz @ rx @ ry @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nrm = np.linalg.norm(normal)
    if nrm == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / nrm


class PanelEnv:
    """Gymnasium-style environment for 4-DOF panel control."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.sandbox.seed)
        self._step_idx = 0
        self._pose = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "z": (cfg.control.z_min + cfg.control.z_max) * 0.5,
        }
        self.conditions = self._sample_conditions()
        self._last_action = np.zeros(4, dtype=np.float32)

    @property
    def observation_dim(self) -> int:
        return 14

    @property
    def action_dim(self) -> int:
        return 4

    def _sample_conditions(self) -> EpisodeConditions:
        s = self.cfg.sandbox
        return EpisodeConditions(
            lat=float(self._rng.uniform(s.lat_min, s.lat_max)),
            lon=float(self._rng.uniform(s.lon_min, s.lon_max)),
            day_of_year=int(self._rng.integers(s.day_min, s.day_max + 1)),
            hour=float(self._rng.uniform(s.hour_min, s.hour_max)),
            ambient_c=float(self._rng.uniform(s.ambient_c_min, s.ambient_c_max)),
            wind_mps=float(self._rng.uniform(s.wind_min, s.wind_max)),
            cloud_cover=float(self._rng.uniform(s.cloud_min, s.cloud_max)),
            g_peak=float(self._rng.uniform(s.g_peak_min, s.g_peak_max)),
        )

    def reset(self, seed: int | None = None) -> tuple[FloatArray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.conditions = self._sample_conditions()
        self._step_idx = 0
        self._pose = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "z": (self.cfg.control.z_min + self.cfg.control.z_max) * 0.5,
        }
        self._last_action = np.zeros(4, dtype=np.float32)
        obs = self._build_observation()
        return obs, {"conditions": self.conditions}

    def _clamp_and_slew(self, action: FloatArray) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(4)

        ctl = self.cfg.control
        dt = self.cfg.sandbox.dt_s
        rate_limits = {
            "pitch": ctl.max_pitch_rate * dt,
            "yaw": ctl.max_yaw_rate * dt,
            "roll": ctl.max_roll_rate * dt,
            "z": ctl.max_z_rate * dt,
        }
        ranges = {
            "pitch": (-ctl.pitch_bound, ctl.pitch_bound),
            "yaw": (-ctl.yaw_bound, ctl.yaw_bound),
            "roll": (-ctl.roll_bound, ctl.roll_bound),
            "z": (ctl.z_min, ctl.z_max),
        }
        targets = {
            "pitch": float(np.clip(action[0], -1.0, 1.0) * ctl.pitch_bound),
            "yaw": float(np.clip(action[1], -1.0, 1.0) * ctl.yaw_bound),
            "roll": float(np.clip(action[2], -1.0, 1.0) * ctl.roll_bound),
            "z": float(ctl.z_min + 0.5 * (np.clip(action[3], -1.0, 1.0) + 1.0) * (ctl.z_max - ctl.z_min)),
        }
        next_pose: dict[str, float] = {}
        for k in _POSE_KEYS:
            delta = targets[k] - self._pose[k]
            max_step = rate_limits[k]
            delta = float(np.clip(delta, -max_step, max_step))
            lo, hi = ranges[k]
            next_pose[k] = float(np.clip(self._pose[k] + delta, lo, hi))
        return next_pose

    def _physics_eval(self, pose: dict[str, float]) -> dict[str, float]:
        c = self.conditions
        t_h = torch.tensor(c.hour, dtype=torch.float32)
        t_noon = torch.tensor(12.0, dtype=torch.float32)
        daylen = torch.tensor(12.0, dtype=torch.float32)
        g_peak = torch.tensor(c.g_peak, dtype=torch.float32)
        g_poa = diurnal_irradiance(t_h=t_h, t_noon=t_noon, daylen=daylen, G_peak=g_peak)

        cc = torch.tensor(c.cloud_cover, dtype=torch.float32)
        gamma_cc = torch.tensor(self.cfg.physics.gamma_CC_init, dtype=torch.float32)
        g_eff = effective_irradiance(
            G_poa=g_poa,
            CC=cc,
            gamma_CC=gamma_cc,
            M_spectral=torch.tensor(1.0, dtype=torch.float32),
        )
        sun = _sun_vector(c.lat, c.day_of_year, c.hour)
        normal = _panel_normal(pose["pitch"], pose["yaw"], pose["roll"])
        incidence = float(np.clip(float(np.dot(sun, normal)), 0.0, 1.0))
        captured = float(g_eff.item()) * incidence

        t_amb_k = celsius_to_kelvin(torch.tensor(c.ambient_c, dtype=torch.float32))
        t_panel_k = faiman_steady_state(
            T_amb_K=t_amb_k,
            G_eff=torch.tensor(captured, dtype=torch.float32),
            U0=torch.tensor(self.cfg.physics.U0_init, dtype=torch.float32),
            U1=torch.tensor(self.cfg.physics.U1_init, dtype=torch.float32),
            WS=torch.tensor(c.wind_mps, dtype=torch.float32),
        )
        eta = sandia_efficiency(
            T_K=t_panel_k,
            eta_ref=torch.tensor(self.cfg.physics.eta_ref, dtype=torch.float32),
            beta_Pmax=torch.tensor(self.cfg.physics.beta_Pmax, dtype=torch.float32),
            T_ref_K=torch.tensor(self.cfg.physics.T_ref_K, dtype=torch.float32),
        )
        return {
            "G_poa": float(g_poa.item()),
            "G_eff": float(g_eff.item()),
            "incidence": incidence,
            "captured_irradiance": captured,
            "T_panel_K": float(t_panel_k.item()),
            "eta": float(eta.item()),
        }

    def _reward(self, metrics: dict[str, float], action_change_norm: float) -> float:
        s = self.cfg.sandbox
        over_temp = max(
            0.0,
            metrics["T_panel_K"] - (self.cfg.physics.T_ref_K + s.reward_temp_margin_K),
        )
        return (
            s.reward_w_capture * s.reward_capture_scale * metrics["captured_irradiance"] * metrics["eta"]
            - s.reward_w_temp * s.reward_temp_scale * over_temp
            - s.pose_change_penalty * action_change_norm
        )

    def _build_observation(self) -> FloatArray:
        c = self.conditions
        m = self._physics_eval(self._pose)
        sun = _sun_vector(c.lat, c.day_of_year, c.hour)
        obs = np.array(
            [
                c.lat / 90.0,
                c.lon / 180.0,
                c.day_of_year / 365.0,
                c.hour / 24.0,
                c.ambient_c / 50.0,
                c.wind_mps / max(1.0, self.cfg.sandbox.wind_max),
                c.cloud_cover,
                m["G_poa"] / 1200.0,
                m["G_eff"] / 1200.0,
                sun[0],
                sun[1],
                sun[2],
                m["incidence"],
                self._pose["z"] / max(1e-6, self.cfg.control.z_max),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(4)
        action_change_norm = float(np.linalg.norm(action_arr - self._last_action))
        next_pose = self._clamp_and_slew(action)
        self._pose = next_pose
        metrics = self._physics_eval(self._pose)
        reward = self._reward(metrics, action_change_norm)
        self._last_action = action_arr
        self._step_idx += 1
        obs = self._build_observation()
        terminated = False
        truncated = self._step_idx >= int(self.cfg.sandbox.episode_steps)
        info: dict[str, Any] = {
            "pose": dict(self._pose),
            "metrics": metrics,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        }
        return obs, float(reward), terminated, truncated, info
