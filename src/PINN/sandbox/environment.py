"""Phase 2 Panel Environment for RL control of PV array.

Simplified environment that interfaces with PINN agent for predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class EpisodeConditions:
    """Domain-randomized environmental conditions for one episode."""

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


def _solar_declination_rad(day_of_year: int) -> float:
    """Solar declination in radians."""
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
        dtype=np.float32,
    )


def _panel_normal(pitch_deg: float, yaw_deg: float, roll_deg: float) -> FloatArray:
    """Panel normal vector after yaw(z)-pitch(x)-roll(y) rotations."""
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
    nrm = np.linalg.norm(normal)
    if nrm == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (normal / nrm).astype(np.float32)


class PanelEnv:
    """Simple 4-DOF panel control environment for Phase 2 sandbox training.
    
    Interfaces with PINN agent for temperature and efficiency predictions.
    """

    def __init__(
        self,
        pinn_agent,
        seed: int = 42,
        dt_s: float = 1.0,
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
        capture_scale: float = 1.0e-3,
        temp_scale: float = 1.0,
        pose_change_penalty: float = 1.0e-3,
    ) -> None:
        """Initialize panel environment.

        Args:
            pinn_agent: SandboxPINNAgent for prediction
            seed: Random seed
            dt_s: Time step (seconds)
            pitch_bound: Max pitch angle (degrees)
            yaw_bound: Max yaw angle (degrees)
            roll_bound: Max roll angle (degrees)
            z_min: Min height (m)
            z_max: Max height (m)
            max_pitch_rate: Max pitch rate (deg/s)
            max_yaw_rate: Max yaw rate (deg/s)
            max_roll_rate: Max roll rate (deg/s)
            max_z_rate: Max height rate (m/s)
            t_ref_k: Reference temperature (K)
            temp_margin_k: Temperature safety margin (K)
            reward_w_capture: Capture weight
            reward_w_temp: Temperature weight
            capture_scale: Irradiance scaling
            temp_scale: Temperature scaling
            pose_change_penalty: Action smoothness penalty
        """
        self.pinn_agent = pinn_agent
        self._rng = np.random.default_rng(seed)
        self._step_idx = 0

        # Control config
        self.dt_s = dt_s
        self.pitch_bound = pitch_bound
        self.yaw_bound = yaw_bound
        self.roll_bound = roll_bound
        self.z_min = z_min
        self.z_max = z_max
        self.max_pitch_rate = max_pitch_rate
        self.max_yaw_rate = max_yaw_rate
        self.max_roll_rate = max_roll_rate
        self.max_z_rate = max_z_rate

        # Physics config
        self.t_ref_k = t_ref_k
        self.temp_margin_k = temp_margin_k
        self.eta_ref = 0.18
        self.beta_pmax = -0.004

        # Reward config
        self.reward_w_capture = reward_w_capture
        self.reward_w_temp = reward_w_temp
        self.capture_scale = capture_scale
        self.temp_scale = temp_scale
        self.pose_change_penalty = pose_change_penalty

        # Initialize state
        self._pose = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "z": (z_min + z_max) * 0.5,
        }
        self._last_action = np.zeros(4, dtype=np.float32)
        self.conditions = self._sample_conditions()

    @property
    def observation_dim(self) -> int:
        """Observation space dimension."""
        return 14

    @property
    def action_dim(self) -> int:
        """Action space dimension."""
        return 4

    def _sample_conditions(self) -> EpisodeConditions:
        """Sample random episode conditions."""
        return EpisodeConditions(
            lat=float(self._rng.uniform(-65.0, 65.0)),
            lon=float(self._rng.uniform(-180.0, 180.0)),
            alt=float(self._rng.uniform(0.0, 2000.0)),
            day_of_year=int(self._rng.integers(1, 366)),
            hour=float(self._rng.uniform(5.0, 19.0)),
            minute=float(self._rng.uniform(0.0, 60.0)),
            month=int(self._rng.integers(1, 13)),
            year=2024,
            ambient_c=float(self._rng.uniform(-15.0, 45.0)),
            wind_mps=float(self._rng.uniform(0.0, 20.0)),
            wind_dir=float(self._rng.uniform(0.0, 360.0)),
            humidity=float(self._rng.uniform(0.0, 1.0)),
            irradiance=float(self._rng.uniform(500.0, 1100.0)),
            cloud_cover=float(self._rng.uniform(0.0, 1.0)),
            pressure=float(self._rng.uniform(80000.0, 105000.0)),
        )

    def reset(self, seed: int | None = None) -> tuple[FloatArray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed

        Returns:
            Observation and info dict
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.conditions = self._sample_conditions()
        self._step_idx = 0
        self._pose = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "z": (self.z_min + self.z_max) * 0.5,
        }
        self._last_action = np.zeros(4, dtype=np.float32)
        obs = self._build_observation()
        return obs, {"conditions": self.conditions}

    def _clamp_and_slew(self, action: FloatArray) -> dict[str, float]:
        """Clamp actions to bounds and apply rate limiting.

        Args:
            action: Raw action [pitch, yaw, roll, z]

        Returns:
            Next pose dict
        """
        action = np.asarray(action, dtype=np.float32).reshape(4)

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

        targets = {
            "pitch": float(np.clip(action[0], -1.0, 1.0) * self.pitch_bound),
            "yaw": float(np.clip(action[1], -1.0, 1.0) * self.yaw_bound),
            "roll": float(np.clip(action[2], -1.0, 1.0) * self.roll_bound),
            "z": float(
                self.z_min
                + 0.5 * (np.clip(action[3], -1.0, 1.0) + 1.0) * (self.z_max - self.z_min)
            ),
        }

        next_pose: dict[str, float] = {}
        for k in ("pitch", "yaw", "roll", "z"):
            delta = targets[k] - self._pose[k]
            max_step = rate_limits[k]
            delta = float(np.clip(delta, -max_step, max_step))
            lo, hi = ranges[k]
            next_pose[k] = float(np.clip(self._pose[k] + delta, lo, hi))

        return next_pose

    def _build_observation(self) -> FloatArray:
        """Build observation from conditions and pose.

        Returns:
            Observation array [14]
        """
        c = self.conditions
        sun = _sun_vector(c.lat, c.day_of_year, c.hour)

        obs = np.array(
            [
                c.lat / 90.0,
                c.lon / 180.0,
                c.day_of_year / 365.0,
                c.hour / 24.0,
                c.ambient_c / 50.0,
                c.wind_mps / 20.0,
                c.cloud_cover,
                c.irradiance / 1200.0,
                sun[0],
                sun[1],
                sun[2],
                self._pose["pitch"] / self.pitch_bound,
                self._pose["yaw"] / self.yaw_bound,
                self._pose["z"] / self.z_max,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_pinn_prediction(self) -> dict[str, float]:
        """Get PINN prediction for current conditions.

        Returns:
            Dict with T_operating and eta
        """
        c = self.conditions
        try:
            # Build input tensors for PINN
            weather = torch.tensor(
                [
                    c.ambient_c + 273.15,  # T_amb [K]
                    c.wind_mps,  # wind_speed [m/s]
                    c.wind_dir,  # wind_dir [deg]
                    c.humidity,  # humidity [frac]
                    c.irradiance,  # irradiance [W/m²]
                    c.cloud_cover,  # cloud_cover [frac]
                    c.pressure,  # pressure [Pa]
                ],
                dtype=torch.float32,
            )

            panel_state = torch.tensor(
                [
                    self._pose["z"],  # pv_height [m]
                    self._pose["pitch"],  # pitch [deg]
                    self._pose["roll"],  # roll [deg]
                    self._pose["yaw"],  # yaw [deg]
                ],
                dtype=torch.float32,
            )

            location = torch.tensor(
                [
                    c.lat,
                    c.lon,
                    c.alt,
                ],
                dtype=torch.float32,
            )

            time = torch.tensor(
                [
                    c.minute,
                    c.hour,
                    c.day_of_year,
                    c.month,
                ],
                dtype=torch.float32,
            )

            # Get PINN prediction
            pred = self.pinn_agent.predict(
                weather=weather,
                panel_state=panel_state,
                location=location,
                time=time,
                include_rk4=False,  # Don't need RK4 during episode
            )

            pinn_out = pred.get("pinn", {})
            return {
                "T_operating": float(pinn_out.get("T_operating", 45.0)),
                "eta": float(pinn_out.get("eta", 0.18)),
            }

        except Exception as e:
            print(f"PINN prediction failed: {e}")
            return {"T_operating": 45.0, "eta": 0.18}

    def _reward(self, pinn_pred: dict[str, float], action_change_norm: float) -> float:
        """Compute reward with multi-component shaping.

        Args:
            pinn_pred: PINN prediction dict
            action_change_norm: L2 norm of action change

        Returns:
            Reward scalar
        """
        c = self.conditions

        # ===== Irradiance Capture Component =====
        sun = _sun_vector(c.lat, c.day_of_year, c.hour)
        normal = _panel_normal(self._pose["pitch"], self._pose["yaw"], self._pose["roll"])
        incidence = float(np.clip(float(np.dot(sun, normal)), 0.0, 1.0))
        captured = c.irradiance * incidence

        # Efficiency bonus (encourages high conversion)
        eta_efficiency = pinn_pred["eta"]
        capture_reward = self.reward_w_capture * self.capture_scale * captured * eta_efficiency

        # ===== Temperature Control Component =====
        over_temp = max(
            0.0,
            pinn_pred["T_operating"] - (self.t_ref_k + self.temp_margin_k),
        )
        temp_penalty = self.reward_w_temp * self.temp_scale * over_temp

        # ===== Action Smoothness Component =====
        # Penalize sudden movements to encourage smooth trajectories
        action_penalty = self.pose_change_penalty * action_change_norm

        # ===== Stability Bonus =====
        # Small reward for maintaining pose (when not needed to move)
        if action_change_norm < 0.1:
            stability_bonus = 0.01
        else:
            stability_bonus = 0.0

        total_reward = capture_reward - temp_penalty - action_penalty + stability_bonus

        return float(total_reward)

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action array [pitch_cmd, yaw_cmd, roll_cmd, z_cmd]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_arr = np.asarray(action, dtype=np.float32).reshape(4)
        action_change_norm = float(np.linalg.norm(action_arr - self._last_action))

        # Apply action
        next_pose = self._clamp_and_slew(action)
        self._pose = next_pose

        # Get PINN prediction
        pinn_pred = self._get_pinn_prediction()

        # Compute reward
        reward = self._reward(pinn_pred, action_change_norm)

        # Update state
        self._last_action = action_arr
        self._step_idx += 1

        # Build observation
        obs = self._build_observation()

        # Episode termination (no early stopping for now)
        terminated = False
        truncated = False

        info = {
            "conditions": self.conditions,
            "pinn_prediction": pinn_pred,
            "reward_components": {
                "captured_irradiance": float(
                    np.clip(float(np.dot(_sun_vector(self.conditions.lat, self.conditions.day_of_year, self.conditions.hour), 
                                                     _panel_normal(self._pose["pitch"], self._pose["yaw"], self._pose["roll"]))), 0.0, 1.0)
                    * self.conditions.irradiance
                ),
                "eta": pinn_pred["eta"],
                "T_operating": pinn_pred["T_operating"],
            },
        }

        return obs, reward, terminated, truncated, info
