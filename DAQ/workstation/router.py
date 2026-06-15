"""workstation/router.py – RL-inspired fidelity routing for Fortran solver tiers.

Mirrors the Q-table routing logic from simv4: given the current sensor state
(rate-of-change of irradiance, wind speed, temperature spread, and a budget
flag) choose the cheapest Fortran solver that is physically adequate.

Tier definitions
----------------
simv1b  – fast SAPM (Single-diode / Sandia Array Performance Model).
          Suitable for clear, steady-state conditions.
simv2b  – Prilliman IIR transient model.
          Suitable for moderate transients or partial cloud cover.
simv3b  – Fuentes RK4 thermal model.
          Required when irradiance is ramping fast (cloud ramp events).
simv1   – HiFi BTE + adjoint full solver.
          Reserved for high wind-speed transients when the budget flag
          allows the extra compute cost.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Routing thresholds (tunable at construction time)
# ---------------------------------------------------------------------------

# Rate-of-change thresholds (W m⁻² s⁻¹)
_G_ROC_MODERATE: float = 5.0
_G_ROC_HIGH: float = 20.0

# Wind-speed rate-of-change threshold (m s⁻¹ s⁻¹)
_WS_ROC_HIGH: float = 1.0

# Normalised temperature spread threshold (sigma / mean_T_amb)
_SIGMA_T_RATIO_HIGH: float = 0.05

# History window length (number of packets) used to compute rates of change.
_HISTORY_WINDOW: int = 10


class FidelityRouter:
    """Route incoming sensor packets to the appropriate Fortran solver tier.

    State features
    --------------
    G_roc         : rate-of-change of G_poa (W m⁻² s⁻¹)
    WS_roc        : rate-of-change of WS (m s⁻¹ s⁻¹)
    sigma_T_ratio : std(T_amb) / mean(T_amb) over the recent history window
    budget_flag   : True when the caller sets ``high_budget=True``

    Routing logic (mirrors simv4 Q-table)
    --------------------------------------
    1. High WS_roc **and** high_budget → ``simv1``  (HiFi BTE+adjoint)
    2. High G_roc  **or**  high sigma_T_ratio → ``simv3b`` (Fuentes RK4)
    3. Moderate G_roc → ``simv2b`` (Prilliman IIR)
    4. Else → ``simv1b`` (fast SAPM)

    Parameters
    ----------
    g_roc_moderate:
        G_poa rate-of-change threshold below which conditions are considered
        moderate rather than high (default: 5 W m⁻² s⁻¹).
    g_roc_high:
        G_poa rate-of-change threshold above which conditions are considered
        a cloud ramp (default: 20 W m⁻² s⁻¹).
    ws_roc_high:
        WS rate-of-change threshold for the HiFi tier (default: 1 m s⁻¹ s⁻¹).
    sigma_t_ratio_high:
        Normalised temperature spread threshold (default: 0.05).
    history_window:
        Number of recent packets used to compute rates of change
        (default: 10).
    """

    def __init__(
        self,
        *,
        g_roc_moderate: float = _G_ROC_MODERATE,
        g_roc_high: float = _G_ROC_HIGH,
        ws_roc_high: float = _WS_ROC_HIGH,
        sigma_t_ratio_high: float = _SIGMA_T_RATIO_HIGH,
        history_window: int = _HISTORY_WINDOW,
    ) -> None:
        self._g_roc_moderate = g_roc_moderate
        self._g_roc_high = g_roc_high
        self._ws_roc_high = ws_roc_high
        self._sigma_t_ratio_high = sigma_t_ratio_high
        self._history_window = history_window

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        packet: dict[str, Any],
        history: list[dict[str, Any]],
        *,
        high_budget: bool = False,
    ) -> str:
        """Return the solver tier name for the given packet and history.

        Parameters
        ----------
        packet:
            The current sensor packet (``PINN_SENSOR_PACKET_SCHEMA``).
        history:
            Ordered list of recent sensor packets (oldest first).  May be
            empty; the router degrades gracefully when history is short.
        high_budget:
            When ``True``, the HiFi BTE+adjoint tier (``simv1``) is eligible
            for high wind-speed transients.  Defaults to ``False``.

        Returns
        -------
        str
            One of ``'simv1b'``, ``'simv2b'``, ``'simv3b'``, ``'simv1'``.
        """
        features = self.compute_state_features(packet, history)
        return self._apply_routing_rules(features, high_budget=high_budget)

    def compute_state_features(
        self,
        packet: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute routing state features from the recent history window.

        Parameters
        ----------
        packet:
            The current (most recent) sensor packet.
        history:
            Ordered list of recent packets (oldest first).

        Returns
        -------
        dict
            Keys: ``G_roc``, ``WS_roc``, ``sigma_T_ratio``.
        """
        window = (history[-self._history_window :] if history else []) + [packet]

        g_roc = self._rate_of_change(window, "G_poa")
        ws_roc = self._rate_of_change(window, "WS")
        sigma_t_ratio = self._sigma_ratio(window, "T_amb")

        return {
            "G_roc": g_roc,
            "WS_roc": ws_roc,
            "sigma_T_ratio": sigma_t_ratio,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_routing_rules(
        self,
        features: dict[str, float],
        *,
        high_budget: bool = False,
    ) -> str:
        """Apply the four-tier routing rules and return the tier name."""
        g_roc = features["G_roc"]
        ws_roc = features["WS_roc"]
        sigma_t = features["sigma_T_ratio"]

        # Rule 1: HiFi tier — high wind transient + budget available
        if ws_roc >= self._ws_roc_high and high_budget:
            return "simv1"

        # Rule 2: Fuentes RK4 — cloud ramp or high temperature variance
        if g_roc >= self._g_roc_high or sigma_t >= self._sigma_t_ratio_high:
            return "simv3b"

        # Rule 3: Prilliman IIR — moderate transient
        if g_roc >= self._g_roc_moderate:
            return "simv2b"

        # Rule 4: Fast SAPM — clear steady-state
        return "simv1b"

    @staticmethod
    def _rate_of_change(
        window: list[dict[str, Any]],
        field: str,
    ) -> float:
        """Estimate the mean absolute rate-of-change of *field* over *window*.

        Uses a first-difference approach.  Returns 0.0 when the window is
        too short or the field is missing.
        """
        values = [
            pkt[field]
            for pkt in window
            if field in pkt and pkt[field] is not None
        ]
        if len(values) < 2:
            return 0.0
        diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        return sum(diffs) / len(diffs)

    @staticmethod
    def _sigma_ratio(
        window: list[dict[str, Any]],
        field: str,
    ) -> float:
        """Compute std(field) / |mean(field)| over *window*.

        Returns 0.0 when the window is empty, the mean is zero, or the field
        is missing from all packets.
        """
        values = [
            pkt[field]
            for pkt in window
            if field in pkt and pkt[field] is not None
        ]
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        if mean == 0.0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance) / abs(mean)
