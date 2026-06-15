"""Physics-derived complexity scoring for route-label generation.

Design doc §2.1 defines a 5-feature complexity score computed from a
window of :class:`SensorPacket` objects. The score is a pure function
of the packet sequence — no hidden state, no learned parameters — and
maps monotonically to the AURA-MFP simulation tiers via the thresholds
stored in :class:`~src.config.RoutingConfig`.

Public API
----------

* :func:`complexity_score` — ``list[SensorPacket] → float in [0, 1]``.
* :func:`score_to_route_label` — ``float → int route index``.
* :func:`extract_features` — per-feature breakdown, useful for analysis.

The features are, in order:

1. **Irradiance rate of change**: ``|ΔG_poa| / Δt`` over the window,
   normalized by 500 W/m²/s and clipped to ``[0, 1]``.
2. **Cloud cover**: ``CC`` of the most recent packet, already in ``[0, 1]``.
3. **Wind speed**: ``WS / 15`` clipped to ``[0, 1]``.
4. **Thermal lag**: ``|T_panel_measured - T_faiman_ss| / 10 K`` clipped
   to ``[0, 1]``. Uses the most recent packet's ambient temperature
   together with the Faiman steady-state formula to estimate ``T_ss``.
   If no panel temperature is available on the packet, the feature is
   zero.
5. **Physics residual magnitude**: ``|τ_eff · dT/dt − (T_ss − T)| / 5.0``
   clipped to ``[0, 1]``, using finite differences on the temperature
   series. Requires at least two packets with panel temperature data.

The five features are combined as a weighted mean, with weights from
:class:`~src.config.RoutingConfig` (default equal weights). The result
is a scalar in ``[0, 1]``.

Threshold → route-index mapping (see :class:`~src.config.RoutingConfig`)
-----------------------------------------------------------------------

``score < 0.05``          → 0  (LOFI)
``0.05 ≤ score < 0.10``   → 1  (SIMV2)
``0.10 ≤ score < 0.16``   → 4  (SIMV4)  — middle tier per design doc §2.3
``0.16 ≤ score < 0.24``   → 2  (SIMV3)
``score ≥ 0.24``          → 3  (SIMV1)

The non-monotonic index order is intentional: it matches the
``("LOFI", "SIMV2", "SIMV3", "SIMV1", "SIMV4")`` order of
:attr:`~src.config.ModelConfig.route_labels`, which in turn matches
AURA-MFP's ``probs[5]`` schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..config import PhysicsConfig, RoutingConfig
from .data import SensorPacket

__all__ = [
    "ComplexityFeatures",
    "complexity_score",
    "extract_features",
    "score_to_route_label",
]


# Feature normalization constants (design doc §2.1).
_DG_DT_NORM: float = 500.0   # W/m²/s
_WS_NORM: float = 15.0       # m/s
_THERMAL_LAG_NORM: float = 10.0  # K
_RESIDUAL_NORM: float = 5.0


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComplexityFeatures:
    """Per-feature breakdown of a complexity-score computation.

    All fields are scalars in ``[0, 1]``.
    """

    dG_dt: float
    cloud_cover: float
    wind_speed: float
    thermal_lag: float
    physics_residual: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    """Clip ``x`` into ``[0, 1]``."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _packet_field(packet: Any, name: str, default: float = 0.0) -> float:
    """Read a scalar field from a :class:`SensorPacket` or dict-like."""
    if isinstance(packet, SensorPacket):
        val = getattr(packet, name, default)
    elif isinstance(packet, dict):
        val = packet.get(name, default)
    else:
        val = getattr(packet, name, default)
    try:
        return float(val) if val is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _panel_temperature(packet: Any) -> float | None:
    """Best-effort read of a measured panel temperature from ``packet``.

    :class:`SensorPacket` does not define a ``T_panel`` attribute but the
    synthetic dataset record dicts and some live drivers attach one.
    Returns ``None`` when the panel temperature is unknown.
    """
    if isinstance(packet, SensorPacket):
        return None
    if isinstance(packet, dict):
        if "T_panel" in packet and packet["T_panel"] is not None:
            try:
                return float(packet["T_panel"])
            except (TypeError, ValueError):
                return None
        return None
    t_panel = getattr(packet, "T_panel", None)
    if t_panel is None:
        return None
    try:
        return float(t_panel)
    except (TypeError, ValueError):
        return None


def _faiman_ss_celsius(
    T_amb_C: float,
    G_poa: float,
    WS: float,
    CC: float,
    physics_cfg: PhysicsConfig,
) -> float:
    """Faiman steady-state panel temperature in °C (scalar, pure Python).

    Mirrors :func:`src.pinn.physics.faiman_steady_state` but operates on
    plain floats and applies the cloud-cover derate from design doc §3.5
    with ``gamma_CC = 1`` (the physics config default). Using the live
    ``gamma_CC_init`` instead of the learned value is intentional: the
    complexity score is a pure function of packet data.
    """
    gamma = float(physics_cfg.gamma_CC_init)
    U0 = float(physics_cfg.U0_init)
    U1 = float(physics_cfg.U1_init)
    cc = max(0.0, min(1.0, CC))
    if cc <= 0.0:
        cc_pow = 0.0
    else:
        cc_pow = cc ** gamma
    g_eff = max(0.0, G_poa) * (1.0 - cc_pow)
    denom = U0 + U1 * max(0.0, WS)
    if denom <= 0.0:
        return T_amb_C
    return T_amb_C + g_eff / denom


def _tau_eff(WS: float, physics_cfg: PhysicsConfig) -> float:
    """Wind-adjusted time constant in seconds (scalar)."""
    U0 = float(physics_cfg.U0_init)
    U1 = float(physics_cfg.U1_init)
    tau0 = float(physics_cfg.tau_0_init)
    denom = U0 + U1 * max(0.0, WS)
    if denom <= 0.0:
        return tau0
    return tau0 * U0 / denom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    packet_window: Sequence[Any],
    physics_cfg: PhysicsConfig | None = None,
) -> ComplexityFeatures:
    """Extract the five complexity features from ``packet_window``.

    ``packet_window`` must contain at least one element; single-packet
    windows produce zero for all time-derivative-dependent features.
    Packets may be :class:`SensorPacket` instances or dicts with the
    same field names; panel temperature is optional.

    Args:
        packet_window: Chronologically ordered list of packets (oldest
            first, newest last).
        physics_cfg: Physics configuration used to evaluate Faiman
            steady-state and ``τ_eff``. Defaults to
            :class:`PhysicsConfig`'s defaults.

    Returns:
        A :class:`ComplexityFeatures` dataclass with each field clipped
        to ``[0, 1]``.

    Raises:
        ValueError: If ``packet_window`` is empty.
    """
    if len(packet_window) == 0:
        raise ValueError("packet_window must contain at least one packet")

    if physics_cfg is None:
        physics_cfg = PhysicsConfig()

    newest = packet_window[-1]
    oldest = packet_window[0]

    # --- Feature 1: |ΔG_poa| / Δt, normalized by 500 W/m²/s ----------
    g_new = _packet_field(newest, "G_poa")
    g_old = _packet_field(oldest, "G_poa")
    t_new = _packet_field(newest, "t_s")
    t_old = _packet_field(oldest, "t_s")
    dt = t_new - t_old
    if dt <= 0.0 or len(packet_window) < 2:
        dg_dt = 0.0
    else:
        dg_dt = abs(g_new - g_old) / dt / _DG_DT_NORM
    f_dG_dt = _clip01(dg_dt)

    # --- Feature 2: CC directly ---------------------------------------
    f_cc = _clip01(_packet_field(newest, "CC"))

    # --- Feature 3: WS / 15 clipped -----------------------------------
    ws_new = _packet_field(newest, "WS")
    f_ws = _clip01(ws_new / _WS_NORM)

    # --- Feature 4: thermal lag |T_panel - T_ss| / 10 K ---------------
    t_panel_meas = _panel_temperature(newest)
    t_amb_new = _packet_field(newest, "T_amb")
    if t_panel_meas is None:
        f_lag = 0.0
    else:
        t_ss_new = _faiman_ss_celsius(
            t_amb_new, g_new, ws_new, f_cc, physics_cfg
        )
        f_lag = _clip01(abs(t_panel_meas - t_ss_new) / _THERMAL_LAG_NORM)

    # --- Feature 5: physics residual magnitude ------------------------
    # Requires at least two measurements of panel temperature plus a
    # positive Δt. When unavailable, the feature is zero.
    if len(packet_window) < 2:
        f_res = 0.0
    else:
        t_panel_new = t_panel_meas
        t_panel_old = _panel_temperature(oldest)
        if (
            t_panel_new is None
            or t_panel_old is None
            or dt <= 0.0
        ):
            f_res = 0.0
        else:
            dT_dt = (t_panel_new - t_panel_old) / dt
            tau = _tau_eff(ws_new, physics_cfg)
            t_ss = _faiman_ss_celsius(
                t_amb_new, g_new, ws_new, f_cc, physics_cfg
            )
            residual = tau * dT_dt - (t_ss - t_panel_new)
            f_res = _clip01(abs(residual) / _RESIDUAL_NORM)

    return ComplexityFeatures(
        dG_dt=f_dG_dt,
        cloud_cover=f_cc,
        wind_speed=f_ws,
        thermal_lag=f_lag,
        physics_residual=f_res,
    )


def complexity_score(
    packet_window: Sequence[Any],
    routing_cfg: RoutingConfig | None = None,
    physics_cfg: PhysicsConfig | None = None,
) -> float:
    """Compute the design-doc §2.1 complexity score for a packet window.

    The score is the weighted mean of the five features produced by
    :func:`extract_features`, using the per-feature weights stored in
    ``routing_cfg`` (default :class:`RoutingConfig` → equal weights).

    The output is deterministic, pure (no hidden state, no learned
    parameters), and always lives in ``[0, 1]``.

    Args:
        packet_window: Chronologically ordered list of packets (at
            least one element).
        routing_cfg: Routing configuration providing feature weights.
            Defaults to :class:`RoutingConfig`.
        physics_cfg: Physics configuration for Faiman steady-state and
            ``τ_eff``. Defaults to :class:`PhysicsConfig`.

    Returns:
        Scalar complexity score in ``[0, 1]``.

    Raises:
        ValueError: If ``packet_window`` is empty or all feature weights
            sum to zero.
    """
    if routing_cfg is None:
        routing_cfg = RoutingConfig()
    feats = extract_features(packet_window, physics_cfg=physics_cfg)

    weights = (
        float(routing_cfg.w_dG_dt),
        float(routing_cfg.w_cloud_cover),
        float(routing_cfg.w_wind_speed),
        float(routing_cfg.w_thermal_lag),
        float(routing_cfg.w_physics_residual),
    )
    if any(w < 0 for w in weights):
        raise ValueError(f"feature weights must be non-negative, got {weights}")
    w_sum = sum(weights)
    if w_sum <= 0.0:
        raise ValueError("feature weights sum to zero; cannot form weighted mean")

    values = (
        feats.dG_dt,
        feats.cloud_cover,
        feats.wind_speed,
        feats.thermal_lag,
        feats.physics_residual,
    )
    score = sum(w * v for w, v in zip(weights, values)) / w_sum
    return _clip01(score)


def score_to_route_label(
    score: float,
    routing_cfg: RoutingConfig | None = None,
) -> int:
    """Map a complexity score to an integer route index.

    The mapping uses the thresholds stored in :class:`RoutingConfig` and
    produces indices consistent with :attr:`ModelConfig.route_labels`
    ``("LOFI", "SIMV2", "SIMV3", "SIMV1", "SIMV4")``:

    ``score < t_lofi``          → 0  (LOFI)
    ``t_lofi ≤ score < t_sim2`` → 1  (SIMV2)
    ``t_sim2 ≤ score < t_sim4`` → 4  (SIMV4)  — middle tier per §2.3
    ``t_sim4 ≤ score < t_sim3`` → 2  (SIMV3)
    ``score ≥ t_sim3``          → 3  (SIMV1)

    The index order is non-monotonic on purpose; it matches AURA-MFP's
    ``probs[5]`` label order.

    Args:
        score: Complexity score, typically from :func:`complexity_score`.
            Values outside ``[0, 1]`` are clipped before dispatch.
        routing_cfg: Thresholds. Defaults to :class:`RoutingConfig`.

    Returns:
        Integer route index in ``[0, 4]``.
    """
    if routing_cfg is None:
        routing_cfg = RoutingConfig()
    s = _clip01(float(score))
    if s < routing_cfg.threshold_lofi:
        return 0  # LOFI
    if s < routing_cfg.threshold_simv2:
        return 1  # SIMV2
    if s < routing_cfg.threshold_simv4:
        return 4  # SIMV4 (middle tier — see class docstring)
    if s < routing_cfg.threshold_simv3:
        return 2  # SIMV3
    return 3  # SIMV1
