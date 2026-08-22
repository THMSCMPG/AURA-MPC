"""Central configuration for PINN-AURA-MFP.

All configurable parameter groups live here as frozen dataclasses. Later
PRs consume these names; do not redefine them elsewhere. Field names and
ordering (notably ``ModelConfig.route_labels``) match the AURA-MFP JSON
interface contract exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProductContract:
    """Operational contract for decision cadence and prediction horizons.

    Attributes:
        decision_cadence_s: Nominal cadence between decisions.
        min_cadence_s: Lower bound on decision cadence.
        max_cadence_s: Upper bound on decision cadence.
        min_horizon_s: Lower bound on prediction horizon.
        max_horizon_s: Upper bound on prediction horizon.
        uncertainty_watchdog: Probability threshold above which the
            watchdog flags the inference as low-confidence.
        max_consecutive_faults: Consecutive ingest faults tolerated before
            the watchdog trips.
        stale_data_threshold_s: Maximum age of the newest packet before it
            is treated as stale.
        max_missing_fraction: Maximum fraction of missing samples allowed
            in a rolling window.
    """

    decision_cadence_s: float = 3.0
    min_cadence_s: float = 1.0
    max_cadence_s: float = 5.0
    min_horizon_s: float = 10.0
    max_horizon_s: float = 1800.0
    uncertainty_watchdog: float = 0.55
    max_consecutive_faults: int = 3
    stale_data_threshold_s: float = 3.0
    max_missing_fraction: float = 0.3


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Dual-head multimodal architecture settings.

    Attributes:
        num_numeric_features: Number of numeric scalar features
            (``t_s, G_poa, T_amb, WS, CC, lat, lon``).
        image_embed_dim: Output dimensionality of the sky-image encoder.
        hidden_dim: Width of the shared trunk.
        num_residual_blocks: Depth of the shared trunk.
        route_hidden: Width of the routing head.
        num_routes: Number of simulation routes.
        route_labels: Route names. Order **must** match AURA-MFP's
            ``probs[5]`` schema.
    """

    num_numeric_features: int = 7
    image_embed_dim: int = 32
    hidden_dim: int = 128
    num_residual_blocks: int = 5
    route_hidden: int = 64
    pose_hidden: int = 64  # width of the pose regression head
    num_routes: int = 5
    route_labels: tuple[str, ...] = ("LOFI", "SIMV2", "SIMV3", "SIMV1", "SIMV4")


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    """Initial values for the learnable extended-thermal physics parameters.

    Attributes:
        tau_0_init: Initial thermal time constant (s).
        U0_init: Initial free-convection coefficient (W/m²/K).
        U1_init: Initial wind-dependent coefficient (W·s/m³/K).
        gamma_CC_init: Initial cloud-cover exponent.
        T_ref_K: Reference temperature for efficiency model (K).
        eta_ref: Reference electrical efficiency at ``T_ref_K``.
        beta_Pmax: Power temperature coefficient (1/K).
    """

    tau_0_init: float = 600.0
    U0_init: float = 25.0
    U1_init: float = 6.84
    gamma_CC_init: float = 1.0
    T_ref_K: float = 298.15
    eta_ref: float = 0.18
    beta_Pmax: float = -0.004


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimisation settings for staged training and routing warm-up.

    Attributes:
        pretrain_epochs: Epochs for data-only pretraining.
        sim_loop_epochs: Epochs for the physics + sim-loop phase.
        finetune_epochs: Epochs for the final finetune phase.
        route_warmup_epochs: Epochs before the routing loss is activated.
        lr: Base learning rate.
        batch_size: Batch size.
        grad_clip: Max gradient-norm for clipping.
        val_split: Fraction of data held out for validation.
        collocation_points: Number of physics collocation points per
            epoch.
        lambda_data: Weight for the data loss term.
        lambda_phys: Weight for the physics residual loss term.
        lambda_IC: Weight for the initial-condition loss term.
        lambda_route: Weight for the routing-classification loss term.
        lr_scheduler_factor: ``ReduceLROnPlateau`` factor.
        lr_scheduler_patience: ``ReduceLROnPlateau`` patience.
    """

    pretrain_epochs: int = 400
    sim_loop_epochs: int = 800
    finetune_epochs: int = 800
    route_warmup_epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 256
    grad_clip: float = 1.0
    val_split: float = 0.15
    collocation_points: int = 4000
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_IC: float = 10.0
    lambda_route: float = 0.5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 250
    lambda_pose: float = 0.0  # 0 = pose head off; set > 0 once Phase 4 labels exist


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Hard bounds and slew-rate limits for panel actuation.

    Attributes:
        pitch_bound: Symmetric pitch bound (deg).
        yaw_bound: Symmetric yaw bound (deg).
        roll_bound: Symmetric roll bound (deg).
        z_min: Minimum panel height (m).
        z_max: Maximum panel height (m).
        max_pitch_rate: Max pitch slew (deg/s).
        max_yaw_rate: Max yaw slew (deg/s).
        max_roll_rate: Max roll slew (deg/s).
        max_z_rate: Max vertical slew (m/s).
    """

    pitch_bound: float = 35.0
    yaw_bound: float = 180.0
    roll_bound: float = 25.0
    z_min: float = 0.0
    z_max: float = 3.0
    max_pitch_rate: float = 3.5
    max_yaw_rate: float = 10.0
    max_roll_rate: float = 3.0
    max_z_rate: float = 0.2


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Physics-derived complexity-scoring thresholds and feature weights.

    The thresholds (0.05, 0.10, 0.16, 0.24) are the heuristic values from
    design doc §2.3; Batch G's CI calibrates them if real data becomes
    available. They partition the unit interval into five route buckets.

    The thresholds map to route *indices* (not names). Per design doc
    §2.3 the middle-complexity tier is SimV4, not SimV3, which is why the
    boundary-to-index table below looks non-monotonic in the index: the
    order of route labels in :attr:`ModelConfig.route_labels` is
    ``("LOFI", "SIMV2", "SIMV3", "SIMV1", "SIMV4")`` to match
    AURA-MFP's ``probs[5]`` schema.

    Threshold → index mapping
    -------------------------
    ``score < t_lofi``          → 0  (LOFI)
    ``t_lofi ≤ score < t_sim2`` → 1  (SIMV2)
    ``t_sim2 ≤ score < t_sim4`` → 4  (SIMV4)  — middle tier, index 4
    ``t_sim4 ≤ score < t_sim3`` → 2  (SIMV3)
    ``score ≥ t_sim3``          → 3  (SIMV1)

    Attributes:
        threshold_lofi: Below this score → LOFI (index 0).
        threshold_simv2: Upper bound of SIMV2 band.
        threshold_simv4: Upper bound of SIMV4 band (middle-tier per §2.3).
        threshold_simv3: Upper bound of SIMV3 band.
        w_dG_dt: Weight for irradiance rate-of-change feature.
        w_cloud_cover: Weight for cloud-cover feature.
        w_wind_speed: Weight for wind-speed feature.
        w_thermal_lag: Weight for thermal-lag feature.
        w_physics_residual: Weight for physics-residual feature.
    """

    threshold_lofi: float = 0.05
    threshold_simv2: float = 0.10
    threshold_simv4: float = 0.16
    threshold_simv3: float = 0.24

    w_dG_dt: float = 1.0
    w_cloud_cover: float = 1.0
    w_wind_speed: float = 1.0
    w_thermal_lag: float = 1.0
    w_physics_residual: float = 1.0


@dataclass(frozen=True, slots=True)
class ImagePipelineConfig:
    """Configuration for :class:`src.pinn.image_pipeline.SkyImagePipeline`.

    Every preprocessing step is independently toggleable. The defaults
    produce a ``(3, 32, 32)`` float32 tensor normalised with ImageNet
    statistics and are safe for any RGB sky camera input.

    Attributes:
        target_size: Output ``(H, W)`` for the image tensor.
        apply_circular_crop: If ``True``, mask pixels outside a centred
            circle (fisheye/all-sky cameras).
        circular_crop_radius_frac: Radius of the kept disc as a fraction
            of ``min(H, W) / 2``.
        apply_histogram_eq: If ``True``, equalise the HSV value channel
            to boost contrast while preserving hue.
        apply_sun_mask: If ``True``, detect and remove the sun disc.
        sun_detection_threshold: Percentile of brightness above which a
            pixel is considered sun-saturated.
        sun_mask_method: One of ``"median_inpaint"``, ``"black_fill"``,
            ``"none"``.
        add_rbr_channel: If ``True``, append a red/blue-ratio channel so
            the output is ``(4, H, W)``. Requires a wider image encoder.
        normalize: If ``True``, normalise with ``mean``/``std``; else the
            output is raw ``[0, 1]`` float32.
        mean: Per-channel mean for normalisation.
        std: Per-channel std for normalisation.
        interpolation: Resampling kernel — ``"area"`` (default) is the
            correct anti-aliased downscaler; ``"lanczos"`` preserves
            slightly more high-frequency detail but is ~2× slower.
    """

    target_size: tuple[int, int] = (32, 32)
    apply_circular_crop: bool = False
    circular_crop_radius_frac: float = 0.95
    apply_histogram_eq: bool = True
    apply_sun_mask: bool = True
    sun_detection_threshold: float = 0.95
    sun_mask_method: str = "median_inpaint"
    add_rbr_channel: bool = False
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "area"


@dataclass(frozen=True, slots=True)
class PathConfig:
    """File-system paths used by training and inference.

    ``aura_mfp_root`` defaults to the value of the ``AURA_MFP_ROOT``
    environment variable when set, else ``None``.

    Attributes:
        training_data_dir: Root of user-provided training data.
        checkpoints_dir: Destination for ``.pt`` checkpoints.
        logs_dir: Destination for JSON-line training logs.
        aura_mfp_root: Root of an AURA-MFP checkout for SimV1-V4 bindings.
    """

    training_data_dir: Path = Path("Training_Data")
    checkpoints_dir: Path = Path("checkpoints")
    logs_dir: Path = Path("logs")
    aura_mfp_root: Path | None = None


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Configuration for phase-2 sandbox RL training and live demo.

    Attributes:
        seed: Base RNG seed for deterministic domain randomization.
        dt_s: Environment step duration in seconds.
        episode_steps: Number of steps per episode.
        train_epochs: Number of policy-optimization epochs.
        episodes_per_epoch: Number of episodes rolled out per epoch.
        discount_gamma: Reward discount factor for returns.
        learning_rate: Policy optimizer learning rate.
        policy_hidden_dim: Hidden width of the policy MLP.
        policy_std: Fixed exploration std used by Gaussian policy.
        reward_w_capture: Weight for captured irradiance × efficiency term.
        reward_w_temp: Weight for over-temperature penalty term.
        reward_temp_margin_K: Margin above ``physics.T_ref_K`` before penalty.
        reward_capture_scale: Scalar to keep capture term numerically stable.
        reward_temp_scale: Scalar to keep temperature term numerically stable.
        lat_min/lat_max: Latitude randomization range in degrees.
        lon_min/lon_max: Longitude randomization range in degrees.
        day_min/day_max: Day-of-year randomization range.
        hour_min/hour_max: Local-hour randomization range.
        ambient_c_min/ambient_c_max: Ambient-temperature range in Celsius.
        wind_min/wind_max: Wind-speed range in m/s.
        cloud_min/cloud_max: Cloud-cover range in [0, 1].
        g_peak_min/g_peak_max: Peak irradiance randomization range in W/m².
        pose_change_penalty: Small action-change regularizer.
        command_log_path: JSONL path written by live mode.
        z_to_height_mm_max: Mapping target for ``z[m] -> height[mm]``.
        validation_samples: Number of sampled points for tier validation.
    """

    seed: int = 42
    dt_s: float = 1.0
    episode_steps: int = 64
    train_epochs: int = 40
    episodes_per_epoch: int = 8
    discount_gamma: float = 0.98
    learning_rate: float = 3e-4
    policy_hidden_dim: int = 128
    policy_std: float = 0.25
    reward_w_capture: float = 1.0
    reward_w_temp: float = 0.4
    reward_temp_margin_K: float = 0.0
    reward_capture_scale: float = 1.0e-3
    reward_temp_scale: float = 1.0
    lat_min: float = -65.0
    lat_max: float = 65.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    day_min: int = 1
    day_max: int = 365
    hour_min: float = 5.0
    hour_max: float = 19.0
    ambient_c_min: float = -15.0
    ambient_c_max: float = 45.0
    wind_min: float = 0.0
    wind_max: float = 20.0
    cloud_min: float = 0.0
    cloud_max: float = 1.0
    g_peak_min: float = 500.0
    g_peak_max: float = 1100.0
    pose_change_penalty: float = 1.0e-3
    command_log_path: Path = Path("results/commands.jsonl")
    z_to_height_mm_max: float = 130.0
    validation_samples: int = 16


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level configuration composing every sub-config group."""

    contract: ProductContract = field(default_factory=ProductContract)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    image_pipeline: ImagePipelineConfig = field(default_factory=ImagePipelineConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


_SECTION_TYPES: dict[str, type[Any]] = {
    "contract": ProductContract,
    "model": ModelConfig,
    "physics": PhysicsConfig,
    "training": TrainingConfig,
    "control": ControlConfig,
    "routing": RoutingConfig,
    "paths": PathConfig,
    "image_pipeline": ImagePipelineConfig,
    "sandbox": SandboxConfig,
}


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Coerce a YAML-loaded value into the dataclass field's annotation."""
    if value is None:
        return None
    annotation_str = str(annotation)
    if "Path" in annotation_str and not isinstance(value, Path):
        return Path(str(value))
    if "tuple" in annotation_str and isinstance(value, list):
        return tuple(value)
    return value


def _build_section(section_cls: type[Any], overrides: dict[str, Any]) -> Any:
    """Instantiate a dataclass section, coercing YAML-friendly types."""
    coerced: dict[str, Any] = {}
    for fld in fields(section_cls):
        if fld.name in overrides:
            coerced[fld.name] = _coerce_value(overrides[fld.name], fld.type)
    return section_cls(**coerced)


def load_config(path: Path | None = None) -> AppConfig:
    """Load an :class:`AppConfig`, optionally overlaying a YAML file.

    If ``path`` is ``None`` the default :class:`AppConfig` is returned. The
    environment variable ``AURA_MFP_ROOT`` (if set) always overrides
    ``paths.aura_mfp_root`` as a last step.

    Args:
        path: Optional path to a YAML file with a subset of the config
            surface.

    Returns:
        A fully populated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
    """
    sections: dict[str, Any] = {}

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        import yaml  # local import so unit tests without yaml still import the module

        raw = yaml.safe_load(p.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config YAML must be a mapping, got {type(raw).__name__}")
        for key, section_cls in _SECTION_TYPES.items():
            overrides = raw.get(key, {}) or {}
            if not isinstance(overrides, dict):
                raise ValueError(f"section '{key}' must be a mapping")
            sections[key] = _build_section(section_cls, overrides)

    cfg = AppConfig(
        contract=sections.get("contract", ProductContract()),
        model=sections.get("model", ModelConfig()),
        physics=sections.get("physics", PhysicsConfig()),
        training=sections.get("training", TrainingConfig()),
        control=sections.get("control", ControlConfig()),
        routing=sections.get("routing", RoutingConfig()),
        paths=sections.get("paths", PathConfig()),
        image_pipeline=sections.get("image_pipeline", ImagePipelineConfig()),
        sandbox=sections.get("sandbox", SandboxConfig()),
    )

    env_root = os.environ.get("AURA_MFP_ROOT")
    if env_root:
        # frozen dataclass → rebuild paths with the override
        paths_kwargs = {f.name: getattr(cfg.paths, f.name) for f in fields(PathConfig)}
        paths_kwargs["aura_mfp_root"] = Path(env_root)
        cfg = AppConfig(
            contract=cfg.contract,
            model=cfg.model,
            physics=cfg.physics,
            training=cfg.training,
            control=cfg.control,
            routing=cfg.routing,
            paths=PathConfig(**paths_kwargs),
            image_pipeline=cfg.image_pipeline,
            sandbox=cfg.sandbox,
        )

    return cfg


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Serialize an :class:`AppConfig` to a YAML-friendly ``dict``.

    ``Path`` objects are converted to strings and tuples to lists so that
    the output round-trips cleanly through ``yaml.safe_dump``.

    Args:
        cfg: Configuration to serialize.

    Returns:
        A nested ``dict`` mirroring the dataclass structure.
    """

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        if is_dataclass(value):
            return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
        return value

    return {f.name: _convert(getattr(cfg, f.name)) for f in fields(cfg)}
