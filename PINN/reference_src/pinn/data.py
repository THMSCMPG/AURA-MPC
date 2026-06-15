"""Data ingestion, validation, and dataset assembly for PINN-AURA-MFP.

Implements design doc §4 (data contract) and §6.6 (ingest + buffer).

Public classes:

* :class:`SensorPacket` – frozen, validated sensor packet matching the
  AURA-MFP JSON input contract.
* :class:`UnifiedDataBuffer` – thread-safe buffer with staleness and
  missing-fraction watchdogs.
* :class:`PinnDataset` – :class:`torch.utils.data.Dataset` with synthetic
  and Jakoplić constructors.
* :class:`JakoplicLoader` – adapter for the Jakoplić Sky-Images + Solar
  Radiation Mendeley dataset.
* :class:`NumericNormalizer` – per-field min-max normaliser with save /
  load.

Exceptions: :class:`SensorValidationError`, :class:`DatasetNotFoundError`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ..config import ProductContract

if TYPE_CHECKING:
    from .image_augmentation import SkyImageAugmentor
    from .image_pipeline import SkyImagePipeline

# ImageNet normalisation constants. Override at the call site if the
# upstream backbone expects different statistics.
IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Field ranges mirror AURA-MFP's ``sanitize_inputs``.
_FIELD_RANGES: dict[str, tuple[float, float]] = {
    "t_s": (0.0, 86400.0),
    "G_poa": (0.0, 1400.0),
    "T_amb": (-40.0, 70.0),
    "WS": (0.0, 60.0),
    "CC": (0.0, 1.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
}

NUMERIC_FIELDS: tuple[str, ...] = ("t_s", "G_poa", "T_amb", "WS", "CC", "lat", "lon")


class SensorValidationError(ValueError):
    """Raised when a :class:`SensorPacket` field is out of contract range."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Jakoplić dataset directory is absent or empty."""


@dataclass(frozen=True)
class SensorPacket:
    """A single multimodal sensor reading matching AURA-MFP's JSON inputs.

    Attributes:
        timestamp: Wall-clock timestamp of the reading.
        t_s: Seconds since local midnight, ``[0, 86400]``.
        G_poa: Plane-of-array irradiance (W/m²), ``[0, 1400]``.
        T_amb: Ambient temperature (°C), ``[-40, 70]``.
        WS: Wind speed (m/s), ``[0, 60]``.
        CC: Cloud cover fraction, ``[0, 1]``.
        lat: Latitude (deg), ``[-90, 90]``.
        lon: Longitude (deg), ``[-180, 180]``.
        sky_image: Optional RGB sky image of shape ``(H, W, 3)`` dtype
            ``uint8``.
        pose: Optional current panel pose
            ``{"pitch", "yaw", "roll", "z"}``.
        fault_flags: Bitmask of active sensor faults from EDGE-AURA-MFP
            (uint16, default ``0`` = no faults). Bit definitions are in
            ``EDGE-AURA-MFP/pi/packet_builder.py::PINN_SENSOR_PACKET_SCHEMA``.
        edge_version: Version string of the EDGE-AURA-MFP producer that
            emitted this packet (default ``"unknown"``).
        sky_image_path: Filesystem path to the raw JPEG written by the Pi
            camera before pipeline processing. ``None`` when no image was
            captured. The image pipeline reads this path when present.
    """

    timestamp: datetime
    t_s: float
    G_poa: float
    T_amb: float
    WS: float
    CC: float
    lat: float
    lon: float
    sky_image: NDArray[np.uint8] | None = None
    pose: dict[str, float] | None = None
    fault_flags: int = 0
    edge_version: str = "unknown"
    sky_image_path: str | None = None

    def validate(self) -> None:
        """Enforce the AURA-MFP field-range contract.

        Raises:
            SensorValidationError: If any numeric field is outside its
                documented range or if ``sky_image`` has the wrong shape
                or dtype.
        """
        for fname in NUMERIC_FIELDS:
            value = float(getattr(self, fname))
            lo, hi = _FIELD_RANGES[fname]
            if not (lo <= value <= hi):
                raise SensorValidationError(
                    f"field '{fname}' out of range: {value} not in [{lo}, {hi}]"
                )
        if not (0 <= self.fault_flags <= 0xFFFF):
            raise SensorValidationError(
                f"fault_flags out of range: {self.fault_flags} not in [0, 65535]"
            )
        if self.sky_image is not None:
            img = self.sky_image
            if not isinstance(img, np.ndarray):
                raise SensorValidationError("sky_image must be a numpy.ndarray or None")
            if img.ndim != 3 or img.shape[2] != 3:
                raise SensorValidationError(
                    f"sky_image must have shape (H, W, 3), got {img.shape}"
                )
            if img.dtype != np.uint8:
                raise SensorValidationError(
                    f"sky_image must be uint8, got {img.dtype}"
                )

    def to_tensor(
        self, image_size: tuple[int, int] = (32, 32)
    ) -> dict[str, torch.Tensor]:
        """Convert the packet into tensors ready for model input.

        The image is resized with nearest-neighbour interpolation (cheap
        and dependency-free) and normalised with :data:`IMAGE_MEAN` /
        :data:`IMAGE_STD`. If no image is present, a zero tensor of the
        requested size is returned.

        Args:
            image_size: Target ``(H, W)`` for the image tensor.

        Returns:
            ``{"numeric": Tensor(7,), "image": Tensor(3, H, W)}``.
        """
        numeric = torch.tensor(
            [float(getattr(self, f)) for f in NUMERIC_FIELDS],
            dtype=torch.float32,
        )
        h, w = image_size
        if self.sky_image is None:
            image = torch.zeros((3, h, w), dtype=torch.float32)
        else:
            arr = self.sky_image.astype(np.float32) / 255.0
            # Nearest-neighbour resize without introducing PIL/torchvision.
            src_h, src_w, _ = arr.shape
            ys = (np.linspace(0, src_h - 1, h)).round().astype(np.int64)
            xs = (np.linspace(0, src_w - 1, w)).round().astype(np.int64)
            resized = arr[np.ix_(ys, xs)]  # (h, w, 3)
            chw = np.transpose(resized, (2, 0, 1))  # (3, h, w)
            mean = np.array(IMAGE_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(IMAGE_STD, dtype=np.float32).reshape(3, 1, 1)
            chw = (chw - mean) / std
            image = torch.from_numpy(chw).to(dtype=torch.float32)
        return {"numeric": numeric, "image": image}


class UnifiedDataBuffer:
    """Thread-safe rolling buffer of :class:`SensorPacket` with watchdogs.

    Enforces the :class:`ProductContract` windows:

    * ``stale_data_threshold_s`` — the gap between successive packets.
    * ``max_missing_fraction`` — the fraction of failed ingests inside a
      rolling 60-sample window.

    Failed ingests increment :meth:`fault_count`; successful ingests reset
    it. The buffer never raises from :meth:`ingest`; instead it records
    the failure and swallows the error so callers can check
    :meth:`fault_count` against
    :attr:`ProductContract.max_consecutive_faults`.
    """

    WINDOW_SIZE: int = 60

    def __init__(self, contract: ProductContract) -> None:
        """Initialise an empty buffer bound to ``contract``.

        Args:
            contract: Product contract with the staleness and
                missing-fraction thresholds.
        """
        self._contract = contract
        self._packets: deque[SensorPacket] = deque(maxlen=4096)
        self._window: deque[bool] = deque(maxlen=self.WINDOW_SIZE)
        self._fault_count: int = 0
        self._lock = threading.Lock()

    def ingest(self, packet: SensorPacket) -> None:
        """Validate and store ``packet``.

        Validation errors, stale timestamps, or a missing-fraction
        breach are all treated as faults: the packet is **not** stored,
        :meth:`fault_count` is incremented, and the rolling window
        records a failure.

        Args:
            packet: Incoming sensor packet.
        """
        with self._lock:
            try:
                packet.validate()
            except SensorValidationError:
                self._record_failure()
                return

            latest = self._packets[-1] if self._packets else None
            if latest is not None:
                gap = (packet.timestamp - latest.timestamp).total_seconds()
                if gap < 0 or gap > self._contract.stale_data_threshold_s:
                    self._record_failure()
                    return

            self._packets.append(packet)
            self._window.append(True)
            self._fault_count = 0

            missing_fraction = 1.0 - (sum(self._window) / len(self._window))
            if missing_fraction > self._contract.max_missing_fraction:
                # Window has drifted into unhealthy territory even though the
                # newest packet validated; treat the buffer as faulted.
                self._fault_count += 1

    def _record_failure(self) -> None:
        self._window.append(False)
        self._fault_count += 1

    def latest(self) -> SensorPacket | None:
        """Return the most recent successfully-ingested packet, if any."""
        with self._lock:
            return self._packets[-1] if self._packets else None

    def fault_count(self) -> int:
        """Return the number of consecutive failed ingests."""
        with self._lock:
            return self._fault_count

    def reset_fault_counter(self) -> None:
        """Reset the consecutive-fault counter to zero."""
        with self._lock:
            self._fault_count = 0


@dataclass
class NumericNormalizer:
    """Per-field min-max normaliser for numeric features.

    Use :meth:`fit` on a list of record dicts once, then call
    :meth:`transform` / :meth:`inverse_transform` on arrays or tensors.
    Persist with :meth:`to_dict` / :meth:`from_dict`.
    """

    fields: tuple[str, ...] = NUMERIC_FIELDS
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)

    def fit(self, records: list[dict[str, Any]]) -> "NumericNormalizer":
        """Fit per-field min/max on ``records``.

        Args:
            records: Sequence of dicts containing at least the keys in
                :attr:`fields`.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If ``records`` is empty.
        """
        if not records:
            raise ValueError("cannot fit NumericNormalizer on empty records")
        for fname in self.fields:
            values = np.array([float(r[fname]) for r in records], dtype=np.float64)
            lo = float(values.min())
            hi = float(values.max())
            if hi == lo:
                # Avoid divide-by-zero; widen the range by 1.0 symmetrically.
                hi = lo + 1.0
            self.mins[fname] = lo
            self.maxs[fname] = hi
        return self

    def _check_fitted(self) -> None:
        if not self.mins or not self.maxs:
            raise RuntimeError("NumericNormalizer must be fit() before use")

    def transform(self, x: NDArray[Any] | torch.Tensor) -> NDArray[Any] | torch.Tensor:
        """Map raw features to ``[0, 1]`` per field.

        Args:
            x: Array or tensor of shape ``(..., len(self.fields))``.

        Returns:
            Same type/shape as ``x`` with values in ``[0, 1]``.
        """
        self._check_fitted()
        # Use float64 arithmetic internally; some fields (e.g. ``t_s`` up to
        # 86400) lose > 1e-3 precision under float32 round-trip. Cast back
        # to the input dtype at the end.
        mins = np.array([self.mins[f] for f in self.fields], dtype=np.float64)
        ranges = np.array(
            [self.maxs[f] - self.mins[f] for f in self.fields], dtype=np.float64
        )
        if isinstance(x, torch.Tensor):
            mins_t = torch.as_tensor(mins, dtype=torch.float64, device=x.device)
            ranges_t = torch.as_tensor(ranges, dtype=torch.float64, device=x.device)
            return ((x.to(torch.float64) - mins_t) / ranges_t).to(dtype=x.dtype)
        arr = np.asarray(x, dtype=np.float64)
        return ((arr - mins) / ranges).astype(np.asarray(x).dtype, copy=False)

    def inverse_transform(
        self, x: NDArray[Any] | torch.Tensor
    ) -> NDArray[Any] | torch.Tensor:
        """Map normalised features back to physical units.

        Args:
            x: Array or tensor of shape ``(..., len(self.fields))``.

        Returns:
            Same type/shape as ``x`` in original units.
        """
        self._check_fitted()
        mins = np.array([self.mins[f] for f in self.fields], dtype=np.float64)
        ranges = np.array(
            [self.maxs[f] - self.mins[f] for f in self.fields], dtype=np.float64
        )
        if isinstance(x, torch.Tensor):
            mins_t = torch.as_tensor(mins, dtype=torch.float64, device=x.device)
            ranges_t = torch.as_tensor(ranges, dtype=torch.float64, device=x.device)
            return (x.to(torch.float64) * ranges_t + mins_t).to(dtype=x.dtype)
        arr = np.asarray(x, dtype=np.float64)
        return (arr * ranges + mins).astype(np.asarray(x).dtype, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise state to a plain dict suitable for JSON/YAML."""
        return {
            "fields": list(self.fields),
            "mins": dict(self.mins),
            "maxs": dict(self.maxs),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "NumericNormalizer":
        """Rehydrate a normaliser from :meth:`to_dict` output."""
        return cls(
            fields=tuple(state["fields"]),
            mins=dict(state["mins"]),
            maxs=dict(state["maxs"]),
        )


def _jitter_image(img: NDArray[np.float32], rng: np.random.Generator) -> NDArray[np.float32]:
    """Apply a small per-channel brightness/contrast jitter (float32 CHW)."""
    brightness = rng.uniform(-0.05, 0.05)
    contrast = rng.uniform(0.95, 1.05)
    out = img * contrast + brightness
    return out.astype(np.float32)


class PinnDataset(Dataset[dict[str, torch.Tensor]]):
    """A :class:`torch.utils.data.Dataset` of PINN training records.

    Each record dict must contain keys ``t_s, G_poa, T_amb, WS, CC, lat,
    lon, T_panel, sky_image`` and an integer ``route_label`` in
    ``[0, 4]``. ``sky_image`` may be ``None``, in which case a zero
    tensor is emitted.

    Items are returned as
    ``{"numeric": Tensor(7,), "image": Tensor(3, H, W),
       "T_panel": Tensor(1,), "route_label": Tensor(1,) long}``.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: tuple[int, int] = (32, 32),
        augment: bool = False,
        seed: int = 0,
        image_pipeline: "SkyImagePipeline | None" = None,
        augmentor: "SkyImageAugmentor | None" = None,
    ) -> None:
        """Build a dataset from ``records``.

        Args:
            records: List of record dicts as described above. The
                ``sky_image`` field may be ``None``, a ``(H, W, 3)``
                uint8 array, a ``(3, H, W)`` float32 array already at
                ``image_size`` (synthetic data path — pipeline is
                skipped), a :class:`pathlib.Path`, or a string path.
            image_size: Target image size ``(H, W)``. If
                ``image_pipeline`` is supplied, its
                :attr:`ImagePipelineConfig.target_size` takes precedence.
            augment: If ``True``, apply Gaussian noise to numeric
                features and — if ``augmentor`` is ``None`` — the legacy
                horizontal-flip + colour-jitter image augmentation. Pass
                a :class:`~src.pinn.image_augmentation.SkyImageAugmentor`
                for the richer, per-tensor augmentation bank.
            seed: Seed used *locally* for the augmentation RNG. Does
                **not** touch global RNG state.
            image_pipeline: Optional
                :class:`~src.pinn.image_pipeline.SkyImagePipeline` used
                to decode raw sky images (paths or full-resolution
                arrays) into the canonical ``(3, H, W)`` tensor.
                ``None`` keeps the pre-Batch-H legacy path and is used
                by the synthetic-data constructor.
            augmentor: Optional
                :class:`~src.pinn.image_augmentation.SkyImageAugmentor`.
                Only applied when ``augment=True``.
        """
        self._records = list(records)
        if image_pipeline is not None:
            self._image_size = image_pipeline.cfg.target_size
        else:
            self._image_size = image_size
        self._augment = augment
        # Local RNG: does not perturb global numpy/torch state.
        self._rng = np.random.default_rng(seed)
        self._image_pipeline = image_pipeline
        self._augmentor = augmentor

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the ``idx``-th item as a tensor dict."""
        rec = self._records[idx]

        numeric_values = np.array(
            [float(rec[f]) for f in NUMERIC_FIELDS], dtype=np.float32
        )

        h, w = self._image_size
        raw_img = rec.get("sky_image")
        image_tensor: torch.Tensor | None = None
        image: NDArray[np.float32]
        if raw_img is None:
            image = np.zeros((3, h, w), dtype=np.float32)
        elif isinstance(raw_img, (str, Path)):
            # A path record: must route through the pipeline.
            if self._image_pipeline is None:
                from .image_pipeline import SkyImagePipeline  # lazy import
                from ..config import ImagePipelineConfig

                self._image_pipeline = SkyImagePipeline(
                    ImagePipelineConfig(target_size=(h, w))
                )
            image_tensor = self._image_pipeline.process(raw_img)
            image = image_tensor.numpy()
        else:
            arr = np.asarray(raw_img)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                # HWC path. If we have a pipeline we always route through
                # it (full preprocessing); otherwise fall back to the
                # legacy nearest-neighbour resize.
                if self._image_pipeline is not None:
                    image_tensor = self._image_pipeline.process(arr)
                    image = image_tensor.numpy()
                else:
                    if arr.dtype == np.uint8:
                        arr = arr.astype(np.float32) / 255.0
                    src_h, src_w, _ = arr.shape
                    ys = (np.linspace(0, src_h - 1, h)).round().astype(np.int64)
                    xs = (np.linspace(0, src_w - 1, w)).round().astype(np.int64)
                    arr = arr[np.ix_(ys, xs)]
                    image = np.transpose(arr, (2, 0, 1)).astype(np.float32)
            elif arr.ndim == 3 and arr.shape[0] in (3, 4):
                # Already CHW float — synthetic/cached tensors. Skip pipeline.
                image = arr.astype(np.float32)
            else:
                raise ValueError(f"unsupported sky_image shape {arr.shape}")

        if self._augment:
            # ~1% of each field's contract range.
            sigmas = np.array(
                [
                    0.01 * (_FIELD_RANGES[f][1] - _FIELD_RANGES[f][0])
                    for f in NUMERIC_FIELDS
                ],
                dtype=np.float32,
            )
            numeric_values = numeric_values + self._rng.normal(
                0.0, sigmas
            ).astype(np.float32)

            if self._augmentor is not None:
                image = self._augmentor(torch.from_numpy(image)).numpy()
            else:
                if self._rng.random() < 0.5:
                    image = image[:, :, ::-1].copy()
                image = _jitter_image(image, self._rng)

        return {
            "numeric": torch.from_numpy(numeric_values),
            "image": torch.from_numpy(image),
            "T_panel": torch.tensor([float(rec["T_panel"])], dtype=torch.float32),
            "route_label": torch.tensor([int(rec["route_label"])], dtype=torch.long),
        }

    # --- Constructors -------------------------------------------------

    @classmethod
    def from_synthetic(
        cls,
        n_samples: int = 1000,
        seed: int = 0,
        routing_cfg: Any | None = None,
        physics_cfg: Any | None = None,
    ) -> "PinnDataset":
        """Generate physically plausible fake records for smoke tests.

        The generator uses a *local* :class:`numpy.random.Generator`
        seeded from ``seed``; it never touches the global NumPy RNG, so
        it does not violate the no-bare-seeds rule.

        Each record's ``route_label`` is derived from a physics-based
        complexity score (design doc §2.1) computed on a two-packet
        window ``(prior, current)``, with the prior packet synthesised
        60 s before the current packet along the same diurnal profile so
        the irradiance-rate-of-change and physics-residual features are
        well-defined.

        Args:
            n_samples: Number of samples to generate.
            seed: Local RNG seed.
            routing_cfg: Optional :class:`src.config.RoutingConfig`.
                Defaults to the stock :class:`RoutingConfig`.
            physics_cfg: Optional :class:`src.config.PhysicsConfig`.
                Defaults to the stock :class:`PhysicsConfig`.

        Returns:
            A fully populated :class:`PinnDataset`.
        """
        # Local imports keep this module importable without the pinn
        # subpackage being fully initialised (e.g. during tests that
        # import only ``data``).
        from ..config import PhysicsConfig as _PhysicsConfig
        from ..config import RoutingConfig as _RoutingConfig
        from .complexity import complexity_score, score_to_route_label

        if routing_cfg is None:
            routing_cfg = _RoutingConfig()
        if physics_cfg is None:
            physics_cfg = _PhysicsConfig()

        # Prior-packet offset for the two-point complexity window (s).
        _PRIOR_DT_S = 60.0

        rng = np.random.default_rng(seed)
        records: list[dict[str, Any]] = []
        for _ in range(n_samples):
            t_s = float(rng.uniform(0.0, 86400.0))
            t_h = t_s / 3600.0
            diurnal = max(0.0, np.cos((t_h - 12.0) * np.pi / 12.0))
            G_poa = float(np.clip(
                950.0 * diurnal + rng.normal(0.0, 30.0), 0.0, 1400.0
            ))
            T_amb = float(np.clip(
                22.0 + 8.0 * np.sin((t_h - 15.0) * np.pi / 12.0)
                + rng.normal(0.0, 1.0),
                -40.0,
                70.0,
            ))
            WS = float(np.clip(rng.lognormal(mean=1.0, sigma=0.5), 0.0, 60.0))
            CC = float(rng.uniform(0.0, 1.0))
            lat = float(rng.uniform(-60.0, 60.0))
            lon = float(rng.uniform(-180.0, 180.0))
            # Crude Faiman-flavoured panel temperature for plausibility.
            T_panel = T_amb + G_poa * (1.0 - CC) / (25.0 + 6.84 * WS)
            sky = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

            # --- Synthesise a prior packet 60 s earlier along the same
            # diurnal profile so the complexity window has two points
            # and the dG/dt feature is meaningful. The prior panel
            # temperature is set to its own Faiman steady state so the
            # physics residual is well-defined (non-zero only when the
            # diurnal forcing changes faster than the thermal time
            # constant).
            t_prior = max(0.0, t_s - _PRIOR_DT_S)
            t_h_prior = t_prior / 3600.0
            diurnal_prior = max(0.0, np.cos((t_h_prior - 12.0) * np.pi / 12.0))
            G_poa_prior = float(np.clip(
                950.0 * diurnal_prior + rng.normal(0.0, 30.0), 0.0, 1400.0
            ))
            T_amb_prior = float(np.clip(
                22.0 + 8.0 * np.sin((t_h_prior - 15.0) * np.pi / 12.0)
                + rng.normal(0.0, 1.0),
                -40.0,
                70.0,
            ))
            T_panel_prior = T_amb_prior + G_poa_prior * (1.0 - CC) / (
                25.0 + 6.84 * WS
            )
            packet_window = [
                {
                    "t_s": t_prior,
                    "G_poa": G_poa_prior,
                    "T_amb": T_amb_prior,
                    "WS": WS,
                    "CC": CC,
                    "T_panel": T_panel_prior,
                },
                {
                    "t_s": t_s,
                    "G_poa": G_poa,
                    "T_amb": T_amb,
                    "WS": WS,
                    "CC": CC,
                    "T_panel": T_panel,
                },
            ]
            score = complexity_score(
                packet_window, routing_cfg=routing_cfg, physics_cfg=physics_cfg
            )
            route_label = score_to_route_label(score, routing_cfg=routing_cfg)

            records.append(
                {
                    "t_s": t_s,
                    "G_poa": G_poa,
                    "T_amb": T_amb,
                    "WS": WS,
                    "CC": CC,
                    "lat": lat,
                    "lon": lon,
                    "T_panel": T_panel,
                    "sky_image": sky,
                    "route_label": route_label,
                    "complexity_score": float(score),
                }
            )
        return cls(records)

    # AURA-MFP parquet schema produced by scripts/generate_synthetic_dataset.py
    # §7.1 of the Day 7 plan. Required columns are input features + T_panel;
    # the rest (efficiency, trajectory, M_spectral, wall_time_ms, tier) are
    # metadata we preserve but do not feed to the model in Batch D.
    _AURA_PARQUET_REQUIRED_COLS: tuple[str, ...] = (
        "t_s",
        "G_poa",
        "T_amb",
        "WS",
        "CC",
        "lat",
        "lon",
        "T_panel",
    )

    @classmethod
    def from_aura_mfp_parquet(cls, path: str | Path) -> "PinnDataset":
        """Load a :class:`PinnDataset` from an AURA-MFP Parquet label file.

        The Parquet file is the one emitted by
        ``scripts/generate_synthetic_dataset.py`` (Day 7). Every input
        feature in :data:`NUMERIC_FIELDS` plus ``T_panel`` must be
        present; schema mismatches raise immediately.

        ``sky_image`` is always emitted as ``None`` (SimV1 does not
        consume images) so the dataset returns zero image tensors. The
        per-row ``route_label`` is read from a ``route_label`` column if
        present, else defaults to ``0`` (LOFI).

        Args:
            path: Path to the Parquet file.

        Returns:
            A fully populated :class:`PinnDataset`.

        Raises:
            DatasetNotFoundError: If ``path`` does not exist.
            ValueError: If the Parquet file is missing required columns
                or is empty.
        """
        p = Path(path)
        if not p.exists():
            raise DatasetNotFoundError(
                f"AURA-MFP parquet file not found: {p!s}"
            )
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas in requirements
            raise DatasetNotFoundError(
                "pandas is required to load an AURA-MFP parquet; install it first."
            ) from exc

        df = pd.read_parquet(p)
        missing = [c for c in cls._AURA_PARQUET_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"AURA-MFP parquet at {p!s} missing required columns: {missing}. "
                f"Required columns: {list(cls._AURA_PARQUET_REQUIRED_COLS)}"
            )
        if df.empty:
            raise ValueError(f"AURA-MFP parquet at {p!s} is empty")

        has_route = "route_label" in df.columns
        records: list[dict[str, Any]] = []
        for row in df.itertuples(index=False):
            records.append(
                {
                    "t_s": float(row.t_s),
                    "G_poa": float(row.G_poa),
                    "T_amb": float(row.T_amb),
                    "WS": float(row.WS),
                    "CC": float(row.CC),
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "T_panel": float(row.T_panel),
                    "sky_image": None,
                    "route_label": int(getattr(row, "route_label", 0)) if has_route else 0,
                }
            )
        return cls(records)

    @classmethod
    def from_jakoplic(
        cls,
        root: Path,
        image_pipeline: "SkyImagePipeline | None" = None,
        match_window_s: float = 30.0,
    ) -> "PinnDataset":
        """Build a dataset from the Jakoplić directory at ``root``.

        Args:
            root: Directory populated according to the layout documented
                in ``Training_Data/README.md``.
            image_pipeline: Optional
                :class:`~src.pinn.image_pipeline.SkyImagePipeline` used
                both by the loader (to verify decoding works) and by
                the emitted :class:`PinnDataset` (to preprocess images
                in ``__getitem__``).
            match_window_s: Maximum time gap (seconds) between a CSV
                row's timestamp and its paired image.

        Returns:
            A :class:`PinnDataset` constructed from the CSV + image
            pairs. Rows with no image within ``match_window_s`` are
            skipped with a logger warning.

        Raises:
            DatasetNotFoundError: If ``root`` is missing, not a
                directory, or contains no usable records.
        """
        loader = JakoplicLoader(
            root=Path(root),
            match_window_s=match_window_s,
            image_pipeline=None,  # emit raw paths; dataset applies pipeline lazily
        )
        records = loader.load()
        return cls(records, image_pipeline=image_pipeline)


@dataclass
class JakoplicLoader:
    """Adapter for the Jakoplić Sky-Images + Solar-Radiation dataset.

    Expected layout (see ``Training_Data/README.md``)::

        <root>/
          images/YYYY-MM-DD/HH_MM_SS.jpg
          radiation.csv     # columns: timestamp, G_poa, T_amb, WS

    For each CSV row, the nearest image within
    :attr:`match_window_s` seconds is used; rows with no image inside
    the window are skipped and a warning is logged. CI never downloads
    the dataset; a user must populate ``<root>`` manually.

    Attributes:
        root: Dataset directory.
        match_window_s: Half-width of the timestamp matching window, in
            seconds. A row matches an image if
            ``abs(row_ts - image_ts) <= match_window_s``.
        image_pipeline: Optional
            :class:`~src.pinn.image_pipeline.SkyImagePipeline`. When
            provided, images are decoded + preprocessed eagerly at load
            time and emitted as ``(C, H, W)`` float32 numpy arrays so
            they bypass the dataset's legacy resize path. When ``None``,
            images are emitted as raw ``(H, W, 3)`` uint8 arrays and the
            consuming :class:`PinnDataset` applies the pipeline lazily
            in ``__getitem__`` (the recommended path when pipeline
            config may change between runs).
    """

    root: Path
    match_window_s: float = 30.0
    image_pipeline: "SkyImagePipeline | None" = None

    _IMAGE_STEM_FMT: str = "%H_%M_%S"

    def load(self) -> list[dict[str, Any]]:
        """Read CSV + paired images into a list of record dicts.

        Returns:
            A list of records suitable for :class:`PinnDataset`. Rows
            whose CSV timestamp has no image within the matching window
            are skipped (with a warning in the module logger).

        Raises:
            DatasetNotFoundError: If the directory or CSV is missing /
                empty. The error message tells the user how to populate
                the directory.
        """
        import logging

        logger = logging.getLogger(__name__)

        root = Path(self.root)
        hint = (
            f"Populate {root!s} per Training_Data/README.md. "
            "Download from Mendeley: "
            "https://data.mendeley.com/datasets/9xt4wrm6jk/1 (Jakoplić, "
            "Sky Images and Solar Radiation Measurement Dataset)."
        )
        if not root.exists() or not root.is_dir():
            raise DatasetNotFoundError(
                f"Jakoplić dataset directory not found. {hint}"
            )
        csv_path = root / "radiation.csv"
        if not csv_path.exists():
            raise DatasetNotFoundError(
                f"Jakoplić radiation.csv missing at {csv_path!s}. {hint}"
            )

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas listed in requirements
            raise DatasetNotFoundError(
                "pandas is required to load the Jakoplić CSV; install it first."
            ) from exc

        df = pd.read_csv(csv_path)
        required_cols = {"timestamp", "G_poa", "T_amb", "WS"}
        missing = required_cols - set(df.columns)
        if missing:
            raise DatasetNotFoundError(
                f"Jakoplić radiation.csv missing columns: {sorted(missing)}. "
                "Expected columns: timestamp, G_poa, T_amb, WS. "
                f"{hint}"
            )
        if df.empty:
            raise DatasetNotFoundError(
                f"Jakoplić radiation.csv at {csv_path!s} is empty. {hint}"
            )

        # Index all available images by their parsed timestamp. The
        # directory layout is images/<YYYY-MM-DD>/<HH_MM_SS>.jpg so we
        # can derive a full datetime from the path alone.
        images_root = root / "images"
        image_index: list[tuple[datetime, Path]] = []
        if images_root.exists() and images_root.is_dir():
            for day_dir in sorted(images_root.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = datetime.strptime(day_dir.name, "%Y-%m-%d")
                except ValueError:
                    continue
                for img_path in sorted(day_dir.iterdir()):
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    try:
                        t = datetime.strptime(img_path.stem, self._IMAGE_STEM_FMT)
                    except ValueError:
                        continue
                    full_ts = day.replace(
                        hour=t.hour, minute=t.minute, second=t.second
                    )
                    image_index.append((full_ts, img_path))

        image_index.sort(key=lambda pair: pair[0])
        image_timestamps = [pair[0] for pair in image_index]

        def _nearest(ts: datetime) -> Path | None:
            if not image_index:
                return None
            import bisect

            pos = bisect.bisect_left(image_timestamps, ts)
            candidates: list[tuple[float, Path]] = []
            if pos < len(image_index):
                candidates.append(
                    (
                        abs((image_index[pos][0] - ts).total_seconds()),
                        image_index[pos][1],
                    )
                )
            if pos > 0:
                candidates.append(
                    (
                        abs((image_index[pos - 1][0] - ts).total_seconds()),
                        image_index[pos - 1][1],
                    )
                )
            candidates.sort(key=lambda x: x[0])
            best_delta, best_path = candidates[0]
            if best_delta > self.match_window_s:
                return None
            return best_path

        records: list[dict[str, Any]] = []
        skipped = 0
        for row in df.itertuples(index=False):
            ts = datetime.fromisoformat(str(row.timestamp))
            t_s = (
                ts.hour * 3600
                + ts.minute * 60
                + ts.second
                + ts.microsecond / 1_000_000.0
            )

            matched_path: Path | None = _nearest(ts)
            if matched_path is None:
                skipped += 1
                logger.warning(
                    "Jakoplić row at %s has no image within %.1fs; skipping.",
                    ts.isoformat(),
                    self.match_window_s,
                )
                continue

            sky_image: Any
            if self.image_pipeline is not None:
                # Decode + preprocess eagerly so the dataset can emit a
                # CHW float array.
                sky_image = self.image_pipeline.process(matched_path).numpy()
            else:
                # Emit the raw path; the dataset's pipeline (or legacy
                # resize) will handle decode on-demand.
                sky_image = matched_path

            records.append(
                {
                    "t_s": float(t_s),
                    "G_poa": float(row.G_poa),
                    "T_amb": float(row.T_amb),
                    "WS": float(row.WS),
                    "CC": float(getattr(row, "CC", 0.0)),
                    "lat": float(getattr(row, "lat", 0.0)),
                    "lon": float(getattr(row, "lon", 0.0)),
                    "T_panel": float(getattr(row, "T_panel", float(row.T_amb))),
                    "sky_image": sky_image,
                    "route_label": int(getattr(row, "route_label", 0)),
                }
            )

        if skipped:
            logger.warning(
                "JakoplicLoader skipped %d/%d rows with no image within %.1fs.",
                skipped,
                len(df),
                self.match_window_s,
            )

        if not records:
            raise DatasetNotFoundError(
                f"No usable rows found in Jakoplić dataset at {root!s}. {hint}"
            )
        return records
