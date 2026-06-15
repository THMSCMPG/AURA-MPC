"""Sky-image preprocessing pipeline (Batch H).

Raw sky-camera image (arbitrary resolution, any common format) → a
``(3, 32, 32)`` ``torch.float32`` tensor suitable for the PINN image
encoder. Every step is a pure function so it can be tested and toggled
independently; :class:`SkyImagePipeline` composes them according to a
:class:`~src.config.ImagePipelineConfig`.

Design constraints (see ``docs/IMAGE_PIPELINE.md``):

* Pure — no hidden state; identical inputs produce identical outputs.
* < 50 ms mean on CPU for a 1080p input (validated in tests).
* Output is always exactly ``(3, H, W)`` float32 (or ``(4, H, W)`` when
  the red/blue-ratio channel is enabled) — the PINN encoder's contract.

Public surface:

* :class:`SkyImagePipeline` — stateful wrapper around a config.
* :func:`circular_crop`, :func:`histogram_equalize`, :func:`detect_sun`,
  :func:`mask_sun`, :func:`compute_rbr`, :func:`anti_aliased_resize`,
  :func:`normalize` — pure, independently testable steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from ..config import ImagePipelineConfig


__all__ = [
    "SkyImagePipeline",
    "circular_crop",
    "histogram_equalize",
    "detect_sun",
    "mask_sun",
    "compute_rbr",
    "anti_aliased_resize",
    "normalize",
]


_INTERPOLATION_MAP: dict[str, int] = {
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Pure functional steps
# ---------------------------------------------------------------------------


def circular_crop(img: NDArray[Any], radius_frac: float) -> NDArray[Any]:
    """Zero out pixels outside a centred circular aperture.

    Intended for fisheye / all-sky cameras whose corner pixels are
    outside the imaging optic and dominated by lens housing.

    Args:
        img: ``(H, W, 3)`` uint8 or float image.
        radius_frac: Radius of the kept disc as a fraction of
            ``min(H, W) / 2``. Must lie in ``(0, 1]``.

    Returns:
        Same shape and dtype as ``img``; pixels outside the disc are 0.

    Raises:
        ValueError: If ``img`` is not ``(H, W, 3)`` or
            ``radius_frac`` is out of ``(0, 1]``.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"circular_crop expects (H, W, 3), got {img.shape}")
    if not (0.0 < radius_frac <= 1.0):
        raise ValueError(f"radius_frac must be in (0, 1], got {radius_frac}")
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    radius = radius_frac * min(h, w) / 2.0
    # Use float mesh so the disc is continuous at any fraction.
    ys = np.arange(h, dtype=np.float32) - cy + 0.5
    xs = np.arange(w, dtype=np.float32) - cx + 0.5
    mask = (ys[:, None] ** 2 + xs[None, :] ** 2) <= (radius ** 2)
    out = img.copy()
    out[~mask] = 0
    return np.asarray(out)


def histogram_equalize(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Per-image histogram equalisation on HSV's V channel.

    Operating on V preserves hue and saturation while stretching the
    luminance distribution — the right trade-off for sky cameras where
    colour carries most cloud-type information.

    Args:
        img: ``(H, W, 3)`` uint8 RGB image.

    Returns:
        ``(H, W, 3)`` uint8 RGB image with equalised luminance.

    Raises:
        ValueError: If ``img`` is not ``(H, W, 3)`` uint8.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"histogram_equalize expects (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"histogram_equalize expects uint8, got {img.dtype}")
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), dtype=np.uint8)


def detect_sun(
    img: NDArray[np.uint8], threshold: float
) -> tuple[int, int, int] | None:
    """Locate the sun disc via a brightness-percentile threshold.

    The sun is the largest saturated-luminance cluster. We threshold at
    ``percentile(brightness, threshold * 100)`` (so ``threshold=0.95``
    picks the brightest 5 % of pixels), find the largest connected
    component, and fit a minimum-enclosing circle.

    Args:
        img: ``(H, W, 3)`` uint8 RGB image.
        threshold: Percentile of brightness in ``(0, 1)``.

    Returns:
        ``(cx, cy, radius)`` in pixels, or ``None`` if no bright cluster
        is present (e.g. an overcast sky).

    Raises:
        ValueError: If ``img`` or ``threshold`` are out of contract.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"detect_sun expects (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"detect_sun expects uint8, got {img.dtype}")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Histogram-based percentile — orders of magnitude faster than
    # np.percentile for uint8 images because the input alphabet has only
    # 256 bins. Constant-time regardless of image size.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(gray.size)
    target = threshold * total
    cumulative = 0.0
    cutoff_idx = 255
    for i in range(256):
        cumulative += hist[i]
        if cumulative >= target:
            cutoff_idx = i
            break
    cutoff = float(cutoff_idx)
    # When the image is (near-)uniform the percentile cutoff collapses to
    # the global max and nothing exceeds it; no sun to mask.
    if cutoff >= 255.0:
        return None
    binary = (gray > cutoff).astype(np.uint8)
    if binary.sum() == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:  # label 0 is background
        return None
    # Pick the largest foreground component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    component = (labels == largest).astype(np.uint8)
    ys, xs = np.nonzero(component)
    if xs.size == 0:
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    return int(round(cx)), int(round(cy)), int(max(1, round(radius)))


def mask_sun(
    img: NDArray[np.uint8], sun_pos: tuple[int, int, int] | None, method: str
) -> NDArray[np.uint8]:
    """Remove the sun disc from an image.

    Args:
        img: ``(H, W, 3)`` uint8 RGB image.
        sun_pos: ``(cx, cy, radius)`` as returned by :func:`detect_sun`,
            or ``None`` to no-op.
        method: ``"median_inpaint"`` (OpenCV TELEA inpainting, smoothest
            result), ``"black_fill"`` (fast, introduces a hard edge), or
            ``"none"`` (return input unchanged).

    Returns:
        Same shape and dtype as ``img``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method not in {"median_inpaint", "black_fill", "none"}:
        raise ValueError(
            f"mask_sun: unknown method {method!r}; expected one of "
            "'median_inpaint', 'black_fill', 'none'"
        )
    if method == "none" or sun_pos is None:
        return np.asarray(img.copy(), dtype=np.uint8)
    cx, cy, radius = sun_pos
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Pad the radius slightly so the inpaint has clean boundary pixels.
    cv2.circle(mask, (cx, cy), max(1, int(radius) + 2), (255,), thickness=-1)
    if method == "black_fill":
        out = img.copy()
        out[mask > 0] = 0
        return np.asarray(out, dtype=np.uint8)
    # median_inpaint: OpenCV's TELEA algorithm — fast and produces a
    # noticeably smoother neighbourhood than black_fill.
    return np.asarray(
        cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA),
        dtype=np.uint8,
    )


def compute_rbr(img: NDArray[Any]) -> NDArray[np.float32]:
    """Compute the red/blue ratio channel.

    RBR is a classic cloud/clear-sky discriminator: clear sky is blue
    (RBR < 1), thick cloud is white-ish (RBR ≈ 1), sun-coloured pixels
    can be ≫ 1. We clamp small-denominator pixels to avoid blow-up and
    cap the output at ``3.0`` (anything above is saturated sun / lens
    flare and carries no cloud information).

    Args:
        img: ``(H, W, 3)`` uint8 RGB image or ``(H, W, 3)`` float image
            in ``[0, 1]``.

    Returns:
        ``(H, W)`` float32 array in ``[0, 3]``.

    Raises:
        ValueError: If ``img`` is not ``(H, W, 3)``.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"compute_rbr expects (H, W, 3), got {img.shape}")
    if img.dtype == np.uint8:
        r = img[..., 0].astype(np.float32)
        b = img[..., 2].astype(np.float32)
    else:
        arr = img.astype(np.float32)
        # Scale [0, 1] floats up to the uint8 range so the clamp below
        # behaves the same way regardless of input dtype.
        r = arr[..., 0] * 255.0
        b = arr[..., 2] * 255.0
    # Add 1 to both channels (not just the denominator) so the ratio is
    # symmetric around solid-grey pixels.
    rbr = (r + 1.0) / (b + 1.0)
    return np.clip(rbr, 0.0, 3.0).astype(np.float32)


def anti_aliased_resize(
    img: NDArray[Any], target_size: tuple[int, int], interpolation: str
) -> NDArray[Any]:
    """Resize ``img`` to ``target_size`` with an anti-aliased kernel.

    ``INTER_AREA`` is the correct default for downscaling: it averages
    source pixels per output cell and avoids moiré on the high-frequency
    cloud textures that cover most of a sky-camera frame. ``LANCZOS4``
    preserves a hair more detail at ~2× the cost.

    Args:
        img: ``(H, W, C)`` uint8 or float array.
        target_size: Output ``(H, W)``.
        interpolation: ``"area"`` or ``"lanczos"``.

    Returns:
        ``(target_H, target_W, C)`` array with the same dtype as ``img``.

    Raises:
        ValueError: If ``interpolation`` is unknown or ``target_size``
            contains non-positive dimensions.
    """
    if interpolation not in _INTERPOLATION_MAP:
        raise ValueError(
            f"anti_aliased_resize: unknown interpolation {interpolation!r}; "
            f"expected one of {sorted(_INTERPOLATION_MAP)}"
        )
    th, tw = target_size
    if th <= 0 or tw <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    # cv2.resize expects (width, height).
    return cv2.resize(img, (tw, th), interpolation=_INTERPOLATION_MAP[interpolation])


def normalize(
    img: NDArray[np.float32],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> NDArray[np.float32]:
    """Apply per-channel ``(x - mean) / std``.

    Args:
        img: ``(3, H, W)`` float32 image in ``[0, 1]``.
        mean: Three-tuple of per-channel means.
        std: Three-tuple of per-channel stds.

    Returns:
        ``(3, H, W)`` float32 image, normalised.

    Raises:
        ValueError: If shapes or dtypes are out of contract.
    """
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"normalize expects (3, H, W), got {img.shape}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must both be length-3")
    if any(s == 0 for s in std):
        raise ValueError("std components must be non-zero")
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    result = (img.astype(np.float32) - mean_arr) / std_arr
    return np.asarray(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


def _load_image_any(source: Any) -> NDArray[np.uint8]:
    """Normalise heterogeneous input into a ``(H, W, 3)`` uint8 ndarray."""
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, (str, Path)):
        # Use PIL rather than cv2.imread so byte-identical input yields
        # byte-identical decoded pixels regardless of the OpenCV build.
        from PIL import Image

        with Image.open(source) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    else:
        # Duck-type PIL.Image.Image without importing eagerly.
        from PIL import Image

        if isinstance(source, Image.Image):
            arr = np.asarray(source.convert("RGB"), dtype=np.uint8)
        else:
            raise TypeError(
                f"unsupported image source type {type(source).__name__}; "
                "expected numpy.ndarray, pathlib.Path, str, or PIL.Image"
            )
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        # Accept float images in [0, 1] for convenience; otherwise clamp.
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _process_single_for_pool(args: tuple[ImagePipelineConfig, Any]) -> NDArray[np.float32]:
    """Top-level worker for :meth:`SkyImagePipeline.process_batch`.

    Returns a numpy array rather than a tensor because tensors do not
    pickle as cheaply as ndarrays across ``multiprocessing.Pool`` IPC.
    """
    cfg, image = args
    return SkyImagePipeline(cfg).process(image).numpy()


class SkyImagePipeline:
    """Compose the preprocessing steps described by :class:`ImagePipelineConfig`.

    The pipeline is a pure function of its config and the input image:
    calling :meth:`process` twice on the same input always returns
    byte-identical tensors.

    Example::

        >>> from src.config import ImagePipelineConfig
        >>> pipe = SkyImagePipeline(ImagePipelineConfig())
        >>> tensor = pipe.process("sky.jpg")
        >>> tensor.shape
        torch.Size([3, 32, 32])
    """

    def __init__(self, cfg: ImagePipelineConfig) -> None:
        """Bind a config to this pipeline instance.

        Args:
            cfg: :class:`ImagePipelineConfig` describing which steps to
                apply.
        """
        self._cfg = cfg

    @property
    def cfg(self) -> ImagePipelineConfig:
        """The bound :class:`ImagePipelineConfig` (read-only)."""
        return self._cfg

    # --- main entry points ------------------------------------------------

    def process(self, image: Any) -> torch.Tensor:
        """Run the full pipeline on one image.

        Args:
            image: A file path, a PIL Image, or a ``(H, W, 3)`` uint8
                numpy array.

        Returns:
            ``(3, target_H, target_W)`` float32 tensor — or
            ``(4, target_H, target_W)`` when
            :attr:`ImagePipelineConfig.add_rbr_channel` is ``True``.
            Values are in ``[0, 1]`` if ``normalize=False``, else
            mean/std-normalised.
        """
        arr = _load_image_any(image)
        cfg = self._cfg

        if cfg.apply_circular_crop:
            arr = circular_crop(arr, cfg.circular_crop_radius_frac)

        if cfg.apply_sun_mask and cfg.sun_mask_method != "none":
            sun_pos = detect_sun(arr, cfg.sun_detection_threshold)
            arr = mask_sun(arr, sun_pos, cfg.sun_mask_method)

        if cfg.apply_histogram_eq:
            arr = histogram_equalize(arr)

        # Compute the RBR channel on the full-resolution uint8 image,
        # then downsample it alongside the RGB channels so both live in
        # the same spatial grid.
        rbr_full: NDArray[np.float32] | None = None
        if cfg.add_rbr_channel:
            rbr_full = compute_rbr(arr)

        resized = anti_aliased_resize(arr, cfg.target_size, cfg.interpolation)
        # -> (target_H, target_W, 3) uint8

        # Convert to CHW float32 in [0, 1].
        chw = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0

        if cfg.normalize:
            chw = normalize(chw, cfg.mean, cfg.std)

        if rbr_full is not None:
            rbr_small = anti_aliased_resize(
                rbr_full, cfg.target_size, cfg.interpolation
            ).astype(np.float32)
            # Concatenate on the channel axis: (3, H, W) + (1, H, W) -> (4, H, W).
            chw = np.concatenate([chw, rbr_small[None, ...]], axis=0)

        return torch.from_numpy(np.ascontiguousarray(chw))

    def process_batch(
        self, images: list[Any], num_workers: int = 4
    ) -> torch.Tensor:
        """Run :meth:`process` over a list of images.

        Args:
            images: List of image sources (paths, arrays, PIL images).
            num_workers: Number of worker processes. ``<= 1`` runs
                in-process.

        Returns:
            ``(B, C, H, W)`` float32 tensor where ``C`` is 3 or 4
            depending on ``add_rbr_channel``.
        """
        if not images:
            c = 4 if self._cfg.add_rbr_channel else 3
            h, w = self._cfg.target_size
            return torch.empty((0, c, h, w), dtype=torch.float32)

        if num_workers is None or num_workers <= 1 or len(images) <= 1:
            tensors = [self.process(img) for img in images]
            return torch.stack(tensors, dim=0)

        # Use a plain Pool with the module-level worker so the config is
        # picklable. We cap ``num_workers`` at the number of images to
        # avoid idle subprocesses.
        import multiprocessing as mp

        n_workers = min(int(num_workers), len(images))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            arrays = pool.map(
                _process_single_for_pool, [(self._cfg, img) for img in images]
            )
        stacked = np.stack(arrays, axis=0)
        return torch.from_numpy(stacked)

    def compute_dataset_stats(
        self, image_paths: list[Path], sample_size: int = 1000
    ) -> dict[str, list[float]]:
        """Estimate per-channel mean/std on a random sample.

        The stats are computed on the **pre-normalised** ``[0, 1]``
        tensor (i.e. with ``cfg.normalize`` temporarily disabled) so
        callers can drop the result into ``cfg.mean`` / ``cfg.std``.

        Args:
            image_paths: Candidate images; a random subset of
                ``sample_size`` entries is used (all of them if there
                are fewer).
            sample_size: Maximum number of images to read.

        Returns:
            ``{"mean": [r, g, b], "std": [r, g, b]}``. When
            ``add_rbr_channel`` is True the lists have length 4.

        Raises:
            ValueError: If ``image_paths`` is empty.
        """
        if not image_paths:
            raise ValueError("compute_dataset_stats requires at least one image")
        # Local RNG: do not touch global numpy state.
        rng = np.random.default_rng(0)
        paths = list(image_paths)
        if len(paths) > sample_size:
            idx = rng.choice(len(paths), size=sample_size, replace=False)
            paths = [paths[int(i)] for i in idx]

        # Clone the config with normalize=False so the returned stats are
        # in the natural [0, 1] space.
        from dataclasses import replace

        stats_cfg = replace(self._cfg, normalize=False)
        stats_pipe = SkyImagePipeline(stats_cfg)

        c = 4 if self._cfg.add_rbr_channel else 3
        running_sum = np.zeros(c, dtype=np.float64)
        running_sqsum = np.zeros(c, dtype=np.float64)
        running_count = 0
        for p in paths:
            tensor = stats_pipe.process(p)
            arr = tensor.numpy()  # (C, H, W)
            flat = arr.reshape(c, -1)
            running_sum += flat.sum(axis=1)
            running_sqsum += (flat ** 2).sum(axis=1)
            running_count += flat.shape[1]
        if running_count == 0:
            raise ValueError("compute_dataset_stats: empty pixel sample")
        mean = running_sum / running_count
        var = np.maximum(running_sqsum / running_count - mean ** 2, 0.0)
        std = np.sqrt(var)
        return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}
