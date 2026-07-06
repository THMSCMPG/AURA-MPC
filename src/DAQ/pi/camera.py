"""pi/camera.py – Pi Camera v2 still capture.

Wraps the ``picamera2`` library to capture a JPEG still image.  Each call
to :meth:`capture` saves the image to a timestamped file and returns its
path.

Requires ``libcamera`` and ``picamera2`` installed on the Pi:
    sudo apt install -y python3-picamera2
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path.home() / "edge-aura" / "images"
_DEFAULT_RESOLUTION = (1920, 1080)


# ── Picamera2 availability ────────────────────────────────────────────
try:  # pragma: no cover — branch selected at import time only
    from picamera2 import Picamera2  # type: ignore[import]
    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore[assignment]
    PICAMERA2_AVAILABLE = False


class Camera:
    """Pi Camera v2 still capture wrapper.

    Parameters
    ----------
    output_dir : Path | str
        Directory where captured images are saved.
    resolution : tuple[int, int]
        Capture resolution ``(width, height)``.
    """

    def __init__(
        self,
        output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
        resolution: tuple[int, int] = _DEFAULT_RESOLUTION,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._resolution = resolution
        self._cam = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        """Initialise the camera hardware."""
        try:
            from picamera2 import Picamera2  # type: ignore[import]
            self._cam = Picamera2()
            config = self._cam.create_still_configuration(
                main={"size": self._resolution}
            )
            self._cam.configure(config)
            self._cam.start()
            self._output_dir.mkdir(parents=True, exist_ok=True)
            log.info("Camera initialised at %dx%d", *self._resolution)
        except ImportError:
            log.error("picamera2 not installed; camera disabled")
            self._cam = None

    def close(self) -> None:
        """Stop and release the camera."""
        if self._cam is not None:
            self._cam.stop()
            self._cam = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    def capture(self, label: str = "") -> Path | None:
        """Capture a still image and save it as a JPEG.

        Parameters
        ----------
        label : str
            Optional label embedded in the filename.

        Returns
        -------
        Path | None
            Full path to the saved image, or *None* if the camera is
            unavailable.
        """
        if self._cam is None:
            log.warning("Camera not available; skipping capture")
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{label}" if label else ""
        path = self._output_dir / f"frame_{ts}{suffix}.jpg"
        self._cam.capture_file(str(path))
        log.debug("Captured image: %s", path)
        return path


# ══════════════════════════════════════════════════════════════════════
# SkyCameraCapture (Edge-Batch B) – background 1 Hz sky imaging
# ══════════════════════════════════════════════════════════════════════

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class SkyCameraCapture:
    """Background 1 Hz still capture for PINN-AURA-MFP sky imaging.

    Spawns a daemon thread that repeatedly captures a JPEG every
    ``1 / framerate`` seconds and writes it to
    ``image_dir/<YYYY-MM-DD>/<HH_MM_SS>.jpg``. The most recent path is
    exposed via :meth:`latest_path`. Capture runs strictly off the
    sensor ingestion thread so a slow capture cannot back-pressure the
    serial pipeline (see Absolute Constraint 3).

    Any exception raised by the backend camera library is caught,
    logged, and recorded as a fault; :meth:`latest_path` returns
    ``None`` until a subsequent capture succeeds.

    On a non-Pi dev machine where ``picamera2`` is unimportable, the
    class falls back to a "fake camera" that writes a synthetic RGB
    test pattern (with a timestamp overlay) so the rest of the
    pipeline is testable end-to-end.
    """

    # A slow capture longer than this is logged but does NOT stall the
    # caller's thread because captures run on a dedicated worker.
    SLOW_CAPTURE_WARN_MS = 500

    def __init__(
        self,
        resolution: tuple = (1920, 1080),
        framerate: int = 1,
        image_dir: Optional[Path] = None,
        *,
        force_fake: Optional[bool] = None,
    ) -> None:
        if framerate <= 0:
            raise ValueError("framerate must be > 0")
        self.resolution = tuple(resolution)
        self.framerate = int(framerate)
        self.image_dir = Path(image_dir) if image_dir is not None else _DEFAULT_OUTPUT_DIR
        # Resolve backend: allow an explicit override for tests.
        if force_fake is None:
            force_fake = _env_flag("EDGE_CAMERA_FAKE") or not PICAMERA2_AVAILABLE
        self._force_fake = bool(force_fake)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Path] = None
        self._fault = False
        self._cam = None

    # ------------------------------------------------------------------
    @property
    def fault(self) -> bool:
        """True if the most recent capture attempt raised."""
        return self._fault

    def latest_path(self) -> Optional[Path]:
        """Return the most recent saved capture path, or ``None``.

        Returns ``None`` if the thread hasn't produced an image yet or
        the camera is in a faulted state.
        """
        with self._lock:
            if self._fault:
                return None
            return self._latest

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the camera and start the background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("SkyCameraCapture already running")
            return
        self._stop_evt.clear()
        try:
            self._open_backend()
        except Exception as exc:  # pylint: disable=broad-except
            log.error("failed to open camera backend: %s", exc)
            self._fault = True
        self._thread = threading.Thread(
            target=self._run, name="SkyCameraCapture", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the capture thread to exit and join it."""
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        self._close_backend()

    def __enter__(self) -> "SkyCameraCapture":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    def _open_backend(self) -> None:
        if self._force_fake:
            log.info("SkyCameraCapture using fake backend (synthetic test pattern)")
            self._cam = None
            return
        # Real backend – guarded so failure just faults, not crashes.
        self._cam = Picamera2()  # type: ignore[operator]
        config = self._cam.create_still_configuration(main={"size": self.resolution})
        self._cam.configure(config)
        self._cam.start()

    def _close_backend(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("camera stop failed: %s", exc)
            self._cam = None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        interval = 1.0 / self.framerate
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            self._capture_once()
            elapsed = time.monotonic() - t0
            # Wait the remainder of the cadence without drift; bail out
            # promptly on stop().
            remaining = max(0.0, interval - elapsed)
            if self._stop_evt.wait(remaining):
                break

    def _capture_once(self) -> None:
        t0 = time.monotonic()
        now = datetime.now(timezone.utc)
        day_dir = self.image_dir / now.strftime("%Y-%m-%d")
        fname = now.strftime("%H_%M_%S") + ".jpg"
        path = day_dir / fname
        try:
            day_dir.mkdir(parents=True, exist_ok=True)
            if self._force_fake or self._cam is None:
                _write_fake_frame(path, self.resolution, now)
            else:
                self._cam.capture_file(str(path))
            with self._lock:
                self._latest = path
                self._fault = False
        except Exception as exc:  # pylint: disable=broad-except
            log.error("camera capture failed: %s", exc)
            with self._lock:
                self._fault = True
        finally:
            ms = (time.monotonic() - t0) * 1000.0
            if ms > self.SLOW_CAPTURE_WARN_MS:
                log.warning("slow camera capture: %.0f ms", ms)

    # ------------------------------------------------------------------
    def capture_once(self) -> Optional[Path]:
        """Synchronous single capture – primarily for health checks."""
        self._capture_once()
        return self.latest_path()


# ── Fake-camera helper ────────────────────────────────────────────────

def _write_fake_frame(path: Path, resolution: tuple, now: datetime) -> None:
    """Write a synthetic RGB test pattern JPEG with a timestamp overlay.

    Uses Pillow when available (always the case on the deployed Pi and
    in CI per ``requirements.txt``). Falls back to a 16 KB placeholder
    when Pillow is unavailable so downstream health checks still see a
    non-trivial file on disk.
    """
    w, h = int(resolution[0]), int(resolution[1])
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]
    except ImportError:
        path.write_bytes(b"FAKECAM " + now.isoformat().encode() + b"\n" + b"\0" * 16384)
        return
    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    step = max(1, w // 16)
    for x in range(0, w, step):
        r = (x * 255) // max(1, w)
        draw.rectangle([(x, 0), (x + step, h)], fill=(r, 128, 255 - r))
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    label = now.strftime("%Y-%m-%d %H:%M:%S UTC  [FAKE CAMERA]")
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    img.save(path, "JPEG", quality=85)
