"""pi/calibration.py – Convert raw sensor counts to physical engineering units.

Each sensor has its own conversion function.  Coefficients default to the
values below and are overridden by ``~/.config/edge-aura/calibration.json``
(written by the interactive wizard ``pi/scripts/calibrate.py``).

Conversions (see also ``docs/PROTOCOL.md``)
-------------------------------------------

    Pyranometer ADC
        irradiance (W/m²) = (raw_counts / 65535) × Vref / sensitivity

    Thermocouple MAX31856
        The driver returns the 19-bit linearised count from register
        LTCBH.  Each LSB is 0.0078125 °C, so
            temperature (°C) = raw × 0.0078125 + offset

    Anemometer RS-485
        speed (m/s) = raw_speed_x100 / 100 × scale + offset
        direction (°) = (raw_dir_deg + offset_deg) mod 360

A per-sensor fault sentinel (see ``pico/protocol/packet.py``) returns
``None`` from the corresponding calibration function to signal the
upstream packet builder that the value must be serialised as JSON
``null``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pico.protocol.packet import (
    THERMOCOUPLE_FAULT_SENTINEL,
    FAULT_PYRANOMETER,
    FAULT_TC_BITS,
    FAULT_ANEMOMETER,
)

log = logging.getLogger(__name__)

# ── Default calibration coefficients ──────────────────────────────────────
_DEFAULTS: dict = {
    "pyranometer": {
        # Vref = post-amplification reference voltage seen by the ADC.
        # A typical LI-200R + transimpedance amp yields ~0.1 V at 1000 W/m².
        "vref": 0.1,
        "sensitivity": 80e-6,   # V per W/m² (LI-200R typical)
    },
    "thermocouple": {
        "lsb_c": 0.0078125,              # MAX31856 linearised LSB
        "offsets_c": [0.0, 0.0, 0.0, 0.0],
    },
    "anemometer": {
        "speed_scale": 1.0,
        "speed_offset": 0.0,
        "dir_offset_deg": 0.0,
    },
}

_CONFIG_PATH = Path.home() / ".config" / "edge-aura" / "calibration.json"


def _load_coefficients() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open() as fh:
                loaded = json.load(fh)
            merged: dict = {}
            for key, defaults in _DEFAULTS.items():
                merged[key] = {**defaults, **loaded.get(key, {})}
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    # Return a deep copy so callers can't mutate the module defaults.
    return {k: dict(v) for k, v in _DEFAULTS.items()}


def save_coefficients(coeffs: dict) -> None:
    """Persist *coeffs* to the calibration JSON file."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CONFIG_PATH.open("w") as fh:
        json.dump(coeffs, fh, indent=2)


class Calibration:
    """Apply calibration coefficients to raw sensor values."""

    def __init__(self, coeffs: Optional[dict] = None) -> None:
        self._coeffs = coeffs if coeffs is not None else _load_coefficients()

    def reload(self) -> None:
        self._coeffs = _load_coefficients()

    # ------------------------------------------------------------------
    def pyranometer(self, raw_counts: int, fault_flags: int = 0) -> Optional[float]:
        """ADC counts → W/m². Returns ``None`` on sensor fault."""
        if fault_flags & FAULT_PYRANOMETER:
            return None
        c = self._coeffs["pyranometer"]
        voltage = (raw_counts / 65535.0) * c["vref"]
        return voltage / c["sensitivity"]

    # ------------------------------------------------------------------
    def thermocouple(self, raw_counts, fault_flags: int = 0) -> list:
        """Per-channel conversion of 4 linearised-count values to °C.

        Returns a list of ``float | None`` (length 4); ``None`` signals
        that channel's fault bit was set or its value equals the
        INT32_MIN sentinel.
        """
        c = self._coeffs["thermocouple"]
        lsb = c["lsb_c"]
        offs = c.get("offsets_c", [0.0] * 4)
        result: list = []
        for i, raw in enumerate(raw_counts):
            if (fault_flags & FAULT_TC_BITS[i]) or raw == THERMOCOUPLE_FAULT_SENTINEL:
                result.append(None)
            else:
                result.append(raw * lsb + offs[i])
        return result

    # ------------------------------------------------------------------
    def anemometer(self, speed_x100: int, dir_deg: int, fault_flags: int = 0):
        """Raw wire → (m/s, °). Returns ``(None, None)`` on sensor fault."""
        if fault_flags & FAULT_ANEMOMETER:
            return (None, None)
        c = self._coeffs["anemometer"]
        speed = (speed_x100 / 100.0) * c["speed_scale"] + c["speed_offset"]
        direction = (dir_deg + c["dir_offset_deg"]) % 360.0
        return (speed, direction)


# ══════════════════════════════════════════════════════════════════════
# Per-sensor wizard-produced calibration files (Edge-Batch B)
# ══════════════════════════════════════════════════════════════════════
#
# The ``CalibrationLoader`` reads the human-readable JSON files written
# by ``pi/scripts/calibrate.py`` from the repo-root ``calibration/``
# directory. Each file describes a single sensor's ``method=linear``
# fit (slope + intercept) against a reference instrument, together with
# the ISO-8601 timestamp and git SHA of the calibration run.
#
# A missing file is **not fatal**: the loader logs a warning, falls back
# to an identity conversion (``calibrated = raw``) and marks the sensor
# as uncalibrated in :pyattr:`CalibrationLoader.uncalibrated`. Downstream
# ``build_sensor_packet`` ORs the *persistent-fault* flag so the
# orchestrator can surface the warning without dropping data.

# Canonical list of sensors the loader knows about.
SENSORS = ("pyranometer", "thermocouple_0", "thermocouple_1",
           "thermocouple_2", "thermocouple_3", "anemometer")

# Datasheet-expected slopes used for the sanity check.
# slope outside [0.5x, 2x] of these values → wizard requires --force.
EXPECTED_SLOPES = {
    # raw counts (0..65535) × slope → W/m² ; with Vref 0.1 V, sensitivity 80 µV/(W/m²)
    # 1000 W/m² ≈ 0.08 V ≈ 52428 counts ⇒ slope ≈ 0.0191 W/m² per count.
    "pyranometer":    0.0191,
    # MAX31856 LSB is 0.0078125 °C per count.
    "thermocouple_0": 0.0078125,
    "thermocouple_1": 0.0078125,
    "thermocouple_2": 0.0078125,
    "thermocouple_3": 0.0078125,
    # Raw m/s × 100 → m/s ⇒ slope 0.01.
    "anemometer":     0.01,
}

# Units recorded in the calibration file, used by `calibrate list`.
SENSOR_UNITS = {
    "pyranometer":    "W/m^2",
    "thermocouple_0": "degC",
    "thermocouple_1": "degC",
    "thermocouple_2": "degC",
    "thermocouple_3": "degC",
    "anemometer":     "m/s",
}

# Repo-root ``calibration/`` directory (two parents up from this file).
_CAL_DIR = Path(__file__).resolve().parent.parent / "calibration"


def calibration_dir() -> Path:
    """Return the directory where wizard-produced calibration files live."""
    return _CAL_DIR


class CalibrationLoader:
    """Load per-sensor linear calibrations from ``calibration/*.json``.

    Parameters
    ----------
    cal_dir:
        Directory containing ``<sensor>.json`` files. Defaults to the
        repo-root ``calibration/`` directory.

    Attributes
    ----------
    calibrations:
        ``{sensor: {"slope": float, "intercept": float, ...}}`` for every
        sensor whose file was found and parsed.
    uncalibrated:
        Set of sensor names for which no file was found (identity
        conversion is used instead).
    """

    IDENTITY = {"slope": 1.0, "intercept": 0.0, "method": "identity"}

    def __init__(self, cal_dir: Optional[Path] = None) -> None:
        self._dir = Path(cal_dir) if cal_dir is not None else _CAL_DIR
        self.calibrations: dict[str, dict] = {}
        self.uncalibrated: set[str] = set()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        for sensor in SENSORS:
            path = self._dir / f"{sensor}.json"
            if not path.exists():
                log.warning("no calibration file for %s at %s – "
                            "using identity (raw = calibrated)", sensor, path)
                self.uncalibrated.add(sensor)
                self.calibrations[sensor] = dict(self.IDENTITY)
                continue
            try:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                log.error("could not read %s: %s – using identity", path, exc)
                self.uncalibrated.add(sensor)
                self.calibrations[sensor] = dict(self.IDENTITY)
                continue
            # Required fields for a usable calibration.
            if "slope" not in data:
                log.error("%s missing 'slope' – using identity", path)
                self.uncalibrated.add(sensor)
                self.calibrations[sensor] = dict(self.IDENTITY)
                continue
            data.setdefault("intercept", 0.0)
            data.setdefault("method", "linear")
            self.calibrations[sensor] = data

    # ------------------------------------------------------------------
    def apply(self, sensor: str, raw: float) -> float:
        """Convert ``raw`` → engineering units for ``sensor``.

        Falls back to identity if ``sensor`` is unknown/uncalibrated.
        """
        cal = self.calibrations.get(sensor, self.IDENTITY)
        return float(raw) * float(cal["slope"]) + float(cal["intercept"])

    # ------------------------------------------------------------------
    def is_calibrated(self, sensor: str) -> bool:
        return sensor not in self.uncalibrated

    def summary(self) -> list[dict]:
        """Return a list suitable for ``calibrate list`` output."""
        rows: list[dict] = []
        for sensor in SENSORS:
            cal = self.calibrations.get(sensor, self.IDENTITY)
            rows.append({
                "sensor":       sensor,
                "calibrated":   sensor not in self.uncalibrated,
                "slope":        cal.get("slope"),
                "intercept":    cal.get("intercept", 0.0),
                "r_squared":    cal.get("r_squared"),
                "calibrated_at": cal.get("calibrated_at"),
                "units":        cal.get("units", SENSOR_UNITS.get(sensor, "")),
            })
        return rows


# ══════════════════════════════════════════════════════════════════════
# Linear fit helper (shared by the wizard and tests)
# ══════════════════════════════════════════════════════════════════════

def linear_fit(raws, reference: float, intercept: bool = False) -> dict:
    """Fit ``physical = slope × raw (+ intercept)`` against a constant reference.

    The wizard records N raw readings while the user holds a reference
    instrument steady at ``reference``. The mapping is therefore a
    single point plus noise; we use least-squares through the origin
    (``intercept=False``) or an ordinary two-parameter fit
    (``intercept=True``). Returns a dict with ``slope``, ``intercept``,
    ``r_squared`` and ``samples``.
    """
    xs = [float(r) for r in raws]
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 samples for a fit")
    mean_x = sum(xs) / n
    y = float(reference)

    if intercept:
        # y is constant ⇒ a two-parameter fit is degenerate. Report the
        # slope that scales the mean raw value to ``y`` and an intercept
        # of zero; r² = 0 because all residuals = 0 only when raw is
        # also constant.
        slope = y / mean_x if mean_x != 0 else 0.0
        b = 0.0
    else:
        sxx = sum(x * x for x in xs)
        if sxx == 0:
            raise ValueError("all raw samples are zero – cannot fit slope")
        slope = y * sum(xs) / sxx
        b = 0.0

    # r² with a constant reference: residuals around the mean raw.
    # Use the coefficient of determination against the reference model:
    # 1 - Σ(y - ŷ)² / Σ(y - ȳ)². Since y is constant, the denominator is
    # 0; define r² as 1 when all predictions land within 1 % of y.
    preds = [slope * x + b for x in xs]
    err = max(abs(p - y) for p in preds)
    r_squared = 1.0 if y != 0 and err / abs(y) < 0.01 else max(0.0, 1.0 - err / max(abs(y), 1.0))

    return {
        "slope":     slope,
        "intercept": b,
        "r_squared": r_squared,
        "samples":   n,
    }


def check_slope_sanity(sensor: str, slope: float) -> Optional[str]:
    """Return a warning string if *slope* is outside [0.5×, 2×] of the datasheet.

    Returns ``None`` if the slope is plausible.
    """
    expected = EXPECTED_SLOPES.get(sensor)
    if expected is None or expected == 0:
        return None
    ratio = abs(slope) / abs(expected)
    if 0.5 <= ratio <= 2.0:
        return None
    return (f"slope {slope:.6g} is {ratio:.2f}× the datasheet expectation "
            f"{expected:.6g} for {sensor} (allowed range 0.5×–2×)")
