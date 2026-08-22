"""workstation/calibration.py – Convert raw sensor counts to physical engineering units.

Each sensor has its own conversion function.  Coefficients default to the
values below and are overridden by ``~/.config/edge-aura/calibration.json``
(written by the interactive wizard ``workstation/scripts/calibrate.py``).

ARCHITECTURE (confirmed 2026-08-22): calibration lives on the WORKSTATION,
not the Pico firmware -- the Pico is a translator, piping raw sensor
values over USB serial; this module (and the wizard that fits its
coefficients) is where raw counts actually become physical values, before
anything reaches decision_server.py.

Conversions
-----------

    T_amb (RP2040 ADC1, 12-bit)
        temperature (°C) = PHYS_MIN + (raw/4095) × (PHYS_MAX - PHYS_MIN),
        with PHYS_MIN/PHYS_MAX overridden per-install by the wizard.
        Defaults match the RANGE the firmware used to hardcode itself
        before calibration moved here (-20..60 °C over the full ADC span)
        -- a reasonable starting point, not a precision calibration.

    WS (RP2040 ADC2, 12-bit)
        speed (m/s) = same linear mapping, default range 0..20 m/s.

    Thermocouple MAX31856 (confirmed hardware, 2026-08-20 -- 5 real
    channels, not counting the weather:bit's onboard BME280)
        The driver returns the 19-bit linearised count from register
        LTCBH.  Each LSB is 0.0078125 °C, so
            temperature (°C) = raw × 0.0078125 + offset
        Wiring/driver work for these still deferred until the 5th
        chip-select pin is assigned (see thermocouple_max31856.py) --
        this conversion function itself doesn't depend on that, kept
        ready for when it's wired up.

LEGACY, not currently wired to any live Pico packet field (kept for
reference, not deleted -- see checklist for the full history):

    Pyranometer ADC
        Dead -- irradiance is manual-entry on the workstation now, not
        sensed at all. The old formula also assumed a 16-bit ADC
        (raw/65535) that was never actually the RP2040's real 12-bit
        ADC -- left as-is below since the whole method is inactive, but
        flagging so nobody copies this formula's bit-depth assumption
        into a still-active conversion by mistake.

    Anemometer RS-485
        Tied to an RS-485 anemometer that doesn't match the confirmed
        SparkFun Weather Meter Kit plan (simple passive reed-switch, not
        RS-485) -- likely superseded, not confirmed either way, wind/rain
        wiring is still deferred until that hardware's in hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Default calibration coefficients ──────────────────────────────────────
_DEFAULTS: dict = {
    "t_amb": {
        # RP2040 ADC1, 12-bit (0-4095) -- see module docstring. These
        # defaults match what used to be hardcoded in firmware; run the
        # wizard for a real calibration once hardware's in hand.
        "phys_min": -20.0,   # °C at raw=0
        "phys_max":  60.0,   # °C at raw=4095
    },
    "ws": {
        # RP2040 ADC2, 12-bit (0-4095).
        "phys_min": 0.0,     # m/s at raw=0
        "phys_max": 20.0,    # m/s at raw=4095
    },
    "thermocouple": {
        "lsb_c": 0.0078125,              # MAX31856 linearised LSB
        "offsets_c": [0.0, 0.0, 0.0, 0.0, 0.0],   # 5 channels, confirmed 2026-08-22
    },
    # ── Legacy, inactive -- see module docstring ──────────────────────
    "pyranometer": {
        "vref": 0.1,
        "sensitivity": 80e-6,
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
    def t_amb(self, raw_count: int) -> float:
        """RP2040 ADC1 12-bit raw count [0,4095] → °C.

        No fault-flag handling here -- firmware no longer flags physical
        out-of-range faults (it has no calibration to check against, see
        module docstring); if you want a fault check, apply it to the
        RESULT of this conversion, in physical units, on the caller side.
        """
        c = self._coeffs["t_amb"]
        t = max(0.0, min(1.0, raw_count / 4095.0))
        return c["phys_min"] + t * (c["phys_max"] - c["phys_min"])

    # ------------------------------------------------------------------
    def ws(self, raw_count: int) -> float:
        """RP2040 ADC2 12-bit raw count [0,4095] → m/s."""
        c = self._coeffs["ws"]
        t = max(0.0, min(1.0, raw_count / 4095.0))
        return c["phys_min"] + t * (c["phys_max"] - c["phys_min"])

    # ------------------------------------------------------------------
    def thermocouple(self, raw_counts, fault_flags: int = 0) -> list:
        """Per-channel conversion of linearised MAX31856 counts to °C.

        Returns a list of ``float | None`` (length matches raw_counts);
        ``None`` signals that channel's value equals the INT32_MIN fault
        sentinel from the driver (see thermocouple_max31856.py).

        NOTE: fault-bit-based masking (via a per-channel fault_flags bit)
        was removed here since it depended on the dead COBS-protocol
        module's FAULT_TC_BITS constants. The sentinel-value check alone
        (INT32_MIN) still catches per-channel SPI read failures -- see
        thermocouple_max31856.py's read(), which sets exactly that
        sentinel on failure. Re-add fault-bit masking here if/when a
        real per-channel fault bit gets defined for the new wire format.
        """
        c = self._coeffs["thermocouple"]
        lsb = c["lsb_c"]
        offs = c.get("offsets_c", [0.0] * len(raw_counts))
        _THERMOCOUPLE_FAULT_SENTINEL = -(1 << 31)  # matches thermocouple_max31856.py
        result: list = []
        for i, raw in enumerate(raw_counts):
            if raw == _THERMOCOUPLE_FAULT_SENTINEL:
                result.append(None)
            else:
                offset = offs[i] if i < len(offs) else 0.0
                result.append(raw * lsb + offset)
        return result

    # ------------------------------------------------------------------
    # Legacy, inactive -- see module docstring.
    def pyranometer(self, raw_counts: int, fault_flags: int = 0) -> Optional[float]:
        """LEGACY, INACTIVE. ADC counts → W/m², assuming a 16-bit ADC that
        was never actually correct for the RP2040's real 12-bit ADC (see
        module docstring) -- irradiance is manual-entry now regardless,
        this method isn't called from anywhere live."""
        c = self._coeffs["pyranometer"]
        voltage = (raw_counts / 65535.0) * c["vref"]
        return voltage / c["sensitivity"]

    # ------------------------------------------------------------------
    def anemometer(self, speed_x100: int, dir_deg: int, fault_flags: int = 0):
        """LEGACY, likely inactive -- see module docstring (RS-485 specific,
        doesn't match the confirmed SparkFun Weather Meter Kit plan)."""
        c = self._coeffs["anemometer"]
        speed = (speed_x100 / 100.0) * c["speed_scale"] + c["speed_offset"]
        direction = (dir_deg + c["dir_offset_deg"]) % 360.0
        return (speed, direction)


# ══════════════════════════════════════════════════════════════════════
# Per-sensor wizard-produced calibration files (Edge-Batch B)
# ══════════════════════════════════════════════════════════════════════
#
# The ``CalibrationLoader`` reads the human-readable JSON files written
# by ``workstation/scripts/calibrate.py`` from the repo-root ``calibration/``
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
SENSORS = ("t_amb", "ws", "thermocouple_0", "thermocouple_1",
           "thermocouple_2", "thermocouple_3", "thermocouple_4")
# Confirmed 2026-08-22: 5 real thermocouples (not counting the weather:bit's
# onboard BME280). thermocouple_4 is listed here ahead of the actual
# firmware support for it (the C firmware doesn't read thermocouples at
# all yet -- driver work still deferred pending the 5th chip-select pin
# assignment, see thermocouple_max31856.py) -- it'll just show as
# "MISSING" in `calibrate list` until that lands, which is fine.
#
# pyranometer and anemometer REMOVED from the active list -- pyranometer
# is dead (manual-entry irradiance), anemometer was RS-485-specific and
# doesn't match the confirmed SparkFun Weather Meter Kit plan. Their
# Calibration class methods still exist (see calibration.py) for
# reference, just not wired to anything live.

# Datasheet-expected slopes used for the sanity check.
# slope outside [0.5x, 2x] of these values → wizard requires --force.
EXPECTED_SLOPES = {
    # RP2040 12-bit ADC (0-4095) × slope -> physical, using the DEFAULT
    # firmware-era range as a starting expectation (-20..60 C over full
    # span for t_amb, 0..20 m/s for ws) -- real slope comes from the
    # wizard's actual fit against a reference instrument.
    "t_amb": (60.0 - -20.0) / 4095.0,   # ~0.01954 degC per count
    "ws":    (20.0 - 0.0) / 4095.0,     # ~0.00488 (m/s) per count
    # MAX31856 LSB is 0.0078125 °C per count.
    "thermocouple_0": 0.0078125,
    "thermocouple_1": 0.0078125,
    "thermocouple_2": 0.0078125,
    "thermocouple_3": 0.0078125,
    "thermocouple_4": 0.0078125,
}

# Units recorded in the calibration file, used by `calibrate list`.
SENSOR_UNITS = {
    "t_amb":          "degC",
    "ws":             "m/s",
    "thermocouple_0": "degC",
    "thermocouple_1": "degC",
    "thermocouple_2": "degC",
    "thermocouple_3": "degC",
    "thermocouple_4": "degC",
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
