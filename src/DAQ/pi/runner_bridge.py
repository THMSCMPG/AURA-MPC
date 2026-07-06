"""pi/runner_bridge.py – Runner bridge: convert packets to CSV and call simv2b.

Transforms a list of validated ``PINN_SENSOR_PACKET_SCHEMA`` dicts into the
CSV format expected by the AURA-MFP runner (simv2b) and optionally invokes
the runner binary to obtain solar performance metrics.

AURA-MFP runner CSV schema
--------------------------
The simv2b runner expects comma-separated columns in this order:

    timestamp, G_poa, T_amb, WS

where ``timestamp`` is an ISO-8601 UTC string and the remaining columns are
floating-point values (or empty for ``null``).

Runner invocation
-----------------
The runner binary path is resolved in this priority order:

1. The ``runner_path`` keyword argument passed to :func:`call_simv2b_runner`.
2. The ``AURA_MFP_SIMV2B_RUNNER`` environment variable.
3. The string ``"simv2b"`` looked up on the system ``PATH``.

If the binary cannot be found, :func:`call_simv2b_runner` returns a dict
with ``rmse``, ``mae``, and ``mbe`` all set to ``None`` rather than raising.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("edge-aura.runner_bridge")

# ── AURA-MFP runner CSV schema ────────────────────────────────────────────────

AURA_MFP_COLUMNS: list[str] = ["timestamp", "G_poa", "T_amb", "WS"]
"""Column names expected by the AURA-MFP simv2b runner."""


# ── CSV conversion ────────────────────────────────────────────────────────────

def packets_to_csv(packets: list[dict[str, Any]]) -> str:
    """Convert PINN sensor packets to AURA-MFP runner CSV format.

    Parameters
    ----------
    packets:
        List of validated ``PINN_SENSOR_PACKET_SCHEMA`` dicts.

    Returns
    -------
    str
        CSV text with a header row followed by one data row per packet.
        ``None`` values are written as empty strings.

    Examples
    --------
    ::

        csv_text = packets_to_csv(validated_packets)
        metrics = call_simv2b_runner(csv_text)
    """
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=AURA_MFP_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for pkt in packets:
        row = {col: ("" if pkt.get(col) is None else pkt[col]) for col in AURA_MFP_COLUMNS}
        writer.writerow(row)
    return out.getvalue()


# ── Runner invocation ─────────────────────────────────────────────────────────

_NULL_METRICS: dict[str, Any] = {"rmse": None, "mae": None, "mbe": None}


def call_simv2b_runner(
    csv_content: str,
    *,
    runner_path: Optional[str] = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Call the simv2b runner with CSV data and return metrics.

    Parameters
    ----------
    csv_content:
        CSV text as produced by :func:`packets_to_csv`.
    runner_path:
        Path to the simv2b binary.  If ``None``, the function tries
        ``AURA_MFP_SIMV2B_RUNNER`` env var then ``simv2b`` on PATH.
    timeout:
        Maximum seconds to wait for the runner subprocess (default 120).

    Returns
    -------
    dict
        Dictionary with keys ``rmse``, ``mae``, ``mbe`` (floats or ``None``).
        Returns ``{"rmse": None, "mae": None, "mbe": None}`` when the runner
        binary is unavailable or returns a non-zero exit code.
    """
    binary = _resolve_runner(runner_path)
    if binary is None:
        log.info(
            "runner_bridge: simv2b binary unavailable — returning null metrics"
        )
        return dict(_NULL_METRICS)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="aura_mfp_"
    ) as fh:
        fh.write(csv_content)
        csv_path = fh.name

    try:
        result = subprocess.run(
            [binary, "--input", csv_path, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.warning("runner_bridge: simv2b timed out after %d s", timeout)
        return dict(_NULL_METRICS)
    except OSError as exc:
        log.warning("runner_bridge: failed to launch simv2b — %s", exc)
        return dict(_NULL_METRICS)
    finally:
        try:
            Path(csv_path).unlink()
        except OSError:
            pass

    if result.returncode != 0:
        log.warning(
            "runner_bridge: simv2b exited %d — stderr: %s",
            result.returncode,
            result.stderr[:200],
        )
        return dict(_NULL_METRICS)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log.warning("runner_bridge: could not parse simv2b output — %s", exc)
        return dict(_NULL_METRICS)

    return {
        "rmse": data.get("rmse"),
        "mae": data.get("mae"),
        "mbe": data.get("mbe"),
    }


def _resolve_runner(runner_path: Optional[str]) -> Optional[str]:
    """Return the resolved path to the simv2b binary, or ``None`` if not found."""
    candidates: list[Optional[str]] = [
        runner_path,
        os.environ.get("AURA_MFP_SIMV2B_RUNNER"),
        shutil.which("simv2b"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None
