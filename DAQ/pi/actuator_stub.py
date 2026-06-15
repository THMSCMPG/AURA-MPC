"""pi/actuator_stub.py – Placeholder actuator driver.

EDGE-Batch C ships the control-feedback path end-to-end but **does
not** include a real motorised panel.  Motorised actuator integration
is explicitly 2027 work (design doc §13 / problem statement "Out of
Scope").

This stub accepts an ``OrchestrationCommand`` dict coming back from the
PINN-AURA-MFP orchestrator, validates the fields, and logs the
commanded pose – it never moves any hardware.

The command schema (minimum subset the edge cares about)
--------------------------------------------------------

    {
      "schema_version": "1.0",
      "timestamp_utc":  "<ISO-8601>",
      "command_id":     "<uuid|int>",       # optional
      "pose": {
          "azimuth_deg":    <float>,
          "elevation_deg":  <float>,
          "tilt_deg":       <float>         # optional
      }
    }

Any extra fields are allowed and ignored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("edge-aura.actuator-stub")


class ActuatorStub:
    """Stub actuator driver – accepts commanded poses and logs them.

    Parameters
    ----------
    log_path:
        File to which each accepted command is appended as JSONL.  Set
        to ``None`` to disable on-disk logging (useful for tests).
    """

    def __init__(self, log_path: Path | str | None = None) -> None:
        self._log_path = Path(log_path) if log_path is not None else None
        self._applied: int = 0
        self._rejected: int = 0

    # ------------------------------------------------------------------
    @property
    def applied(self) -> int:
        return self._applied

    @property
    def rejected(self) -> int:
        return self._rejected

    # ------------------------------------------------------------------
    def apply(self, command: dict[str, Any]) -> bool:
        """Accept *command*, log the pose, return ``True`` on success.

        The stub never fails a valid command; it only rejects commands
        that don't contain a well-formed ``pose``.
        """
        pose = command.get("pose")
        if not isinstance(pose, dict):
            log.warning("actuator-stub: rejecting command with no pose: %s", command)
            self._rejected += 1
            return False
        az = pose.get("azimuth_deg")
        el = pose.get("elevation_deg")
        if not isinstance(az, (int, float)) or not isinstance(el, (int, float)):
            log.warning("actuator-stub: rejecting command with bad pose: %s", pose)
            self._rejected += 1
            return False

        log.info(
            "actuator-stub: would move to az=%.2f° el=%.2f° (tilt=%s) – no hardware connected",
            float(az), float(el), pose.get("tilt_deg"),
        )
        self._applied += 1

        if self._log_path is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(command, separators=(",", ":")) + "\n")
            except OSError as exc:
                log.error("actuator-stub: failed to write log %s: %s", self._log_path, exc)
        return True


# ══════════════════════════════════════════════════════════════════════
# CLI – read OrchestrationCommand JSONL on stdin, log to the given file
# ══════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EDGE-AURA-MFP actuator stub")
    p.add_argument("--log-path", default=None,
                   help="File to append accepted commands as JSONL.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    stub = ActuatorStub(log_path=args.log_path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as exc:
            log.error("actuator-stub: malformed JSON on stdin: %s", exc)
            continue
        stub.apply(cmd)
    log.info("actuator-stub exit – applied=%d rejected=%d",
             stub.applied, stub.rejected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
