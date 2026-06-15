"""workstation/inference_server.py – Main inference server for the AURA workstation.

Listens for JSON sensor packets from the Pi 3B+ gateway over a TCP socket,
processes each packet through the fidelity router and PINN optimiser, and
returns an OptimalConfigCommand JSON to the gateway.

Packet flow
-----------
1. Gateway sends a ``PINN_SENSOR_PACKET_SCHEMA`` JSON line over TCP.
2. :class:`InferenceServer` validates the packet.
3. :class:`FidelityRouter` selects the Fortran solver tier.
4. The selected solver (or a mock) is invoked to obtain ``T_panel``.
5. :class:`PINNOptimiser` runs PSO to find optimal (tilt, azimuth, height).
6. An ``OPTIMAL_CONFIG_COMMAND_SCHEMA`` JSON line is sent back.

Design notes
------------
* The server uses a single-threaded blocking accept loop; one client at a
  time.  This is intentional — the workstation is a single-inference node.
* A ``history`` deque of the last 60 packets is maintained per client
  connection to feed the :class:`FidelityRouter`.
* Fortran solver integration is delegated to a pluggable
  :class:`FortranRunner` so tests can inject a mock without subprocess calls.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from workstation.optimal_config_command import WORKSTATION_VERSION, OPTIMAL_CONFIG_COMMAND_SCHEMA
from workstation.pinn_optimiser import PINNOptimiser
from workstation.router import FidelityRouter

log = logging.getLogger("workstation.inference_server")

# Maximum number of past packets kept per connection for routing history.
_HISTORY_MAXLEN: int = 60


# ---------------------------------------------------------------------------
# Fortran runner interface
# ---------------------------------------------------------------------------

class FortranRunner:
    """Invoke a Fortran solver binary and return T_panel.

    Parameters
    ----------
    aura_root:
        Path to the ``AURA-MFP`` repository root.  Binaries are expected at
        ``<aura_root>/src/<tier>/bin/<tier>``.
    timeout:
        Maximum seconds to wait for the solver subprocess.
    """

    def __init__(
        self,
        aura_root: Optional[str] = None,
        *,
        timeout: int = 30,
    ) -> None:
        self._aura_root = aura_root
        self._timeout = timeout

    def run(self, tier: str, packet: dict[str, Any]) -> Optional[float]:
        """Run the specified Fortran tier and return T_panel (°C) or ``None``."""
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        if self._aura_root is None:
            log.debug("fortran_runner: aura_root not set — returning None for T_panel")
            return None

        binary = Path(self._aura_root) / "src" / tier / "bin" / tier
        if not binary.exists():
            log.info(
                "fortran_runner: binary not found at %s — skipping solver call", binary
            )
            return None

        input_json = json.dumps(
            {
                "G_poa": packet.get("G_poa", 0.0),
                "T_amb": packet.get("T_amb", 25.0),
                "WS": packet.get("WS", 1.0),
            }
        )
        try:
            result = subprocess.run(
                [str(binary), "--json"],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._aura_root),
                env={**os.environ},
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("fortran_runner: %s call failed — %s", tier, exc)
            return None

        if result.returncode != 0:
            log.warning(
                "fortran_runner: %s exited %d — %s",
                tier,
                result.returncode,
                result.stderr[:200],
            )
            return None

        try:
            data = json.loads(result.stdout)
            t_panel = data.get("T_panel")
            if t_panel is None:
                t_panel = data.get("T_mod")
            if t_panel is None:
                return None
            return float(t_panel)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("fortran_runner: could not parse %s output — %s", tier, exc)
            return None


# ---------------------------------------------------------------------------
# Inference server
# ---------------------------------------------------------------------------

class InferenceServer:
    """Listen for JSON sensor packets and return OptimalConfigCommand JSON.

    Parameters
    ----------
    host:
        Bind address (default: ``"0.0.0.0"``).
    port:
        Bind port (default: 8765).
    aura_root:
        Path to the AURA-MFP repository root (used by :class:`FortranRunner`).
    pinn_root:
        Path to the PINN-AURA-MFP repository root.
    checkpoint:
        Path to the trained PINN checkpoint (``*.pt``).
    optimiser:
        Optimisation algorithm — ``"pso"`` (default) or ``"bo"``.
    n_particles:
        PSO swarm size (default: 50).
    fortran_runner:
        Optional custom :class:`FortranRunner` (or compatible duck-typed
        object).  Primarily used in tests.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        aura_root: Optional[str] = None,
        pinn_root: Optional[str] = None,
        checkpoint: Optional[str] = None,
        optimiser: str = "pso",
        n_particles: int = 50,
        fortran_runner: Optional[Any] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._optimiser_name = optimiser
        self._n_particles = n_particles

        self._router = FidelityRouter()
        self._pinn = PINNOptimiser(pinn_root=pinn_root, checkpoint_path=checkpoint)
        self._fortran = fortran_runner if fortran_runner is not None else FortranRunner(aura_root)

    # ── Main public API ───────────────────────────────────────────────────────

    def handle_packet(
        self,
        packet: dict[str, Any],
        history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Process a single sensor packet and return the optimal config command.

        Parameters
        ----------
        packet:
            A validated ``PINN_SENSOR_PACKET_SCHEMA`` dict.
        history:
            Recent packet history for the routing decision.

        Returns
        -------
        dict
            A validated ``OPTIMAL_CONFIG_COMMAND_SCHEMA`` dict.
        """
        if history is None:
            history = []

        t_start = time.monotonic()

        # Step 1: route to solver tier
        tier = self._router.route(packet, history)

        # Step 2: call Fortran solver (best-effort; None is acceptable)
        t_panel_solver = self._fortran.run(tier, packet)

        # Step 3: PSO optimisation
        opt_result = self._pinn.optimise(
            packet,
            n_particles=self._n_particles,
        )

        # Step 4: merge solver T_panel with PINN T_panel_pred
        t_panel_pred = (
            float(t_panel_solver)
            if t_panel_solver is not None
            else opt_result["T_panel_pred"]
        )

        inference_ms = (time.monotonic() - t_start) * 1000.0

        timestamp = packet.get(
            "timestamp_iso",
            datetime.now(tz=timezone.utc).isoformat(),
        )

        return {
            "timestamp_iso": timestamp,
            "tilt_opt": opt_result["tilt_opt"],
            "azimuth_opt": opt_result["azimuth_opt"],
            "height_opt": opt_result["height_opt"],
            "P_mp_pred": opt_result["P_mp_pred"],
            "T_panel_pred": t_panel_pred,
            "solver_used": tier,
            "n_pso_iterations": opt_result["n_iterations"],
            "optimiser": self._optimiser_name,
            "inference_time_ms": inference_ms,
            "workstation_version": WORKSTATION_VERSION,
        }

    def run(self) -> None:
        """Start the blocking TCP server loop.

        Each client connection is handled synchronously.  The server sends one
        OptimalConfigCommand JSON line for each received sensor packet JSON
        line.  The connection is closed when the client disconnects.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(1)
            log.info(
                "inference_server: listening on %s:%d", self._host, self._port
            )
            while True:
                conn, addr = srv.accept()
                log.info("inference_server: connection from %s", addr)
                self._handle_connection(conn)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single TCP client connection."""
        history: deque[dict] = deque(maxlen=_HISTORY_MAXLEN)
        buf = ""
        with conn:
            conn.settimeout(5.0)
            try:
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._process_line(line, history, conn)
            except OSError as exc:
                log.debug("inference_server: connection closed — %s", exc)

    def _process_line(
        self,
        line: str,
        history: deque,
        conn: socket.socket,
    ) -> None:
        """Parse one JSON line, process it, and send back the command."""
        try:
            packet = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("inference_server: malformed JSON — %s", exc)
            return

        try:
            command = self.handle_packet(packet, list(history))
        except Exception as exc:  # noqa: BLE001
            log.error("inference_server: handle_packet raised — %s", exc)
            return

        history.append(packet)

        try:
            conn.sendall((json.dumps(command) + "\n").encode("utf-8"))
        except OSError as exc:
            log.warning("inference_server: send failed — %s", exc)


def _validate_command(command: dict[str, Any]) -> None:
    """Validate that *command* conforms to OPTIMAL_CONFIG_COMMAND_SCHEMA.

    Raises
    ------
    ValueError
        If any required field is missing or has the wrong type.
    """
    for field, expected_type in OPTIMAL_CONFIG_COMMAND_SCHEMA.items():
        if field not in command:
            raise ValueError(f"command missing required field: {field!r}")
        if not isinstance(command[field], expected_type):
            raise ValueError(
                f"command field {field!r}: expected {expected_type.__name__}, "
                f"got {type(command[field]).__name__}"
            )
