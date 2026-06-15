"""pi/orchestrator_bridge.py – Bridge SensorPackets to PINN-AURA-MFP.

Reads newline-delimited JSON ``SensorPacket`` records from an input
source (by default ``sys.stdin``, which is what the daemon pipes into
us) and forwards them to the PINN-AURA-MFP orchestrator via one of
three transports:

* ``subprocess_pipe`` – primary mode.  Spawns ::

      python -m scripts.predict --mode live --checkpoint $AURA_EDGE_MODEL_PATH

  as a subprocess (``cwd=$AURA_EDGE_PINN_ROOT``), pipes JSON lines to
  its stdin, and reads back ``OrchestrationCommand`` JSON lines from
  its stdout.

* ``unix_socket`` – connect to ``/var/run/aura_sensors.sock`` (or
  ``AURA_EDGE_SOCKET_PATH``); the orchestrator is configured to read
  from it.  Use for multi-process deployments on the same host.

* ``tcp_socket`` – same but TCP for remote orchestrators; ``host:port``
  come from ``AURA_EDGE_TCP_HOST`` / ``AURA_EDGE_TCP_PORT``.

For every ``OrchestrationCommand`` received on the return path the
bridge:

1. Appends the command to ``logs/commands.jsonl`` (configurable).
2. Forwards it to :class:`pi.actuator_stub.ActuatorStub`.

Config is driven entirely by environment variables so the ``systemd``
unit stays a one-line ``ExecStart``:

============================== ===========================================
``AURA_EDGE_BRIDGE_MODE``      ``subprocess_pipe`` | ``unix_socket`` | ``tcp_socket``
``AURA_EDGE_PINN_ROOT``        clone of PINN-AURA-MFP (subprocess mode)
``AURA_EDGE_MODEL_PATH``       PINN checkpoint file (subprocess mode)
``AURA_EDGE_SOCKET_PATH``      override UNIX socket path
``AURA_EDGE_TCP_HOST``         TCP orchestrator host
``AURA_EDGE_TCP_PORT``         TCP orchestrator port
``AURA_EDGE_COMMAND_LOG``      override ``logs/commands.jsonl`` destination
============================== ===========================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any, Optional

from pi.actuator_stub import ActuatorStub

log = logging.getLogger("edge-aura.bridge")

# Sensible defaults that are still overridable via env / CLI.
DEFAULT_UNIX_SOCKET = "/var/run/aura_sensors.sock"
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 9100
DEFAULT_COMMAND_LOG = Path("logs/commands.jsonl")

VALID_MODES = ("subprocess_pipe", "unix_socket", "tcp_socket")


# ══════════════════════════════════════════════════════════════════════
# Bridge implementation
# ══════════════════════════════════════════════════════════════════════

class OrchestratorBridge:
    """Forward SensorPackets to PINN-AURA-MFP and fan commands back.

    Parameters
    ----------
    mode:
        One of :data:`VALID_MODES`.
    pinn_root, model_path:
        Required for ``subprocess_pipe`` mode.
    socket_path:
        UNIX socket path (``unix_socket`` mode).
    tcp_host, tcp_port:
        Remote orchestrator endpoint (``tcp_socket`` mode).
    command_log_path:
        File to append every received :class:`OrchestrationCommand` to
        as JSONL.  Parent directories are created on demand.
    actuator:
        Optional :class:`ActuatorStub` to forward commands to.  A
        default stub is constructed when ``None``.
    python_exe:
        Override the Python interpreter used to spawn the PINN subprocess
        (defaults to :data:`sys.executable`).  Primarily for tests.
    """

    def __init__(
        self,
        mode: str = "subprocess_pipe",
        *,
        pinn_root: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        socket_path: str = DEFAULT_UNIX_SOCKET,
        tcp_host: str = DEFAULT_TCP_HOST,
        tcp_port: int = DEFAULT_TCP_PORT,
        command_log_path: Path | str = DEFAULT_COMMAND_LOG,
        actuator: Optional[ActuatorStub] = None,
        python_exe: Optional[str] = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown bridge mode: {mode!r} (valid: {VALID_MODES})")
        self._mode = mode
        self._pinn_root = Path(pinn_root) if pinn_root else None
        self._model_path = Path(model_path) if model_path else None
        self._socket_path = socket_path
        self._tcp_host = tcp_host
        self._tcp_port = int(tcp_port)
        self._command_log_path = Path(command_log_path)
        self._actuator = actuator if actuator is not None else ActuatorStub()
        self._python_exe = python_exe or sys.executable

        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._sock_file: Optional[IO[bytes]] = None
        self._commands_received: int = 0
        self._packets_sent: int = 0
        self._reader_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def commands_received(self) -> int:
        return self._commands_received

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._mode == "subprocess_pipe":
            self._open_subprocess()
        elif self._mode == "unix_socket":
            self._open_unix_socket()
        else:
            self._open_tcp_socket()

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._sock_file is not None:
            try:
                self._sock_file.close()
            finally:
                self._sock_file = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    def __enter__(self) -> "OrchestratorBridge":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ─── Transport setup ─────────────────────────────────────────────
    def _open_subprocess(self) -> None:
        if self._pinn_root is None or not self._pinn_root.exists():
            raise RuntimeError(
                "subprocess_pipe mode requires AURA_EDGE_PINN_ROOT pointing "
                "at a PINN-AURA-MFP clone"
            )
        if self._model_path is None:
            raise RuntimeError(
                "subprocess_pipe mode requires AURA_EDGE_MODEL_PATH "
                "pointing at a trained checkpoint"
            )
        cmd = [
            self._python_exe, "-m", "scripts.predict",
            "--mode", "live",
            "--checkpoint", str(self._model_path),
        ]
        log.info("bridge: spawning PINN orchestrator: %s (cwd=%s)",
                 " ".join(cmd), self._pinn_root)
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._pinn_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=1,
            text=True,
        )
        self._start_reader(self._proc.stdout)

    def _open_unix_socket(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self._socket_path)
        self._sock = sock
        self._sock_file = sock.makefile("rwb", buffering=0)
        self._start_reader(self._sock.makefile("r", encoding="utf-8"))

    def _open_tcp_socket(self) -> None:
        sock = socket.create_connection((self._tcp_host, self._tcp_port), timeout=10)
        self._sock = sock
        self._sock_file = sock.makefile("rwb", buffering=0)
        self._start_reader(self._sock.makefile("r", encoding="utf-8"))

    def _start_reader(self, stream: IO) -> None:
        self._reader_thread = threading.Thread(
            target=self._read_commands, args=(stream,),
            name="orchestrator-bridge-reader", daemon=True,
        )
        self._reader_thread.start()

    # ─── Send / receive ──────────────────────────────────────────────
    def send_packet(self, packet: dict[str, Any]) -> None:
        """Serialise *packet* as a JSON line and forward to the orchestrator."""
        line = json.dumps(packet, separators=(",", ":")) + "\n"
        if self._mode == "subprocess_pipe":
            if self._proc is None or self._proc.stdin is None:
                raise RuntimeError("bridge subprocess not open")
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                log.error("bridge: subprocess stdin write failed: %s", exc)
                return
        else:
            if self._sock_file is None:
                raise RuntimeError("bridge socket not open")
            try:
                self._sock_file.write(line.encode("utf-8"))
            except OSError as exc:
                log.error("bridge: socket write failed: %s", exc)
                return
        self._packets_sent += 1

    def _read_commands(self, stream: IO) -> None:
        """Background loop – forwards commands to the log and actuator."""
        for raw in stream:
            if self._stop.is_set():
                break
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            raw = raw.strip()
            if not raw:
                continue
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("bridge: malformed command JSON: %s", exc)
                continue
            self._handle_command(cmd)

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        self._commands_received += 1
        self._append_command_log(cmd)
        try:
            self._actuator.apply(cmd)
        except Exception as exc:  # noqa: BLE001 – actuator never crashes the bridge
            log.error("bridge: actuator failed on command: %s", exc)

    def _append_command_log(self, cmd: dict[str, Any]) -> None:
        try:
            self._command_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._command_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(cmd, separators=(",", ":")) + "\n")
        except OSError as exc:
            log.error("bridge: failed to append command log %s: %s",
                      self._command_log_path, exc)

    # ------------------------------------------------------------------
    def run(self, source: Optional[IO] = None) -> int:
        """Pump JSON-line ``SensorPacket`` records from *source* into the bridge.

        Blocks until the source closes (EOF) or :meth:`close` runs.
        Returns the number of packets forwarded.
        """
        if source is None:
            source = sys.stdin
        for raw in source:
            raw = raw.strip()
            if not raw:
                continue
            try:
                pkt = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("bridge: malformed sensor packet JSON: %s", exc)
                continue
            self.send_packet(pkt)
        return self._packets_sent


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

def _config_from_env_and_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDGE-AURA-MFP orchestrator bridge")
    p.add_argument("--mode",
                   choices=list(VALID_MODES),
                   default=os.environ.get("AURA_EDGE_BRIDGE_MODE", "subprocess_pipe"))
    p.add_argument("--pinn-root",
                   default=os.environ.get("AURA_EDGE_PINN_ROOT"))
    p.add_argument("--model-path",
                   default=os.environ.get("AURA_EDGE_MODEL_PATH"))
    p.add_argument("--socket-path",
                   default=os.environ.get("AURA_EDGE_SOCKET_PATH", DEFAULT_UNIX_SOCKET))
    p.add_argument("--tcp-host",
                   default=os.environ.get("AURA_EDGE_TCP_HOST", DEFAULT_TCP_HOST))
    p.add_argument("--tcp-port", type=int,
                   default=int(os.environ.get("AURA_EDGE_TCP_PORT", DEFAULT_TCP_PORT)))
    p.add_argument("--command-log",
                   default=os.environ.get("AURA_EDGE_COMMAND_LOG",
                                          str(DEFAULT_COMMAND_LOG)))
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _config_from_env_and_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    bridge = OrchestratorBridge(
        mode=args.mode,
        pinn_root=args.pinn_root,
        model_path=args.model_path,
        socket_path=args.socket_path,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        command_log_path=args.command_log,
    )
    with bridge:
        sent = bridge.run(source=sys.stdin)
    log.info("bridge exit – %d packets sent, %d commands received",
             sent, bridge.commands_received)
    return 0


if __name__ == "__main__":
    sys.exit(main())
