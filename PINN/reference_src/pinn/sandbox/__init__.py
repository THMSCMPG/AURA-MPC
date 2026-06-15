"""Physics-based 3D RL sandbox package for phase-2 training + live visualization."""

from .environment import PanelEnv
from .live import JsonlSensorStream, run_live_stream, sensor_packet_from_record
from .training import train_policy, validate_against_fortran
from .viewer import map_command_to_twin, read_latest_command, run_viewer

__all__ = [
    "PanelEnv",
    "JsonlSensorStream",
    "run_live_stream",
    "sensor_packet_from_record",
    "train_policy",
    "validate_against_fortran",
    "map_command_to_twin",
    "read_latest_command",
    "run_viewer",
]
