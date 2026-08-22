"""MPC candidate-evaluation sandbox: live/replay decision loop + validation.

REWORKED this session -- PanelEnv (Gym-style RL environment) and
PolicyNetwork/SandboxTrainer (policy-gradient training) were removed along
with the live weight-updating RL approach they supported. See runtime.py's
module docstring for the full decision history.
"""

from .integration import ComparisonMetrics, PINNValidator, RK4TRANValidator, SandboxPINNAgent
from .matlab_bridge import MatlabSimulationBridge
from .runtime import ClosedLoopRuntime, CandidateResult, SessionConditions
from .viewer import Viewer3D, ViewerFactory

__all__ = [
    "SandboxPINNAgent",
    "PINNValidator",
    "RK4TRANValidator",
    "ComparisonMetrics",
    "ClosedLoopRuntime",
    "CandidateResult",
    "SessionConditions",
    "MatlabSimulationBridge",
    "Viewer3D",
    "ViewerFactory",
]
