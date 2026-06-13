# AURA-MPC — Model Predictive Control

**Atmosphere-Unified Radiation Assessment with Multi-Fidelity for Photovoltaics**

AURA-MPC is a research monorepo that couples edge sensor hardware, a physics-informed neural network (PINN), and a Fortran multi-fidelity photovoltaic simulator into a closed-loop model predictive control system. Sensor data captured by a Pico RP2040 / Pi 3B+ edge node is routed through the PINN orchestrator, which selects the appropriate simulation fidelity tier and emits actuator setpoints in real time.

## Modules

| Module | Language | Purpose |
|--------|----------|---------|
| [`modules/aura-mfp`](modules/aura-mfp/) | Fortran | Multi-fidelity PV simulator — eight tiers simv0 → simv4 (including b-variants) |
| [`modules/pinn-aura-mfp`](modules/pinn-aura-mfp/) | Python / PyTorch | Physics-informed neural network with dual heads for panel temperature and 5-way simulator routing |
| [`modules/edge-aura-mfp`](modules/edge-aura-mfp/) | MicroPython / C / Python | Pico RP2040 firmware, Pi 3B+ daemon, and deploy scripts |

## Simulation Tiers

| Tier | Description |
|------|-------------|
| simv0 | Baseline single-diode model, no temperature correction |
| simv1 | Faiman thermal model; BTE-NS-T coupled atmospheric solver |
| simv1b | simv1 with spectral inversion variant |
| simv2 | Sandia empirical PV model |
| simv2b | simv2 with IIR filter smoothing |
| simv3 | Sandia model + energy-balance equation |
| simv3b | simv3 with Fuentes thermal correction |
| simv4 | Highest fidelity — Fortran solver consuming exported PINN weights via `pinn_weights_module.f90` |

## Quick Start

```bash
# Clone
git clone https://github.com/THMSCMPG/AURA-MPC.git
cd AURA-MPC

# Build all Fortran tiers
cd modules/aura-mfp
make all

# Install PINN dependencies
cd ../pinn-aura-mfp
pip install -e .[dev]
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/overview.md](docs/overview.md) | System architecture and data-flow narrative |
| [docs/AURA-MFP/INTEGRATION.md](docs/AURA-MFP/INTEGRATION.md) | Build, run, and plot the Fortran simulator |
| [docs/AURA-MFP/BENCHMARKS.md](docs/AURA-MFP/BENCHMARKS.md) | Simulation tier benchmark reference |
| [docs/PINN-AURA-MFP/INTEGRATION.md](docs/PINN-AURA-MFP/INTEGRATION.md) | PINN install, training, and weight export |
| [docs/PINN-AURA-MFP/ORCHESTRATOR.md](docs/PINN-AURA-MFP/ORCHESTRATOR.md) | Async orchestrator design and routing logic |
| [docs/PINN-AURA-MFP/BENCHMARKS.md](docs/PINN-AURA-MFP/BENCHMARKS.md) | PINN test, lint, and Sandia evaluation commands |
| [docs/EDGE-AURA-MFP/HARDWARE.md](docs/EDGE-AURA-MFP/HARDWARE.md) | Bill of materials and pin mapping |
| [docs/EDGE-AURA-MFP/PROTOCOL.md](docs/EDGE-AURA-MFP/PROTOCOL.md) | Binary frame and SensorPacket JSON schema |
| [docs/EDGE-AURA-MFP/DEPLOYMENT.md](docs/EDGE-AURA-MFP/DEPLOYMENT.md) | Flash Pico and install Pi daemon |
| [docs/EDGE-AURA-MFP/INTEGRATION.md](docs/EDGE-AURA-MFP/INTEGRATION.md) | Full-stack bring-up and bridge modes |
| [docs/EDGE-AURA-MFP/SIMULATION.md](docs/EDGE-AURA-MFP/SIMULATION.md) | Hardware-free simulation |
| [docs/EDGE-AURA-MFP/FIELD_READINESS.md](docs/EDGE-AURA-MFP/FIELD_READINESS.md) | Pre-deployment checklist and troubleshooting |

## Citation

```bibtex
@software{campagna2026aura_mpc,
  author    = {Campagna, Thomas},
  title     = {{AURA-MPC}: Atmosphere-Unified Radiation Assessment with
               Multi-Fidelity for Photovoltaics Model Predictive Control},
  year      = {2026},
  url       = {https://github.com/THMSCMPG/AURA-MPC},
  note      = {Research monorepo — edge sensing, PINN orchestration,
               and Fortran multi-fidelity PV simulation}
}
```

## License

See [LICENSE](LICENSE).
