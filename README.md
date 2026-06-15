# AURA-MPC — Model Predictive Control

**Atmosphere-Unified Radiation Assessment with Multi-Fidelity for Photovoltaics**

AURA-MPC is a research monorepo that couples edge sensor hardware, a physics-informed neural network (PINN), and a Fortran simulator into a closed-loop model predictive control system. Sensor data captured by a Pico RP2040 / Pi 3B+ edge node is routed through the PINN orchestrator, which selects the appropriate simulation fidelity tier and emits actuator setpoints in real time.

## Quick Start

```bash
# Clone
git clone https://github.com/THMSCMPG/AURA-MPC.git
cd AURA-MPC

# Build all Fortran tiers
chmod +x make.sh
./make all

# Install PINN dependencies
cd PINN
pip install -e .[dev]
```

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
