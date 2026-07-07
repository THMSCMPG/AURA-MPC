# AURA-MPC Mathematical and Algorithmic Manual
## Lab Journal Reference Guide

This document presents the mathematical formulations, derivations, numerical methods, and control algorithms used across the AURA-MPC (AURA Multi-Fidelity PV Control) ecosystem. It spans from the baseline physical simulators (AURA-MFP) to the neural architecture (PINN-AURA-MFP) and the edge deployment modules (EDGE-AURA-MFP). 

---
## 1. RK4TRAN
Using variation implementations of Runge Kutta, we are able to predict future incremental steps in behavior of the complex system. We can use synthetic data alongside real data to train the PINN. In addition to this, we can use these models to predict real time data that can feed into the PINN. By randomly shuffling the data rng alongside Monte Carlo Uncertainty Quantification techniques, we are able to produce realistic synthetic data that avoids the pitfalls of common fitting techniques.
### 1.1 Runge-Kutta 4 (RK4)

For a state model

$$\frac{d\mathbf{x}}{dt} = f(t,\mathbf{x},\mathbf{u},\boldsymbol{\theta})$$

the fixed-step RK4 update over $\Delta t$ is:

$$\mathbf{k}_1 = f(t_n,\mathbf{x}_n,\mathbf{u}_n,\boldsymbol{\theta})$$
$$\mathbf{k}_2 = f\left(t_n+\frac{\Delta t}{2},\mathbf{x}_n+\frac{\Delta t}{2}\mathbf{k}_1,\mathbf{u}_n,\boldsymbol{\theta}\right)$$
$$\mathbf{k}_3 = f\left(t_n+\frac{\Delta t}{2},\mathbf{x}_n+\frac{\Delta t}{2}\mathbf{k}_2,\mathbf{u}_n,\boldsymbol{\theta}\right)$$
$$\mathbf{k}_4 = f\left(t_n+\Delta t,\mathbf{x}_n+\Delta t\mathbf{k}_3,\mathbf{u}_n,\boldsymbol{\theta}\right)$$

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{\Delta t}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)$$

For the thermal submodel, a common single-state form is:

$$\frac{dT_{\text{panel}}}{dt} = \frac{T_{\text{ss}} - T_{\text{panel}}}{\tau_{\text{eff}}}, \quad T_{\text{ss}} = T_{\text{amb}} + \frac{\alpha G_{\text{poa}}}{U_0 + U_1 WS}$$

RK4 is used when deterministic replay and fixed compute budget are more important than adaptive local error control.

### 1.2 Runge-Kutta-Felberg (RK45)

RK45 computes two embedded solutions of different order per step:

$$\mathbf{x}_{n+1}^{(5)} \quad \text{and} \quad \mathbf{x}_{n+1}^{(4)}$$

with local truncation error estimate:

$$\mathbf{e}_{n+1} = \mathbf{x}_{n+1}^{(5)} - \mathbf{x}_{n+1}^{(4)}$$

Step acceptance uses an error norm against tolerance $\epsilon$:

$$\left\lVert \mathbf{e}_{n+1} \right\rVert \le \epsilon \Rightarrow \text{accept step}$$

Adaptive time-step control:

$$\Delta t_{\text{new}} = s \cdot \Delta t \left(\frac{\epsilon}{\left\lVert \mathbf{e}_{n+1} \right\rVert}\right)^{1/5}$$

where $s \in (0,1)$ is a safety factor. RK45 is preferred for stiff-transition periods (rapid irradiance/wind changes) to reduce both under-resolution and unnecessary over-sampling.

### 1.3 Monte Carlo Uncertainty Quantification

Uncertain inputs are modeled as random variables:

$$\boldsymbol{\xi} = [G_{\text{poa}}, T_{\text{amb}}, WS, \tau_{\text{eff}}, U_0, U_1, \ldots]$$

with prior distributions $p(\boldsymbol{\xi})$. For $N$ samples:

$$\boldsymbol{\xi}^{(i)} \sim p(\boldsymbol{\xi}), \quad y^{(i)} = \mathcal{M}(\boldsymbol{\xi}^{(i)})$$

Estimated moments and quantiles:

$$\hat{\mu}_y = \frac{1}{N}\sum_{i=1}^N y^{(i)}, \quad \hat{\sigma}_y^2 = \frac{1}{N-1}\sum_{i=1}^N \left(y^{(i)}-\hat{\mu}_y\right)^2$$

$$\text{CI}_{95\%}(y) \approx \left[q_{0.025}(y), q_{0.975}(y)\right]$$

In AURA-MPC, Monte Carlo outputs are used to:
1. Generate synthetic trajectories for PINN regularization.
2. Estimate uncertainty bands used by watchdog logic.
3. Rank sensitivity of control-relevant parameters before retraining.

## 2. PINN: Neural Network Architecture & Physics-Informed Loss

The PINN fuses weather variables and sky camera embeddings through a shared trunk to predict temperature, routing probabilities, and actuator setpoints.

### 2.1 Network Architecture
- **Shared Trunk**: Fuses numeric features $x_{\text{num}} \in \mathbb{R}^7$ and image embeddings $x_{\text{img}} \in \mathbb{R}^{32}$.
- **Temperature Head**: Outputs the predicted normalized temperature $\hat{T}$.
- **Routing Head**: Emits the 5-way routing probability vector.
- **Pose Head**: Outputs the 4-DoF actuator setpoints.

### 2.2 Log-Space and Sigmoid Parameter Constraints
To enforce physical bounds structurally during training, the learnable parameters are constrained using mapping functions:

$$\tau_{\text{eff}} = \exp\left(\text{log\_tau}\right)$$

$$U_0 = \exp\left(\text{log\_U0}\right)$$

$$U_1 = \exp\left(\text{log\_U1}\right)$$

$$\gamma_{\text{CC}} = \text{sigmoid}\left(\text{raw\_gamma\_CC}\right)$$

This ensures that parameters remain within physically realistic bounds regardless of optimizer updates.

### 2.3 Total Loss Formulation
The total loss minimized during training is a weighted combination of five distinct terms:

$$L_{\text{total}} = \lambda_{\text{data}} L_{\text{data}} + \lambda_{\text{phys}} L_{\text{phys}} + \lambda_{\text{IC}} L_{\text{IC}} + \lambda_{\text{route}} L_{\text{route}} + \lambda_{\text{pose}} L_{\text{pose}}$$

#### 2.3.1 Data Loss ($L_{\text{data}}$)
Computes the mean squared error (MSE) of the predicted temperature $\hat{T}$ relative to the ground-truth panel temperature $T_{\text{panel}}$:

$$L_{\text{data}} = \frac{1}{B} \sum_{i=1}^B \left(\hat{T}_i - T_{\text{panel}, i}\right)^2$$

#### 2.3.2 Physics Residual Loss ($L_{\text{phys}}$)
Enforces the Faiman thermal ODE at $N_{\text{coll}}$ collocation points. The residual is defined as:

$$r_i = \tau_{\text{eff}} \frac{\partial \hat{T}}{\partial t} - \left(T_{\text{ss}} - \hat{T}\right)$$

The derivative $\frac{\partial \hat{T}}{\partial t}$ is computed using PyTorch autograd. The loss term scales this residual by $\tau_0$ to keep it $O(1)$:

$$L_{\text{phys}} = \frac{1}{N_{\text{coll}}} \sum_{j=1}^{N_{\text{coll}}} \left( \frac{r_j}{\tau_0} \right)^2$$

#### 2.3.3 Initial Condition Loss ($L_{\text{IC}}$)
Enforces that at zero irradiance and wind speed, the panel temperature matches ambient:

$$L_{\text{IC}} = \frac{1}{B} \sum_{i=1}^B \left(\hat{T}_{\text{IC}, i} - T_{\text{amb}, i}\right)^2$$

#### 2.3.4 Class-Weighted Routing Loss ($L_{\text{route}}$)
To prevent the model from collapsing into majority classes, cross-entropy is weighted using the inverse class frequencies:

$$L_{\text{route}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=1}^5 w_c \cdot \mathbb{I}(y_i = c) \cdot \log\left(\text{softmax}(\text{logits})_{i, c}\right)$$

Where the normalized weights are:

$$w_c = \frac{1 / N_c}{\sum_{j=1}^5 1 / N_j} \cdot 5$$

#### 2.3.5 Bounded Pose Loss ($L_{\text{pose}}$)
When optimal pose labels $P_{\text{opt}} = [\text{pitch}, \text{yaw}, \text{roll}, z]^T$ are available, the pose loss is computed as a normalized MSE:

$$L_{\text{pose}} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^4 \left( \frac{\hat{P}_{i, k} - P_{\text{opt}, i, k}}{\sigma_k} \right)^2$$

Where the normalization scale vector is $\sigma = [35.0, 180.0, 25.0, 3.0]^T$.

---

## 3. DAQ: Edge Communication & Orchestration

The edge node operates on a Pico RP2040 and a Pi 3B+ daemon. It serializes sensor packets, validates inputs, and implements the real-time orchestration loop.

### 3.1 COBS Encoding
The Pico RP2040 serializes telemetry data using Consistent Overhead Byte Stuffing (COBS). COBS eliminates the zero byte (`0x00`) from the payload, reserving it as a frame delimiter:

$$\text{COBS}(P) = [d_1, p_{1,1}, \dots, d_2, p_{2,1}, \dots, 0x00]$$

Where each overhead byte $d_i$ indicates the offset to the next zero byte in the stream, allowing robust, non-blocking frame synchronization.

### 3.2 CRC16-CCITT Verification
To protect against transmission errors over UART, each frame is appended with a 16-bit Cyclic Redundancy Check checksum using the CCITT polynomial:

$$G(x) = x^{16} + x^{12} + x^5 + 1$$

The calculation uses a bitwise shift register:

$$\text{CRC}_{k} = \left(\text{CRC}_{k-1} \ll 1\right) \oplus \left( \text{if MSB}(\text{CRC}_{k-1} \oplus \text{byte}) \text{ then } 0x1021 \text{ else } 0 \right)$$

If the received checksum does not match, the packet is discarded and the fault counter is incremented.

### 3.3 Real-Time Orchestration & Watchdog logic
The orchestrator runs on the Pi 3B+ and executes the following control loop at cadence $\Delta t$:
1. **Ingest & Validate**: Reads the latest sensor packet.
2. **Watchdog Evaluation**: engagement threshold check:
   $$\text{tripped} = (\text{uncertainty} > \text{watchdog\_threshold}) \lor (\text{fault\_count} \ge \text{max\_faults})$$
3. **Override / Fallback**:
   - If tripped, engage fallback: set mode to SIMV4 and hold the last-known-good pose.
   - Else, update actuator targets using the model's predicted pose or the heuristic equations.
4. **Slew-Rate Limiting**:
   $$x_k^{n+1} = x_k^n + \text{clip}\left(x_{target, k} - x_k^n, -R_k \Delta t, R_k \Delta t\right)$$
   Where $R_k$ is the slew rate limit for axis $k$.
5. **Transmit**: Writes the command to the actuator stub.

---

## 4. Updated Next-Step Plan (Post-Journal Update)

1. **Lock physics baseline**: Freeze the current RK4/RK45 + thermal parameter set and produce a reproducible baseline dataset split (train/validation/stress).
2. **Uncertainty calibration pass**: Run Monte Carlo sweeps over sensor noise and weather perturbations; set watchdog uncertainty thresholds from empirical percentiles rather than fixed heuristics.
3. **PINN retraining cycle**: Retrain with refreshed collocation strategy (clear-sky, variable-cloud, gust events), then compare data-loss vs physics-loss Pareto tradeoff.
4. **MPC-in-the-loop validation**: Validate closed-loop trajectories in SIL and HIL, focusing on constraint violations, actuator slew saturation, and fallback-trigger frequency.
5. **Edge DAQ hardening**: Add synchronized timestamp checks, packet-loss accounting, and drift monitoring across Pico + Pi clocks under long-duration runs.
6. **Field trial readiness gate**: Define pass/fail criteria for 72-hour autonomous operation (uptime, CRC error rate, watchdog activations, net energy gain).

---

## 5. Updated Bill of Materials (No Links / No Prices)

| Category | Item | Qty | Purpose |
|---|---|---:|---|
| Edge Compute | Raspberry Pi 4 Model B (4 GB or 8 GB) | 1 | Edge orchestration, logging, MPC inference |
| Edge Compute | High-endurance microSD card (64 GB+) | 1 | Reliable local data logging |
| Edge Compute | Active heatsink/fan kit for Pi | 1 | Thermal stability for continuous operation |
| Sensor MCU | Raspberry Pi Pico RP2040 | 1 | Deterministic sensor polling and framing |
| Sensor Interface | Logic-level UART/I2C breakout wiring set | 1 set | Clean bus wiring between nodes |
| Power | 5 V regulated supply for Pi (3 A) | 1 | Stable SBC power rail |
| Power | 5 V regulated supply for Pico/sensors | 1 | Isolated sensor-domain power |
| Meteorology | Pyranometer or calibrated irradiance sensor | 1 | POA irradiance measurement |
| Meteorology | Ambient temperature and RH sensor (industrial grade) | 1 | Weather state for thermal model |
| Meteorology | Wind speed sensor (anemometer) | 1 | Convective cooling input |
| Meteorology | Wind direction vane (optional but recommended) | 1 | Advanced flow-aware control features |
| PV Thermal | Backsheet/module temperature probe (PT100/PT1000 or thermistor) | 2 | Ground truth for model fitting and validation |
| Mechanical | 4-DoF actuator assembly (pitch/yaw/roll/z) | 1 | Physical panel reorientation |
| Mechanical | Motor drivers matched to actuator type | 1 set | Motion control execution |
| Control Safety | Limit switches/end-stop sensors | 1 set | Hard motion boundary enforcement |
| Control Safety | Emergency stop relay or cutoff chain | 1 | Safety interlock |
| Enclosure | IP65 electronics enclosure with cable glands | 1 | Outdoor protection |
| Enclosure | DIN rail / terminal blocks / fusing | 1 set | Serviceable power and wiring layout |
| Comms | USB-UART adapter (debug and commissioning) | 1 | Field diagnostics |
| Comms | Optional LTE modem for remote telemetry | 1 | Backhaul when Wi-Fi unavailable |

---

## 6. Top 10 Papers to Read for AURA-MPC

1. Fehlberg, E. (1969). **Low-order classical Runge-Kutta formulas with stepsize control**. NASA Technical Report R-315.
2. Dormand, J. R., & Prince, P. J. (1980). **A family of embedded Runge-Kutta formulae**. *Journal of Computational and Applied Mathematics*.
3. Saltelli, A. (2002). **Making best use of model evaluations to compute sensitivity indices**. *Computer Physics Communications*.
4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations**. *Journal of Computational Physics*.
5. Karniadakis, G. E., et al. (2021). **Physics-informed machine learning**. *Nature Reviews Physics*.
6. Mayne, D. Q., Rawlings, J. B., Rao, C. V., & Scokaert, P. O. M. (2000). **Constrained model predictive control: Stability and optimality**. *Automatica*.
7. Qin, S. J., & Badgwell, T. A. (2003). **A survey of industrial model predictive control technology**. *Control Engineering Practice*.
8. Mathur, V., Saini, Y., Giri, V., et al. (2021). **Weather Station Using Raspberry Pi**. *Proceedings of the 2021 International Conference on Intelligent Information Processing (IEEE)*.
9. De Soto, W., Klein, S. A., & Beckman, W. A. (2006). **Improvement and validation of a model for photovoltaic array performance**. *Solar Energy*.
10. Faiman, D. (2008). **Assessing the outdoor operating temperature of photovoltaic modules**. *Progress in Photovoltaics: Research and Applications*.
