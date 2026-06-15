# AURA-MPC Mathematical and Algorithmic Manual
## Lab Journal Reference Guide

This document presents the mathematical formulations, derivations, numerical methods, and control algorithms used across the AURA-MPC (AURA Multi-Fidelity PV Control) ecosystem. It spans from the baseline physical simulators (AURA-MFP) to the neural architecture (PINN-AURA-MFP) and the edge deployment modules (EDGE-AURA-MFP).

---
## 1. RK4TRAN


## 9. PINN: Neural Network Architecture & Physics-Informed Loss

The PINN fuses weather variables and sky camera embeddings through a shared trunk to predict temperature, routing probabilities, and actuator setpoints.

### 9.1 Network Architecture
- **Shared Trunk**: Fuses numeric features $x_{\text{num}} \in \mathbb{R}^7$ and image embeddings $x_{\text{img}} \in \mathbb{R}^{32}$.
- **Temperature Head**: Outputs the predicted normalized temperature $\hat{T}$.
- **Routing Head**: Emits the 5-way routing probability vector.
- **Pose Head**: Outputs the 4-DoF actuator setpoints.

### 9.2 Log-Space and Sigmoid Parameter Constraints
To enforce physical bounds structurally during training, the learnable parameters are constrained using mapping functions:

$$\tau_{\text{eff}} = \exp\left(\text{log\_tau}\right)$$

$$U_0 = \exp\left(\text{log\_U0}\right)$$

$$U_1 = \exp\left(\text{log\_U1}\right)$$

$$\gamma_{\text{CC}} = \text{sigmoid}\left(\text{raw\_gamma\_CC}\right)$$

This ensures that parameters remain within physically realistic bounds regardless of optimizer updates.

### 9.3 Total Loss Formulation
The total loss minimized during training is a weighted combination of five distinct terms:

$$L_{\text{total}} = \lambda_{\text{data}} L_{\text{data}} + \lambda_{\text{phys}} L_{\text{phys}} + \lambda_{\text{IC}} L_{\text{IC}} + \lambda_{\text{route}} L_{\text{route}} + \lambda_{\text{pose}} L_{\text{pose}}$$

#### 9.3.1 Data Loss ($L_{\text{data}}$)
Computes the mean squared error (MSE) of the predicted temperature $\hat{T}$ relative to the ground-truth panel temperature $T_{\text{panel}}$:

$$L_{\text{data}} = \frac{1}{B} \sum_{i=1}^B \left(\hat{T}_i - T_{\text{panel}, i}\right)^2$$

#### 9.3.2 Physics Residual Loss ($L_{\text{phys}}$)
Enforces the Faiman thermal ODE at $N_{\text{coll}}$ collocation points. The residual is defined as:

$$r_i = \tau_{\text{eff}} \frac{\partial \hat{T}}{\partial t} - \left(T_{\text{ss}} - \hat{T}\right)$$

The derivative $\frac{\partial \hat{T}}{\partial t}$ is computed using PyTorch autograd. The loss term scales this residual by $\tau_0$ to keep it $O(1)$:

$$L_{\text{phys}} = \frac{1}{N_{\text{coll}}} \sum_{j=1}^{N_{\text{coll}}} \left( \frac{r_j}{\tau_0} \right)^2$$

#### 9.3.3 Initial Condition Loss ($L_{\text{IC}}$)
Enforces that at zero irradiance and wind speed, the panel temperature matches ambient:

$$L_{\text{IC}} = \frac{1}{B} \sum_{i=1}^B \left(\hat{T}_{\text{IC}, i} - T_{\text{amb}, i}\right)^2$$

#### 9.3.4 Class-Weighted Routing Loss ($L_{\text{route}}$)
To prevent the model from collapsing into majority classes, cross-entropy is weighted using the inverse class frequencies:

$$L_{\text{route}} = -\frac{1}{B} \sum_{i=1}^B \sum_{c=1}^5 w_c \cdot \mathbb{I}(y_i = c) \cdot \log\left(\text{softmax}(\text{logits})_{i, c}\right)$$

Where the normalized weights are:

$$w_c = \frac{1 / N_c}{\sum_{j=1}^5 1 / N_j} \cdot 5$$

#### 9.3.5 Bounded Pose Loss ($L_{\text{pose}}$)
When optimal pose labels $P_{\text{opt}} = [\text{pitch}, \text{yaw}, \text{roll}, z]^T$ are available, the pose loss is computed as a normalized MSE:

$$L_{\text{pose}} = \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^4 \left( \frac{\hat{P}_{i, k} - P_{\text{opt}, i, k}}{\sigma_k} \right)^2$$

Where the normalization scale vector is $\sigma = [35.0, 180.0, 25.0, 3.0]^T$.

---

## 10. DAQ: Edge Communication & Orchestration

The edge node operates on a Pico RP2040 and a Pi 3B+ daemon. It serializes sensor packets, validates inputs, and implements the real-time orchestration loop.

### 10.1 COBS Encoding
The Pico RP2040 serializes telemetry data using Consistent Overhead Byte Stuffing (COBS). COBS eliminates the zero byte (`0x00`) from the payload, reserving it as a frame delimiter:

$$\text{COBS}(P) = [d_1, p_{1,1}, \dots, d_2, p_{2,1}, \dots, 0x00]$$

Where each overhead byte $d_i$ indicates the offset to the next zero byte in the stream, allowing robust, non-blocking frame synchronization.

### 10.2 CRC16-CCITT Verification
To protect against transmission errors over UART, each frame is appended with a 16-bit Cyclic Redundancy Check checksum using the CCITT polynomial:

$$G(x) = x^{16} + x^{12} + x^5 + 1$$

The calculation uses a bitwise shift register:

$$\text{CRC}_{k} = \left(\text{CRC}_{k-1} \ll 1\right) \oplus \left( \text{if MSB}(\text{CRC}_{k-1} \oplus \text{byte}) \text{ then } 0x1021 \text{ else } 0 \right)$$

If the received checksum does not match, the packet is discarded and the fault counter is incremented.

### 10.3 Real-Time Orchestration & Watchdog logic
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
