# AURA-MPC — Remaining Work Checklist

---

## 1. Real-world execution on Loki (blocks almost everything downstream)

Nothing in this section has changed — still fully code-complete and
validated in a sandbox, none of it has touched the real cluster yet.

- [ ] **Run `build_lattice_pools.py` for real, with real network access —
  not `--offline-stub`.** Every test so far used the offline stub
  (placeholder zero elevations) since this sandbox has no internet
  access to query Open-Elevation. The real 1050-point pool (550
  structured + 500 scattered land points) needs real elevation data
  before any real generation run.
- [ ] **Run the actual v3 generation + streaming training sprint(s) on
  Loki.** The one item everything else depends on: a real trained PINN
  checkpoint, meaningful (non-random) MPC predictions, and real
  diagnostic plots all wait on this.
- [ ] **Get real SLURM specifics from your own account** — partition
  name(s), exact walltime/memory limits (`sinfo`/`sacctmgr`, not public
  info). Every `.slurm` script has a placeholder
  `#SBATCH --partition=PARTITION_NAME_HERE` waiting to be filled in.
- [ ] **Confirm Loki's SLURM config allows node-sharing** for the
  generation array job (multiple 1-core tasks packed onto one 8-core
  node) — assumed as a standard default, not confirmed against your
  actual allocation.
- [ ] **Confirm real multi-core throughput numbers on actual Loki
  hardware.** Every benchmark so far (the 58x data-loading speedup, the
  generation-vs-training pace ratio) was measured on a 1-core sandbox.
  The *ratio* should transfer reasonably well; absolute per-location
  minutes haven't been confirmed on Loki's real (older Xeon L5420)
  hardware.
- [ ] **Confirm the full chain end-to-end under real SLURM**:
  `select_training_locations.py` → generation array job → single
  trainer watcher, picking up chunks via the shared NFS directory and
  `.ready` markers. Validated only via simulated/short sandbox tests so
  far, not a real multi-node SLURM run.

---

## 2. DAQ4MPC hardware — Pico-only build

**Confirmed architecture**: Pico-only (Pi 3B+ dropped), USB serial to the
workstation (not Pico W), calibration lives entirely on the workstation
(the Pico just translates raw sensor counts). Components: Pico +
SparkFun weather:bit (Qwiic, onboard BME280) + SparkFun Weather Meter
Kit + SparkFun OpenLog + MAX31856 thermocouple amplifier (5 real
channels).

**Genuinely still open, all hardware-dependent (correctly deferred, not
forgotten) unless noted:**

- [ ] **Wind/rain/vane wiring decision.** Once the weather:bit board is
  in hand: tap its actual traces for the P1 (wind direction)/P2
  (rain)/P8 (wind speed) pads directly, vs. bypass it for these three
  sensors and wire the raw Weather Meter Kit heads straight to the
  Pico's own GPIO/ADC pins. (The BME280 side is separately confirmed —
  clean I2C over Qwiic, not in question.)
- [ ] **5th thermocouple chip-select GPIO pin assignment**, then port
  the (already channel-count-fixed) MicroPython reference driver to the
  Pico's C firmware. Needs to avoid conflicting with the weather:bit's
  I2C pins and OpenLog's UART pins, already assigned.
- [ ] **Move `T_amb`/`WS` off analog ADC** once the above two are
  decided — currently still placeholder RP2040 ADC channels in
  firmware; the real plan is BME280 (I2C) for `T_amb` and the Weather
  Meter Kit's digital pulse for `WS`.
- [ ] **Physically wire OpenLog and confirm it logs correctly** — the
  firmware-side logic is ready (dual-emit to USB CDC + UART0), genuinely
  just needs real hardware to test against.
- [ ] **Update schematics/BOM** to reflect the confirmed architecture.
- [ ] **Full rewrite of `health_check.py`'s and `simulate.py`'s
  live-hardware/mock paths** against the new raw-packet format — both
  currently just import cleanly again, their actual sensor-reading logic
  still reflects the old dead architecture.
- [ ] **Real hardware test** — everything validated against a
  pty-simulated Pico and the real wire format as closely as could be
  confirmed from source, not an actual physical device.

---

## 3. MPC runtime — loose ends from the sandbox/ rework

- [ ] **No real EDGE hardware test.** Everything validated against
  mocked/simulated packets, not an actual physical device sending real
  packets — same item as in Section 2, listed here too since it also
  blocks confidence in the MPC decision loop specifically.
- [ ] **Confirm the "light" predicted-vs-actual logging is actually
  sufficient once real sessions start.** `ClosedLoopRuntime.record_outcome()`
  exists and is tested (pre/post efficiency, good/bad self-grading, tied
  to a `decision_id`) — worth revisiting once you have a real session's
  worth of data to see if it captures what the eventual paper analysis
  needs, or wants more fields.

---

## 4. Workstation Application

- [ ] **Write up new Matlab Workstation Application** at root "workstation/" directory
- [ ] **Workstation parameters** include:
    - [ ] being the primary application source that runs the machine learning code
    - [ ] allows user to view the 3D render of the solar panel, the live data being collected, and the decisions
    - [ ] much like run_prototype_simulation.m and run_simulation.m
    - [ ] present plots, replay series, run RK4TRAIN smoke tests of training, run callibration sequencing
    - [ ] part of callibration sequencing confirms Pico USB connection and all sensors are green
- [ ] **Auto-detect the Pico's USB connection** rather than requiring an
  explicit `--serial-port` argument — part of your stated "application
  that detects a connection" vision, not yet built.

