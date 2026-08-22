"""workstation/__init__.py – Workstation-side calibration, packet schema, and
calibration wizard for EDGE-AURA-MFP. Runs on the workstation, not the
Pico -- the Pico only emits raw sensor counts over USB serial (see
DAQ4MPC/pico/), everything here applies calibration and builds/validates
the PINN_SENSOR_PACKET_SCHEMA packets that reach decision_server.py."""
