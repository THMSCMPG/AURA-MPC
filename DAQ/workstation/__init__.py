"""workstation – Workstation-side inference layer for the AURA edge loop.

Receives JSON sensor packets from the Pi 3B+ gateway, routes them to the
appropriate Fortran solver tier, runs PINN-based PSO optimisation, and
returns an OptimalConfigCommand to the gateway.
"""
