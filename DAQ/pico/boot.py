"""pico/boot.py – Early initialisation, runs before main.py.

MicroPython executes boot.py once immediately after power-on / reset.
Keep this file lean: disable USB mass-storage so the filesystem is not
accidentally corrupted, tune the garbage collector, and configure the CPU
frequency.
"""

import gc
import machine

# ── Raise CPU clock to 133 MHz (default is 125 MHz; max stable ~250 MHz) ──
machine.freq(133_000_000)

# ── Garbage collector: run more aggressively on the small heap ─────────────
gc.enable()
gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())

# ── Disable USB mass-storage in production builds ─────────────────────────
# Uncomment once the filesystem is stable so accidental PC connections
# cannot corrupt sensor logs stored on the Pico's flash.
#
# import usb_cdc
# usb_cdc.disable()
