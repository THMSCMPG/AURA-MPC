"""pico/config.py – Hardware pin map for the RP2040.

All physical pin assignments live here so that main.py and driver modules
never have to hard-code GPIO numbers.  Change this file to adapt the
firmware to a different board layout.
"""

# ── I²C0  –  DS3231 RTC ────────────────────────────────────────────────────
I2C_ID       = 0
I2C_SDA_PIN  = 20  # GP20
I2C_SCL_PIN  = 21  # GP21
I2C_FREQ     = 400_000   # 400 kHz fast-mode

# ── SPI0  –  MAX31856 thermocouple amplifier bank (up to 4 devices) ────────
SPI_ID       = 0
SPI_SCK_PIN  = 2   # GP2
SPI_MOSI_PIN = 3   # GP3
SPI_MISO_PIN = 4   # GP4
# One chip-select per channel (TC0..TC3), active-low.
SPI_CS_PINS  = (5, 6, 7, 8)
SPI_BAUD     = 5_000_000  # 5 MHz

# ── ADC  –  Pyranometer ────────────────────────────────────────────────────
ADC_PIN      = 26  # GP26 = ADC0

# ── UART1  –  RS-485 ModBus anemometer ────────────────────────────────────
RS485_UART_ID  = 1
RS485_TX_PIN   = 12  # GP12 (moved – GP8 now a TC CS pin)
RS485_RX_PIN   = 13  # GP13
RS485_DE_PIN   = 14  # GP14
RS485_BAUD     = 9600
RS485_MODBUS_ADDR = 0x01  # anemometer slave address

# ── UART0  –  Host link to Raspberry Pi ────────────────────────────────────
HOST_UART_ID  = 0
HOST_TX_PIN   = 0   # GP0
HOST_RX_PIN   = 1   # GP1
HOST_BAUD     = 115_200

# ── Watchdog ───────────────────────────────────────────────────────────────
WDT_TIMEOUT_MS = 8000   # 8 s  (RP2040 maximum is 8388 ms)

# ── Sensor loop timing ─────────────────────────────────────────────────────
SAMPLE_INTERVAL_MS = 1000   # read all sensors every 1 s

# ── Fault management ──────────────────────────────────────────────────────
# Consecutive errors before a sensor's fault flag is latched permanently
# (for the remainder of the boot session).
FAULT_PERSIST_THRESHOLD = 10

# One-line persistent fault record written to the Pico's flash on the first
# time a sensor exceeds FAULT_PERSIST_THRESHOLD.  Kept tiny – flash has
# limited erase-write cycles, so we never rewrite an existing record.
FAULT_LOG_PATH = "/faults.log"
