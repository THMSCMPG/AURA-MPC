/* pico/uart_output.h -- UART0 and USB CDC packet emitter.
 *
 * Every emitted JSON packet is written to two sinks in sequence:
 *
 *   1. USB CDC (stdio_usb)          -- PRIMARY link to the workstation
 *      (Pico-only design, confirmed this session: USB serial instead of
 *      a Pi/Pico-W network link -- decision_server.py's workstation-side
 *      serial bridge reads this directly)
 *   2. UART0 (GP0/GP1, 115200 baud) -- to SparkFun OpenLog, local
 *      black-box logging independent of the workstation link
 *
 * Each packet is terminated with a newline ('\n') on both sinks so that
 * the receiver can frame individual JSON objects by line -- this is
 * exactly what OpenLog needs (it just writes whatever it receives on
 * UART straight to a file, no protocol of its own) and exactly what a
 * plain pyserial readline() loop needs on the workstation side.
 *
 * Initialisation
 * --------------
 * Call uart_output_init() once during startup (after pico-sdk stdio_init()
 * and the UART peripheral are ready).  The function configures UART0 with
 * the baud rate and pin assignments from config.h.
 */

#ifndef AURA_UART_OUTPUT_H
#define AURA_UART_OUTPUT_H

/* Initialise UART0 for JSON stream output (to OpenLog). */
void uart_output_init(void);

/* Write a NUL-terminated JSON string followed by '\n' to both USB CDC
 * (primary, workstation) and UART0 (secondary, OpenLog).               */
void uart_output_emit(const char *json_str);

#endif /* AURA_UART_OUTPUT_H */
