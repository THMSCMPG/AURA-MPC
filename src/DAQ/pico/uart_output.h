/* pico/uart_output.h – UART0 and USB CDC packet emitter.
 *
 * Every emitted JSON packet is written to two sinks in sequence:
 *
 *   1. UART0 (GP0/GP1, 115200 baud) – production link to the Raspberry Pi
 *   2. USB CDC (stdio_usb)           – development/debugging mirror
 *
 * Each packet is terminated with a newline ('\n') on both sinks so that
 * the receiver can frame individual JSON objects by line.
 *
 * Initialisation
 * --------------
 * Call uart_output_init() once during startup (after pico-sdk stdio_init()
 * and the UART peripheral are ready).  The function configures UART0 with
 * the baud rate and pin assignments from config.h.
 */

#ifndef AURA_UART_OUTPUT_H
#define AURA_UART_OUTPUT_H

/* Initialise UART0 for JSON stream output. */
void uart_output_init(void);

/* Write a NUL-terminated JSON string followed by '\n' to both UART0
 * and USB CDC.                                                         */
void uart_output_emit(const char *json_str);

#endif /* AURA_UART_OUTPUT_H */
