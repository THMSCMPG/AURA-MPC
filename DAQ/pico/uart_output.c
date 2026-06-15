/* pico/uart_output.c – UART0 and USB CDC packet emitter. */

#include "uart_output.h"
#include "config.h"

#include "hardware/uart.h"
#include "hardware/gpio.h"
#include "pico/stdio.h"

#include <string.h>

void uart_output_init(void)
{
    uart_init(JSON_UART_ID, JSON_BAUD);
    gpio_set_function(JSON_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(JSON_RX_PIN, GPIO_FUNC_UART);
}

void uart_output_emit(const char *json_str)
{
    if (!json_str) { return; }

    /* UART0 – production link */
    uart_puts(JSON_UART_ID, json_str);
    uart_putc_raw(JSON_UART_ID, '\n');

    /* USB CDC – development mirror (printf routes to stdio_usb) */
    printf("%s\n", json_str);
}
