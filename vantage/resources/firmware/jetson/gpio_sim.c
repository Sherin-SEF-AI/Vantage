/* gpio_sim.c: a stand-in for the GPIO lines on a machine without them (Vantage's host check): every write is
 * accepted and nothing is driven; the program's --stats still measure when each edge would have gone out. */
#include "gpio.h"

static uint32_t current;

int gpio_open(const char *chip, const unsigned *offsets, unsigned n, uint32_t idle)
{
    (void)chip;
    (void)offsets;
    (void)n;
    current = idle;
    return 0;
}

int gpio_write(uint32_t state)
{
    current = state;
    return 0;
}

void gpio_close(void) {}
