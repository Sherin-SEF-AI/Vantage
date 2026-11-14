/* gpio.h: the trigger lines as one output port (bit i = line i), over libgpiod or the simulation backend. */
#ifndef TB_GPIO_H
#define TB_GPIO_H
#include <stdint.h>
int gpio_open(const char *chip, const unsigned *offsets, unsigned n, uint32_t idle);
int gpio_write(uint32_t state);
void gpio_close(void);
#endif
