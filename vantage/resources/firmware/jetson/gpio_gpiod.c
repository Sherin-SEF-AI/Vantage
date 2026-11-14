/* gpio_gpiod.c: the trigger lines through libgpiod. Version 1 API (libgpiod 1.6, what JetPack 5 and 6 install:
 * Ubuntu 20.04 / 22.04) by default; build with -DTB_GPIOD_V2 for libgpiod 2.x. Outputs only. */
#include <gpiod.h>
#include <stddef.h>
#include <stdio.h>

#include "gpio.h"

static unsigned n_lines;

#ifndef TB_GPIOD_V2
static struct gpiod_chip *chip_;
static struct gpiod_line_bulk bulk;

int gpio_open(const char *chip, const unsigned *offsets, unsigned n, uint32_t idle)
{
    int vals[32];
    unsigned i;
    chip_ = chip[0] == '/' ? gpiod_chip_open(chip) : gpiod_chip_open_by_name(chip);
    if (!chip_)
        return -1;
    if (gpiod_chip_get_lines(chip_, (unsigned int *)offsets, n, &bulk) != 0)
        return -1;
    for (i = 0; i < n; i++)
        vals[i] = (idle >> i) & 1;
    n_lines = n;
    return gpiod_line_request_bulk_output(&bulk, "vantage-trigger", vals);
}

int gpio_write(uint32_t state)
{
    int vals[32];
    unsigned i;
    for (i = 0; i < n_lines; i++)
        vals[i] = (state >> i) & 1;
    return gpiod_line_set_value_bulk(&bulk, vals);
}

void gpio_close(void)
{
    gpiod_line_release_bulk(&bulk);
    gpiod_chip_close(chip_);
}
#else
static struct gpiod_line_request *req;
static unsigned offs[32];

int gpio_open(const char *chip, const unsigned *offsets, unsigned n, uint32_t idle)
{
    char path[128];
    struct gpiod_chip *c;
    struct gpiod_line_settings *ls;
    struct gpiod_line_config *lc;
    struct gpiod_request_config *rc;
    enum gpiod_line_value vals[32];
    unsigned i;
    if (chip[0] == '/')
        c = gpiod_chip_open(chip);
    else {
        snprintf(path, sizeof path, "/dev/%s", chip);
        c = gpiod_chip_open(path);
    }
    if (!c)
        return -1;
    ls = gpiod_line_settings_new();
    lc = gpiod_line_config_new();
    rc = gpiod_request_config_new();
    gpiod_line_settings_set_direction(ls, GPIOD_LINE_DIRECTION_OUTPUT);
    for (i = 0; i < n; i++) {
        offs[i] = offsets[i];
        vals[i] = ((idle >> i) & 1) ? GPIOD_LINE_VALUE_ACTIVE : GPIOD_LINE_VALUE_INACTIVE;
    }
    gpiod_line_config_add_line_settings(lc, offs, n, ls);
    gpiod_line_config_set_output_values(lc, vals, n);
    gpiod_request_config_set_consumer(rc, "vantage-trigger");
    req = gpiod_chip_request_lines(c, rc, lc);
    gpiod_line_settings_free(ls);
    gpiod_line_config_free(lc);
    gpiod_request_config_free(rc);
    gpiod_chip_close(c);
    n_lines = n;
    return req ? 0 : -1;
}

int gpio_write(uint32_t state)
{
    enum gpiod_line_value vals[32];
    unsigned i;
    for (i = 0; i < n_lines; i++)
        vals[i] = ((state >> i) & 1) ? GPIOD_LINE_VALUE_ACTIVE : GPIOD_LINE_VALUE_INACTIVE;
    return gpiod_line_request_set_values(req, vals);
}

void gpio_close(void) { gpiod_line_request_release(req); }
#endif
