/* tb_rp2040.h: the arithmetic between the scheduling core's cycle timeline and the two PIO programs (trigger.pio).
 * Pure functions, compiled into the firmware and into Vantage's host check, which runs them against a cycle-exact
 * simulation of the PIO programs. Cycle 0 is the cycle both state machines were enabled together. */
#ifndef TB_RP2040_H
#define TB_RP2040_H

#include <stdint.h>

#ifndef TB_RP_CAP_X0
#define TB_RP_CAP_X0 0xffffffffu        /* x when the capture machine starts (pio_encode_mov_not(x, null)) */
#endif

/* tb_trigger: the delay word of the first edge after the start (the first OUT finds the OSR empty after the restart
 * and stalls one cycle to fill it: RP2040 datasheet 3.2.3.3, 3.5.4.2), and of an edge after the previous one */
static inline uint32_t tb_rp_first_word(int64_t edge)
{
    return edge >= 3 ? (uint32_t)(edge - 3) : 0u;
}

static inline uint32_t tb_rp_word(int64_t prev, int64_t edge)
{
    int64_t d = edge - prev - 3;          /* out x (1) + the loop (d + 1) + out pins (1) */
    return d > 0 ? (uint32_t)d : 0u;      /* edges closer than 3 cycles: the core's min_gap prevents it */
}

/* tb_capture: the cycle of a PPS rising edge from the pushed count. k counts the captures before this one (each PPS
 * pulse adds two cycles outside the count); the 32-bit count wraps every 2^33 cycles (68.7 s at 125 MHz) and
 * now_hint, the cycle the CPU believes it is (from the 1 MHz timer), picks the wrap. The result is the edge at the
 * pin to -1..0 cycles: the pin is tested every second cycle behind the 2-cycle input synchroniser. */
typedef struct {
    uint32_t k;
} tb_rp_cap_t;

static inline int64_t tb_rp_capture(tb_rp_cap_t *c, uint32_t x, int64_t now_hint)
{
    uint32_t low = (uint32_t)(TB_RP_CAP_X0 - x);
    int64_t base = 2 * (int64_t)low + 2 * (int64_t)c->k - 4;
    int64_t span = (int64_t)1 << 33;
    int64_t w = (now_hint - base + span / 2) / span;
    c->k++;
    return base + (w > 0 ? w : 0) * span;
}

#endif
