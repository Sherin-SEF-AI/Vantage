/* trigger_core.h: the trigger box's scheduling core (Vantage, generated with the project's sync plan).
 *
 * Target-independent and free of hardware access: the RP2040, STM32 and Jetson programs and the host simulation
 * all compile this same file. It turns the plan (per trigger line: pulses per second, phase after the PPS edge,
 * pulse width) into pulse times on the board's own timer, and disciplines that timeline to a GNSS PPS input:
 *
 * - Ticks are the board's timer counts, a 64-bit count that never wraps (the hardware layer extends its counter).
 * - The plan's times are in nominal ticks x 2^TB_Q (24 fraction bits), computed exactly by the generator.
 * - The PPS discipline: a least-squares line through the last TB_WIN PPS captures against their second numbers.
 *   Its slope is the crystal's frequency (eps, Q40: local ticks per true tick - 1), its value at the newest second
 *   the anchor of that second; pulse k is placed at anchor + offset_k * (1 + eps), rounded to the nearest tick.
 *   With 16 captures the anchor's noise is about half the PPS jitter and the frequency's about 1/18 of it per second.
 * - Lock: the first PPS re-anchors the schedule (a step), the second gives the frequency, three PPS in a row within
 *   lock_ticks of the prediction (and four in the fit) make it LOCKED. Without PPS for a second and a quarter the
 *   anchor advances on its own (HOLDOVER: the learned frequency keeps the schedule); before any PPS it runs on the
 *   nominal clock (FREE). A PPS far off the prediction is ignored once (a glitch); two in a row, a second apart,
 *   re-anchor (a step).
 *
 * Only trigger pulses come out of this: nothing here drives anything else.
 */
#ifndef TRIGGER_CORE_H
#define TRIGGER_CORE_H

#include <stdint.h>

#define TB_MAX_LINES 8
#define TB_Q 24                       /* fraction bits of tick quantities */
#define TB_EQ 40                      /* fraction bits of the frequency error eps */
#define TB_WIN 16                     /* PPS captures in the frequency and phase fit */
#define TB_WIN_SPAN 64                /* ... no older than this many seconds before the newest */

enum { TB_FREE = 0, TB_LOCKING = 1, TB_LOCKED = 2, TB_HOLDOVER = 3 };

typedef struct {
    uint32_t bursts;                  /* bursts per second: the line's frame rate (Hz) */
    uint32_t burst;                   /* pulses per burst: 1 for cameras (the self-test uses more) */
    uint64_t phase_q;                 /* first pulse after the second starts, nominal ticks x 2^TB_Q */
    uint64_t period_q;                /* burst to burst */
    uint64_t spacing_q;               /* pulse to pulse within a burst */
    uint32_t width;                   /* pulse width, ticks */
    uint32_t mask;                    /* the line's bit in the output state */
} tb_line_t;

typedef struct {
    uint32_t hz;                      /* nominal timer ticks per second */
    uint32_t n_lines;
    const tb_line_t *lines;
    uint32_t lock_ticks;              /* |PPS residual| at most this counts towards lock */
    uint32_t step_ticks;              /* a residual beyond this is a glitch, or (twice in a row) a step */
    uint32_t min_gap;                 /* merged edges closer than this become one state change */
} tb_plan_t;

typedef struct {
    int64_t sec;                      /* second of the next pulse (the anchor's numbering) */
    uint32_t k;                       /* its index within the second */
    int64_t rise, fall;               /* the pulse being output (merged edges) */
    int64_t done_fall;                /* the end of the last pulse output (a floor for the next one) */
    uint8_t phase;                    /* 0 fetched (rise to come), 1 high (fall to come), 2 idle */
} tb_line_state_t;

typedef struct {
    const tb_plan_t *plan;
    int64_t anchor;                   /* tick at which second anchor_sec starts (integer part) */
    uint32_t anchor_frac;             /* its fraction x 2^TB_Q */
    int64_t anchor_sec;
    int64_t eps;                      /* (local ticks per true second) / hz - 1, x 2^TB_EQ */
    int state;
    int good;                         /* PPS in a row within lock_ticks */
    int64_t win_sec[TB_WIN], win_c[TB_WIN];   /* the last PPS captures and the seconds they start */
    uint32_t n_win, i_win;
    int64_t last_out;                 /* the last merged edge handed out */
    int64_t last_pps;                 /* tick of the last accepted PPS, -1 none */
    int64_t pending;                  /* a far-off PPS waiting for its confirmation, -1 none */
    int64_t resid;                    /* the last PPS's residual, ticks (capture minus prediction) */
    uint32_t pps, missed, steps, outliers;
    uint32_t out;                     /* the merged output state */
    tb_line_state_t ln[TB_MAX_LINES];
} tb_state_t;

void tb_init(tb_state_t *s, const tb_plan_t *p, int64_t now);
/* a PPS edge captured at this tick; 1 when the schedule stepped (per-line users refetch unstarted pulses) */
int tb_on_pps(tb_state_t *s, int64_t capture);
void tb_poll(tb_state_t *s, int64_t now);
int64_t tb_second_q(const tb_state_t *s);
int64_t tb_pulse_tick(const tb_state_t *s, uint32_t line, int64_t sec, uint32_t k);
/* per line (one timer channel per line): the next pulse's rise and fall ticks */
void tb_line_next(tb_state_t *s, uint32_t line, int64_t *rise, int64_t *fall);
void tb_line_started(tb_state_t *s, uint32_t line);   /* the fetched pulse's rise went out */
void tb_line_ended(tb_state_t *s, uint32_t line);     /* ... and its fall */
/* all lines merged (one port: PIO, GPIO): the next edge's tick and the output state from it on */
void tb_next_edge(tb_state_t *s, int64_t *tick, uint32_t *state);
/* the frequency error estimate in parts per billion (local clock fast: positive) */
int32_t tb_ppb(const tb_state_t *s);
const char *tb_state_name(int state);

#endif
