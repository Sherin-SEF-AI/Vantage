/* trigger_core.c: the trigger box's scheduling core (see trigger_core.h). Integer arithmetic only, the same on a
 * Cortex-M0+, a Cortex-M4, an ARM64 Linux host and the x86 host simulation. Signed scaling divides (truncating
 * towards zero) rather than shifting, so the Rust port gives the same ticks. */
#include "trigger_core.h"

#define ONE_Q ((int64_t)1 << TB_Q)
#define HALF_Q ((int64_t)1 << (TB_Q - 1))
#define EPS_MAX ((int64_t)1 << (TB_EQ - 10))     /* 1/1024: a frequency error beyond ~1000 ppm is not a crystal */

static int64_t fdiv_q(int64_t x)                 /* floor(x / 2^TB_Q) */
{
    return x >= 0 ? (x >> TB_Q) : -((-x + ONE_Q - 1) >> TB_Q);
}

static int64_t abs64(int64_t x) { return x < 0 ? -x : x; }

int64_t tb_second_q(const tb_state_t *s)
{
    int64_t hz = (int64_t)s->plan->hz;
    return hz * ONE_Q + (hz * s->eps) / 65536;   /* hz (1 + eps), x 2^TB_Q */
}

int32_t tb_ppb(const tb_state_t *s)
{
    return (int32_t)((s->eps * 1000000000) / ((int64_t)1 << TB_EQ));
}

const char *tb_state_name(int state)
{
    switch (state) {
    case TB_LOCKING: return "locking";
    case TB_LOCKED: return "locked";
    case TB_HOLDOVER: return "holdover";
    default: return "free";
    }
}

static void set_anchor(tb_state_t *s, int64_t rel)   /* the anchor moved by rel (x 2^TB_Q, from the old anchor) */
{
    int64_t whole = fdiv_q(rel);
    s->anchor += whole;
    s->anchor_frac = (uint32_t)(rel - whole * ONE_Q);
}

void tb_init(tb_state_t *s, const tb_plan_t *p, int64_t now)
{
    uint32_t i;
    s->plan = p;
    s->anchor = now + (int64_t)(p->hz / 10);     /* the free-running schedule starts 100 ms after boot */
    s->anchor_frac = 0;
    s->anchor_sec = 0;
    s->eps = 0;
    s->state = TB_FREE;
    s->good = 0;
    s->last_pps = -1;
    s->pending = -1;
    s->resid = 0;
    s->pps = s->missed = s->steps = s->outliers = 0;
    s->out = 0;
    s->last_out = now;
    s->n_win = s->i_win = 0;
    for (i = 0; i < TB_MAX_LINES; i++) {
        tb_line_state_t *l = &s->ln[i];
        l->sec = 0;
        l->k = 0;
        l->rise = l->fall = 0;
        l->done_fall = now;
        l->phase = 2;
    }
}

int64_t tb_pulse_tick(const tb_state_t *s, uint32_t line, int64_t sec, uint32_t k)
{
    const tb_line_t *L = &s->plan->lines[line];
    uint32_t b = k / L->burst, j = k % L->burst;
    int64_t off = (int64_t)(L->phase_q + (uint64_t)b * L->period_q + (uint64_t)j * L->spacing_q);
    int64_t corr = ((off >> TB_Q) * s->eps) / 65536;          /* off x eps, x 2^TB_Q */
    int64_t rel = (sec - s->anchor_sec) * tb_second_q(s) + off + corr + (int64_t)s->anchor_frac;
    return s->anchor + fdiv_q(rel + HALF_Q);
}

void tb_line_next(tb_state_t *s, uint32_t line, int64_t *rise, int64_t *fall)
{
    tb_line_state_t *l = &s->ln[line];
    const tb_line_t *L = &s->plan->lines[line];
    uint32_t n = L->bursts * L->burst;
    int64_t floor_ = l->done_fall + (int64_t)s->plan->min_gap;
    for (;;) {
        int64_t t = tb_pulse_tick(s, line, l->sec, l->k);
        if (++l->k >= n) {
            l->k = 0;
            l->sec++;
        }
        if (t > floor_) {
            l->rise = t;
            l->fall = t + (int64_t)L->width;
            l->phase = 0;
            *rise = l->rise;
            *fall = l->fall;
            return;
        }
    }
}

void tb_line_started(tb_state_t *s, uint32_t line) { s->ln[line].phase = 1; }

void tb_line_ended(tb_state_t *s, uint32_t line)
{
    tb_line_state_t *l = &s->ln[line];
    if (l->fall > l->done_fall)
        l->done_fall = l->fall;
    l->phase = 2;
}

static void step_to(tb_state_t *s, int64_t c)
{
    uint32_t i;
    s->anchor = c;
    s->anchor_frac = 0;
    s->anchor_sec += 1;
    for (i = 0; i < s->plan->n_lines; i++) {
        tb_line_state_t *l = &s->ln[i];
        int64_t floor_ = c - (int64_t)s->plan->min_gap - 1;  /* nothing before the PPS edge that moved it ... */
        if (s->last_out > floor_)
            floor_ = s->last_out;                           /* ... nor before an edge already handed out */
        if (l->phase == 0)
            l->phase = 2;                                   /* fetched but not started: dropped */
        if (l->done_fall < floor_)
            l->done_fall = floor_;
        l->sec = s->anchor_sec - 1;
        l->k = 0;
    }
    s->good = 0;
    s->n_win = 0;
    s->state = TB_LOCKING;
}

static void win_add(tb_state_t *s, int64_t sec, int64_t c)
{
    uint32_t i = (s->i_win + s->n_win) % TB_WIN;
    if (s->n_win == TB_WIN) {
        s->i_win = (s->i_win + 1) % TB_WIN;
        i = (s->i_win + TB_WIN - 1) % TB_WIN;
    } else {
        s->n_win++;
    }
    s->win_sec[i] = sec;
    s->win_c[i] = c;
}

/* The least-squares line through the window's captures against their second numbers: its slope is the local
 * ticks per true second (eps), its value at the newest second the anchor. Deviations from the nominal line keep
 * the sums small: |x| <= TB_WIN_SPAN, |y| <= 1e-3 hz TB_WIN_SPAN. */
static void fit(tb_state_t *s)
{
    int64_t hz = (int64_t)s->plan->hz, n = 0, sx = 0, sxx = 0, sy = 0, sxy = 0, last_c, last_s, num, den, b_q, a_q;
    uint32_t j;
    last_s = s->win_sec[(s->i_win + s->n_win - 1) % TB_WIN];
    last_c = s->win_c[(s->i_win + s->n_win - 1) % TB_WIN];
    for (j = 0; j < s->n_win; j++) {
        uint32_t i = (s->i_win + j) % TB_WIN;
        int64_t x = s->win_sec[i] - last_s, y;
        if (x < -TB_WIN_SPAN)
            continue;                                       /* too old to share a line with the newest */
        y = s->win_c[i] - last_c - x * hz;
        n++;
        sx += x;
        sxx += x * x;
        sy += y;
        sxy += x * y;
    }
    den = n * sxx - sx * sx;
    if (n < 2 || den == 0)
        return;
    num = n * sxy - sx * sy;
    b_q = (num / den) * ONE_Q + ((num % den) * ONE_Q) / den;  /* ticks per second over hz, x 2^TB_Q */
    a_q = (sy * ONE_Q - b_q * sx) / n;                      /* the line at the newest second, from last_c */
    s->eps = b_q * 65536 / hz;
    if (s->eps > EPS_MAX)
        s->eps = EPS_MAX;
    if (s->eps < -EPS_MAX)
        s->eps = -EPS_MAX;
    s->anchor = last_c;
    s->anchor_frac = 0;
    set_anchor(s, a_q);
}

int tb_on_pps(tb_state_t *s, int64_t c)
{
    int64_t sq = tb_second_q(s);
    int64_t rq = (c - s->anchor) * ONE_Q - (int64_t)s->anchor_frac - sq;   /* capture minus prediction */
    int64_t step_q = (int64_t)s->plan->step_ticks * ONE_Q;
    int64_t r = fdiv_q(rq + HALF_Q);
    s->pps++;
    s->resid = r;
    if (s->state == TB_FREE || s->last_pps < 0) {           /* the first PPS: the schedule jumps to it */
        step_to(s, c);
        win_add(s, s->anchor_sec, c);
        s->last_pps = c;
        s->pending = -1;
        return 1;
    }
    /* far off the prediction: a glitch, or (confirmed by the next PPS a second later) a real jump; while the
     * frequency is not yet known (one capture) any PPS a whole number of seconds on is taken */
    if ((rq > step_q || rq < -step_q) && s->n_win >= 2) {
        if (s->pending >= 0 && abs64(c - s->pending - fdiv_q(sq + HALF_Q)) <= (int64_t)s->plan->step_ticks) {
            int64_t p = s->pending;
            step_to(s, c);
            win_add(s, s->anchor_sec - 1, p);
            win_add(s, s->anchor_sec, c);
            fit(s);
            s->steps++;
            s->last_pps = c;
            s->pending = -1;
            return 1;
        }
        s->pending = c;
        s->outliers++;
        return 0;
    }
    s->pending = -1;
    {
        /* the second it starts: the one predicted, or more when PPS went missing in between */
        int64_t gap = fdiv_q((c - s->anchor) * ONE_Q - (int64_t)s->anchor_frac + sq / 2);
        int64_t secs = gap / fdiv_q(sq);
        if (secs < 1)
            secs = 1;
        s->anchor_sec += secs;
    }
    win_add(s, s->anchor_sec, c);
    fit(s);
    if (s->state == TB_HOLDOVER)
        s->state = TB_LOCKING;
    s->good = abs64(r) <= (int64_t)s->plan->lock_ticks ? s->good + 1 : 0;
    if (s->good >= 3 && s->n_win >= 4)
        s->state = TB_LOCKED;
    s->last_pps = c;
    return 0;
}

void tb_poll(tb_state_t *s, int64_t now)
{
    for (;;) {
        int64_t sq = tb_second_q(s);
        int64_t late = s->anchor + fdiv_q((int64_t)s->anchor_frac + sq + sq / 4);   /* 1.25 s after the anchor */
        if (now <= late)
            return;
        set_anchor(s, (int64_t)s->anchor_frac + sq);
        s->anchor_sec++;
        if (s->state != TB_FREE)
            s->missed++;
        if (s->state == TB_LOCKED)
            s->state = TB_HOLDOVER;
        s->good = 0;
    }
}

void tb_next_edge(tb_state_t *s, int64_t *tick, uint32_t *state)
{
    uint32_t i, n = s->plan->n_lines;
    int64_t best = INT64_MAX;
    for (i = 0; i < n; i++) {
        tb_line_state_t *l = &s->ln[i];
        int64_t r, f, t;
        if (l->phase == 2)
            tb_line_next(s, i, &r, &f);
        t = l->phase == 0 ? l->rise : l->fall;
        if (t < best)
            best = t;
    }
    if (n == 0) {
        *tick = s->anchor + fdiv_q((int64_t)s->anchor_frac + tb_second_q(s));
        *state = s->out;
        return;
    }
    for (i = 0; i < n; i++) {
        tb_line_state_t *l = &s->ln[i];
        int64_t t = l->phase == 0 ? l->rise : l->fall;
        if (t - best < (int64_t)s->plan->min_gap || t == best) {
            if (l->phase == 0) {
                s->out |= s->plan->lines[i].mask;
                l->phase = 1;
            } else {
                s->out &= ~s->plan->lines[i].mask;
                tb_line_ended(s, i);
            }
        }
    }
    s->last_out = best;
    *tick = best;
    *state = s->out;
}
