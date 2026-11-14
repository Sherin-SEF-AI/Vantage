/* host_sim.c: runs the trigger box's scheduling core (trigger_core.c, with this project's trigger_plan.h) on the
 * build machine against a simulated board: a timer whose crystal is off by `ppm`, a GNSS PPS with Gaussian jitter
 * (`jitter_ns`, 1 sigma) and a dropout, PPS captures quantised to `res` ticks. It prints every PPS and every pulse
 * rise as CSV lines; the generator compares each rise with the plan.
 *
 *   cc -O2 -std=c99 -ffp-contract=off -I. trigger_core.c host_sim.c -lm -o host_sim
 *   ./host_sim seconds=20 ppm=30 jitter_ns=20 seed=1 drop_start=8 drop_len=4 api=edges plan=normal
 *
 * Output lines:  P,<second>,<capture tick>,<true ps>,<stepped>,<state>,<residual ticks>
 *                R,<line>,<tick>,<true ps>,<state>
 *                S,<pps>,<missed>,<steps>,<outliers>,<late>,<final ppb>,<final state>
 * The random numbers (xorshift64*, Irwin-Hall Gaussian) and the arithmetic match the Rust host simulation exactly.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trigger_core.h"
#include "trigger_plan.h"

static uint64_t rng_state;

static double uniform(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}

static double gauss(void)
{
    double s = 0.0;
    int i;
    for (i = 0; i < 12; i++)
        s += uniform();
    return s - 6.0;
}

static double hz_true;      /* local ticks per true second */
static double l0;           /* local tick at true time 0 */

static double local_of(double t_ns) { return l0 + hz_true * (t_ns * 1e-9); }
static double true_ns_of(int64_t tick) { return ((double)tick - l0) / hz_true * 1e9; }
static long long ps(double t_ns) { return (long long)floor(t_ns * 1000.0 + 0.5); }

static double arg(int argc, char **argv, const char *key, double dflt)
{
    size_t n = strlen(key);
    int i;
    for (i = 1; i < argc; i++)
        if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=')
            return atof(argv[i] + n + 1);
    return dflt;
}

static const char *sarg(int argc, char **argv, const char *key, const char *dflt)
{
    size_t n = strlen(key);
    int i;
    for (i = 1; i < argc; i++)
        if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=')
            return argv[i] + n + 1;
    return dflt;
}

int main(int argc, char **argv)
{
    double seconds = arg(argc, argv, "seconds", 20.0);
    double ppm = arg(argc, argv, "ppm", 0.0);
    double jitter = arg(argc, argv, "jitter_ns", 0.0);
    double pps_off = arg(argc, argv, "pps_offset_ns", 300e6);
    double drop_start = arg(argc, argv, "drop_start", -1.0);
    double drop_len = arg(argc, argv, "drop_len", 0.0);
    double res = arg(argc, argv, "res", 1.0);
    int lines_api = strcmp(sarg(argc, argv, "api", "edges"), "lines") == 0;
    const tb_plan_t *plan = strcmp(sarg(argc, argv, "plan", "normal"), "selftest") == 0 ? &TB_PLAN_SELFTEST : &TB_PLAN;
    tb_state_t st;
    double end_ns = seconds * 1e9, poll_ns = 0.0, pps_ns;
    long pps_k = 0;
    int64_t edge_tick = 0;
    uint32_t edge_state = 0, prev_state = 0, i;
    int waiting[TB_MAX_LINES];
    int64_t nrise[TB_MAX_LINES], nfall[TB_MAX_LINES];
    long late = 0;

    rng_state = (uint64_t)arg(argc, argv, "seed", 1.0) * 0x9E3779B97F4A7C15ULL + 1;
    l0 = arg(argc, argv, "l0", 0.0);
    hz_true = (double)plan->hz * (1.0 + ppm * 1e-6);
    tb_init(&st, plan, (int64_t)floor(local_of(0.0)));
    if (lines_api) {
        for (i = 0; i < plan->n_lines; i++) {
            tb_line_next(&st, i, &nrise[i], &nfall[i]);
            waiting[i] = 1;
        }
    } else {
        tb_next_edge(&st, &edge_tick, &edge_state);
    }
    pps_ns = pps_off + (jitter > 0 ? jitter * gauss() : 0.0);
    for (;;) {
        /* the next event: PPS, poll (every ms) or the core's next edge */
        double t_edge;
        int64_t tick = 0;
        uint32_t line = 0;
        int is_rise = 0;
        if (lines_api) {
            int64_t best = INT64_MAX;
            for (i = 0; i < plan->n_lines; i++) {
                int64_t t = waiting[i] ? nrise[i] : nfall[i];
                if (t < best) {
                    best = t;
                    line = i;
                }
            }
            tick = best;
            is_rise = waiting[line];
        } else {
            tick = edge_tick;
        }
        t_edge = true_ns_of(tick);
        if (pps_ns <= t_edge && pps_ns <= poll_ns) {
            if (pps_ns > end_ns)
                break;
            {
                long sec = pps_k;
                int dropped = drop_start >= 0 && sec >= (long)drop_start && sec < (long)(drop_start + drop_len);
                if (!dropped) {
                    int64_t cap = (int64_t)(floor(local_of(pps_ns) / res) * res);
                    int stepped = tb_on_pps(&st, cap);
                    printf("P,%ld,%lld,%lld,%d,%s,%lld\n", sec, (long long)cap, ps(pps_ns), stepped,
                           tb_state_name(st.state), (long long)st.resid);
                    if (stepped && lines_api)
                        for (i = 0; i < plan->n_lines; i++)
                            if (waiting[i])
                                tb_line_next(&st, i, &nrise[i], &nfall[i]);
                }
            }
            pps_k++;
            pps_ns = pps_off + (double)pps_k * 1e9 + (jitter > 0 ? jitter * gauss() : 0.0);
            continue;
        }
        if (poll_ns < t_edge) {
            if (poll_ns > end_ns)
                break;
            tb_poll(&st, (int64_t)floor(local_of(poll_ns)));
            poll_ns += 1e6;
            continue;
        }
        if (t_edge > end_ns)
            break;
        /* the edge goes out */
        if (lines_api) {
            if (is_rise) {
                printf("R,%u,%lld,%lld,%s\n", line, (long long)tick, ps(t_edge), tb_state_name(st.state));
                tb_line_started(&st, line);
                waiting[line] = 0;
            } else {
                tb_line_ended(&st, line);
                tb_line_next(&st, line, &nrise[line], &nfall[line]);
                waiting[line] = 1;
            }
        } else {
            int64_t next_tick;
            for (i = 0; i < plan->n_lines; i++)
                if ((edge_state & plan->lines[i].mask) && !(prev_state & plan->lines[i].mask))
                    printf("R,%u,%lld,%lld,%s\n", i, (long long)tick, ps(t_edge), tb_state_name(st.state));
            prev_state = edge_state;
            tb_next_edge(&st, &next_tick, &edge_state);
            if (next_tick <= tick) {             /* the hardware cannot go back: the edge is late */
                late++;
                next_tick = tick + 1;
            }
            edge_tick = next_tick;
        }
    }
    printf("S,%u,%u,%u,%u,%ld,%d,%s\n", st.pps, st.missed, st.steps, st.outliers, late, (int)tb_ppb(&st),
           tb_state_name(st.state));
    return 0;
}
