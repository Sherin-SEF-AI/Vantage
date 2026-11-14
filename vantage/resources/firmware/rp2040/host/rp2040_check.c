/* rp2040_check.c: the firmware's PIO arithmetic (tb_rp2040.h) and scheduling core on the build machine, for
 * Vantage's cycle-exact simulation of trigger.pio.
 *   rp2040_check edges N   prints the first N edges as "tick state word": the core's edge, the output state and
 *                          the delay word main.c writes for it
 *   rp2040_check capture   reads "x now_hint" lines (pushed counts) and prints the recovered PPS edge cycles */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trigger_core.h"
#include "trigger_plan.h"
#include "tb_rp2040.h"

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "edges")) {
        tb_state_t st;
        int n = atoi(argv[2]), i;
        int64_t prev = 0;
        tb_init(&st, &TB_PLAN, 0);
        for (i = 0; i < n; i++) {
            int64_t e;
            uint32_t s, w;
            tb_next_edge(&st, &e, &s);
            w = i == 0 ? tb_rp_first_word(e) : tb_rp_word(prev, e);
            printf("%lld %u %u\n", (long long)e, (unsigned)(s ^ TB_OUT_INVERT), (unsigned)w);
            prev = e;
        }
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "capture")) {
        tb_rp_cap_t cap = {0};
        unsigned long x;
        long long hint;
        while (scanf("%lu %lld", &x, &hint) == 2)
            printf("%lld\n", (long long)tb_rp_capture(&cap, (uint32_t)x, hint));
        return 0;
    }
    fprintf(stderr, "usage: rp2040_check edges N | capture\n");
    return 2;
}
