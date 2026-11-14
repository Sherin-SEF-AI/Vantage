//! The Rust twin of host_sim.c: the scheduling core (trigger-core) against a simulated board (a crystal `ppm` off,
//! a PPS with Gaussian jitter and a dropout, captures quantised to `res` ticks), printing the same CSV lines as the C
//! simulation, number for number: Vantage compares the two outputs. std only, no crates.
//!
//!   cargo run --release -p hostsim -- seconds=20 ppm=30 jitter_ns=20 seed=1 drop_start=8 drop_len=4 api=edges
use trigger_core::{Core, Plan, MAX_LINES};

include!("../../plan.rs");

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(2685821657736338717) >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    fn gauss(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.uniform();
        }
        s - 6.0
    }
}

fn arg(args: &[String], key: &str, dflt: f64) -> f64 {
    args.iter()
        .find_map(|a| a.strip_prefix(&format!("{key}=")).map(|v| v.parse().unwrap_or(0.0)))
        .unwrap_or(dflt)
}

fn sarg<'a>(args: &'a [String], key: &str, dflt: &'a str) -> &'a str {
    args.iter().find_map(|a| a.strip_prefix(&format!("{key}="))).unwrap_or(dflt)
}

fn ps(t_ns: f64) -> i64 {
    (t_ns * 1000.0 + 0.5).floor() as i64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seconds = arg(&args, "seconds", 20.0);
    let ppm = arg(&args, "ppm", 0.0);
    let jitter = arg(&args, "jitter_ns", 0.0);
    let pps_off = arg(&args, "pps_offset_ns", 300e6);
    let drop_start = arg(&args, "drop_start", -1.0);
    let drop_len = arg(&args, "drop_len", 0.0);
    let res = arg(&args, "res", 1.0);
    let lines_api = sarg(&args, "api", "edges") == "lines";
    let plan: Plan = if sarg(&args, "plan", "normal") == "selftest" { PLAN_SELFTEST } else { PLAN };
    let mut rng = Rng((arg(&args, "seed", 1.0) as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
    let l0 = arg(&args, "l0", 0.0);
    let hz_true = plan.hz as f64 * (1.0 + ppm * 1e-6);
    let local_of = |t_ns: f64| l0 + hz_true * (t_ns * 1e-9);
    let true_ns_of = |tick: i64| (tick as f64 - l0) / hz_true * 1e9;
    let n = plan.lines.len();
    let mut st = Core::new(plan, local_of(0.0).floor() as i64);
    let end_ns = seconds * 1e9;
    let mut poll_ns = 0.0;
    let mut pps_k: i64 = 0;
    let (mut edge_tick, mut edge_state, mut prev_state) = (0i64, 0u32, 0u32);
    let mut waiting = [false; MAX_LINES];
    let mut nrise = [0i64; MAX_LINES];
    let mut nfall = [0i64; MAX_LINES];
    let mut late: i64 = 0;
    if lines_api {
        for i in 0..n {
            let (r, f) = st.line_next(i);
            nrise[i] = r;
            nfall[i] = f;
            waiting[i] = true;
        }
    } else {
        let (t, s) = st.next_edge();
        edge_tick = t;
        edge_state = s;
    }
    let mut pps_ns = pps_off + if jitter > 0.0 { jitter * rng.gauss() } else { 0.0 };
    loop {
        let tick;
        let (mut line, mut is_rise) = (0usize, false);
        if lines_api {
            let mut best = i64::MAX;
            for i in 0..n {
                let t = if waiting[i] { nrise[i] } else { nfall[i] };
                if t < best {
                    best = t;
                    line = i;
                }
            }
            tick = best;
            is_rise = waiting[line];
        } else {
            tick = edge_tick;
        }
        let t_edge = true_ns_of(tick);
        if pps_ns <= t_edge && pps_ns <= poll_ns {
            if pps_ns > end_ns {
                break;
            }
            let sec = pps_k;
            let dropped = drop_start >= 0.0 && sec >= drop_start as i64 && sec < (drop_start + drop_len) as i64;
            if !dropped {
                let cap = ((local_of(pps_ns) / res).floor() * res) as i64;
                let stepped = st.on_pps(cap);
                println!("P,{},{},{},{},{},{}", sec, cap, ps(pps_ns), stepped as i32, st.state.name(), st.resid);
                if stepped && lines_api {
                    for i in 0..n {
                        if waiting[i] {
                            let (r, f) = st.line_next(i);
                            nrise[i] = r;
                            nfall[i] = f;
                        }
                    }
                }
            }
            pps_k += 1;
            pps_ns = pps_off + pps_k as f64 * 1e9 + if jitter > 0.0 { jitter * rng.gauss() } else { 0.0 };
            continue;
        }
        if poll_ns < t_edge {
            if poll_ns > end_ns {
                break;
            }
            st.poll(local_of(poll_ns).floor() as i64);
            poll_ns += 1e6;
            continue;
        }
        if t_edge > end_ns {
            break;
        }
        if lines_api {
            if is_rise {
                println!("R,{},{},{},{}", line, tick, ps(t_edge), st.state.name());
                st.line_started(line);
                waiting[line] = false;
            } else {
                st.line_ended(line);
                let (r, f) = st.line_next(line);
                nrise[line] = r;
                nfall[line] = f;
                waiting[line] = true;
            }
        } else {
            for i in 0..n {
                let m = plan.lines[i].mask;
                if edge_state & m != 0 && prev_state & m == 0 {
                    println!("R,{},{},{},{}", i, tick, ps(t_edge), st.state.name());
                }
            }
            prev_state = edge_state;
            let (mut next_tick, s) = st.next_edge();
            edge_state = s;
            if next_tick <= tick {
                late += 1;
                next_tick = tick + 1;
            }
            edge_tick = next_tick;
        }
    }
    println!("S,{},{},{},{},{},{},{}", st.pps, st.missed, st.steps, st.outliers, late, st.ppb(), st.state.name());
}
