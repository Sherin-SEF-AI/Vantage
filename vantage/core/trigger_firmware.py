"""Trigger-box firmware from the sync plan: buildable projects that fire every camera trigger line at the plan's times,
phase-locked to a GNSS PPS, for four targets, with each project's scheduling core proved on this machine.

The plan (core.timing). Per trigger line (a sync letter): its cameras, their frame rate (pulses per second) and the
trigger time after the sync pulse (the top of the reference lidar's rotation, which a PTP/PPS-locked lidar puts on
the whole second). Pulse k of every second goes out at phase + k/rate after the PPS edge, the line's latency (the
camera's FSYNC-to-exposure delay, a setting) taken off. A line's rate must be a whole number of pulses per second
(PPS discipline restarts the pattern every second). Without a spinning lidar the cameras' sync letters still make
lines, fired on the whole second (phase 0). The plan's assigned letters or its suggested ones (a setting).

The scheduling core (resources/firmware/common/trigger_core.c, and its Rust twin) is one piece of integer code all
targets share: the plan in timer ticks x 2^24; the PPS discipline a least-squares line through the last 16 PPS
captures against their second numbers (its slope the crystal's frequency, its value at the newest second that
second's anchor: about half the PPS jitter on the anchor, a sixteenth of it on the frequency); lock after three PPS
within 2 µs; holdover on the learned frequency when the PPS stops; a step (re-anchor) only on two consistent far-off
PPS in a row. Targets:
- rp2040: C with the pico-sdk. PIO state machine tb_trigger puts every edge on a 125 MHz cycle from a queue the CPU
  fills (d + 3 cycles after the previous edge); tb_capture timestamps the PPS to one cycle (a count every second
  cycle, two cycles per PPS pulse outside the count, added back). Lines on consecutive GPIOs (PIO OUT pins).
- stm32f411: C on CMSIS registers (Nucleo-F411RE or Black Pill). TIM2/TIM5, 32-bit at 100 MHz: PPS input capture on
  TIM2_CH1, one output-compare channel per line, so the timer places each edge itself.
- rust_rp2040: Rust, the same core as a no_std crate (tested and simulated on the host with cargo, offline) and
  rp2040-hal firmware with the same PIO programs.
- jetson: userspace C with libgpiod: clock_nanosleep to each edge on the system clock (which chrony/phc2sys keep on
  GNSS), or the PPS device's stamps through the same core; its jitter is the kernel's wake-up latency, measured.
Each project has a PPS input, a gPTP/PPS note, the trigger outputs on the pins chosen, a self-test pattern (line n
gives n + 1 pulses of 2 ms every second), a README with build and flash commands and the wiring table, and the host
simulation. Nothing in them drives anything but the trigger lines.

Proof on this machine. host_check compiles each project's core with gcc (and cargo for Rust) and runs it against a
simulated board: (1) an exact clock: every rise must be the plan's time rounded to the timer tick, |error| <= 0.5
tick; (2) a crystal 30 ppm off with a PPS of 20 ns jitter (1 sigma) and a 10 s dropout: errors while locked, in
holdover and after re-lock; (3) the per-line API (STM32) and the merged-edge API (PIO, GPIO); (4) the self-test
plan. For the RP2040 the delay words the firmware computes run through a cycle-exact simulation of the PIO programs
(semantics of RP2040 datasheet 3.2.2, 3.4.2, 3.5.4.2), and the capture arithmetic is checked against simulated PPS
edges. The Rust core must print the same output as the C core, number for number. The Jetson program runs here with
a stand-in for the GPIO lines and its edge lateness is measured (on this x86 desktop, not a Jetson).

Limits. No cross-compiler is installed here (arm-none-eabi-gcc, the thumbv6m Rust target, the pico-sdk): the
firmware itself is not compiled or flashed here; the README says what to install. Hardware edge placement is exact
to the tick on the RP2040 (PIO) and STM32 (output compare); the PPS capture adds its input synchroniser's fixed
delay (2 cycles on the RP2040, a few timer clocks on the STM32), not corrected. Crystal phase noise, temperature
transients and PLL jitter are not modelled; the PPS is modelled as Gaussian. Lines on TIM5 of the STM32 are started
by TIM2's trigger output; a fixed offset of a tick or two between the timers is not corrected. During a step (the
first lock, or a re-anchor) pulses already queued keep the old timeline.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np

SCHEMA = 'vantage.trigger_fw/1'
METHOD_REVISION = 1
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'firmware')
Q = 24                                   # fraction bits of the plan's tick quantities (trigger_core.h TB_Q)
LETTERS = 'ABCDEFGH'
MAX_LINES = 8

# Clock figures. RP2040 datasheet (Raspberry Pi Ltd, build 2025-02-20, read 2026-11-12): 2.15 clk_sys 125 MHz in
# the SDK's default set-up, up to 133 MHz (1.1 'Dual Cortex M0+ processor cores, up to 133MHz'); 3.2.2 'Each
# instruction takes precisely one cycle, unless it explicitly stalls'; 2.3.1.4 / GPIO 'bypass to save two cycles of
# input' (the input synchroniser's latency). STM32F411: 100 MHz maximum, TIM2 and TIM5 32-bit on APB1 whose timer
# clock is twice the APB1 clock when APB1 is divided (DS10314, RM0383 6.2): 100 MHz at APB1 = 50 MHz.
TARGET_IDS = ('rp2040', 'stm32f411', 'rust_rp2040', 'jetson')
TARGETS = {
    'rp2040': {
        'name': 'RP2040 · C (pico-sdk, PIO)', 'short': 'RP2040 C', 'hz': 125_000_000, 'min_gap': 3, 'capture_res': 2,
        'lang': 'C', 'hardware': 'Raspberry Pi Pico (RP2040)',
        'resolution': '8 ns: one PIO cycle at 125 MHz (RP2040 datasheet 3.2.2: one instruction per cycle)',
        'jitter': 'none from software: the PIO places every edge on its cycle; the PPS is timestamped to 2 cycles '
                  '(16 ns) behind the 2-cycle input synchroniser (a fixed 16 ns, not corrected)',
        'needs': 'pico-sdk 1.5+ (PICO_SDK_PATH), CMake 3.13+, arm-none-eabi-gcc, picotool or a UF2 drag-and-drop'},
    'stm32f411': {
        'name': 'STM32F411 · C (CMSIS, timer output compare)', 'short': 'STM32 C', 'hz': 100_000_000, 'min_gap': 1,
        'capture_res': 1, 'lang': 'C', 'hardware': 'Nucleo-F411RE or WeAct "Black Pill" STM32F411CE',
        'resolution': '10 ns: one tick of TIM2/TIM5 at 100 MHz (DS10314, RM0383 6.2)',
        'jitter': 'none from software: output compare sets each line on its tick; the PPS input capture is exact to '
                  'a tick plus a fixed delay of a few timer clocks (the input stage), not corrected',
        'needs': 'arm-none-eabi-gcc, the CMSIS headers of STM32CubeF4, st-flash (stlink) or OpenOCD'},
    'rust_rp2040': {
        'name': 'RP2040 · Rust (rp2040-hal, PIO)', 'short': 'RP2040 Rust', 'hz': 125_000_000, 'min_gap': 3,
        'capture_res': 2, 'lang': 'Rust', 'hardware': 'Raspberry Pi Pico (RP2040)',
        'resolution': '8 ns: one PIO cycle at 125 MHz (the same PIO programs as the C project)',
        'jitter': 'as the C project: none from software; the PPS to 2 cycles behind the input synchroniser',
        'needs': 'rustup target add thumbv6m-none-eabi; cargo install flip-link elf2uf2-rs (or probe-rs); network '
                 'for crates.io on the first build'},
    'jetson': {
        'name': 'Jetson / Linux GPIO · C (libgpiod)', 'short': 'Jetson', 'hz': 1_000_000_000, 'min_gap': 20_000,
        'capture_res': 1, 'lang': 'C', 'hardware': 'NVIDIA Jetson (Orin, Xavier) or any Linux board with GPIO',
        'resolution': '1 ns on paper (the system clock); in practice the kernel\'s wake-up latency',
        'jitter': 'userspace: each edge is late by the thread\'s wake-up latency plus the GPIO write, tens of µs '
                  'typically and hundreds at worst on a stock kernel (measure with --stats and cyclictest); '
                  'measured here on this x86 desktop with a stand-in for the lines',
        'needs': 'libgpiod-dev (and pps-tools for --pps) on the Jetson; chrony or ptp4l + phc2sys disciplining the '
                 'clock to GNSS'},
}
LOCK_NS = 2_000                          # |PPS residual| under this counts towards lock
STEP_NS = 50_000                         # a PPS this far off is a glitch, or (twice) a step
SELFTEST = {'phase_ms': 100.0, 'spacing_ms': 20.0, 'width_ms': 2.0}

# RP2040 Pico: GP0-GP22 and GP26-GP28 on the header; GP23 (SMPS mode), GP24 (VBUS sense) and GP25 (LED) are not
RP_PINS = tuple(range(0, 23)) + (26, 27, 28)
# STM32F411 pins with a 32-bit timer channel (DS10314 alternate-function table): pin -> [(timer, channel, AF)]
STM32_PINS = {
    'PA0': [(5, 1, 2)], 'PA1': [(2, 2, 1), (5, 2, 2)], 'PA2': [(2, 3, 1), (5, 3, 2)], 'PA3': [(2, 4, 1), (5, 4, 2)],
    'PB3': [(2, 2, 1)], 'PB10': [(2, 3, 1)]}
STM32_PPS_PINS = ('PA15', 'PA5', 'PA0')          # TIM2_CH1, AF1
STM32_GPIO = tuple(f'P{p}{i}' for p in 'ABC' for i in range(16))
STM32_BOARDS = {'nucleo': {'name': 'Nucleo-F411RE', 'hse_mhz': 8, 'bypass': 1, 'led': 'PA5', 'led_low': 0},
                'blackpill': {'name': 'WeAct Black Pill F411CE', 'hse_mhz': 25, 'bypass': 0, 'led': 'PC13',
                              'led_low': 1}}
STM32_NOTES = {'PA2': 'PA2/PA3 are the Nucleo\'s ST-LINK virtual COM port (USART2): open SB13/SB14 or pick other pins',
               'PA3': 'PA2/PA3 are the Nucleo\'s ST-LINK virtual COM port (USART2): open SB13/SB14 or pick other pins',
               'PA5': 'PA5 drives the Nucleo\'s green LED (LD2)',
               'PB3': 'PB3 is SWO/JTDO: fine with SWD debugging, not with JTAG'}

LIMITS = {'width_us': (1.0, 20000.0), 'latency_us': (0.0, 20000.0), 'rp_base_pin': (0, 28), 'rp_pps_pin': (0, 28),
          'rp_selftest_pin': (0, 28), 'sim_ppm': (-200.0, 200.0), 'sim_jitter_ns': (0.0, 5000.0),
          'sim_seconds': (10, 300), 'sim_drop_s': (0, 120), 'seed': (0, 2 ** 31 - 1), 'jetson_seconds': (1, 30)}
INTS = ('rp_base_pin', 'rp_pps_pin', 'rp_selftest_pin', 'sim_seconds', 'sim_drop_s', 'seed', 'jetson_seconds')
CHOICES = {'target': TARGET_IDS, 'groups': ('plan', 'suggested'), 'stm_board': tuple(STM32_BOARDS),
           'jetson_clock': ('realtime', 'tai')}

ASSUMPTIONS = (
    'Pulse k of every second goes out at the line\'s trigger time + k / rate after the PPS edge: the reference '
    'lidar is phase-locked to PPS/PTP so that its sync pulse falls on the whole second (core.timing).',
    'A trigger line runs at a whole number of pulses per second; its cameras share its rate.',
    'The trigger is the pulse\'s rising edge (active-low lines invert it); the camera\'s FSYNC-to-exposure latency '
    'is the latency setting, taken off every pulse.',
    'Edges are exact to the timer tick on the RP2040 (PIO) and the STM32 (output compare); the Jetson\'s are as late '
    'as the kernel wakes the thread.',
    'The PPS is captured to the timer\'s resolution behind a fixed input delay (not corrected); crystal phase noise, '
    'temperature transients and PLL jitter are not modelled in the host simulation; its PPS jitter is Gaussian.',
    'Nothing here is compiled for the boards themselves on this machine (no cross-compiler); the scheduling core '
    'each project uses is compiled and simulated here.')


# =========================================================================== settings
def _default_stm_pins():
    return ['PB3', 'PB10', 'PA0', 'PA1', 'PA2', 'PA3']


@dataclass
class FirmwareSettings:
    target: str = 'rp2040'               # the target the page shows and exports first (all four are generated)
    groups: str = 'plan'                 # 'plan': the cameras' sync letters; 'suggested': the plan's suggestion
    width_us: float = 100.0              # trigger pulse width
    latency_us: float = 0.0              # FSYNC-to-exposure latency taken off every pulse
    active_low: bool = False
    rp_base_pin: int = 2                 # RP2040: the first line's GPIO; the others follow it
    rp_pps_pin: int = 15
    rp_selftest_pin: int = 22
    stm_board: str = 'nucleo'
    stm_pins: list = field(default_factory=_default_stm_pins)   # STM32: the lines' pins in line order
    stm_pps_pin: str = 'PA15'
    stm_selftest_pin: str = 'PB0'
    jetson_chip: str = 'gpiochip0'
    jetson_offsets: list = field(default_factory=list)          # libgpiod line offsets in line order ([] = 0, 1, ...)
    jetson_clock: str = 'realtime'
    sim_ppm: float = 30.0                # host simulation: the crystal's frequency error
    sim_jitter_ns: float = 20.0          # ... the PPS jitter (1 sigma)
    sim_seconds: int = 60
    sim_drop_s: int = 10                 # ... a PPS dropout this long, starting at a third of the run
    seed: int = 1
    jetson_seconds: int = 4              # the Jetson program's measured run here

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'FirmwareSettings':
        s = FirmwareSettings()
        d = d if isinstance(d, dict) else {}
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    v = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    v = min(max(v, lo), hi)
                    setattr(s, k, int(round(v)) if k in INTS else v)
        for k, opts in CHOICES.items():
            if d.get(k) in opts:
                setattr(s, k, d[k])
        if isinstance(d.get('active_low'), bool):
            s.active_low = d['active_low']
        if isinstance(d.get('stm_pins'), (list, tuple)):
            s.stm_pins = [str(p).upper()[:5] for p in d['stm_pins'][:MAX_LINES]]
        for k in ('stm_pps_pin', 'stm_selftest_pin'):
            if isinstance(d.get(k), str) and d[k].upper() in STM32_GPIO:
                setattr(s, k, d[k].upper())
        if isinstance(d.get('jetson_chip'), str) and re.fullmatch(r'[\w./-]{1,64}', d['jetson_chip']):
            s.jetson_chip = d['jetson_chip']
        if isinstance(d.get('jetson_offsets'), (list, tuple)):
            offs = []
            for v in d['jetson_offsets'][:MAX_LINES]:
                try:
                    offs.append(min(max(int(v), 0), 1023))
                except (TypeError, ValueError):
                    break
            s.jetson_offsets = offs
        return s


def settings_of(project) -> FirmwareSettings:
    return FirmwareSettings.from_dict(getattr(project.analysis, 'trigger_fw', None))


# =========================================================================== the plan
@dataclass
class FwLine:
    letter: str
    tags: list
    rate: int                            # pulses per second
    phase_ns: Fraction                   # first pulse after the PPS edge, latency taken off
    trigger_ms: float                    # the plan's trigger time (before the latency)
    period_ms: float
    worst_ms: float | None               # the plan's worst misalignment on the line


@dataclass
class FwPlan:
    lines: list
    lidar: dict | None
    width_ns: int
    notes: list
    problems: list                       # what stops the firmware from being generated
    source: str                          # 'plan' | 'suggested' | 'letters'
    project: str = ''
    variant: str = ''


def _rate_of(fps_list):
    """The line's pulses per second: the fastest camera's frame rate, when it is a whole number."""
    f = max(fps_list)
    n = round(f)
    return n if n >= 1 and abs(f - n) < 1e-6 else None


def firmware_plan(project, variant=None, settings: FirmwareSettings | None = None) -> FwPlan:
    """The trigger lines the firmware drives, from the sync plan (core.timing)."""
    from . import timing as TM
    st = settings or settings_of(project)
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    var = var or project.variant
    res = TM.plan(project, var)
    notes, problems = [], []
    lat = Fraction(st.latency_us).limit_denominator(1000) * 1000
    width_ns = int(round(st.width_us * 1000))
    groups, source = [], 'letters'
    if res.lidar is not None:
        if st.groups == 'suggested':
            groups, _ = TM.evaluate_groups(res, res.suggestion['letters'])
            source = 'suggested'
        else:
            groups, source = res.groups, 'plan'
        lidar = {'tag': res.lidar.tag, 'model': res.lidar.name, 'rate_hz': res.lidar.rate_hz,
                 'phase_deg': res.lidar.phase_deg}
        if abs(res.lidar.rate_hz - round(res.lidar.rate_hz)) > 1e-9:
            notes.append(f'The lidar turns at {res.lidar.rate_hz:g} Hz, not a whole number: its sync pulse does not '
                         f'fall on every PPS second, so the triggers are locked to the PPS, not to its sweep.')
    else:
        lidar = None
        notes.append(res.reason + ' The trigger lines fire on the whole second (phase 0) at their cameras\' rate.')
    lines = []
    if groups:
        for g in groups:
            fps = [c['fps'] for c in res.cams if c['uid'] in g['uids']]
            lines.append((g['letter'], g['tags'], fps, float(g['trigger_ms']), g.get('worst_ms')))
    else:
        by = {}
        for c in res.cams:
            if c['triggered']:
                by.setdefault(c['sync'], []).append(c)
        for L in sorted(by):
            lines.append((L, [c['tag'] for c in by[L]], [c['fps'] for c in by[L]], 0.0, None))
    out = []
    for L, tags, fps, trig, worst in lines:
        rate = _rate_of(fps)
        if rate is None:
            problems.append(f'Line {L} ({", ".join(tags)}): {max(fps):g} fps is not a whole number of pulses per '
                            f'second; a PPS-locked trigger cannot run it.')
            continue
        if len(set(fps)) > 1:
            notes.append(f'Line {L} drives cameras at {", ".join(f"{f:g}" for f in sorted(set(fps)))} fps: it '
                         f'pulses at {rate} Hz, the fastest.')
        period = Fraction(10 ** 9, rate)
        ph = (Fraction(trig).limit_denominator(10 ** 9) * 10 ** 6 - lat) % period
        out.append(FwLine(L, list(tags), rate, ph, trig, float(period) / 1e6, worst))
    if len(out) > MAX_LINES:
        problems.append(f'{len(out)} trigger lines; the firmware drives at most {MAX_LINES}.')
    for ln in out:
        if width_ns * 2 + 50_000 > float(Fraction(10 ** 9, ln.rate)):
            problems.append(f'Line {ln.letter}: a {st.width_us:g} µs pulse does not fit {ln.rate} pulses a second '
                            f'with room to spare; shorten the pulse.')
    if not out and not problems:
        problems.append('No camera is on a trigger line with a hardware trigger input: nothing to fire.')
    untrig = [c['tag'] for c in res.cams if not c['triggered']]
    if untrig:
        notes.append(f'Not triggered (no line or no trigger input): {", ".join(untrig)}.')
    return FwPlan(out, lidar, width_ns, notes, problems, source, getattr(project, 'name', ''), var.name)


def schedule_ns(ln: FwLine, seconds: int = 1) -> list:
    """The line's pulse times in ns after the first PPS edge (exact Fractions), over this many seconds."""
    per = Fraction(10 ** 9, ln.rate)
    return [s * 10 ** 9 + ln.phase_ns + k * per for s in range(seconds) for k in range(ln.rate)]


# =========================================================================== target set-up
def _qticks(ns: Fraction, hz: int) -> int:
    """ns -> nominal ticks x 2^Q, floored (the generator's exact arithmetic)."""
    return math.floor(Fraction(ns) * hz / 10 ** 9 * (1 << Q))


def core_lines(plan: FwPlan, hz: int, selftest=False) -> list:
    """The tb_line_t rows of a target clock: {bursts, burst, phase_q, period_q, spacing_q, width, mask}."""
    rows = []
    for i, ln in enumerate(plan.lines):
        if selftest:
            rows.append({'bursts': 1, 'burst': i + 1, 'phase_q': _qticks(Fraction(SELFTEST['phase_ms']) * 10 ** 6, hz),
                         'period_q': _qticks(Fraction(10 ** 9), hz),
                         'spacing_q': _qticks(Fraction(SELFTEST['spacing_ms']) * 10 ** 6, hz),
                         'width': int(round(SELFTEST['width_ms'] * 1e-3 * hz)), 'mask': 1 << i})
        else:
            rows.append({'bursts': ln.rate, 'burst': 1, 'phase_q': _qticks(ln.phase_ns, hz),
                         'period_q': _qticks(Fraction(10 ** 9, ln.rate), hz), 'spacing_q': 0,
                         'width': max(1, int(round(plan.width_ns * hz / 1e9))), 'mask': 1 << i})
    return rows


def pin_assignment(plan: FwPlan, target: str, st: FirmwareSettings) -> tuple:
    """({line letter: pin text}, extra {name: pin}, [problems], [notes]) for a target."""
    n = len(plan.lines)
    probs, notes, pins, extra = [], [], {}, {}
    if target in ('rp2040', 'rust_rp2040'):
        base = st.rp_base_pin
        use = list(range(base, base + n))
        extra = {'PPS input': f'GP{st.rp_pps_pin}', 'Self-test (to GND)': f'GP{st.rp_selftest_pin}', 'LED': 'GP25'}
        bad = [p for p in use if p not in RP_PINS]
        if bad:
            probs.append(f'GP{bad[0]} is not on the Pico\'s header (GP0-GP22, GP26-GP28): the {n} lines need '
                         f'{n} consecutive GPIOs from GP{base}.')
        for name, p in (('PPS', st.rp_pps_pin), ('self-test', st.rp_selftest_pin)):
            if p in use:
                probs.append(f'The {name} input GP{p} is one of the trigger lines GP{base}-GP{base + n - 1}.')
            if p not in RP_PINS:
                probs.append(f'The {name} input GP{p} is not on the Pico\'s header.')
        if st.rp_pps_pin == st.rp_selftest_pin:
            probs.append('The PPS and self-test inputs are the same GPIO.')
        pins = {ln.letter: f'GP{base + i}' for i, ln in enumerate(plan.lines)}
    elif target == 'stm32f411':
        board = STM32_BOARDS[st.stm_board]
        extra = {'PPS input (TIM2_CH1)': st.stm_pps_pin, 'Self-test (to GND)': st.stm_selftest_pin,
                 'Status UART TX (USART1)': 'PA9', 'LED': board['led']}
        if st.stm_pps_pin not in STM32_PPS_PINS:
            probs.append(f'The PPS must be on a TIM2_CH1 pin ({", ".join(STM32_PPS_PINS)}), not {st.stm_pps_pin}.')
        chosen = list(st.stm_pins[:n]) + [p for p in _default_stm_pins() if p not in st.stm_pins][:max(0, n - len(
            st.stm_pins))]
        used_ch, used_pins = {(2, 1)}, set()
        for ln, p in zip(plan.lines, chosen):
            opts = [o for o in STM32_PINS.get(p, []) if (o[0], o[1]) not in used_ch]
            if p == st.stm_pps_pin or p in used_pins:
                probs.append(f'Line {ln.letter}: {p} is already in use.')
                continue
            if not opts:
                probs.append(f'Line {ln.letter}: {p} has no free TIM2/TIM5 channel (pins: {", ".join(STM32_PINS)}).')
                continue
            t, ch, af = opts[0]
            used_ch.add((t, ch))
            used_pins.add(p)
            pins[ln.letter] = f'{p} (TIM{t}_CH{ch}, AF{af})'
            if p in STM32_NOTES and st.stm_board == 'nucleo' or p == 'PB3':
                notes.append(f'{p}: {STM32_NOTES[p]}.')
        for name, p in (('self-test', st.stm_selftest_pin), ('LED', board['led']), ('UART', 'PA9')):
            if any(v.split()[0] == p for v in pins.values()) or p == st.stm_pps_pin:
                probs.append(f'The {name} pin {p} is also a trigger line or the PPS.')
        if len(plan.lines) > 7:
            probs.append('The STM32 project drives at most 7 lines (TIM2_CH2-CH4 and TIM5_CH1-CH4).')
    else:
        offs = list(st.jetson_offsets) + list(range(len(st.jetson_offsets), n))
        offs = offs[:n]
        if len(set(offs)) < len(offs):
            probs.append('Two trigger lines share a GPIO line offset.')
        pins = {ln.letter: f'{st.jetson_chip} line {o}' for ln, o in zip(plan.lines, offs)}
        extra = {'PPS': 'chrony / phc2sys (or --pps /dev/ppsN)'}
    return pins, extra, probs, notes


def _stm_line_hw(plan, st):
    """[(port letter, pin number, timer, channel)] per line (after pin_assignment found no problem)."""
    out, used = [], {(2, 1)}
    chosen = list(st.stm_pins[:len(plan.lines)]) + [p for p in _default_stm_pins() if p not in st.stm_pins]
    for ln, p in zip(plan.lines, chosen):
        t, ch, af = next(o for o in STM32_PINS[p] if (o[0], o[1]) not in used)
        used.add((t, ch))
        out.append((p[1], int(p[2:]), t, ch))
    return out


# =========================================================================== code generation
def _c_lines(rows, name):
    body = ',\n'.join(f'    {{{r["bursts"]}u, {r["burst"]}u, {r["phase_q"]}ull, {r["period_q"]}ull, '
                      f'{r["spacing_q"]}ull, {r["width"]}u, 0x{r["mask"]:x}u}}' for r in rows) or '    {0}'
    return f'static const tb_line_t {name}[] = {{\n{body}\n}};\n'


def plan_header(plan: FwPlan, target: str, st: FirmwareSettings) -> str:
    """trigger_plan.h: the plan in the target's ticks, the self-test plan and the target's pin macros."""
    T = TARGETS[target]
    hz = T['hz']
    n = len(plan.lines)
    rows, test = core_lines(plan, hz), core_lines(plan, hz, selftest=True)
    lock, step = int(LOCK_NS * hz // 10 ** 9), int(STEP_NS * hz // 10 ** 9)
    inv = ((1 << n) - 1) if st.active_low else 0
    out = [f'/* trigger_plan.h: generated by Vantage ({SCHEMA}) for {plan.project} · {plan.variant}, target '
           f'{T["name"]}.\n * Ticks of {hz:,} Hz; tick quantities x 2^{Q}. Do not edit: generate again from the '
           f'project. */',
           '#ifndef TRIGGER_PLAN_H', '#define TRIGGER_PLAN_H', '#include "trigger_core.h"', '',
           f'#define TB_HZ {hz}u', f'#define TB_N_LINES {n}u', f'#define TB_OUT_INVERT 0x{inv:x}u',
           f'#define TB_ACTIVE_LOW {1 if st.active_low else 0}', '']
    for i, ln in enumerate(plan.lines):
        out.append(f'/* line {i} = {ln.letter}: {", ".join(ln.tags)}; {ln.rate} Hz; first pulse '
                   f'{float(ln.phase_ns) / 1e6:.6f} ms after the PPS (the plan\'s trigger {ln.trigger_ms:.6f} ms'
                   f'{f", latency {st.latency_us:g} µs off" if st.latency_us else ""}) */')
    out += ['', _c_lines(rows, 'TB_LINES'), _c_lines(test, 'TB_LINES_SELFTEST'),
            f'static const tb_plan_t TB_PLAN = {{{hz}u, {n}u, TB_LINES, {lock}u, {step}u, {T["min_gap"]}u}};',
            f'static const tb_plan_t TB_PLAN_SELFTEST = {{{hz}u, {n}u, TB_LINES_SELFTEST, {lock}u, {step}u, '
            f'{T["min_gap"]}u}};', '']
    if target in ('rp2040', 'rust_rp2040'):
        out += [f'#define TB_BASE_PIN {st.rp_base_pin}u', f'#define TB_PPS_PIN {st.rp_pps_pin}u',
                f'#define TB_SELFTEST_PIN {st.rp_selftest_pin}u', '#define TB_LED_PIN 25u', '']
    elif target == 'stm32f411':
        board = STM32_BOARDS[st.stm_board]
        hw = _stm_line_hw(plan, st) if plan.lines else []
        arr = lambda vals, fmt: ', '.join(fmt(v) for v in vals) or '0'
        out += [f'/* {board["name"]}: HSE {board["hse_mhz"]} MHz{" (bypass: the ST-LINK MCO)" if board["bypass"] else ""} */',
                f'#define TB_HSE_MHZ {board["hse_mhz"]}u', f'#define TB_HSE_BYPASS {board["bypass"]}',
                f'#define TB_PPS_PORT \'{st.stm_pps_pin[1]}\'', f'#define TB_PPS_PIN {int(st.stm_pps_pin[2:])}u',
                f'#define TB_SELFTEST_PORT \'{st.stm_selftest_pin[1]}\'',
                f'#define TB_SELFTEST_PIN {int(st.stm_selftest_pin[2:])}u',
                f'#define TB_LED_PORT \'{board["led"][1]}\'', f'#define TB_LED_PIN {int(board["led"][2:])}u',
                f'#define TB_LED_ACTIVE_LOW {board["led_low"]}',
                f'static const char TB_LINE_PORT[] = {{{arr(hw, lambda h: repr(h[0]))}}};',
                f'static const uint8_t TB_LINE_PIN[] = {{{arr(hw, lambda h: str(h[1]))}}};',
                f'static const uint8_t TB_LINE_TIMER[] = {{{arr(hw, lambda h: str(h[2]))}}};',
                f'static const uint8_t TB_LINE_CHANNEL[] = {{{arr(hw, lambda h: str(h[3]))}}};', '']
    else:
        offs = (list(st.jetson_offsets) + list(range(len(st.jetson_offsets), MAX_LINES)))[:MAX_LINES]
        out += [f'#define TB_GPIO_CHIP "{st.jetson_chip}"',
                f'static const unsigned TB_GPIO_OFFSETS[TB_MAX_LINES] = {{{", ".join(map(str, offs))}}};', '']
    out += ['#endif', '']
    return '\n'.join(out)


def plan_rs(plan: FwPlan, st: FirmwareSettings) -> str:
    """plan.rs for the Rust workspace (the same numbers as the C header of the RP2040)."""
    hz = TARGETS['rust_rp2040']['hz']
    lock, step = int(LOCK_NS * hz // 10 ** 9), int(STEP_NS * hz // 10 ** 9)

    def arr(rows):
        return ',\n'.join(f'    trigger_core::Line {{ bursts: {r["bursts"]}, burst: {r["burst"]}, phase_q: {r["phase_q"]}, '
                          f'period_q: {r["period_q"]}, spacing_q: {r["spacing_q"]}, width: {r["width"]}, '
                          f'mask: 0x{r["mask"]:x} }}' for r in rows)
    n = len(plan.lines)
    inv = ((1 << n) - 1) if st.active_low else 0
    return (f'// plan.rs: generated by Vantage ({SCHEMA}) for {plan.project} · {plan.variant}. Ticks of {hz} Hz.\n'
            + ''.join(f'// line {i} = {ln.letter}: {", ".join(ln.tags)}; {ln.rate} Hz; first pulse '
                      f'{float(ln.phase_ns) / 1e6:.6f} ms after the PPS\n' for i, ln in enumerate(plan.lines))
            + f'#[allow(dead_code)]\npub const HZ: u32 = {hz};\n#[allow(dead_code)]\npub const N_LINES: usize = {n};\n'
            f'#[allow(dead_code)]\npub const OUT_INVERT: u32 = 0x{inv:x};\n'
            f'#[allow(dead_code)]\npub const BASE_PIN: u8 = {st.rp_base_pin};\n#[allow(dead_code)]\n'
            f'pub const PPS_PIN: u8 = {st.rp_pps_pin};\n#[allow(dead_code)]\npub const SELFTEST_PIN: u8 = '
            f'{st.rp_selftest_pin};\n'
            f'pub const LINES: [trigger_core::Line; {n}] = [\n{arr(core_lines(plan, hz))}\n];\n'
            f'pub const LINES_SELFTEST: [trigger_core::Line; {n}] = [\n{arr(core_lines(plan, hz, True))}\n];\n'
            f'pub const PLAN: trigger_core::Plan = trigger_core::Plan {{ hz: {hz}, lines: &LINES, lock_ticks: {lock}, '
            f'step_ticks: {step}, min_gap: {TARGETS["rust_rp2040"]["min_gap"]} }};\n'
            f'pub const PLAN_SELFTEST: trigger_core::Plan = trigger_core::Plan {{ hz: {hz}, lines: &LINES_SELFTEST, '
            f'lock_ticks: {lock}, step_ticks: {step}, min_gap: {TARGETS["rust_rp2040"]["min_gap"]} }};\n')


def _res(*parts) -> bytes:
    with open(os.path.join(RES_DIR, *parts), 'rb') as fh:
        return fh.read()


def _subst(data: bytes, plan: FwPlan, st: FirmwareSettings, extra=None) -> bytes:
    n = len(plan.lines)
    vals = {'PROJECT': plan.project, 'VARIANT': plan.variant, 'BASE_PIN': str(st.rp_base_pin),
            'LAST_PIN': str(st.rp_base_pin + max(n, 1) - 1), 'PPS_PIN': str(st.rp_pps_pin),
            'SELFTEST_PIN': (st.stm_selftest_pin if extra == 'stm32f411' else f'GP{st.rp_selftest_pin}'),
            'JETSON_CLOCK': st.jetson_clock,
            'RUST_PIN_SETUP': '\n    '.join(
                [f'let _l{i} = pins.gpio{st.rp_base_pin + i}.into_function::<hal::gpio::FunctionPio0>();'
                 for i in range(n)]
                + [f'let _pps = pins.gpio{st.rp_pps_pin}.into_function::<hal::gpio::FunctionPio0>();',
                   f'let mut selftest = pins.gpio{st.rp_selftest_pin}.into_pull_up_input();'])}
    text = data.decode('utf-8')
    for k, v in vals.items():
        text = text.replace(f'@@{k}@@', v)
    return text.encode('utf-8')


TREE_FILES = {
    'rp2040': [('rp2040', 'main.c'), ('rp2040', 'trigger.pio'), ('rp2040', 'tb_rp2040.h'),
               ('rp2040', 'CMakeLists.txt'), ('rp2040', 'pico_sdk_import.cmake')],
    'stm32f411': [('stm32f411', 'main.c'), ('stm32f411', 'startup.c'), ('stm32f411', 'stm32f411.ld'),
                  ('stm32f411', 'Makefile')],
    'jetson': [('jetson', 'trigger_jetson.c'), ('jetson', 'gpio.h'), ('jetson', 'gpio_gpiod.c'),
               ('jetson', 'gpio_sim.c'), ('jetson', 'Makefile'), ('jetson', 'vantage-trigger.service')],
    'rust_rp2040': [('rust', 'Cargo.toml'), ('rust', 'core', 'Cargo.toml'), ('rust', 'core', 'src', 'lib.rs'),
                    ('rust', 'hostsim', 'Cargo.toml'), ('rust', 'hostsim', 'src', 'main.rs'),
                    ('rust', 'firmware', 'Cargo.toml'), ('rust', 'firmware', 'src', 'main.rs'),
                    ('rust', 'firmware', 'memory.x'), ('rust', 'firmware', 'build.rs'),
                    ('rust', 'firmware', '.cargo', 'config.toml')],
}
COMMON_C = [('common', 'trigger_core.c'), ('common', 'trigger_core.h'), ('common', 'host_sim.c'),
            ('common', 'Makefile.host')]


def generate_tree(plan: FwPlan, target: str, st: FirmwareSettings) -> dict:
    """{relative path: bytes} of a complete project for the target."""
    files = {}
    if target == 'rust_rp2040':
        for parts in TREE_FILES[target]:
            rel = '/'.join(parts[1:])
            files[rel] = _subst(_res(*parts), plan, st, target)
        files['plan.rs'] = plan_rs(plan, st).encode('utf-8')
        files['firmware/trigger.pio'] = _res('rp2040', 'trigger.pio')                # the programs main.rs inlines
        files['c/trigger_core.c'] = _res('common', 'trigger_core.c')          # the C twin, for the comparison
        files['c/trigger_core.h'] = _res('common', 'trigger_core.h')
        files['c/host_sim.c'] = _res('common', 'host_sim.c')
        files['c/trigger_plan.h'] = plan_header(plan, 'rp2040', st).encode('utf-8')
    else:
        for parts in TREE_FILES[target] + COMMON_C:
            files[parts[-1]] = _subst(_res(*parts), plan, st, target)
        files['trigger_plan.h'] = plan_header(plan, target, st).encode('utf-8')
        if target == 'rp2040':
            files['host/rp2040_check.c'] = _res('rp2040', 'host', 'rp2040_check.c')
    files['README.md'] = readme(plan, target, st).encode('utf-8')
    files['schedule.csv'] = schedule_csv(plan).encode('utf-8')
    return files


def write_tree(files: dict, out_dir: str) -> list:
    paths = []
    for rel, data in sorted(files.items()):
        p = os.path.join(out_dir, *rel.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def schedule_csv(plan: FwPlan) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['line', 'cameras', 'pulse', 'time_after_pps_ns', 'time_after_pps_ms'])
    for ln in plan.lines:
        for k, t in enumerate(schedule_ns(ln)):
            w.writerow([ln.letter, ' '.join(ln.tags), k, f'{float(t):.3f}', f'{float(t) / 1e6:.6f}'])
    return buf.getvalue()


PTP_NOTE = (
    'The schedule is laid on the GNSS PPS: pulse k of every second goes out at the line\'s phase + k / rate after the '
    'PPS edge, and the board disciplines its own crystal to the PPS (lock after three PPS within 2 µs, holdover on '
    'the learned frequency when the PPS drops out). Lidars and radars on the same rig are usually disciplined by '
    'gPTP (IEEE 802.1AS) from a grandmaster that itself follows the same GNSS: then the PPS edge, the lidar\'s sync '
    'pulse and these triggers share one time base. Without a PPS from the receiver at the trigger box, take it from '
    'the PTP clock: on a Jetson or a PC with a PTP hardware clock, ptp4l keeps the NIC\'s clock on the grandmaster and '
    'its PPS output (testptp -p / ts2phc, or the PHC\'s periodic output pin) gives the box a PPS that is the gPTP '
    'second. The box\'s trigger then lands within the PPS\'s own error of the lidar\'s sweep.')


def readme(plan: FwPlan, target: str, st: FirmwareSettings) -> str:
    T = TARGETS[target]
    pins, extra, probs, notes = pin_assignment(plan, target, st)
    lines_tab = '\n'.join(f'| {ln.letter} | {pins.get(ln.letter, "?")} | {", ".join(ln.tags)} | {ln.rate} Hz | '
                          f'{float(ln.phase_ns) / 1e6:.4f} ms | {plan.width_ns / 1000:g} µs |' for ln in plan.lines)
    extra_tab = '\n'.join(f'| {k} | {v} |' for k, v in extra.items())
    build = {
        'rp2040': ('```sh\nexport PICO_SDK_PATH=~/pico-sdk          # git clone https://github.com/raspberrypi/pico-sdk\n'
                   'mkdir build && cd build && cmake .. && make -j\n'
                   '# hold BOOTSEL, plug the Pico in, then copy build/vantage_trigger.uf2 to the RPI-RP2 drive, or:\n'
                   'picotool load -x vantage_trigger.uf2\n```'),
        'stm32f411': ('```sh\n# the Arm GNU Toolchain (arm-none-eabi-gcc) and STM32CubeF4\'s CMSIS headers:\n'
                      'git clone --depth 1 https://github.com/STMicroelectronics/STM32CubeF4 ~/STM32CubeF4\n'
                      'make CUBE=~/STM32CubeF4\nmake flash            # st-flash (stlink tools); or: make flash-openocd\n```'),
        'rust_rp2040': ('```sh\nrustup target add thumbv6m-none-eabi\ncargo install flip-link elf2uf2-rs\n'
                        'cd firmware && cargo build --release\n'
                        'elf2uf2-rs -d target/thumbv6m-none-eabi/release/vantage-trigger   # Pico in BOOTSEL mode\n'
                        '# the core and its host simulation, on the build machine (no crates needed):\n'
                        'cargo test --offline && cargo run --release -p hostsim -- seconds=30 ppm=30 jitter_ns=20\n```'),
        'jetson': ('```sh\nsudo apt install libgpiod-dev pps-tools\nmake                         # or: make trigger_jetson_pps\n'
                   'gpioinfo                     # find the chip and the line offsets of your header pins\n'
                   'sudo ./trigger_jetson --stats --seconds 10      # a trial: prints how late the edges were\n'
                   'sudo make install && sudo systemctl enable --now vantage-trigger\n```'),
    }[target]
    host = ('```sh\nmake -f Makefile.host check     # or: make host_sim && ./host_sim seconds=60 ppm=30 jitter_ns=20\n```'
            if target != 'rust_rp2040' else '```sh\ncargo test --offline\n```')
    return f"""# Trigger box · {T['name']}

Generated by Vantage for **{plan.project} · {plan.variant}** from its sync plan ({plan.source} trigger groups).
Board: {T['hardware']}. **Only trigger pulses come out of this project: it drives nothing else.**

## Trigger lines

| Line | Pin | Cameras | Rate | First pulse after the PPS | Width |
|---|---|---|---:|---:|---:|
{lines_tab}

| Signal | Pin |
|---|---|
{extra_tab}

Pulses are {'active low' if st.active_low else 'active high'}; the rising edge of an active-high pulse is the
trigger.{f' The FSYNC-to-exposure latency of {st.latency_us:g} µs is taken off every pulse.' if st.latency_us else ''}
Ground the board to the cameras' trigger ground; use a level shifter or an opto-isolator where the camera's trigger
input is not 3.3 V. The complete schedule of one second is in `schedule.csv`.

{''.join(f'- {n}' + chr(10) for n in notes + plan.notes)}
## Timing

- Resolution: {T['resolution']}.
- Jitter: {T['jitter']}.
- PPS discipline: a least-squares line through the last 16 PPS edges (frequency and phase); lock after three PPS
  within 2 µs; a PPS more than 50 µs off is ignored once and, if the next one
  agrees with it, the schedule steps to it; without PPS for 1.25 s the board holds over on the learned frequency.

## PPS and gPTP

{PTP_NOTE}

## Self-test

Tie the self-test input low at power-up (or start the Jetson program with `--selftest`): line n gives n + 1 pulses
of 2 ms, 20 ms apart, 100 ms after every PPS second, so a scope or an LED on each output shows which line is wired
where.

## Build and flash

Needs: {T['needs']}.

{build}

## Host check

The scheduling core compiles on any machine and runs against a simulated board (crystal error, PPS jitter, a
dropout); Vantage ran it when it generated this project:

{host}
"""


# =========================================================================== PIO simulation
M32 = 0xFFFFFFFF


def parse_pio(text: str) -> dict:
    """{program name: {'instr', 'labels', 'wrap_target', 'wrap'}} from a .pio file (the subset trigger.pio uses)."""
    progs, cur, in_c = {}, None, False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith('% c-sdk') or s.startswith('% rust'):
            in_c = True
            continue
        if in_c:
            in_c = not s.startswith('%}')
            continue
        line = s.split(';', 1)[0].strip()
        if not line:
            continue
        if line.startswith('.program'):
            cur = {'name': line.split()[1], 'instr': [], 'labels': {}, 'wrap_target': 0, 'wrap': None}
            progs[cur['name']] = cur
        elif line == '.wrap_target':
            cur['wrap_target'] = len(cur['instr'])
        elif line == '.wrap':
            cur['wrap'] = len(cur['instr']) - 1
        else:
            m = re.match(r'^(\w+):\s*(.*)$', line)
            if m:
                cur['labels'][m.group(1)] = len(cur['instr'])
                line = m.group(2).strip()
                if not line:
                    continue
            cur['instr'].append(line)
    for p in progs.values():
        if p['wrap'] is None:
            p['wrap'] = len(p['instr']) - 1
    return progs


class PioSM:
    """One PIO state machine, cycle by cycle, for the instructions trigger.pio uses (jmp with no condition, x-- or
    pin; out x / pins with autopull at 32; in x; push noblock). Semantics after the RP2040 datasheet: one cycle per
    instruction (3.2.2); JMP X-- tests X non-zero before the decrement and always decrements (3.4.2); autopull on an
    OUT finding the OSR exhausted refills it and stalls that cycle, an OUT that exhausts it refills it in the same
    cycle, non-OUT cycles refill an exhausted OSR (3.5.4.2); after a restart the OSR is exhausted (3.2.3.3). The JMP
    pin is sampled behind the 2-cycle input synchroniser (pin_in(cycle) sees the pin as it was 2 cycles earlier)."""

    def __init__(self, prog, pin_in=None, fifo=4, autopull=False):
        self.p = prog
        self.pc = 0
        self.x = 0
        self.osr, self.osr_n = 0, 32
        self.isr = 0
        self.tx, self.rx = deque(), deque()
        self.fifo = fifo
        self.autopull = autopull
        self.pin_in = pin_in or (lambda c: 0)
        self.edges, self.pushes = [], []
        self.stalls = 0
        self.cycle = 0

    def step(self):
        cyc = self.cycle
        self.cycle += 1
        ins = self.p['instr'][self.pc]
        op, *rest = ins.replace(',', ' ').split()
        if self.autopull and op != 'out' and self.osr_n >= 32 and self.tx:
            self.osr, self.osr_n = self.tx.popleft(), 0
        nxt = self.p['wrap_target'] if self.pc == self.p['wrap'] else self.pc + 1
        if op == 'jmp':
            cond, tgt = ('', rest[0]) if len(rest) == 1 else rest
            if cond == '':
                take = True
            elif cond == 'x--':
                take = self.x != 0
                self.x = (self.x - 1) & M32
            elif cond == 'pin':
                take = bool(self.pin_in(cyc))
            else:
                raise ValueError(f'jmp condition {cond} is not simulated')
            self.pc = self.p['labels'][tgt] if take else nxt
        elif op == 'out':
            dest, n = rest[0], int(rest[1])
            if self.autopull and self.osr_n >= 32:
                if self.tx:
                    self.osr, self.osr_n = self.tx.popleft(), 0
                self.stalls += 1
                return
            v = self.osr if n == 32 else self.osr & ((1 << n) - 1)
            self.osr = 0 if n == 32 else self.osr >> n
            self.osr_n = min(32, self.osr_n + n)
            if self.autopull and self.osr_n >= 32 and self.tx:
                self.osr, self.osr_n = self.tx.popleft(), 0
            if dest == 'x':
                self.x = v
            elif dest == 'pins':
                self.edges.append((cyc, v))
            else:
                raise ValueError(f'out {dest} is not simulated')
            self.pc = nxt
        elif op == 'in':
            self.isr = self.x if rest[0] == 'x' else 0
            self.pc = nxt
        elif op == 'push':
            if len(self.rx) < self.fifo:
                self.rx.append(self.isr)
                self.pushes.append((cyc, self.isr))
            self.isr = 0
            self.pc = nxt
        else:
            raise ValueError(f'{ins} is not simulated')

    def run(self, until):
        """Step to cycle `until`; a delay loop (jmp x-- to itself) is skipped in one go."""
        while self.cycle < until:
            ins = self.p['instr'][self.pc]
            if ins.startswith('jmp x--') and self.p['labels'].get(ins.split()[-1]) == self.pc and self.x > 1 and \
                    not (self.autopull and self.osr_n >= 32 and self.tx):
                k = min(self.x - 1, until - self.cycle)
                self.x -= k
                self.cycle += k
                continue
            self.step()


# =========================================================================== host checks
def toolchains() -> dict:
    """What this machine can build: gcc (host simulation), cargo (Rust core), cross-compilers (firmware)."""
    have = lambda c: shutil.which(c) is not None
    out = {'gcc': have('gcc') or have('cc'), 'cargo': have('cargo'), 'arm-none-eabi-gcc': have('arm-none-eabi-gcc'),
           'cmake': have('cmake'), 'thumbv6m-none-eabi': False}
    if have('rustup'):
        try:
            r = subprocess.run(['rustup', 'target', 'list', '--installed'], capture_output=True, text=True, timeout=20)
            out['thumbv6m-none-eabi'] = 'thumbv6m-none-eabi' in r.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def _cc():
    return shutil.which('gcc') or shutil.which('cc')


def _run(cmd, cwd, timeout=300, env=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'{" ".join(cmd[:3])}… failed ({r.returncode}): {(r.stderr or r.stdout)[-1500:]}')
    return r.stdout


def parse_sim(text: str) -> dict:
    pps, rises, summ = [], [], None
    for line in text.splitlines():
        f = line.split(',')
        if f[0] == 'P':
            pps.append({'sec': int(f[1]), 'cap': int(f[2]), 'true_ps': int(f[3]), 'stepped': int(f[4]),
                        'state': f[5], 'resid': int(f[6])})
        elif f[0] == 'R':
            rises.append((int(f[1]), int(f[2]), int(f[3]), f[4]))
        elif f[0] == 'S':
            summ = {'pps': int(f[1]), 'missed': int(f[2]), 'steps': int(f[3]), 'outliers': int(f[4]),
                    'late': int(f[5]), 'ppb': int(f[6]), 'state': f[7]}
    return {'pps': pps, 'rises': rises, 'summary': summ}


def _offsets_ns(plan: FwPlan, selftest: bool) -> list:
    """Per line, its pulse offsets within the second (float ns, sorted)."""
    out = []
    for i, ln in enumerate(plan.lines):
        if selftest:
            out.append(np.array([SELFTEST['phase_ms'] * 1e6 + j * SELFTEST['spacing_ms'] * 1e6 for j in range(i + 1)]))
        else:
            out.append(np.array([float(ln.phase_ns + Fraction(k * 10 ** 9, ln.rate)) for k in range(ln.rate)]))
    return out


def rise_errors(sim: dict, plan: FwPlan, pps_offset_ns: float, selftest=False) -> np.ndarray:
    """(line, true ns, error ns, state index) per rise: the error against the nearest planned pulse of the true
    (jitter-free) PPS seconds."""
    offs = _offsets_ns(plan, selftest)
    states = {'free': 0, 'locking': 1, 'locked': 2, 'holdover': 3}
    rows = []
    for line, tick, tps, state in sim['rises']:
        t = tps / 1000.0
        s = math.floor((t - pps_offset_ns) / 1e9)
        cand = np.concatenate([pps_offset_ns + (s + d) * 1e9 + offs[line] for d in (-1, 0, 1)])
        e = t - cand[np.argmin(np.abs(t - cand))]
        rows.append((line, t, e, states[state]))
    return np.array(rows, float).reshape(-1, 4)


def _stats(e) -> dict:
    e = np.asarray(e, float)
    if not len(e):
        return {'n': 0}
    a = np.abs(e)
    return {'n': int(len(e)), 'mean_ns': float(e.mean()), 'std_ns': float(e.std()), 'p99_ns': float(np.percentile(a, 99)),
            'max_ns': float(a.max())}


SCENARIOS = ('exact', 'disciplined', 'lines', 'selftest')


def _sim_args(name, T, st):
    drop0 = st.sim_seconds // 3
    common = [f'seconds={st.sim_seconds}', f'res={T["capture_res"]}', f'seed={st.seed}']
    if name == 'exact':
        return ['seconds=8', 'ppm=0', 'jitter_ns=0', 'pps_offset_ns=0', 'l0=0', 'res=1', 'api=edges'], 0.0
    if name == 'selftest':
        return ['seconds=8', 'ppm=0', 'jitter_ns=0', 'pps_offset_ns=0', 'l0=0', 'res=1', 'plan=selftest'], 0.0
    api = 'lines' if name == 'lines' else 'edges'
    return common + [f'ppm={st.sim_ppm:g}', f'jitter_ns={st.sim_jitter_ns:g}', 'pps_offset_ns=300000000',
                     'l0=12345', f'drop_start={drop0}', f'drop_len={st.sim_drop_s}', f'api={api}'], 3e8


def evaluate(name, sim, plan, T, st, offset) -> dict:
    """The scenario's figures and its pass/fail."""
    hz = T['hz']
    tick_ns = 1e9 / hz
    selftest = name == 'selftest'
    E = rise_errors(sim, plan, offset, selftest)
    out = {'scenario': name, 'summary': sim['summary'], 'rises': int(len(E))}
    first = next((p for p in sim['pps']), None)
    if name in ('exact', 'selftest'):
        # every rise after the first PPS must be the plan's time rounded to the tick
        offs = _offsets_ns(plan, selftest) if selftest else None
        worst_tick = 0.0
        mism = 0
        for line, tick, tps, state in sim['rises']:
            if state != 'locked':
                continue
            if selftest:
                t = tps / 1000.0
                s = math.floor(t / 1e9)
                cand = [Fraction(s * 10 ** 9) + Fraction(o).limit_denominator(10 ** 6) for o in offs[line]]
                exact = min(cand, key=lambda c: abs(float(c) - t))
            else:
                ln = plan.lines[line]
                per = Fraction(10 ** 9, ln.rate)
                t = Fraction(tps, 1000)
                k = round((t - ln.phase_ns) / per)
                exact = ln.phase_ns + k * per
            ex_ticks = exact * hz / 10 ** 9
            d = abs(tick - ex_ticks)
            worst_tick = max(worst_tick, float(d))
            mism += tick != math.floor(ex_ticks + Fraction(1, 2))
        lk = E[E[:, 3] == 2]
        out.update({'locked_rises': int(len(lk)), 'max_tick_error': worst_tick, 'ticks_off_rounding': int(mism),
                    'max_error_ns': float(np.abs(lk[:, 2]).max()) if len(lk) else None})
        out['ok'] = bool(len(lk)) and worst_tick <= 0.5 + 1e-4 and sim['summary']['late'] == 0
        if selftest:
            counts = [int(np.sum((E[:, 0] == i) & (E[:, 1] >= 6e9) & (E[:, 1] < 7e9))) for i in range(len(plan.lines))]
            out['pulses_in_second_6'] = counts
            out['ok'] = out['ok'] and counts == [i + 1 for i in range(len(plan.lines))]
        return out
    locked = E[:, 3] == 2
    hold = E[:, 3] == 3
    out['locked'] = _stats(E[locked, 2])
    out['holdover'] = _stats(E[hold, 2])
    lock_t = next((p['true_ps'] / 1e12 for p in sim['pps'] if p['state'] == 'locked'), None)
    out['lock_after_s'] = None if lock_t is None else lock_t - offset / 1e9
    relock = [p for p in sim['pps'] if p['sec'] >= st.sim_seconds // 3 + st.sim_drop_s]
    out['relock_resid_ns'] = None if not relock else relock[0]['resid'] * tick_ns
    out['final_ppb'] = sim['summary']['ppb']
    out['true_ppb'] = st.sim_ppm * 1000.0
    # bound while locked: the PPS jitter (filtered, 6 sigma is generous) + a tick + the capture's quantisation
    bound = 6 * st.sim_jitter_ns + 2 * tick_ns + T['capture_res'] * tick_ns
    out['locked_bound_ns'] = bound
    out['ok'] = (out['locked'].get('n', 0) > 0 and out['locked']['max_ns'] <= bound
                 and sim['summary']['late'] == 0 and abs(out['final_ppb'] - out['true_ppb']) < 200)
    return out


def _build_c_sim(tree_dir, cc):
    _run([cc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off', '-I.', 'trigger_core.c',
          'host_sim.c', '-lm', '-o', 'host_sim'], tree_dir)
    return os.path.join(tree_dir, 'host_sim')


def pio_chain_check(tree_dir: str, plan: FwPlan, cc: str, n_edges: int = 40) -> dict:
    """The RP2040 chain on simulated PIO: (1) the core's first edges -> the firmware's delay words (tb_rp2040.h,
    compiled) -> tb_trigger cycle by cycle: every pin change on the core's cycle with the core's state; (2) PPS edges
    at known cycles -> tb_capture cycle by cycle (the counter started near its wrap) -> the firmware's capture
    arithmetic (compiled): each edge recovered to -1..0 cycles."""
    progs = parse_pio(open(os.path.join(tree_dir, 'trigger.pio'), encoding='utf-8').read())
    x0 = 1700                                 # the capture counter starts near zero, so it wraps between captures
    exe = os.path.join(tree_dir, 'host', 'rp2040_check')
    _run([cc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', f'-DTB_RP_CAP_X0={x0}u', '-I.', '-Ihost',
          'trigger_core.c', 'host/rp2040_check.c', '-o', exe], tree_dir)
    out = _run([exe, 'edges', str(n_edges)], tree_dir).split()
    vals = [int(v) for v in out]
    edges = [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals), 3)]     # (tick, state, word)
    sm = PioSM(progs['tb_trigger'], fifo=8, autopull=True)
    for _, state, word in edges:
        sm.tx.extend([word, state])
    sm.run(edges[-1][0] + 1)                  # the queue holds every edge: a stall here would be a slip
    stalls = sm.stalls - 1                    # the first OUT's own stall (the OSR starts exhausted) is planned
    got = sm.edges
    trig_ok = len(got) == len(edges) and all(g == (e[0], e[1]) for g, e in zip(got, edges)) and stalls == 0
    trig_err = max((abs(g[0] - e[0]) for g, e in zip(got, edges)), default=None)
    # capture: PPS rises (raw, at the pin) at known cycles, 300-cycle pulses, uneven spacing
    rng = np.random.default_rng(7)
    rises = list(np.cumsum(rng.integers(1100, 1400, 12)) + 101)
    width = 300
    raw = lambda c: any(r <= c < r + width for r in rises)
    cap = PioSM(progs['tb_capture'], pin_in=lambda c: raw(c - 2), fifo=len(rises))   # the CPU drains it
    cap.x = x0
    cap.run(int(rises[-1]) + 50)
    xs = [v for _, v in cap.pushes]
    inp = '\n'.join(f'{x} {int(r) + 5}' for x, r in zip(xs, rises)) + '\n'
    r = subprocess.run([exe, 'capture'], input=inp, cwd=tree_dir, capture_output=True, text=True, timeout=60)
    est = [int(v) for v in r.stdout.split()]
    d = [int(rr) - e for rr, e in zip(rises, est)]
    cap_ok = len(est) == len(rises) and all(-1 <= v <= 0 for v in d)
    wrapped = any(b > a for a, b in zip(xs, xs[1:]))
    return {'trigger_edges': len(edges), 'trigger_matched': bool(trig_ok), 'trigger_max_cycle_error': trig_err,
            'trigger_stalls': stalls, 'captures': len(est), 'capture_errors_cycles': sorted(set(d)),
            'capture_counter_wrapped': bool(wrapped), 'ok': bool(trig_ok and cap_ok and wrapped)}


def _cargo_env(work):
    env = dict(os.environ)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    env['CARGO_TARGET_DIR'] = os.path.join(base, 'vantage', 'cargo-trigger')
    env.setdefault('CARGO_NET_OFFLINE', 'true')
    return env


def rust_check(tree_dir: str, plan: FwPlan, st: FirmwareSettings, cc: str | None) -> dict:
    """cargo test (the core's unit tests) and the Rust host simulation, compared line for line with the C host
    simulation of the same plan."""
    env = _cargo_env(tree_dir)
    t0 = time.perf_counter()
    test_out = _run(['cargo', 'test', '--offline', '--release', '-q'], tree_dir, timeout=600, env=env)
    _run(['cargo', 'build', '--offline', '--release', '-q', '-p', 'hostsim'], tree_dir, timeout=600, env=env)
    exe = os.path.join(env['CARGO_TARGET_DIR'], 'release', 'hostsim')
    T = TARGETS['rust_rp2040']
    out = {'cargo_test': 'passed', 'tests': len(re.findall(r'test result: ok', test_out)) or None,
           'build_seconds': time.perf_counter() - t0, 'compared': {}}
    c_exe = _build_c_sim(os.path.join(tree_dir, 'c'), cc) if cc else None
    same = True
    for name in ('disciplined', 'lines', 'exact'):
        args, _ = _sim_args(name, T, st)
        rs = _run([exe] + args, tree_dir)
        if c_exe:
            cs = _run([c_exe] + args, tree_dir)
            eq = rs == cs
            same &= eq
            out['compared'][name] = {'lines': rs.count('\n'), 'identical': eq}
    out['identical_to_c'] = same if c_exe else None
    out['ok'] = same
    return out


def jetson_measure(tree_dir: str, cc: str, seconds: int = 4) -> dict:
    """Build the Jetson program against the stand-in GPIO and run it here: how late its edges go out."""
    _run([cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I.', 'trigger_jetson.c', 'trigger_core.c',
          'gpio_sim.c', '-lpthread', '-o', 'trigger_jetson_sim'], tree_dir)
    txt = _run(['./trigger_jetson_sim', '--stats', '--seconds', str(seconds), '--prio', '0'], tree_dir,
               timeout=seconds + 30)
    m = re.search(r'edges=(\d+) late_us p50=([\d.]+) p90=([\d.]+) p99=([\d.]+) p999=([\d.]+) max=([\d.]+) '
                  r'state=(\w+) pps=(\d+)', txt)
    if not m:
        raise RuntimeError(f'unexpected output: {txt[-300:]}')
    v = m.groups()
    return {'edges': int(v[0]), 'late_us': {'p50': float(v[1]), 'p90': float(v[2]), 'p99': float(v[3]),
                                            'p999': float(v[4]), 'max': float(v[5])},
            'state': v[6], 'pps': int(v[7]), 'seconds': seconds, 'rt_priority': False,
            'machine': 'this build machine (x86-64 desktop kernel, no real-time priority), not a Jetson',
            'ok': int(v[0]) > 0}


def host_check(files: dict, target: str, plan: FwPlan, st: FirmwareSettings, progress=None, cancel=None) -> dict:
    """Build and simulate one generated project here. {'ok', 'scenarios': {...}, 'pio'?, 'rust'?, 'jetson'?,
    'skipped'?: reason}."""
    tc = toolchains()
    cc = _cc()
    T = TARGETS[target]
    out = {'target': target, 'toolchains': tc, 'scenarios': {}, 'cross_compiled': False,
           'cross_compile_note': f'not compiled for the board here: needs {T["needs"]}'}
    if not plan.lines or plan.problems:
        out['skipped'] = 'nothing to simulate: ' + ('; '.join(plan.problems) or 'no trigger line')
        out['ok'] = False
        return out
    with tempfile.TemporaryDirectory(prefix='vantage-fw-') as work:
        write_tree(files, work)
        if target == 'rust_rp2040':
            if not tc['cargo']:
                out['skipped'] = 'cargo is not installed: the Rust core was not tested here'
                out['ok'] = None
                return out
            out['rust'] = rust_check(work, plan, st, cc)
            sim_dir = os.path.join(work, 'c')
        else:
            sim_dir = work
        if cc is None:
            out['skipped'] = 'no C compiler (gcc) here: the host simulation did not run'
            out['ok'] = None
            return out
        exe = _build_c_sim(sim_dir, cc)
        for i, name in enumerate(SCENARIOS):
            if cancel and cancel():
                return None
            if progress:
                progress(i / len(SCENARIOS), f'{T["short"]}: {name}')
            args, offset = _sim_args(name, T, st)
            sim = parse_sim(_run([exe] + args, sim_dir))
            out['scenarios'][name] = evaluate(name, sim, plan, T, st, offset)
        if target == 'rp2040':
            out['pio'] = pio_chain_check(work, plan, cc)
        if target == 'jetson':
            out['jetson'] = jetson_measure(work, cc, st.jetson_seconds)
        oks = [s['ok'] for s in out['scenarios'].values()] + [out.get(k, {}).get('ok', True) for k in
                                                               ('pio', 'rust', 'jetson')]
        out['ok'] = all(oks)
    return out


# =========================================================================== the study
@dataclass
class FirmwareResult:
    settings: FirmwareSettings
    plan: FwPlan
    trees: dict                          # target -> {relative path: bytes}
    pins: dict                           # target -> (pins, extra, problems, notes)
    checks: dict                         # target -> host_check result (or None)
    findings: list
    metrics: dict
    variant_uid: str = ''


def run(project, engine=None, settings: FirmwareSettings | None = None, variant=None, progress=None, cancel=None,
        targets=None, simulate=True) -> FirmwareResult | None:
    """Generate every target's project from the sync plan and prove each one's core here."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    var = project.variant_by_uid(variant) if isinstance(variant, str) else variant
    var = var or project.variant
    plan = firmware_plan(project, var, st)
    targets = list(targets or TARGET_IDS)
    trees, pins, checks = {}, {}, {}
    for i, t in enumerate(targets):
        if cancel and cancel():
            return None
        pins[t] = pin_assignment(plan, t, st)
        trees[t] = generate_tree(plan, t, st) if plan.lines else {}
        if simulate and plan.lines and not plan.problems:
            sub = (lambda f, text='', i=i: progress((i + f) / len(targets), text)) if progress else None
            try:
                checks[t] = host_check(trees[t], t, plan, st, sub, cancel)
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                checks[t] = {'target': t, 'ok': False, 'error': str(e)[:2000]}
            if checks[t] is None:
                return None
    findings = _findings(plan, pins, checks, st)
    m = {'lines': len(plan.lines), 'targets': len(targets), 'seconds': time.perf_counter() - t0,
         'proved': [t for t, c in checks.items() if c and c.get('ok')],
         'failed': [t for t, c in checks.items() if c and c.get('ok') is False]}
    return FirmwareResult(st, plan, trees, pins, checks, findings, m, var.uid)


def _findings(plan, pins, checks, st) -> list:
    out = [{'severity': 'critical', 'message': p} for p in plan.problems]
    for t, (_, _, probs, notes) in pins.items():
        out += [{'severity': 'serious', 'message': f'{TARGETS[t]["short"]}: {p}', 'target': t} for p in probs]
        out += [{'severity': 'note', 'message': f'{TARGETS[t]["short"]}: {n}', 'target': t} for n in notes]
    for t, c in checks.items():
        if not c:
            continue
        if c.get('error'):
            out.append({'severity': 'serious', 'message': f'{TARGETS[t]["short"]}: the host check failed: '
                                                          f'{c["error"][:300]}', 'target': t})
        elif c.get('skipped'):
            out.append({'severity': 'warning', 'message': f'{TARGETS[t]["short"]}: {c["skipped"]}', 'target': t})
        elif c.get('ok') is False:
            bad = [k for k, s in c['scenarios'].items() if not s['ok']] + [k for k in ('pio', 'rust', 'jetson')
                                                                           if c.get(k, {}).get('ok') is False]
            out.append({'severity': 'critical', 'message': f'{TARGETS[t]["short"]}: the host simulation does not '
                                                           f'meet the plan ({", ".join(bad)})', 'target': t})
        else:
            d = c['scenarios'].get('disciplined', {})
            lk = d.get('locked', {})
            out.append({'severity': 'good', 'message': (
                f'{TARGETS[t]["short"]}: every pulse on its tick with an exact clock; with a {st.sim_ppm:g} ppm '
                f'crystal and {st.sim_jitter_ns:g} ns PPS jitter, locked within {lk.get("max_ns", 0):.0f} ns '
                f'(p99 {lk.get("p99_ns", 0):.0f} ns)'), 'target': t})
    out += [{'severity': 'note', 'message': n} for n in plan.notes]
    return out


# =========================================================================== output
def _r(v, d=3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), d)


def _check_doc(c):
    if not c:
        return None
    doc = {k: v for k, v in c.items() if k != 'toolchains'}
    doc['toolchains'] = c.get('toolchains')
    return json.loads(json.dumps(doc, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def summary(res: FirmwareResult) -> dict:
    out = {'lines': [{'line': ln.letter, 'cameras': ln.tags, 'rate_hz': ln.rate,
                      'first_pulse_ms': _r(float(ln.phase_ns) / 1e6, 6)} for ln in res.plan.lines],
           'source': res.plan.source, 'problems': res.plan.problems, 'targets': {}}
    for t in res.trees or res.pins:
        c = res.checks.get(t) or {}
        d = (c.get('scenarios') or {}).get('disciplined', {})
        out['targets'][t] = {
            'name': TARGETS[t]['name'], 'files': len(res.trees.get(t, {})), 'host_check': c.get('ok'),
            'resolution': TARGETS[t]['resolution'],
            'locked_max_error_ns': _r(d.get('locked', {}).get('max_ns'), 1),
            'locked_p99_ns': _r(d.get('locked', {}).get('p99_ns'), 1),
            'holdover_max_error_ns': _r(d.get('holdover', {}).get('max_ns'), 1),
            'exact_max_tick_error': _r((c.get('scenarios') or {}).get('exact', {}).get('max_tick_error'), 6)}
        if c.get('jetson'):
            out['targets'][t]['measured_late_us'] = c['jetson']['late_us']
        if c.get('rust'):
            out['targets'][t]['rust_identical_to_c'] = c['rust'].get('identical_to_c')
        if c.get('pio'):
            out['targets'][t]['pio_chain'] = c['pio'].get('ok')
    return out


def to_doc(res: FirmwareResult, project=None) -> dict:
    var = project.variant_by_uid(res.variant_uid) if project is not None else None
    return {
        'schema': SCHEMA, 'method_revision': METHOD_REVISION,
        'project': getattr(project, 'name', res.plan.project), 'variant': var.name if var else res.plan.variant,
        'conventions': {'time': 'ns after the GNSS PPS edge (the whole second); a line\'s pulse k at its first pulse '
                                '+ k / rate', 'ticks': 'each target\'s timer ticks; tick quantities x 2^24',
                        'error': 'a rise\'s true time minus the plan\'s, against the jitter-free PPS seconds'},
        'settings': res.settings.to_dict(),
        'lidar': res.plan.lidar,
        'lines': [{'line': ln.letter, 'cameras': ln.tags, 'rate_hz': ln.rate,
                   'first_pulse_ns': float(ln.phase_ns), 'plan_trigger_ms': ln.trigger_ms,
                   'period_ms': ln.period_ms, 'plan_worst_misalignment_ms': ln.worst_ms}
                  for ln in res.plan.lines],
        'width_ns': res.plan.width_ns,
        'targets': {t: {'name': TARGETS[t]['name'], 'hardware': TARGETS[t]['hardware'], 'clock_hz': TARGETS[t]['hz'],
                        'resolution': TARGETS[t]['resolution'], 'jitter': TARGETS[t]['jitter'],
                        'needs': TARGETS[t]['needs'], 'pins': res.pins[t][0], 'other_pins': res.pins[t][1],
                        'pin_problems': res.pins[t][2], 'files': sorted(res.trees.get(t, {})),
                        'host_check': _check_doc(res.checks.get(t))} for t in res.pins},
        'findings': res.findings, 'notes': res.plan.notes, 'problems': res.plan.problems,
        'assumptions': list(ASSUMPTIONS),
    }


def files(res: FirmwareResult, project=None) -> list:
    out = [('trigger_firmware.json', (json.dumps(to_doc(res, project), indent=1, ensure_ascii=False) + '\n').encode()),
           ('trigger_schedule.csv', schedule_csv(res.plan).encode())]
    for t, tree in res.trees.items():
        out += [(f'firmware/{t}/{rel}', data) for rel, data in sorted(tree.items())]
    return out


def export(res: FirmwareResult, project, out_dir) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, *name.split('/'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def exit_code(res: FirmwareResult | None) -> int:
    """Batch: 0 every target generated and proved here, 1 a plan problem or a failed host check, 2 generated but
    not proved (a toolchain missing), 3 cancelled."""
    if res is None:
        return 3
    if res.plan.problems or res.metrics['failed']:
        return 1
    if len(res.metrics['proved']) < len(res.trees):
        return 2
    return 0


def issues(project, vg=None) -> list:
    """Cheap: a trigger line the firmware cannot run (a frame rate that is not a whole number of pulses a second)."""
    from .rules import Issue
    try:
        st = settings_of(project)
        if not getattr(project.analysis, 'trigger_fw', None):
            return []
        plan = firmware_plan(project, None, st)
    except Exception:                   # noqa: BLE001 - a project the plan cannot read is judged elsewhere
        return []
    return [Issue('warning', f'Trigger firmware: {p}', 'Trigger firmware', [], 'Open Embedded ▸ Trigger firmware')
            for p in plan.problems[:3]]


# =========================================================================== validation
def validation_checks() -> list:
    """Spec-compatible checks for vantage.validation (the lead registers them)."""
    def _exact(ctx):
        from ..validation.suite import Outcome
        from . import samples
        p = samples.tigor_sample_project()
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        st = FirmwareSettings(groups='suggested')
        plan = firmware_plan(p, None, st)
        tree = generate_tree(plan, 'stm32f411', st)
        c = host_check(tree, 'stm32f411', plan, st)
        ex = c['scenarios']['exact']
        return Outcome(float(ex['max_tick_error']), {'rises': ex['rises'], 'lines': len(plan.lines),
                                                     'ticks off the rounded plan': ex['ticks_off_rounding']},
                       ok=bool(ex['ok']))

    def _pio(ctx):
        from ..validation.suite import Outcome
        from . import samples
        p = samples.tigor_sample_project()
        p.active_uid = next(v for v in p.variants if v.sensors).uid
        st = FirmwareSettings(groups='suggested')
        plan = firmware_plan(p, None, st)
        with tempfile.TemporaryDirectory() as work:
            write_tree(generate_tree(plan, 'rp2040', st), work)
            r = pio_chain_check(work, plan, _cc())
        err = max([r['trigger_max_cycle_error'] or 0] + [abs(v) for v in r['capture_errors_cycles']])
        return Outcome(float(err), {k: v for k, v in r.items() if k != 'ok'}, ok=bool(r['ok']))

    return [
        {'id': 'trigger_fw.exact_ticks', 'model': 'timing',
         'title': 'Generated trigger firmware puts every pulse on the plan\'s tick',
         'quantity': 'largest |rise tick − plan time × clock| over 6 s of the sample rig\'s suggested lines, the '
                     'STM32 project\'s scheduling core compiled with gcc and run against an exact simulated clock',
         'reference': 'the sync plan\'s trigger times in exact rational arithmetic (Python Fractions), rounded to '
                      'the 100 MHz tick', 'tolerance': 0.5, 'unit': 'ticks', 'refs': (), 'fn': _exact,
         'cost': 'study'},
        {'id': 'trigger_fw.pio_cycles', 'model': 'timing',
         'title': 'RP2040 PIO programs place edges and timestamp the PPS to the cycle',
         'quantity': 'largest cycle error of the trigger edges (core ticks through the firmware\'s delay words and '
                     'a cycle-exact PIO simulation) and of the recovered PPS edges (capture program and the '
                     'firmware\'s arithmetic, across a counter wrap)',
         'reference': 'the core\'s edge ticks; PPS edges placed at known cycles (RP2040 datasheet 3.2.2, 3.4.2, '
                      '3.5.4.2 instruction semantics)', 'tolerance': 1.0, 'unit': 'cycles', 'refs': (), 'fn': _pio,
         'cost': 'study'},
    ]
