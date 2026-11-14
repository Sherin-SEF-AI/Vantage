"""Trigger-box firmware (core.trigger_firmware): the plan, the generated trees, the scheduling core proved on the host
(gcc; cargo for the Rust twin), the PIO programs cycle by cycle, an injected error caught, and the page."""
import json
import os
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from vantage.core import samples
from vantage.core import timing as TM
from vantage.core import trigger_firmware as FW

HAVE_GCC = FW._cc() is not None
HAVE_CARGO = shutil.which('cargo') is not None
needs_gcc = pytest.mark.skipif(not HAVE_GCC, reason='needs a C compiler')


@pytest.fixture(scope='module')
def proj():
    p = samples.tigor_sample_project()
    p.active_uid = next(v for v in p.variants if v.sensors).uid
    return p


@pytest.fixture(scope='module')
def plan(proj):
    return FW.firmware_plan(proj, None, FW.FirmwareSettings(groups='suggested'))


# ---------------------------------------------------------------------- settings and the plan
def test_settings_tolerate_junk_and_clamp():
    s = FW.FirmwareSettings.from_dict({'width_us': 'x', 'latency_us': 1e9, 'target': 'avr', 'groups': 'suggested',
                                       'rp_base_pin': 99.4, 'stm_pins': ['pa0', 5, None], 'stm_pps_pin': 'PZ9',
                                       'jetson_offsets': [3, '4', 'x'], 'jetson_chip': '../../etc;rm', 'seed': -3,
                                       'active_low': 'yes', 'sim_seconds': 1})
    assert s.width_us == 100.0 and s.latency_us == 20000.0 and s.target == 'rp2040' and s.groups == 'suggested'
    assert s.rp_base_pin == 28 and s.stm_pins == ['PA0', '5', 'NONE'] and s.stm_pps_pin == 'PA15'
    assert s.jetson_offsets == [3, 4] and s.jetson_chip == 'gpiochip0' and s.seed == 0 and s.active_low is False
    assert s.sim_seconds == 10
    assert FW.FirmwareSettings.from_dict(None) == FW.FirmwareSettings()
    assert FW.FirmwareSettings.from_dict(s.to_dict()) == s


def test_plan_follows_the_sync_plan(proj):
    res = TM.plan(proj)
    fp = FW.firmware_plan(proj, None, FW.FirmwareSettings())
    assert fp.source == 'plan' and not fp.problems
    assert [ln.letter for ln in fp.lines] == [g['letter'] for g in res.groups]
    for ln, g in zip(fp.lines, res.groups):
        assert ln.rate == 30
        assert float(ln.phase_ns) / 1e6 == pytest.approx(g['trigger_ms'] % (1000 / 30), abs=1e-6)
    sug = FW.firmware_plan(proj, None, FW.FirmwareSettings(groups='suggested'))
    assert sug.source == 'suggested' and len(sug.lines) == res.suggestion['lines']
    # the latency moves every pulse earlier, modulo the line's period
    lat = FW.firmware_plan(proj, None, FW.FirmwareSettings(groups='suggested', latency_us=250.0))
    for a, b in zip(sug.lines, lat.lines):
        assert float((a.phase_ns - b.phase_ns) % Fraction(10 ** 9, 30)) == pytest.approx(250_000.0)


def test_rates_and_widths_that_cannot_run(proj):
    assert FW._rate_of([29.97]) is None and FW._rate_of([30.0, 15.0]) == 30
    fp = FW.firmware_plan(proj, None, FW.FirmwareSettings(width_us=17000.0))
    assert any('does not fit' in p for p in fp.problems)


def test_schedule_and_quantisation(plan):
    ln = plan.lines[0]
    t = FW.schedule_ns(ln, 2)
    assert len(t) == 60 and t[1] - t[0] == Fraction(10 ** 9, 30) and t[30] - t[0] == 10 ** 9
    rows = FW.core_lines(plan, 125_000_000)
    assert rows[0]['phase_q'] / 2 ** FW.Q == pytest.approx(float(ln.phase_ns) * 0.125, abs=1e-6)
    assert rows[0]['period_q'] == (125_000_000 << FW.Q) // 30
    test = FW.core_lines(plan, 125_000_000, selftest=True)
    assert [r['burst'] for r in test] == list(range(1, len(plan.lines) + 1))


# ---------------------------------------------------------------------- trees
def test_trees_hold_what_each_target_needs(plan):
    st = FW.FirmwareSettings(groups='suggested')
    need = {'rp2040': {'main.c', 'trigger.pio', 'tb_rp2040.h', 'CMakeLists.txt', 'pico_sdk_import.cmake',
                       'trigger_core.c', 'trigger_core.h', 'trigger_plan.h', 'host_sim.c', 'README.md'},
            'stm32f411': {'main.c', 'startup.c', 'stm32f411.ld', 'Makefile', 'Makefile.host', 'trigger_core.c',
                          'trigger_plan.h', 'README.md'},
            'jetson': {'trigger_jetson.c', 'gpio_gpiod.c', 'gpio_sim.c', 'gpio.h', 'Makefile',
                       'vantage-trigger.service', 'trigger_plan.h', 'README.md'},
            'rust_rp2040': {'Cargo.toml', 'core/src/lib.rs', 'hostsim/src/main.rs', 'firmware/src/main.rs',
                            'firmware/Cargo.toml', 'firmware/memory.x', 'firmware/.cargo/config.toml', 'plan.rs',
                            'README.md'}}
    for t, files in need.items():
        tree = FW.generate_tree(plan, t, st)
        assert files <= set(tree), (t, files - set(tree))
        text = {k: v.decode('utf-8') for k, v in tree.items()}
        assert not any('@@' in v for v in text.values()), t
        rd = text['README.md']
        for words in ('## Trigger lines', '## PPS and gPTP', '## Self-test', '## Build and flash', 'Only trigger pulses'):
            assert words in rd, (t, words)
        for ln in plan.lines:
            assert f'| {ln.letter} |' in rd
    hdr = FW.generate_tree(plan, 'rp2040', st)['trigger_plan.h'].decode()
    assert '#define TB_HZ 125000000u' in hdr and '#define TB_BASE_PIN 2u' in hdr and 'TB_PLAN_SELFTEST' in hdr
    stm = FW.generate_tree(plan, 'stm32f411', st)['trigger_plan.h'].decode()
    assert 'TB_LINE_TIMER[] = {2, 2, 5, 5}' in stm and "TB_LINE_PORT[] = {'B', 'B', 'A', 'A'}" in stm
    rs = FW.generate_tree(plan, 'rust_rp2040', st)['firmware/src/main.rs'].decode()
    assert 'pins.gpio2.into_function' in rs and 'pins.gpio15.into_function' in rs


def test_pin_assignment_problems(plan):
    st = FW.FirmwareSettings(rp_base_pin=21)
    assert any('not on the Pico' in p for p in FW.pin_assignment(plan, 'rp2040', st)[2])
    st = FW.FirmwareSettings(rp_pps_pin=3)
    assert any('PPS input GP3 is one of the trigger lines' in p for p in FW.pin_assignment(plan, 'rp2040', st)[2])
    st = FW.FirmwareSettings(stm_pins=['PB3', 'PB3', 'PA0', 'PA1'])
    assert any('already in use' in p for p in FW.pin_assignment(plan, 'stm32f411', st)[2])
    st = FW.FirmwareSettings(stm_pins=['PC7', 'PB3', 'PA0', 'PA1'])
    assert any('no free TIM2/TIM5' in p for p in FW.pin_assignment(plan, 'stm32f411', st)[2])
    ok = FW.pin_assignment(plan, 'stm32f411', FW.FirmwareSettings())
    assert not ok[2] and ok[0]['A'].startswith('PB3 (TIM2_CH2')
    assert any('Two trigger lines' in p for p in FW.pin_assignment(plan, 'jetson',
                                                                     FW.FirmwareSettings(jetson_offsets=[5, 5]))[2])


# ---------------------------------------------------------------------- the PIO programs
def _progs():
    return FW.parse_pio(open(os.path.join(FW.RES_DIR, 'rp2040', 'trigger.pio'), encoding='utf-8').read())


def test_pio_trigger_program_spacing_and_stall():
    sm = FW.PioSM(_progs()['tb_trigger'], fifo=8, autopull=True)
    words = [(5, 1), (0, 0), (100, 1), (7, 3), (2_000_000, 0)]
    for d, s in words:
        sm.tx.extend([d, s])
    sm.run(2_000_200)
    cyc = [c for c, _ in sm.edges]
    assert cyc[0] == 5 + 3                                   # the first OUT stalls to fill the exhausted OSR
    assert [b - a for a, b in zip(cyc, cyc[1:])] == [d + 3 for d, _ in words[1:]]
    assert [v for _, v in sm.edges] == [s for _, s in words]
    assert sm.stalls >= 2                                    # the start, then the empty queue at the end


def test_pio_capture_counts_two_cycles_per_count():
    progs = _progs()
    rises = [1000, 2203, 3507, 4800, 6111]
    raw = lambda c: any(r <= c < r + 250 for r in rises)
    cap = FW.PioSM(progs['tb_capture'], pin_in=lambda c: raw(c - 2), fifo=16)
    cap.x = 0xFFFFFFFF
    cap.run(7000)
    assert len(cap.pushes) == len(rises)
    for k, ((_, x), r) in enumerate(zip(cap.pushes, rises)):
        est = 2 * (0xFFFFFFFF - x) + 2 * k - 4               # tb_rp2040.h's formula
        assert -1 <= r - est <= 0


@needs_gcc
def test_pio_chain_with_the_firmware_arithmetic(plan):
    st = FW.FirmwareSettings(groups='suggested')
    with tempfile.TemporaryDirectory() as d:
        FW.write_tree(FW.generate_tree(plan, 'rp2040', st), d)
        r = FW.pio_chain_check(d, plan, FW._cc(), n_edges=30)
    assert r['ok'] and r['trigger_matched'] and r['trigger_max_cycle_error'] == 0 and r['trigger_stalls'] == 0
    assert r['capture_counter_wrapped'] and set(r['capture_errors_cycles']) <= {-1, 0}


# ---------------------------------------------------------------------- the host simulation
@pytest.fixture(scope='module')
def checks(plan):
    st = FW.FirmwareSettings(groups='suggested', sim_seconds=40, sim_drop_s=8, jetson_seconds=1)
    if not HAVE_GCC:
        pytest.skip('needs a C compiler')
    return {t: FW.host_check(FW.generate_tree(plan, t, st), t, plan, st)
            for t in FW.TARGET_IDS if t != 'rust_rp2040' or HAVE_CARGO}


def test_every_pulse_on_its_tick_with_an_exact_clock(checks, plan):
    for t, c in checks.items():
        ex = c['scenarios']['exact']
        assert ex['ok'] and ex['max_tick_error'] <= 0.5 + 1e-4 and ex['ticks_off_rounding'] == 0, t
        assert ex['locked_rises'] >= len(plan.lines) * 30 * 4
        assert c['scenarios']['exact']['summary']['late'] == 0


def test_disciplined_crystal_locks_holds_over_and_relocks(checks):
    for t, c in checks.items():
        d = c['scenarios']['disciplined']
        assert d['ok'], t
        assert d['locked']['n'] > 1000 and d['locked']['max_ns'] < 60, (t, d['locked'])
        assert d['holdover']['n'] > 100 and d['holdover']['max_ns'] < 80
        assert d['lock_after_s'] == pytest.approx(4.0, abs=0.01)
        assert d['summary']['missed'] == 8 and d['summary']['state'] == 'locked'
        assert abs(d['final_ppb'] - 30_000) < 20
        # the per-line API (one timer channel per line) gives the very same pulses as the merged edges
        assert c['scenarios']['lines']['locked'] == d['locked']


def test_selftest_pattern(checks, plan):
    for c in checks.values():
        s = c['scenarios']['selftest']
        assert s['ok'] and s['pulses_in_second_6'] == [i + 1 for i in range(len(plan.lines))]


def test_rp2040_and_jetson_extras(checks):
    assert checks['rp2040']['pio']['ok']
    j = checks['jetson']['jetson']
    assert j['ok'] and j['edges'] > 50 and j['late_us']['p50'] >= 0
    for c in checks.values():
        assert c['ok'] and c['cross_compiled'] is False and 'needs' in c['cross_compile_note']


@pytest.mark.skipif(not (HAVE_GCC and HAVE_CARGO), reason='needs gcc and cargo')
def test_rust_core_matches_c_number_for_number(checks):
    r = checks['rust_rp2040']['rust']
    assert r['cargo_test'] == 'passed' and r['identical_to_c'] is True
    assert all(v['identical'] and v['lines'] > 500 for v in r['compared'].values())


@needs_gcc
def test_an_injected_rounding_error_is_caught(plan):
    """The exact-clock check is independent of the core: truncating instead of rounding breaks it."""
    st = FW.FirmwareSettings(groups='suggested')
    tree = FW.generate_tree(plan, 'stm32f411', st)
    src = tree['trigger_core.c'].decode()
    assert 'fdiv_q(rel + HALF_Q)' in src
    tree['trigger_core.c'] = src.replace('fdiv_q(rel + HALF_Q)', 'fdiv_q(rel)').encode()
    c = FW.host_check(tree, 'stm32f411', plan, st)
    assert c['scenarios']['exact']['ok'] is False and c['scenarios']['exact']['ticks_off_rounding'] > 0
    # and a schedule a millisecond late is far outside the locked bound
    tree = FW.generate_tree(plan, 'stm32f411', st)
    hdr = tree['trigger_plan.h'].decode()
    rows = FW.core_lines(plan, 100_000_000)
    hdr = hdr.replace(f'{rows[0]["phase_q"]}ull', f'{rows[0]["phase_q"] + (100_000 << FW.Q)}ull', 1)
    tree['trigger_plan.h'] = hdr.encode()
    c = FW.host_check(tree, 'stm32f411', plan, st)
    assert c['scenarios']['disciplined']['ok'] is False


# ---------------------------------------------------------------------- the firmware sources compile (syntax)
STUB_CMSIS = r'''
#pragma once
#include <stdint.h>
#define __IO volatile
typedef struct { __IO uint32_t CR1, CR2, SMCR, DIER, SR, EGR, CCMR1, CCMR2, CCER, CNT, PSC, ARR, RCR, CCR1, CCR2, CCR3,
                 CCR4; } TIM_TypeDef;
typedef struct { __IO uint32_t MODER, OTYPER, OSPEEDR, PUPDR, IDR, ODR, BSRR, LCKR, AFR[2]; } GPIO_TypeDef;
typedef struct { __IO uint32_t CR, PLLCFGR, CFGR, CIR, AHB1RSTR, AHB2RSTR, r0, r1, APB1RSTR, APB2RSTR, r2, r3,
                 AHB1ENR, AHB2ENR, r4, r5, APB1ENR, APB2ENR; } RCC_TypeDef;
typedef struct { __IO uint32_t CR, CSR; } PWR_TypeDef;
typedef struct { __IO uint32_t ACR; } FLASH_TypeDef;
typedef struct { __IO uint32_t SR, DR, BRR, CR1, CR2, CR3, GTPR; } USART_TypeDef;
extern TIM_TypeDef *TIM2, *TIM5; extern GPIO_TypeDef *GPIOA, *GPIOB, *GPIOC; extern RCC_TypeDef *RCC;
extern PWR_TypeDef *PWR; extern FLASH_TypeDef *FLASH; extern USART_TypeDef *USART1; extern uint32_t SystemCoreClock;
typedef enum { TIM2_IRQn = 28, TIM5_IRQn = 50 } IRQn_Type;
void NVIC_SetPriority(IRQn_Type, uint32_t); void NVIC_EnableIRQ(IRQn_Type); uint32_t SysTick_Config(uint32_t);
void __NOP(void); void __disable_irq(void); void __enable_irq(void);
#define TIM_SR_UIF 1u
#define TIM_SR_CC1IF 2u
#define TIM_DIER_UIE 1u
#define TIM_DIER_CC1IE 2u
#define TIM_EGR_UG 1u
#define TIM_CR1_CEN 1u
#define TIM_CR2_MMS (7u << 4)
#define TIM_CR2_MMS_0 (1u << 4)
#define TIM_SMCR_SMS 7u
#define TIM_SMCR_SMS_1 2u
#define TIM_SMCR_SMS_2 4u
#define TIM_SMCR_TS (7u << 4)
#define TIM_CCMR1_CC1S_0 1u
#define TIM_CCER_CC1E 1u
#define TIM_CCER_CC1P 2u
#define RCC_CR_HSEON (1u << 16)
#define RCC_CR_HSERDY (1u << 17)
#define RCC_CR_HSEBYP (1u << 18)
#define RCC_CR_PLLON (1u << 24)
#define RCC_CR_PLLRDY (1u << 25)
#define RCC_PLLCFGR_PLLM_Pos 0
#define RCC_PLLCFGR_PLLN_Pos 6
#define RCC_PLLCFGR_PLLP_Pos 16
#define RCC_PLLCFGR_PLLQ_Pos 24
#define RCC_PLLCFGR_PLLSRC_HSE (1u << 22)
#define RCC_CFGR_HPRE_DIV1 0u
#define RCC_CFGR_PPRE1_DIV2 (4u << 10)
#define RCC_CFGR_PPRE2_DIV1 0u
#define RCC_CFGR_SW_PLL 2u
#define RCC_CFGR_SWS (3u << 2)
#define RCC_CFGR_SWS_PLL (2u << 2)
#define RCC_APB1ENR_PWREN (1u << 28)
#define RCC_APB1ENR_TIM2EN 1u
#define RCC_APB1ENR_TIM5EN 8u
#define RCC_APB2ENR_USART1EN 16u
#define RCC_AHB1ENR_GPIOAEN 1u
#define RCC_AHB1ENR_GPIOBEN 2u
#define RCC_AHB1ENR_GPIOCEN 4u
#define PWR_CR_VOS (3u << 14)
#define FLASH_ACR_ICEN (1u << 9)
#define FLASH_ACR_DCEN (1u << 10)
#define FLASH_ACR_PRFTEN (1u << 8)
#define FLASH_ACR_LATENCY_3WS 3u
#define USART_CR1_TE 8u
#define USART_CR1_UE (1u << 13)
#define USART_SR_TXE (1u << 7)
'''

STUB_PICO = r'''
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
typedef unsigned int uint;
typedef struct { volatile uint32_t fdebug; } pio_hw_t;
typedef pio_hw_t *PIO;
extern PIO pio0;
#define PIO_FDEBUG_TXSTALL_LSB 24
#define GPIO_IN 0
#define GPIO_OUT 1
typedef struct { uint32_t a, b, c, d; } pio_sm_config;
typedef struct { const uint16_t *instructions; uint8_t length; int8_t origin; } pio_program_t;
enum pio_src_dest { pio_x = 1, pio_null = 3 };
enum pio_fifo_join { PIO_FIFO_JOIN_TX = 1, PIO_FIFO_JOIN_RX = 2 };
bool set_sys_clock_khz(uint32_t, bool); bool stdio_init_all(void); void gpio_init(uint); void gpio_set_dir(uint, bool);
void gpio_pull_up(uint); bool gpio_get(uint); void gpio_put(uint, bool); void sleep_ms(uint32_t);
uint64_t time_us_64(void); void watchdog_enable(uint32_t, bool); void watchdog_update(void);
uint pio_add_program(PIO, const pio_program_t *); void pio_sm_put(PIO, uint, uint32_t); uint32_t pio_sm_get(PIO, uint);
uint pio_sm_get_tx_fifo_level(PIO, uint); bool pio_sm_is_rx_fifo_empty(PIO, uint);
void pio_set_sm_mask_enabled(PIO, uint32_t, bool); void pio_sm_clear_fifos(PIO, uint); void pio_sm_restart(PIO, uint);
void pio_sm_exec(PIO, uint, uint); void pio_sm_set_pins_with_mask(PIO, uint, uint32_t, uint32_t);
void pio_enable_sm_mask_in_sync(PIO, uint32_t); uint pio_encode_jmp(uint); uint pio_encode_mov_not(enum pio_src_dest, enum pio_src_dest);
void sm_config_set_out_pins(pio_sm_config *, uint, uint); void sm_config_set_out_shift(pio_sm_config *, bool, bool, uint);
void sm_config_set_in_shift(pio_sm_config *, bool, bool, uint); void sm_config_set_fifo_join(pio_sm_config *, enum pio_fifo_join);
void sm_config_set_clkdiv_int_frac(pio_sm_config *, uint16_t, uint8_t); void sm_config_set_jmp_pin(pio_sm_config *, uint);
void pio_gpio_init(PIO, uint); int pio_sm_set_consecutive_pindirs(PIO, uint, uint, uint, bool);
int pio_sm_init(PIO, uint, uint, const pio_sm_config *);
'''

STUB_GPIOD1 = r'''
struct gpiod_chip; struct gpiod_line;
struct gpiod_line_bulk { struct gpiod_line *lines[64]; unsigned int num_lines; };
struct gpiod_chip *gpiod_chip_open(const char *); struct gpiod_chip *gpiod_chip_open_by_name(const char *);
int gpiod_chip_get_lines(struct gpiod_chip *, unsigned int *, unsigned int, struct gpiod_line_bulk *);
int gpiod_line_request_bulk_output(struct gpiod_line_bulk *, const char *, const int *);
int gpiod_line_set_value_bulk(struct gpiod_line_bulk *, const int *);
void gpiod_line_release_bulk(struct gpiod_line_bulk *); void gpiod_chip_close(struct gpiod_chip *);
'''

STUB_GPIOD2 = r'''
#include <stddef.h>
struct gpiod_chip; struct gpiod_line_settings; struct gpiod_line_config; struct gpiod_request_config;
struct gpiod_line_request;
enum gpiod_line_value { GPIOD_LINE_VALUE_INACTIVE = 0, GPIOD_LINE_VALUE_ACTIVE = 1 };
enum gpiod_line_direction { GPIOD_LINE_DIRECTION_OUTPUT = 3 };
struct gpiod_chip *gpiod_chip_open(const char *); void gpiod_chip_close(struct gpiod_chip *);
struct gpiod_line_settings *gpiod_line_settings_new(void); void gpiod_line_settings_free(struct gpiod_line_settings *);
int gpiod_line_settings_set_direction(struct gpiod_line_settings *, enum gpiod_line_direction);
struct gpiod_line_config *gpiod_line_config_new(void); void gpiod_line_config_free(struct gpiod_line_config *);
int gpiod_line_config_add_line_settings(struct gpiod_line_config *, const unsigned int *, size_t, struct gpiod_line_settings *);
int gpiod_line_config_set_output_values(struct gpiod_line_config *, const enum gpiod_line_value *, size_t);
struct gpiod_request_config *gpiod_request_config_new(void); void gpiod_request_config_free(struct gpiod_request_config *);
void gpiod_request_config_set_consumer(struct gpiod_request_config *, const char *);
struct gpiod_line_request *gpiod_chip_request_lines(struct gpiod_chip *, struct gpiod_request_config *, struct gpiod_line_config *);
int gpiod_line_request_set_values(struct gpiod_line_request *, const enum gpiod_line_value *);
void gpiod_line_request_release(struct gpiod_line_request *);
'''


def _syntax(cwd, files, *flags):
    r = subprocess.run([FW._cc(), '-fsyntax-only', '-std=c11', '-Wall', '-Wextra', '-Werror', '-Istub', '-I.', *flags,
                        *files], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


@needs_gcc
def test_firmware_sources_compile_against_stub_sdk_headers(plan):
    """A syntax and type check of the firmware's own C against minimal stand-ins for the SDK headers (CMSIS,
    pico-sdk, libgpiod v1 and v2): not the real SDKs, which are not installed here."""
    st = FW.FirmwareSettings(groups='suggested')
    with tempfile.TemporaryDirectory() as d:
        for t in ('stm32f411', 'rp2040', 'jetson'):
            root = os.path.join(d, t)
            FW.write_tree(FW.generate_tree(plan, t, st), root)
            os.makedirs(os.path.join(root, 'stub'))
        open(os.path.join(d, 'stm32f411', 'stub', 'stm32f4xx.h'), 'w').write(STUB_CMSIS)
        _syntax(os.path.join(d, 'stm32f411'), ['main.c', 'startup.c', 'trigger_core.c'])
        pio = open(os.path.join(d, 'rp2040', 'trigger.pio')).read()
        csdk = pio[pio.index('% c-sdk {') + len('% c-sdk {'):pio.index('%}')]
        stub = os.path.join(d, 'rp2040', 'stub')
        os.makedirs(os.path.join(stub, 'pico'))
        os.makedirs(os.path.join(stub, 'hardware'))
        open(os.path.join(stub, 'pico', 'stdlib.h'), 'w').write(STUB_PICO)
        for h in ('clocks', 'gpio', 'pio', 'watchdog'):
            open(os.path.join(stub, 'hardware', h + '.h'), 'w').write('#include "pico/stdlib.h"\n')
        open(os.path.join(stub, 'trigger.pio.h'), 'w').write(
            '#include "pico/stdlib.h"\nstatic const pio_program_t tb_trigger_program = {0, 3, -1};\n'
            'static const pio_program_t tb_capture_program = {0, 9, -1};\n'
            'static inline pio_sm_config tb_trigger_program_get_default_config(uint o) { pio_sm_config c = {o, 0, 0, 0}; return c; }\n'
            'static inline pio_sm_config tb_capture_program_get_default_config(uint o) { pio_sm_config c = {o, 0, 0, 0}; return c; }\n'
            + csdk)
        _syntax(os.path.join(d, 'rp2040'), ['main.c', 'trigger_core.c'])
        jet = os.path.join(d, 'jetson')
        open(os.path.join(jet, 'stub', 'gpiod.h'), 'w').write(STUB_GPIOD1)
        _syntax(jet, ['trigger_jetson.c', 'gpio_gpiod.c', 'trigger_core.c'])
        open(os.path.join(jet, 'stub', 'gpiod.h'), 'w').write(STUB_GPIOD2)
        _syntax(jet, ['gpio_gpiod.c'], '-DTB_GPIOD_V2')


# ---------------------------------------------------------------------- the study, exports, validation
@needs_gcc
def test_run_summary_exports(proj, tmp_path):
    st = FW.FirmwareSettings(groups='suggested', sim_seconds=20, sim_drop_s=4, jetson_seconds=1)
    res = FW.run(proj, settings=st, targets=['stm32f411', 'jetson'])
    assert res.metrics['proved'] == ['stm32f411', 'jetson'] and FW.exit_code(res) == 0
    s = FW.summary(res)
    json.dumps(s)
    assert s['targets']['jetson']['measured_late_us']['p50'] >= 0
    doc = FW.to_doc(res, proj)
    assert doc['schema'] == FW.SCHEMA and len(doc['lines']) == len(res.plan.lines)
    json.dumps(doc)
    paths = FW.export(res, proj, str(tmp_path))
    names = {os.path.relpath(p, tmp_path).replace(os.sep, '/') for p in paths}
    assert {'trigger_firmware.json', 'trigger_schedule.csv', 'firmware/stm32f411/main.c',
            'firmware/jetson/trigger_jetson.c'} <= names
    cancelled = FW.run(proj, settings=st, targets=['stm32f411'], cancel=lambda: True)
    assert cancelled is None and FW.exit_code(cancelled) == 3
    bad = FW.run(proj, settings=FW.FirmwareSettings(width_us=17000.0), simulate=False)
    assert FW.exit_code(bad) == 1


def test_issues_only_when_configured(proj):
    p = proj.snapshot()
    assert FW.issues(p) == []
    p.analysis.trigger_fw = {'width_us': 17000.0}
    iss = FW.issues(p)
    assert iss and 'does not fit' in iss[0].message


@needs_gcc
def test_validation_checks_pass():
    from vantage.validation.suite import Ctx, ValidationSettings
    ctx = Ctx(None, None, None, ValidationSettings())
    for spec in FW.validation_checks():
        out = spec['fn'](ctx)
        assert out.ok and out.error <= spec['tolerance'], (spec['id'], out)


# ---------------------------------------------------------------------- the page
from test_gui import app, win, pump, settle, _wait   # noqa: E402,F401  (fixtures)


def _open(win, name, cls):
    if name not in win._page_widgets:
        pg = cls(win.ctx)
        win.pages.addWidget(pg)
        win._page_widgets[name] = pg
    win.set_page(name)
    return win._page_widgets[name]


@needs_gcc
def test_page_generates_checks_edits_and_exports(app, win, tmp_path):
    from vantage.ui.workspaces.trigger_fw_ws import TriggerFirmwarePage
    pg = _open(win, 'trigger_fw', TriggerFirmwarePage)
    pump(app, 0.3)
    assert pg.lines_table.rowCount() == len(pg.plan.lines) >= 1
    assert pg.chart.lines
    # an undoable setting: the pulse width
    f = pg.fields['width_us']
    f.lineEdit().setText('250')
    f.editingFinished.emit()
    pump(app, 0.3)
    assert FW.settings_of(win.store.project).width_us == 250.0
    win.store.undo.undo()
    pump(app, 0.3)
    assert FW.settings_of(win.store.project).width_us == 100.0
    pg.groups.buttons['suggested'].click()
    pump(app, 0.3)
    assert FW.settings_of(win.store.project).groups == 'suggested'
    assert pg.plan.source in ('suggested', 'letters')        # 'letters' when the active variant has no lidar
    # a short host check, through the window's job runner
    def fn(p):
        p.analysis.trigger_fw = dict(p.analysis.trigger_fw, sim_seconds=12, sim_drop_s=2, jetson_seconds=1)
    win.store.edit('analysis', fn, 'short simulation')
    pump(app, 0.3)
    pg.b_run.click()
    t0 = __import__('time').time()
    while pg.result is None and __import__('time').time() - t0 < 300:
        pump(app, 0.2)
    assert pg.result is not None and pg.res_table.rowCount() == 4
    assert pg.result.metrics['proved'], pg.result.metrics
    assert pg.file_list.count() > 5 and 'Trigger box' in pg.file_view.toPlainText()
    paths = pg.export(str(tmp_path))
    assert any(p.endswith('trigger_firmware.json') for p in paths)
    assert os.path.isfile(os.path.join(tmp_path, 'firmware', 'rp2040', 'main.c'))


@pytest.mark.skipif(shutil.which('rustfmt') is None, reason='needs rustfmt')
def test_rust_firmware_parses(plan, tmp_path):
    """The RP2040 Rust firmware cannot be built here (no thumbv6m target, no crates): rustfmt at least parses it."""
    tree = FW.generate_tree(plan, 'rust_rp2040', FW.FirmwareSettings(groups='suggested'))
    for name in ('firmware/src/main.rs', 'firmware/build.rs', 'plan.rs', 'core/src/lib.rs'):
        p = tmp_path / name.replace('/', '_')
        p.write_bytes(tree[name])
        r = subprocess.run(['rustfmt', '--edition', '2021', '--emit', 'stdout', str(p)], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0 and 'error' not in r.stderr, (name, r.stderr[-1500:])
