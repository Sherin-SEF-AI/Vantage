"""54 Scenarios (episode gt040, chapter 4): three of the built-in scenarios, each a 12 s drive with ground truth (in
$VANTAGE_GT_DRIVES, read only), at real speed and held after their events: the front camera with every object boxed,
the roof lidar from above, and the scenario's events from its own run.json on a timeline.

Figures narrated, each read from the drive's own files while recording and checked with r.expect: the walker enters
the ego's lane at 2.05 s, the ego stops at 3.86 s 5.2 m from it (run.json scenario.events: enters_ego_lane t_s 2.05,
ego_stopped t_s 3.8625 distance_m 5.2325), the walker's 194 views all labelled pedestrian (index.json: its object in
F, ML, MR, R and QL); the cyclist in the ego's lane from 1.26 to 2.14 s (enters 1.2625, leaves 2.1375), 206 views all
labelled bicycle; the cut-in sent across at 2 s, in the ego's lane at 3.03 s with an 8.9 m gap (lane_change_ordered
2.0; cut_in 3.025, gap_m 8.8599); seed 1 (run.json scene.seed). Times are rounded half up from run.json's values, as
the simcore report states them.
"""

META = {
    'id': 'gt-scenarios',
    'part': 2,
    'title': 'Scenarios',
    'promise': 'The same events every time, for the same seed.',
    'menu': 'Simulate ▸ CARLA ▸ Scenario  ·  --carla-scenario',
}

LINES = [
    # 0
    ("Scenarios replay the same way for the same seed. A pedestrian crossing: the walker enters the ego's lane at "
     "2.05 s.",
     "Scenarios replay the same way, for the same seed. A pedestrian crossing: the walker enters the ego's lane at "
     "two point oh five seconds."),
    # 1
    ("The ego stops at 3.86 s, 5.2 m short, and all 194 views of the walker are labelled pedestrian.",
     "The ego stops at three point eight six seconds, five point two metres short, and all hundred and ninety-four "
     "views of the walker are labelled pedestrian."),
    # 2
    ("A cyclist: in the ego's lane from 1.26 to 2.14 s, labelled bicycle in all 206 views.",
     "A cyclist: in the ego's lane from one point two six to two point one four seconds, labelled bicycle in all two "
     "hundred and six views."),
    # 3
    ("A cut-in: sent across at 2 s, the car is in the ego's lane at 3.03 s, 8.9 m ahead. All three ran with seed 1.",
     "A cut-in: sent across at two seconds, the car is in the ego's lane at three point oh three, eight point nine "
     "metres ahead. All three ran with seed one."),
]


# (drive, label, actor role, play from, hold at, timeline span); each scenario's actor is object 1 of its drive
DRIVES = [('scenario_pedestrian_crossing', 'Pedestrian crossing', 'pedestrian', 0.5, 3.9, (0.0, 6.0)),
          ('scenario_cyclist_crossing', 'Cyclist crossing', 'bicycle', 0.6, 2.6, (0.0, 6.0)),
          ('scenario_cut_in', 'Cut-in', 'car', 1.6, 4.9, (0.0, 6.0))]
TEXT = {'start': ('start', '{who} starts, {distance_m:.1f} m ahead'),
        'enters_ego_lane': ('in lane', 'enters the ego\'s lane, {distance_m:.1f} m ahead'),
        'ego_stopped': ('ego stops', 'the ego stops, {distance_m:.1f} m short'),
        'leaves_ego_lane': ('out of lane', 'leaves the ego\'s lane'),
        'done': ('across', 'across the road'),
        'lane_change_ordered': ('sent across', 'sent across, {longitudinal_m:.1f} m ahead in the next lane'),
        'cut_in': ('in lane', 'in the ego\'s lane, {gap_m:.1f} m gap'),
        'cut_in_complete': ('settled', 'settled, {lateral_m:.2f} m from the lane centre')}


def events(run, who):
    out = []
    for e in run['scenario']['events']:
        short, text = TEXT[e['what']]
        out.append({'t': e['t_s'], 'ts': fmt_t(e['t_s']), 'short': short, 'text': text.format(who=who, **e),
                    'what': e['what'], 'e': e})
    return out


def fmt_t(x):
    import gtfootage as G
    return G.fmt(x, 2)


def views(drive, oid):
    n, cams, classes = 0, set(), set()
    for tag, cam in drive.doc['cameras'].items():
        for f in cam['frames']:
            for o in f['objects']:
                if o['id'] == oid:
                    n += 1
                    cams.add(tag)
                    classes.add(o['class'])
    return n, cams, classes


def scene(stage, r):
    import gtfootage as G

    v = G.start(stage, r)
    lays = []
    for name, label, role, t_a, t_b, span in DRIVES:
        d = G.Drive(name)
        sc = d.run['scenario']
        r.expect((d.run['scene']['seed'], sc['actors'][0]['gt_id']), (1, 1), f'{name}: seed 1, its actor object 1')
        who = {'pedestrian': 'the walker', 'bicycle': 'the cyclist', 'car': 'the car'}[role]
        ev = events(d.run, who)
        lay = G.Scenario(d, label, ev, span, hi=1)
        lays.append((d, lay, ev, t_a, t_b, role))
    (dp, lp, ep, pa, pb, _), (dc, lc, ec, ca, cb, _), (dk, lk, ek, ka, kb, _) = lays
    E = {e['what']: e for e in ep}
    r.expect((E['enters_ego_lane']['ts'], E['ego_stopped']['ts'], f"{E['ego_stopped']['e']['distance_m']:.1f}"),
             ('2.05', '3.86', '5.2'), 'pedestrian crossing events')
    n_p, cams_p, cl_p = views(dp, 1)
    print('walker', n_p, sorted(cams_p), cl_p, flush=True)
    r.expect((n_p, len(cams_p), cl_p), (194, 5, {'pedestrian'}), 'the walker\'s views')
    C = {e['what']: e for e in ec}
    r.expect((C['enters_ego_lane']['ts'], C['leaves_ego_lane']['ts']), ('1.26', '2.14'), 'cyclist in the lane')
    n_c, cams_c, cl_c = views(dc, 1)
    print('cyclist', n_c, sorted(cams_c), cl_c, flush=True)
    r.expect((n_c, cl_c), (206, {'bicycle'}), 'the cyclist\'s views')
    K = {e['what']: e for e in ek}
    r.expect((fmt_t(K['lane_change_ordered']['t']), K['cut_in']['ts'], f"{K['cut_in']['e']['gap_m']:.1f}"),
             ('2.00', '3.03', '8.9'), 'cut-in events')

    # ---- 0, 1: the pedestrian crossing
    v.use(lp)
    v.clock.pause(pa)
    r.lower_third()
    r.hold(0.3)
    r.say(0)
    r.until_word(0, 'A pedestrian crossing')
    G.play_until(r, v, pb)
    r.callout(lambda: lp.rect('events'), '', dur=max(1.5, r.speech_left() + 0.3), pad=8, dim=0.25)
    r.finish_speech(0.1)
    r.say(1)
    r.until_word(1, 'and all')
    src = lp.obj_rect('camera', 1)
    r.figure('Views of the walker', n_p, '', 0, dur=max(2.4, r.speech_left() + 0.4), src=src,
             note=f'all labelled pedestrian · {len(cams_p)} cameras')
    r.finish_speech(0.2)
    dp.release()

    # ---- 2: the cyclist crossing
    v.use(lc)
    v.clock.pause(ca)
    r.crossfade(0.5)
    r.say(2)
    r.until_word(2, 'in the ego')
    G.play_until(r, v, cb)
    src = lc.obj_rect('camera', 1)
    r.figure('Views of the cyclist', n_c, '', 0, dur=max(2.0, r.speech_left() + 0.4), src=src,
             note='all labelled bicycle')
    r.finish_speech(0.2)
    dc.release()

    # ---- 3: the cut-in
    v.use(lk)
    v.clock.pause(ka)
    r.crossfade(0.5)
    r.say(3)
    r.until_word(3, 'sent across')
    G.play_until(r, v, kb)
    r.callout(lambda: lk.rect('events'), '', dur=max(1.5, r.speech_left() + 0.3), pad=8, dim=0.25)
    r.finish_speech(1.0)
    dk.release()
