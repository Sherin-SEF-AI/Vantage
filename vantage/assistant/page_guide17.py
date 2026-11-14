"""The page guide's entries for round 17's pages (ui/pages17.py), kept beside the table that adds them: page_guide
appends them to its list, so the assistant reads them like every other page."""
from __future__ import annotations


def guides(G, Need, Next, NEED_CAMERAS, NEED_LIDAR):
    from .page_guide import _has
    NEED_RADAR = Need('radar', 'a radar on the variant', _has('radar'), 'Vehicle ▸ Add radar', {'page': 'viewports'})
    return [
        G('thermal', 'Thermal cameras at night',
          'Thermal (LWIR) cameras against visible cameras at night: detection, recognition and identification range '
          'per class and condition, the hot-road contrast inversion, and glass blocking LWIR.',
          needs=(NEED_CAMERAS,), result='result', module='thermal_ir', settings='thermal_ir', stale='stale',
          run='b_run', run_text='Run', auto=True, export='b_export', before=('viewports', 'perception'),
          after=(Next('perception', 'The perception range uses the thermal camera at night too.'),),
          ask='Which camera sees a pedestrian first at night, and what limits the visible cameras?'),
        G('event_camera', 'Event cameras',
          'Each event camera\'s event rate and data rate over a drive and whether its link keeps up, how sharply it '
          'places a moving edge where the RGB camera smears it, and a tunnel exit where the RGB exposure catches up.',
          needs=(NEED_CAMERAS,), result='res', module='event_camera', settings='event_camera', stale='stale',
          run='b_run', run_text='Run', auto=True, export='b_export', before=('isp', 'timing'),
          after=(Next('recording', 'The event cameras\' data in the recording budget.'),),
          ask='Does each event camera\'s link keep up at speed, and does it see the stopped car at the tunnel exit '
              'before the RGB camera?'),
        G('fmcw', 'FMCW lidar',
          'What a coherent FMCW lidar adds: the radial velocity of every return, the car\'s own velocity and the '
          'moving points from one sweep, and immunity to sunlight and other lidars, against a time-of-flight lidar.',
          needs=(NEED_LIDAR,), result='result', module='fmcw_lidar', settings='fmcw_lidar', stale='stale',
          run='b_run', run_text='Run', auto=True, export='b_export', before=('coverage',),
          after=(Next('coverage', 'Where the lidars\' beams land around the car.'),),
          ask='How much range does each lidar keep in full sun, and which moving objects does Doppler miss?'),
        G('imaging_radar', 'Imaging radar',
          'What each radar\'s MIMO array resolves and hides, how far each class is detected over the road, and the '
          'ghosts the bumper cavity and the road make, beside the zone model.',
          needs=(NEED_RADAR,), result='result', module='imaging_radar', settings='imaging_radar', stale='stale',
          run='b_run', run_text='Run', auto=True, export='b_export', before=('coverage', 'materials'),
          after=(Next('materials', 'The fascia\'s loss in front of each radar.'),),
          ask='Which radar ghosts targets through its bumper, and what mounting change would stop it?'),
        G('sotif', 'SOTIF matrix',
          'ISO 21448 triggering conditions (weather, sun, dirt, blind volume, housings, tunnel exit) against the '
          'rig\'s functional insufficiencies, measured from the app\'s studies, and the highest speed the ODD '
          'supports.',
          needs=(NEED_CAMERAS,), result='result', module='sotif', settings='sotif', stale='stale', run='b_run',
          run_text='Run', auto=True, export='b_export', before=('perception', 'exposure', 'volume'),
          after=(Next('requirements', 'Add the proposed range requirements to the scorecard.', 'b_props', here=True),
                 Next('cleaning', 'Lens dirt: a washer layout mitigates it.')),
          ask='What binds the ODD speed limit, and what would raise it?'),
        G('cleaning', 'Cleaning and de-icing',
          'Washer and air nozzles and lens heaters on the cameras and lidars: what each covers, how often it cleans, '
          'how long the tank lasts, whether heaters clear ice at speed, and the power and harness they add.',
          result='result', module='cleaning', settings='cleaning', stale='stale', auto=True, export='b_export',
          before=('exposure', 'power'), after=(Next('power', 'The power budget with the cleaning\'s loads.'),
                                               Next('sotif', 'The lens-dirt condition counts the washers.'))),
        G('framing', 'Frame',
          'Turn the roof rig into a buildable T-slot aluminium frame from cited catalogue profiles: members, plates '
          'under every sensor, roof-rail feet, a cut list, stock plan, BOM, CNC plate DXFs, STL parts and a STEP '
          'assembly.',
          needs=(Need('roof_rig', 'rig parts on the roof (a roof rack or a mast) or sensors above the roof',
                      lambda p, w: bool(p.rig) or any(s.visible for s in p.variant.sensors) or None,
                      'Vehicle ▸ Add rig part ▸ Roof rack or Sensor mast', {'page': 'viewports'}),),
          result='result', module='framing', settings='framing', auto=True, export='b_export',
          before=('brackets',), after=(Next('fea', 'The frame\'s modes and the road jitter each sensor gets.'),
                                       Next('aero', 'The frame\'s drag and the EV range it costs.'))),
        G('fea', 'Frame modes',
          'A quadratic-tetrahedron modal analysis of the frame (sensors as rigid masses, feet fixed or on roof '
          'springs): its lowest modes, and the road (ISO 8608) through them to each sensor\'s angular jitter beside '
          'the one-mode beam estimate; a 130 km/h crosswind case.',
          result='result', module='fea', settings='fea', stale='stale', run='b_run', run_text='Run',
          export='b_export', before=('framing',),
          after=(Next('aero', 'Vortex lock-in against these modes.'),
                 Next('thermo_vib', 'Thermal warping and vibration of the rig.')),
          ask='Which sensor shakes most on the road, and what would stiffen its mount?'),
        G('aero', 'Aerodynamics',
          'The rig\'s drag from cited bluff-body coefficients (not CFD), the EV range it costs at 100 and 130 km/h, '
          'vortex-shedding lock-in speeds against the frame\'s modes, and the crosswind load on each sensor.',
          result='result', module='aero', settings='aero', auto=True, export='b_export', before=('framing', 'fea'),
          after=(Next('power', 'The range lost beside the rig\'s electrical load.'),),
          ask='How much range does the rig cost at highway speed, and which part drags most?'),
        G('trigger_fw', 'Trigger firmware',
          'Generate trigger-box firmware from the sync plan for RP2040 (C, PIO), STM32F411 (C), RP2040 (Rust) and '
          'Jetson (libgpiod), and prove each project\'s scheduling core on this machine against a crystal and a PPS.',
          needs=(NEED_CAMERAS,), result='result', module='trigger_firmware', settings='trigger_firmware',
          run='b_run', run_text='Generate and check', export='b_export', before=('timing', 'clock_tree'),
          after=(Next('edge', 'The in-car watchdog that checks the triggered streams.'),),
          ask='How close to the plan does each board fire, and what happens when the PPS drops out?'),
        G('edge', 'Edge watchdog',
          'The in-car service that watches every camera stream (rate, drops, stalls, frozen and dark images, sync '
          'skew) and knocked cameras, and alerts locally; its configuration and systemd unit, and dry runs on a '
          'recording with faults written in.',
          needs=(NEED_CAMERAS,), result='result', module='edge_daemon', settings='edge_daemon', run='b_run',
          run_text='Dry run', export='b_export', before=('rig', 'drivelog'),
          after=(Next('fleet_twin', 'Knocks and drift across the fleet over time.'),),
          ask='Which faults did the dry run catch, how fast, and were there false alarms?'),
        G('thermal_intrinsics', 'Thermal intrinsics',
          'How each camera\'s focal length, principal point and distortion move with temperature, fitted from '
          'calibrations at several temperatures, with intervals; a capture compensated for its temperature.',
          needs=(NEED_CAMERAS,), result='result', module='thermal_intrinsics', settings='thermal_intrinsics',
          export='b_export', before=('thermo_vib',), after=(Next('thermo_vib', 'Thermal warping of the rig.'),),
          ask='How many pixels does each camera\'s focal length move over the operating range?'),
        G('lab', 'Lab',
          'Publish the project to the team\'s lab server as immutable versions, sync and merge with colleagues, '
          'compare versions, and pool fleets and perception records across teams.',
          needs=(Need('lab_server', 'a lab server address and a token from its admin', lambda p, w: None,
                      'run one with vantage --lab-server ROOT; its admin gives each team a token'),),
          before=('fleet_twin',),
          ask='What changed between the last two versions, and who published them?'),
    ]
