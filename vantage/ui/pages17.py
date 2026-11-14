"""Round 17's pages (new sensor modalities, neural replay, the user's body in simulation, mechanical design, embedded
hardware, safety cases, the lab), registered from one table: the main window adds each page, its action (with
tooltip and icon) and its place at the end of its menu, so a page is wired by one entry here.

    key     page id (PAGE_CLASSES, set_page, the page guide)
    module  ui/workspaces module, cls its page class
    label   the page's name (the assistant's page table and its guide)
    core    the core module whose summary(page.<result>) the assistant reads (none: the page keeps no result);
            result: the page's attribute holding it (default 'result')
    menu    top-level menu title; text, icon (ui/icon_data), tip; docks: the docks the page shows (default none)"""
PAGES = [
    dict(key='thermal', label='Thermal cameras at night', core='thermal_ir', module='thermal_ws', cls='ThermalPage',
         menu='Analysis', text='Thermal cameras at night…',
         icon='sun', tip='How far thermal (LWIR) cameras see pedestrians, cyclists and cars at night, in fog, rain and '
                         'on a hot road, next to the visible cameras in UN R112 headlamp light'),
    dict(key='event_camera', label='Event cameras', core='event_camera', result='res', module='event_ws',
         cls='EventPage', menu='Analysis', text='Event cameras…', icon='timeline',
         tip='Neuromorphic event cameras on the rig: their event rate and bandwidth over a drive, motion blur against '
             'the RGB cameras\' smear, and a tunnel exit with a stopped car ahead; CARLA\'s DVS camera against the '
             'model'),
    dict(key='fmcw', label='FMCW lidar', core='fmcw_lidar', module='fmcw_ws', cls='FmcwPage',
         menu='Analysis', text='FMCW lidar…', icon='sensor',
         tip='Radial velocity per lidar point: ego motion and moving points from one sweep; the sun and other '
             'lidars against a time-of-flight lidar'),
    dict(key='imaging_radar', label='Imaging radar', core='imaging_radar', module='imaging_radar_ws',
         cls='ImagingRadarPage', menu='Analysis', text='Imaging radar…', icon='sensor',
         tip='Every radar as a 4D imaging radar: its virtual array\'s resolution and sidelobes, detection range per '
             'class over the road, and the ghosts the bumper and the road make'),
    dict(key='sotif', label='SOTIF matrix', core='sotif', module='sotif_ws', cls='SotifPage',
         menu='Analysis', text='SOTIF matrix…', icon='checklist',
         tip='ISO 21448 triggering conditions against the rig\'s functional insufficiencies, and the ODD\'s speed '
             'bound (supports a safety case; not a certification)'),
    dict(key='cleaning', label='Cleaning and de-icing', core='cleaning', module='cleaning_ws', cls='CleaningPage',
         menu='Analysis', text='Cleaning and de-icing…',
         icon='weather', tip='Washer and air nozzles and lens heaters: coverage, clean frequency, tank over a shift, '
                             'ice at speed, power and harness'),
    dict(key='framing', label='Frame', core='framing', module='framing_ws', cls='FramingPage',
         menu='Analysis', text='Frame…', icon='frame',
         tip='T-slot aluminium frame for the roof rig: cut list, BOM, CNC plates, STL, STEP'),
    dict(key='fea', label='Frame modes', core='fea', module='fea_ws', cls='FEAPage',
         menu='Analysis', text='Frame modes…', icon='mesh',
         tip='Tetrahedral modal analysis of the frame: modes, road jitter per sensor, wind case'),
    dict(key='aero', label='Aerodynamics', core='aero', module='aero_ws', cls='AeroPage',
         menu='Analysis', text='Aerodynamics…', icon='weather',
         tip='Rig drag and EV range loss, vortex lock-in against the frame\'s modes, wind on each sensor (not CFD)'),
    dict(key='trigger_fw', label='Trigger firmware', core='trigger_firmware', module='trigger_fw_ws',
         cls='TriggerFirmwarePage', menu='Embedded', text='Trigger firmware…', icon='sync',
         tip='Firmware for a trigger box that fires every camera line at the sync plan\'s times, locked to the GNSS '
             'PPS (RP2040, STM32, Rust, Jetson), proved here by simulation'),
    dict(key='edge', label='Edge watchdog', core='edge_daemon', module='edge_ws', cls='EdgePage',
         menu='Embedded', text='Edge watchdog…', icon='eye',
         tip='The in-car service that watches the camera streams and alerts on faults and knocks'),
    dict(key='thermal_intrinsics', label='Thermal intrinsics', core='thermal_intrinsics',
         module='thermal_intrinsics_ws', cls='ThermalIntrinsicsPage', menu='Calibrate', text='Thermal intrinsics…',
         icon='lens', tip='Focal length, principal point and distortion against temperature, from calibrations at '
                          'several temperatures; compensate a capture for its temperature'),
    dict(key='lab', label='Lab', module='lab_ws', cls='LabPage',
         menu='File', text='Lab…', icon='history',
         tip='Lab server: publish versions, sync with your team, pooled fleets and records'),
]


def by_menu(menu: str) -> list:
    return [p for p in PAGES if p['menu'] == menu]
