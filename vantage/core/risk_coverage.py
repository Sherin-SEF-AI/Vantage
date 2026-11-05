"""Risk-weighted coverage: the rig's coverage weighted by where hazards really appear around a car in the seconds before
a crash, from cited crash statistics and the Euro NCAP test scenarios: "this rig covers 96 % of the area but 81 % of the
pre-crash pedestrian exposure; the gap is front-right 10–20 m (turning-right-at-junction scenarios)".

Hazard data (tables below, every number from the report or protocol itself, with table and page)
-----------------------------------------------------------------------------------------------
* NHTSA's pre-crash scenario typology: Najm, Smith and Yanagisawa (2007), "Pre-Crash Scenario Typology for Crash
  Avoidance Research", DOT HS 810 767 (2004 GES: 37 scenarios, 5,942,000 police-reported light-vehicle crashes,
  economic cost and functional years lost; Tables 13–15, PDF p.37–39), and its update Swanson, Foderaro, Yanagisawa,
  Najm and Azeredo (2019), "Statistics of Light-Vehicle Pre-Crash Scenarios Based on 2011–2015 National Crash Data",
  DOT HS 812 745 (Table 12, PDF p.83: crashes, fatal crashes, comprehensive cost, equivalent lives).
* Pedestrian crash geometry: Swanson, Yanagisawa, Najm, Foderaro and Azeredo (2016), "Crash Avoidance Needs and
  Countermeasure Profiles for Safety Applications Based on Light-Vehicle-to-Pedestrian Communications", DOT HS 812 312
  (Table 4, PDF p.27: vehicle manoeuvre × pedestrian action, 2011–2012 GES and FARS; costs Tables 5–6).
* Euro NCAP's test scenarios, the current (2026–2028) Crash Avoidance protocols: Frontal Collisions v1.2 (July 2026),
  Low Speed Collisions v1.2 (July 2026), Lane Departure Collisions v1.2 (July 2026): speeds, paths, impact locations,
  start distances and the turn paths (Frontal §1.2.4: clothoid–arc–clothoid, R 9.00 / 11.75 / 14.75 m to the far side,
  8.00 m to the near side). Every scenario row cites its section and page.

Geometry: where the conflicting road user IS before the conflict
---------------------------------------------------------------
Each scenario is a kinematic template (the Euro NCAP scenario itself, or the one an NHTSA scenario maps to, said per
row): the ego and the target move as the protocol drives them (constant speeds; the lead car of CCRb braking at
4 m/s² from a 1.0 s headway; a pedestrian or cyclist standing at its start point until it starts; the ego on the
protocol's turn path; the ego drifting sideways at the protocol's lateral speeds; starting from a stop 2.9 m from
the impact point), and meet at t = 0 with the target's reference point at the protocol's impact location (% of the
ego's width from its right edge; % of its length on a side). For every test point (the protocol's speed grid and
impact locations, each equally) and every time before the conflict τ in [ttc_min, ttc_max] (0.5–4 s, sampled evenly),
the target's centre is placed in the ego's vehicle frame (ISO 8855, rear axle) at −τ. These points, weighted, are the
hazard density: per class (pedestrian, cyclist, vehicle, motorcyclist), on core.regions' grid and as a fine ground
map.

Weights (selectable, every one traced to its table row)
* frequency: the NHTSA crashes of the row (vehicle and cyclist rows from Swanson 2019 or Najm 2007; pedestrians from
  the 2016 pedestrian table), split over the templates the row maps to as stated per row;
* cost: the row's cost (Najm: economic, year-2000 unit costs; Swanson and the pedestrian table: comprehensive, 2010
  dollars); fatal: fatal crashes (Swanson 2019, the pedestrian table); fyl: functional years lost (Najm 2007);
* ncap: every Euro NCAP scenario equally (each test point of it equally), a test-design weighting, not exposure.
Motorcyclists: the NHTSA light-vehicle typology does not separate them, so their density is the NCAP one under every
weighting (said so).

Coverage, weighted
-----------------
At every hazard point the app's perception-range test (core.perception.object_pixels: in the image and the lens
circle, the centre not hidden by the body, at least the class's minimum pixel height, through each camera's lens
model): 'resolved' by any camera (default), 'seen' by any camera, or 'redundant' (resolved by two or more). Weighted
coverage = Σ w · covered / Σ w per class; the unweighted figure is the share of the ground (cells of cell_m out to
area_m, off the body) resolved for the same class. Per region of the grid: the hazard share, its covered share, and
the uncovered hazard with the scenarios that put it there.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .regions import RegionGrid, default_grid

SCHEMA = 'vantage.risk_coverage/1'
FRAME = 'vehicle (ISO 8855, rear axle)'
CLASSES = ('pedestrian', 'cyclist', 'vehicle', 'motorcyclist')
CLASS_LABEL = {'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'vehicle': 'Vehicle', 'motorcyclist': 'Motorcyclist'}
WEIGHTS = ('frequency', 'cost', 'fatal', 'fyl', 'ncap')
WEIGHT_LABEL = {'frequency': 'Crash frequency (NHTSA)', 'cost': 'Crash cost (NHTSA)', 'fatal': 'Fatal crashes (NHTSA)',
                'fyl': 'Functional years lost (NHTSA 2007)', 'ncap': 'Euro NCAP scenarios, equally'}
DATASETS = ('swanson2019', 'najm2007')
METRICS = ('resolved', 'seen', 'redundant')
METRIC_LABEL = {'resolved': 'Resolved by a camera (perception range)', 'seen': 'In a camera\'s view',
                'redundant': 'Resolved by two or more cameras'}
# typical target sizes (length, width) in metres for placing a target's centre from its reference point; the
# coverage test uses the perception study's class heights and widths
TARGET_SIZE = {'pedestrian': (0.5, 0.5), 'cyclist': (1.8, 0.6), 'vehicle': (4.5, 1.8), 'motorcyclist': (2.2, 0.8)}
MOTORCYCLIST_CLASS = {'name': 'Motorcyclist', 'height': 1.5, 'width': 0.8, 'base': 0.0}   # typical; min_px: Cyclist's
LANE_OFFSET_M = 1.75          # Frontal v1.2 §1.2.4 / §3.1.2.1: the ego's and the oncoming lane 1.75 m from the centre line
SFS_ACCEL = 1.0               # m/s², ego starting from a stop (the pedal profile is CA 102's, not reproduced): assumed
SFS_DIST_M = 2.9              # Low Speed v1.2 §3.1.1: the ego's front 2.9 m from the impact point at the start
DOOR_AT = 0.55                # rear edge of the front door at this share of the length from the rear (typical)
ASSUMPTIONS = (
    'Constant speeds before the conflict (no braking, no evasive steering), except where the protocol prescribes a '
    'profile (CCRb\'s braking lead, a start from a stop at 1 m/s² assumed, the VRU standing until it starts). '
    'NHTSA rows give no geometry: each maps to the Euro NCAP template of the same configuration (stated per row) '
    'with that template\'s speeds. A turn-into-path or lateral-direction junction crash is placed as a crossing from '
    'the side it comes from at turning speeds (10–25 km/h). Cyclist rows (no breakdown in the reports) are split '
    'equally over the crossing and longitudinal templates (no manoeuvre) or the turning ones (manoeuvre). Target sizes '
    'are typical (pedestrian 0.5 m, cyclist 1.8 m, car 4.5 m, motorcycle 2.2 m long). The US data (NHTSA) weights US '
    'crashes; Euro NCAP\'s scenarios are test designs weighted equally.')
LIMITS_TEXT = (
    'Rows without a road user of these classes (vehicle failure, control loss, road-edge departure, animal, object, '
    'non-collision, evasive action, hit-and-run, other) are left out and their share is reported. Coverage is the '
    'cameras\' (lidar and radar are not counted), in clear weather, by the perception study\'s pixel thresholds; the '
    'hazard points are the targets\' centres (their extent is not spread over the ground).')

LIMITS = {'ttc_min': (0.0, 10.0), 'ttc_max': (0.1, 20.0), 'ttc_samples': (2, 200), 'cell_m': (0.1, 5.0),
          'area_m': (5.0, 300.0), 'map_m': (10.0, 300.0)}
INTS = ('ttc_samples',)


# =========================================================================== the published tables
# (generated from the reports by script; checked against their printed totals in tests/test_risk_coverage.py)
# Najm, Smith and Yanagisawa (2007), DOT HS 810 767: 2004 GES. Columns: scenario number (Tables 17/18, PDF p.80-82),
# name, crashes (Table 13, PDF p.37, unrounded), economic cost in $ million (Table 14, PDF p.38, year-2000 unit costs),
# functional years lost (Table 15, PDF p.39).
NAJM2007 = (
    (1, 'Vehicle Failure', 42147, 1051.0, 26000),
    (2, 'Control Loss With Prior Vehicle Action', 102617, 1970.0, 49000),
    (3, 'Control Loss Without Prior Vehicle Action', 528930, 15796.0, 478000),
    (4, 'Running Red Light', 253618, 6627.0, 135000),
    (5, 'Running Stop Sign', 48296, 1310.0, 28000),
    (6, 'Road Edge Departure With Prior Vehicle Maneuver', 67528, 1144.0, 34000),
    (7, 'Road Edge Departure Without Prior Vehicle Maneuver', 333706, 9005.0, 270000),
    (8, 'Road Edge Departure While Backing Up', 65809, 350.0, 6000),
    (9, 'Animal Crash With Prior Vehicle Maneuver', 23322, 120.0, 2000),
    (10, 'Animal Crash Without Prior Vehicle Maneuver', 305102, 1632.0, 24000),
    (11, 'Pedestrian Crash With Prior Vehicle Maneuver', 17118, 843.0, 24000),
    (12, 'Pedestrian Crash Without Prior Vehicle Maneuver', 39324, 4022.0, 144000),
    (13, 'Pedalcyclist Crash With Prior Vehicle Maneuver', 18325, 523.0, 11000),
    (14, 'Pedalcyclist Crash Without Prior Vehicle Maneuver', 24071, 1301.0, 39000),
    (15, 'Backing Up Into Another Vehicle', 130701, 947.0, 9000),
    (16, 'Vehicle(s) Turning – Same Direction', 221791, 2810.0, 47000),
    (17, 'Vehicle(s) Parking – Same Direction', 48138, 623.0, 11000),
    (18, 'Vehicle(s) Changing Lanes – Same Direction', 338309, 4247.0, 71000),
    (19, 'Vehicle(s) Drifting – Same Direction', 97973, 1383.0, 37000),
    (20, 'Vehicle(s) Making a Maneuver – Opposite Direction', 15472, 943.0, 32000),
    (21, 'Vehicle(s) Not Making a Maneuver – Opposite Direction', 123699, 6407.0, 206000),
    (22, 'Following Vehicle Making a Maneuver', 85373, 1212.0, 18000),
    (23, 'Lead Vehicle Accelerating', 18722, 273.0, 4000),
    (24, 'Lead Vehicle Moving at Lower Constant Speed', 209610, 3910.0, 78000),
    (25, 'Lead Vehicle Decelerating', 428067, 6390.0, 100000),
    (26, 'Lead Vehicle Stopped', 974855, 15388.0, 240000),
    (27, 'LTAP/OD at Signalized Junctions', 220206, 5749.0, 121000),
    (28, 'Vehicle Turning Right at Signalized Junctions', 34951, 355.0, 4000),
    (29, 'LTAP/OD at Non-Signalized Junctions', 189816, 5137.0, 113000),
    (30, 'Straight Crossing Paths at Non-Signalized Junctions', 263840, 7290.0, 174000),
    (31, 'Vehicle(s) Turning at Non-Signalized Junctions', 434892, 7343.0, 138000),
    (32, 'Evasive Action With Prior Vehicle Maneuver', 13120, 198.0, 4000),
    (33, 'Evasive Action Without Prior Vehicle Maneuver', 56199, 1349.0, 36000),
    (34, 'Non-Collision Incident', 45910, 592.0, 13000),
    (35, 'Object Crash With Prior Vehicle Maneuver', 30301, 155.0, 3000),
    (36, 'Object Crash Without Prior Vehicle Maneuver', 54526, 687.0, 19000),
    (37, 'Other', 35859, 764.0, 21000),
)
NAJM2007_TOTALS = {'crashes': 5942000, 'cost_musd': 120000, 'fyl': 2767000}  # stated (abstract p.3; Sec 4.4)
# Swanson, Foderaro, Yanagisawa, Najm and Azeredo (2019), DOT HS 812 745: yearly average of 2011-2015 FARS and GES.
# Columns: Table 12 row (PDF p.83, B-3; None for the unnumbered rows), name, crashes (light vehicle in the critical
# event), fatal crashes, comprehensive cost in $ million (2010 dollars, Blincoe et al. 2015; crashes where the light
# vehicle made the critical action), equivalent lives.
SWANSON2019 = (
    (1, 'Vehicle Failure', 39359, 216, 5029, 550),
    (2, 'Control Loss/Vehicle Action', 73952, 405, 7758, 848),
    (3, 'Control Loss/No Vehicle Action', 399439, 4124, 69749, 7626),
    (4, 'Road Edge Departure/Maneuver', 90382, 252, 6029, 659),
    (5, 'Road edge departure/No Maneuver', 472182, 6284, 91707, 10027),
    (6, 'Road Edge Departure/Backing', 70025, 43, 1827, 200),
    (7, 'Animal/Maneuver', 2833, 7, 171, 19),
    (8, 'Animal/No Maneuver', 295273, 96, 6060, 663),
    (9, 'Pedestrian/Maneuver', 28018, 323, 7417, 811),
    (10, 'Pedestrian/No Maneuver', 42507, 3409, 39926, 4365),
    (11, 'Pedal Cyclist/Maneuver', 23019, 62, 3923, 429),
    (12, 'Pedal Cyclist/No Maneuver', 24908, 456, 8223, 899),
    (13, 'Backing into Vehicle', 113685, 30, 2407, 263),
    (14, 'Turning/Same Direction', 194303, 222, 9181, 1004),
    (15, 'Parking/Same Direction', 34898, 97, 2409, 263),
    (16, 'Changing Lanes/Same Direction', 348464, 360, 14473, 1582),
    (17, 'Drifting/Same Direction', 120223, 196, 6872, 751),
    (18, 'Opposite Direction/Maneuver', 4897, 305, 3969, 434),
    (19, 'Opposite Direction/No Maneuver', 96095, 2983, 44285, 4842),
    (20, 'Rear-End/Striking Maneuver', 57224, 88, 3272, 358),
    (21, 'Rear-End/Lead Vehicle Accelerating (LVA)', 22008, 25, 1274, 139),
    (22, 'Rear-End/Lead Vehicle Moving (LVM)', 214001, 647, 17863, 1953),
    (23, 'Rear-End/LVD', 412536, 196, 23464, 2566),
    (24, 'Rear-End/LVS', 1050558, 667, 60641, 6630),
    (25, 'Right Turn Into Path (RTIP)', 91191, 59, 4308, 471),
    (26, 'Right Turn Across Path (RTAP)', 23451, 18, 1114, 122),
    (27, 'Straight Crossing Paths (SCP)', 434374, 2206, 63557, 6949),
    (28, 'Left Turn Across Path, Lat. Dir. (LTAP/LD)', 193102, 555, 19911, 2177),
    (29, 'Left Turn Into Path (LTIP)', 80585, 55, 3905, 427),
    (30, 'Left Turn Across Path, Opp. Dir. (LTAP/OD)', 329410, 1192, 42610, 4659),
    (31, 'Avoidance/Maneuver', 21152, 31, 983, 108),
    (32, 'Avoidance/No Maneuver', 79713, 133, 5236, 573),
    (None, 'Rollover', 1069, 24, 360, 39),
    (33, 'Noncollision - No Impact', 10496, 173, 2293, 251),
    (34, 'Object/Maneuver', 16417, 15, 515, 56),
    (35, 'Object/No Maneuver', 80088, 151, 3688, 403),
    (None, 'Hit and Run', 19540, 6, 532, 58),
    (None, 'Other - Rear-End', 102, 1, 12, 1),
    (None, 'Other - Sideswipe', 141, 0, 5, 1),
    (None, 'Other - Turn Across Path', 13127, 5, 341, 37),
    (None, 'Other - Turn Into Path', 5581, 7, 234, 26),
    (None, 'Other', 26953, 73, 1208, 132),
)
SWANSON2019_TOTALS = {'crashes': 5657279, 'fatal': 26197, 'cost_musd': 588743, 'equivalent_lives': 64372}  # Table 12 Total row
# Swanson, Yanagisawa, Najm, Foderaro and Azeredo (2016), DOT HS 812 312: yearly average of 2011-2012 GES and FARS,
# light vehicle strikes a pedestrian (first event). Columns: vehicle manoeuvre, pedestrian action, crashes (GES) and
# fatal crashes (FARS) (Table 4, PDF p.27), comprehensive cost in $ million (2010 dollars; Tables 5-6, PDF p.30-31;
# None where the report prints none).
V2P2016 = (
    ('Going Straight', 'Crossing Road', 23558, 2029, 45582.062),
    ('Turning Left', 'Crossing Road', 14427, 114, 6048.459),
    ('Turning Right', 'Crossing Road', 5123, 33, 2048.5),
    ('Going Straight', 'Adjacent to Road', 4243, 363, 8391.304),
    ('Going Straight', 'In-Road', 2326, 481, 9450.946),
    ('Going Straight', 'None/Other/Unknown', 2276, 106, 2800.222),
    ('Starting', 'Crossing Road', 1285, 9, 476.916),
    ('Backing Up', 'Crossing Road', 1090, 5, 369.745),
    ('Backing Up', 'None/Other/Unknown', 824, 9, 327.722),
    ('Backing Up', 'Adjacent to Road', 653, 12, 372.668),
    ('Starting', 'In-Road', 518, 2, 137.174),
    ('Changing Lanes', 'Crossing Road', 510, 31, 793.623),
    ('Going Straight', 'Non-Applicable', 495, 28, 661.284),
    ('Turning Left', 'Adjacent to Road', 492, 3, 201.622),
    ('Turning Right', 'Adjacent to Road', 478, 4, 227.854),
    ('Turning Left', 'In-Road', 464, 3, 147.901),
    ('Turning Left', 'None/Other/Unknown', 462, 4, 250.509),
    ('Non-Applicable', 'In-Road', 436, 19, 458.797),
    ('Parking', 'Crossing Road', 405, 2, 204.329),
    ('Non-Applicable', 'Crossing Road', 361, 5, None),
    ('Parking', 'In-Road', 283, 3, 125.703),
    ('Starting', 'Adjacent to Road', 257, 1, 60.897),
    ('Non-Applicable', 'None/Other/Unknown', 255, 11, 272.278),
    ('Backing Up', 'In-Road', 239, 12, 312.083),
    ('Parking', 'Adjacent to Road', 216, 2, 93.634),
    ('Parking', 'None/Other/Unknown', 199, 2, None),
    ('Non-Applicable', 'Adjacent to Road', 194, 2, None),
    ('Turning Right', 'None/Other/Unknown', 180, 1, None),
    ('Starting', 'None/Other/Unknown', 172, 3, None),
    ('Changing Lanes', 'In-Road', 134, 14, 283.31),
    ('Changing Lanes', 'Adjacent to Road', 127, 9, 185.405),
    ('Turning Right', 'Non-Applicable', 119, 0, None),
    ('Turning Left', 'Non-Applicable', 43, 1, None),
    ('Turning Right', 'In-Road', 38, 3, 66.813),
    ('Starting', 'Non-Applicable', 15, 0, None),
    ('Changing Lanes', 'None/Other/Unknown', 13, 3, None),
    ('Non-Applicable', 'Non-Applicable', 5, 0, None),
    ('Backing Up', 'Non-Applicable', 0, 1, None),
    ('Changing Lanes', 'Non-Applicable', 0, 2, None),
    ('Parking', 'Non-Applicable', 0, 0, None),
    ('Unknown', 'Adjacent to Road', 0, 0, None),
    ('Unknown', 'Crossing Road', 0, 9, None),
    ('Unknown', 'In-Road', 0, 1, None),
    ('Unknown', 'Non-Applicable', 0, 0, None),
    ('Unknown', 'None/Other/Unknown', 0, 4, None),
)
V2P2016_TOTALS = {'crashes': 62917, 'fatal': 3336.5, 'cost_musd': 81197.282}  # Table 4 and 5 totals


NHTSA_SOURCES = {
    'najm2007': {'citation': 'Najm, W. G., Smith, J. D., and Yanagisawa, M. (2007). Pre-Crash Scenario Typology for '
                             'Crash Avoidance Research. DOT HS 810 767. NHTSA / Volpe Center.',
                 'url': 'https://rosap.ntl.bts.gov/view/dot/6281/dot_6281_DS1.pdf',
                 'data': '2004 GES, police-reported crashes with at least one light vehicle',
                 'tables': 'Table 13 (frequency, PDF p.37), Table 14 (economic cost, p.38), Table 15 (functional years '
                           'lost, p.39); scenario numbers Tables 17-18 (p.80-82)',
                 'cost': 'economic cost, year-2000 unit costs (Blincoe et al. 2002)'},
    'swanson2019': {'citation': 'Swanson, E., Foderaro, F., Yanagisawa, M., Najm, W. G., and Azeredo, P. (2019). '
                                'Statistics of Light-Vehicle Pre-Crash Scenarios Based on 2011-2015 National Crash '
                                'Data. DOT HS 812 745. NHTSA.',
                    'url': 'https://rosap.ntl.bts.gov/view/dot/41932/dot_41932_DS1.pdf',
                    'data': 'yearly average of 2011-2015 FARS and GES, light vehicle in the critical event',
                    'tables': 'Table 12 (PDF p.83, B-3): crashes, fatal crashes, comprehensive cost, equivalent lives',
                    'cost': 'comprehensive cost, 2010 dollars (Blincoe et al. 2015, DOT HS 812 013); for crashes '
                            'where the light vehicle made the critical action'},
    'v2p2016': {'citation': 'Swanson, E., Yanagisawa, M., Najm, W. G., Foderaro, F., and Azeredo, P. (2016). Crash '
                            'Avoidance Needs and Countermeasure Profiles for Safety Applications Based on '
                            'Light-Vehicle-to-Pedestrian Communications. DOT HS 812 312. NHTSA.',
                'url': 'https://rosap.ntl.bts.gov/view/dot/12373/dot_12373_DS1.pdf',
                'data': 'yearly average of 2011-2012 GES and FARS, light vehicle strikes a pedestrian (first event)',
                'tables': 'Table 4 (PDF p.27): crashes and fatal crashes; Tables 5-6 (p.30-31): comprehensive cost',
                'cost': 'comprehensive cost, 2010 dollars'},
}
NCAP_SOURCES = {
    'FC': {'title': 'Euro NCAP Crash Avoidance – Frontal Collisions Test Protocol', 'version': '1.2',
           'date': 'July 2026 (implementation January 2026)',
           'url': 'https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Crash_Avoidance_Frontal_Collisions_v1_2_'
                  '220d87e28a.pdf'},
    'LSC': {'title': 'Euro NCAP Crash Avoidance – Low Speed Collisions Test Protocol', 'version': '1.2',
            'date': 'July 2026 (implementation January 2026)',
            'url': 'https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Crash_Avoidance_Low_Speed_Collisions_v1_2_'
                   '3d24fa41ea.pdf'},
    'LDC': {'title': 'Euro NCAP Crash Avoidance – Lane Departure Collisions Test Protocol', 'version': '1.2',
            'date': 'July 2026 (implementation January 2026)',
            'url': 'https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Crash_Avoidance_Lane_Departure_Collisions_'
                   'v1_2_4c2613beca.pdf'},
}
# the turn paths of Frontal v1.2 §1.2.4 (PDF p.15): ego speed -> (clothoid angle α°, arc radius m, arc angle β°)
TURN_FAR = {10: (20.62, 9.00, 48.75), 15: (20.93, 11.75, 48.14), 20: (21.79, 14.75, 46.42), 25: (21.79, 14.75, 46.42)}
TURN_NEAR = {10: (22.85, 8.00, 44.30)}


def _rng(a, b, step):
    return tuple(float(x) for x in np.arange(a, b + 1e-9, step))


# Euro NCAP scenarios (current 2026 protocols). kind: straight | turn | turn_target. face: where the target meets the
# ego (front, rear, left, right); pct: impact location, % of the width from the right edge (front, rear) or of the
# length from the rear (left, right); heading: the target's heading in the ego frame at the conflict (deg, 0 = same
# direction, 180 = oncoming, +90 = moving left, i.e. coming from the right); start_m: the target stands this far from
# the impact point until it starts (None: it moves throughout); for turns, line_y: the target's path, parallel to the
# main road, this far left of its centre line, and heading relative to the main road.
NCAP = (
    # ---- Frontal Collisions v1.2: car to car
    dict(id='CCRs', cls='vehicle', kind='straight', ego=_rng(10, 80, 10), tgt=(0.0,), heading=0, face='front',
         pct=(100, 75, 50, 25, 0), src='FC', sec='3.1.1.1 (CCRs table); 1.2.5.1', page='p.29 (PDF 31)',
         what='Car-to-car rear, stationary target'),
    dict(id='CCRm', cls='vehicle', kind='straight', pairs=((30, 20), (40, 20), (50, 20), (60, 20), (70, 20), (80, 20),
                                                             (90, 30), (100, 40), (110, 50), (120, 60), (130, 70)),
         heading=0, face='front', pct=(100, 75, 50, 25, 0), src='FC', sec='3.1.1.1 (CCRm table)', page='p.29 (PDF 31)',
         what='Car-to-car rear, moving target (20 km/h; ego − 60 km/h above 80 km/h)'),
    dict(id='CCRb', cls='vehicle', kind='straight', ego=_rng(30, 80, 10), brake=(1.0, 4.0), heading=0, face='front',
         pct=(100, 75, 50, 25, 0), src='FC', sec='3.1.1.1 (CCRb table and text)', page='p.29-30 (PDF 31-32)',
         what='Car-to-car rear, braking target: same speed, 1.0 s headway, −4 m/s² (standard range 30–80 km/h)'),
    dict(id='CCFhos', cls='vehicle', kind='straight', pairs=((30, 50), (40, 50), (50, 50), (60, 70), (70, 70),
                                                              (80, 70)),
         heading=180, face='front', pct=(75, 50, 25), src='FC', sec='3.1.1.2 (CCFhos table, Fig 3-1)',
         page='p.30 (PDF 32)', what='Car-to-car front, head-on, straight'),
    dict(id='CCFhol', cls='vehicle', kind='straight', pairs=((30, 50), (40, 50), (50, 50), (60, 70), (70, 70),
                                                              (80, 70)),
         heading=180, face='front', pct=(75, 50, 25), lane_change=(3.5, 1.5, {50: 44.0, 70: 60.0}), src='FC',
         sec='3.1.1.2 (CCFhol table, Fig 3-2, lane-change table: offset 3.5 m, L 44 / 60 m, ends at TTC [1.5] s)',
         page='p.31 (PDF 33)', what='Car-to-car front, oncoming car changing into the ego lane from the left'),
    dict(id='CCFtap', cls='vehicle', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(30.0, 45.0, 60.0),
         line_y=LANE_OFFSET_M, heading=180, pct=(50,), src='FC',
         sec='3.1.2.1; 1.2.1.1; 1.2.4 (turn path), Fig 3-5, 3-6', page='p.33-34 (PDF 35-36)',
         what='Ego turning left across the path of an oncoming car (lanes 1.75 m from the centre line)'),
    dict(id='CCCscp-near', cls='vehicle', kind='straight', ego=_rng(20, 60, 10), tgt=_rng(20, 60, 10), heading=90,
         face='front', pct=(50,), src='FC', sec='3.1.3.1, Fig 3-9..3-11', page='p.36-38 (PDF 38-40)',
         what='Car-to-car crossing from the near side (right)'),
    dict(id='CCCscp-far', cls='vehicle', kind='straight', ego=_rng(20, 60, 10), tgt=_rng(20, 60, 10), heading=-90,
         face='front', pct=(50,), src='FC', sec='3.1.3.1, Fig 3-9..3-11', page='p.36-38 (PDF 38-40)',
         what='Car-to-car crossing from the far side (left)'),
    # ---- motorcyclist
    dict(id='CMRs', cls='motorcyclist', kind='straight', ego=_rng(10, 80, 10), tgt=(0.0,), heading=0, face='front',
         pct=(75, 50, 25), src='FC', sec='3.1.1.3 (CMRs table, Fig 3-3); 1.2.5.6', page='p.32 (PDF 34)',
         what='Car-to-motorcyclist rear, stationary'),
    dict(id='CMRb', cls='motorcyclist', kind='straight', ego=_rng(30, 80, 10), brake=(1.0, 4.0), heading=0,
         face='front', pct=(75, 50, 25), src='FC', sec='3.1.1.3 (CMRb table, Fig 3-4)', page='p.32-33 (PDF 34-35)',
         what='Car-to-motorcyclist rear, braking (1.0 s headway, −4 m/s²)'),
    dict(id='CMFtap', cls='motorcyclist', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(30.0, 45.0, 60.0),
         line_y=LANE_OFFSET_M, heading=180, pct=(50,), src='FC', sec='3.1.2.2; 1.2.1.2; 1.2.4, Fig 3-7, 3-8',
         page='p.34-35 (PDF 36-37)', what='Ego turning left across the path of an oncoming motorcycle'),
    dict(id='CMCscp-near', cls='motorcyclist', kind='straight', ego=_rng(20, 60, 10), tgt=_rng(20, 60, 10),
         heading=90, face='left', pct=(90,), src='FC', sec='3.1.3.2; 1.2.5.7, Fig 3-12..3-14',
         page='p.39-41 (PDF 41-43)', what='Motorcycle crossing from the near side, striking the ego\'s side'),
    dict(id='CMCscp-far', cls='motorcyclist', kind='straight', ego=_rng(20, 60, 10), tgt=_rng(20, 60, 10),
         heading=-90, face='right', pct=(90,), src='FC', sec='3.1.3.2; 1.2.5.7, Fig 3-12..3-14',
         page='p.39-41 (PDF 41-43)', what='Motorcycle crossing from the far side, striking the ego\'s side'),
    # ---- pedestrians
    dict(id='CPLA', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(5.0,), heading=0, face='front',
         pct=(50,), src='FC', sec='3.2.1.1, Fig 3-15', page='p.42-44 (PDF 44-46)',
         what='Pedestrian walking ahead along the ego\'s path'),
    dict(id='CPNA', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(5.0,), heading=90, face='front',
         pct=(25, 50, 75), start_m=5.0, src='FC', sec='3.2.3.1, Fig 3-22 (E 4.0 m + G 1.0 m)',
         page='p.52-53 (PDF 54-55)', what='Adult walking across from the near side (right)'),
    dict(id='CPFA', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(8.0,), heading=-90, face='front',
         pct=(50,), start_m=7.5, src='FC', sec='3.2.3.1, Fig 3-23 (D 6.0 m + F 1.5 m)', page='p.53-54 (PDF 55-56)',
         what='Adult running across from the far side (left)'),
    dict(id='CPNCO', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(5.0,), heading=90, face='front',
         pct=(50,), start_m=5.0, src='FC', sec='3.2.3.1, Fig 3-24 (E 4.0 m + G 1.0 m), behind parked cars',
         page='p.54-55 (PDF 56-57)', what='Child from the near side behind parked cars'),
    dict(id='CPTAfs', cls='pedestrian', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(5.0,), line_y=9.5,
         heading=0, pct=(50,), start_m=10.5, src='FC', sec='3.2.2.1, Fig 3-17; 1.2.4 (E 9.5 m + G 1.0 m)',
         page='p.46-47 (PDF 48-49)', what='Ego turning left; pedestrian crossing the road turned into, same direction'),
    dict(id='CPTAfo', cls='pedestrian', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(5.0,), line_y=9.5,
         heading=180, pct=(50,), start_m=7.0, src='FC', sec='3.2.2.1, Fig 3-18; 1.2.4 (E 6.0 m + G 1.0 m)',
         page='p.46-48 (PDF 48-50)', what='Ego turning left; pedestrian crossing the road turned into, opposite'),
    dict(id='CPTAns', cls='pedestrian', kind='turn', side='right', ego=(10.0,), tgt=(5.0,), line_y=-9.5, heading=0,
         pct=(50,), start_m=7.0, src='FC', sec='3.2.2.1, Fig 3-20; 1.2.4 (E 6.0 m + G 1.0 m)',
         page='p.46-50 (PDF 48-52)', what='Ego turning right; pedestrian crossing the road turned into, same direction'),
    dict(id='CPTAno', cls='pedestrian', kind='turn', side='right', ego=(10.0,), tgt=(5.0,), line_y=-9.5, heading=180,
         pct=(50,), start_m=10.5, src='FC', sec='3.2.2.1, Fig 3-19; 1.2.4 (E 9.5 m + G 1.0 m)',
         page='p.46-49 (PDF 48-51)', what='Ego turning right; pedestrian crossing the road turned into, opposite'),
    # ---- cyclists
    dict(id='CBLA', cls='cyclist', kind='straight', ego=_rng(20, 60, 10), tgt=(15.0,), heading=0, face='front',
         pct=(50,), src='FC', sec='3.2.1.2, Fig 3-16', page='p.45 (PDF 47)',
         what='Cyclist riding ahead along the ego\'s path'),
    dict(id='CBNA', cls='cyclist', kind='straight', ego=_rng(10, 60, 10), tgt=(15.0,), heading=90, face='front',
         pct=(50, 75), start_m=25.0, src='FC', sec='3.2.3.2, Fig 3-25 (steady state D 17 m, acceleration < 8 m)',
         page='p.56 (PDF 58)', what='Cyclist crossing from the near side (right)'),
    dict(id='CBNAO', cls='cyclist', kind='straight', ego=_rng(10, 60, 10), tgt=(10.0,), heading=90, face='front',
         pct=(50, 75), start_m=11.5, src='FC', sec='3.2.3.2, Fig 3-26 (H 4.5 m, G ≤ 7 m), behind an obstruction',
         page='p.57 (PDF 59)', what='Cyclist crossing from the near side behind an obstruction'),
    dict(id='CBFA', cls='cyclist', kind='straight', ego=_rng(10, 60, 10), tgt=(20.0,), heading=-90, face='front',
         pct=(25, 50), start_m=30.0, src='FC', sec='3.2.3.2, Fig 3-27 (O 22 m, N < 8 m)', page='p.58 (PDF 60)',
         what='Cyclist crossing from the far side (left)'),
    dict(id='CBTAfs', cls='cyclist', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(15.0,), line_y=5.0,
         heading=0, pct=(100,), src='FC', sec='3.2.2.2, Fig 3-21; 1.2.5.10 (path 5.00 m from the centre line)',
         page='p.50-51 (PDF 52-53)', what='Ego turning left across a same-direction cyclist on the far side'),
    dict(id='CBTAfo', cls='cyclist', kind='turn', side='left', ego=(10.0, 15.0, 20.0), tgt=(15.0,), line_y=2.75,
         heading=180, pct=(50,), src='FC', sec='3.2.2.2, Fig 3-21 (path 2.75 m from the centre line)',
         page='p.50-51 (PDF 52-53)', what='Ego turning left across an oncoming cyclist'),
    dict(id='CBTAns', cls='cyclist', kind='turn', side='right', ego=(10.0,), tgt=(15.0,), line_y=-5.0, heading=0,
         pct=(0,), src='FC', sec='3.2.2.2, Fig 3-21 (path 5.00 m from the centre line)', page='p.50-51 (PDF 52-53)',
         what='Ego turning right across a same-direction cyclist on the near side'),
    dict(id='CBTAno', cls='cyclist', kind='turn', side='right', ego=(10.0,), tgt=(15.0,), line_y=-5.0, heading=180,
         pct=(50,), src='FC', sec='3.2.2.2, Fig 3-21 (path 5.00 m from the centre line)', page='p.50-51 (PDF 52-53)',
         what='Ego turning right across a contraflow cyclist on the near side'),
    # ---- Low Speed Collisions v1.2
    dict(id='CCCscp-SfS', cls='vehicle', kind='straight', ego_profile='sfs', ego=(0.0,), tgt=_rng(20, 60, 10),
         heading=-90, face='front', pct=(50,), src='LSC', sec='3.1.1; 1.2.2.1, Fig 3-1, 3-2 (ego 2.9 m from the '
                                                           'impact point)', page='p.19-21 (PDF 22-24)',
         what='Ego starting from a stop; car crossing from the left'),
    dict(id='CMCscp-SfS', cls='motorcyclist', kind='straight', ego_profile='sfs', ego=(0.0,), tgt=_rng(20, 80, 10),
         heading=-90, face='front', pct=(50,), src='LSC', sec='3.1.2; 1.2.2.3, Fig 3-3, 3-4', page='p.22-23 (PDF 25-26)',
         what='Ego starting from a stop; motorcycle crossing from the left'),
    dict(id='CBNAO-SfS', cls='cyclist', kind='straight', ego_profile='sfs', ego=(0.0,), tgt=(10.0, 15.0, 20.0),
         heading=90, face='front', pct=(50,), src='LSC', sec='3.2.1.1, Fig 3-9', page='p.27-28 (PDF 30-31)',
         what='Ego starting from a stop; cyclist from the right behind an obstruction'),
    dict(id='CPMRCm', cls='pedestrian', kind='straight', ego_profile='reverse', ego=(4.0, 8.0), tgt=(5.0, 8.0),
         heading=90, face='rear', pct=(50,), start_m=11.5, rear_gap=1.5, src='LSC',
         sec='3.2.2.1, Fig 3-10 (child starts 10 m from the centre line, F 1.5 m; path 1.0-2.0 m behind the ego; '
             'ego reversing at CPMRCs\' 4 / 8 km/h)', page='p.29-30 (PDF 32-33)',
         what='Reversing; a child walking across behind from the right'),
    dict(id='CPMRCs', cls='pedestrian', kind='straight', ego_profile='reverse', ego=(4.0, 8.0), tgt=(0.0,), heading=0,
         face='rear', pct=(25, 50, 75), src='LSC', sec='3.2.2.2, Fig 3-11', page='p.31-32 (PDF 34-35)',
         what='Reversing towards a child standing behind'),
    dict(id='CBDA', cls='cyclist', kind='straight', ego_profile='parked', ego=(0.0,), tgt=(10.0, 15.0, 20.0), heading=0,
         face='left', pct=(DOOR_AT * 100,), gap_m=1.0, src='LSC', sec='3.2.3; 1.2.2.4; 5.2.3, Fig 3-13 (path 1 m from '
                                                                   'the widest structure; driver side)',
         page='p.35-36 (PDF 38-39)', what='Parked; a cyclist passing on the driver side as the door opens'),
    # ---- Lane Departure Collisions v1.2
    dict(id='ELK-On-C2C', cls='vehicle', kind='straight', ego=_rng(50, 100, 10), same_speed=True, lat=(0.3, 0.4, 0.5,
                                                                                                         0.6),
         heading=180, face='front', pct=(90,), src='LDC', sec='3.2.1, Fig 3-3, 3-4; 1.2.1.1', page='p.17-18 (PDF 20-21)',
         what='Ego drifting left into an oncoming car'),
    dict(id='ELK-Ov-C2C', cls='vehicle', kind='straight', ego=_rng(50, 130, 10), dv=10.0, lat=(0.2, 0.3, 0.4, 0.5, 0.6,
                                                                                               0.7),
         heading=0, face='left', pct=(25,), src='LDC', sec='3.2.2, Fig 3-5, 3-6; 1.2.1.3', page='p.18-20 (PDF 21-23)',
         what='Ego drifting left into a car overtaking it (+10 km/h)'),
    dict(id='ELK-On-C2M', cls='motorcyclist', kind='straight', ego=_rng(50, 100, 10), same_speed=True,
         lat=(0.3, 0.4, 0.5, 0.6), heading=180, face='front', pct=(110,), src='LDC', sec='3.2.3, Fig 3-7, 3-8; 1.2.1.2',
         page='p.20-21 (PDF 23-24)', what='Ego drifting left into an oncoming motorcycle'),
    dict(id='ELK-Ov-C2M', cls='motorcyclist', kind='straight', ego=_rng(50, 130, 10), dv=10.0,
         lat=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7), heading=0, face='left', pct=(25,), src='LDC',
         sec='3.2.4, Fig 3-9, 3-10; 1.2.1.4', page='p.21-23 (PDF 24-26)',
         what='Ego drifting left into a motorcycle overtaking it (+10 km/h)'),
)
NCAP_BY_ID = {s['id']: s for s in NCAP}

# templates the NHTSA rows map to that are not Euro NCAP scenarios as such (stated per row)
EXTRA = (
    dict(id='turn-target-LTAP', cls='vehicle', kind='turn_target', side='left', ego=_rng(30, 60, 10),
         tgt=(10.0, 15.0, 20.0), line_y=LANE_OFFSET_M, pct=(50,), src=None,
         what='An oncoming car turning left across the ego\'s path (CCFtap with the roles swapped)'),
    dict(id='cross-near-turning', cls='vehicle', kind='straight', ego=_rng(20, 60, 10), tgt=(10.0, 15.0, 20.0, 25.0),
         heading=90, face='front', pct=(50,), src=None,
         what='A car from the right at turning speed (CCCscp geometry, 10–25 km/h)'),
    dict(id='cross-far-turning', cls='vehicle', kind='straight', ego=_rng(20, 60, 10), tgt=(10.0, 15.0, 20.0, 25.0),
         heading=-90, face='front', pct=(50,), src=None,
         what='A car from the left at turning speed (CCCscp geometry, 10–25 km/h)'),
    dict(id='lane-change-left', cls='vehicle', kind='straight', ego=_rng(30, 130, 10), dv=10.0,
         lat=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7), heading=0, face='left', pct=(25,), src=None,
         what='Ego moving left into a car beside and behind it (ELK overtaking geometry)'),
    dict(id='lane-change-right', cls='vehicle', kind='straight', ego=_rng(30, 130, 10), dv=10.0,
         lat=(-0.2, -0.3, -0.4, -0.5, -0.6, -0.7), heading=0, face='right', pct=(25,), src=None,
         what='Ego moving right into a car beside and behind it (mirrored ELK overtaking geometry)'),
    dict(id='parking-side', cls='vehicle', kind='straight', ego=(10.0, 20.0, 30.0), dv=0.0, lat=(0.3, -0.3),
         heading=0, face='side', pct=(50,), src=None, what='Ego moving sideways into a car alongside at low speed'),
    dict(id='reverse-vehicle', cls='vehicle', kind='straight', ego_profile='reverse', ego=(4.0, 8.0), tgt=(0.0,),
         heading=0, face='rear', pct=(25, 50, 75), src=None,
         what='Reversing into a stationary car behind (CPMRCs geometry with a car)'),
    dict(id='CPNA-start', cls='pedestrian', kind='straight', ego=(5.0, 10.0), tgt=(5.0,), heading=90, face='front',
         pct=(25, 50, 75), start_m=5.0, src=None, what='Starting off; pedestrian crossing from the right (CPNA at 5–10 '
                                                       'km/h)'),
    dict(id='CPFA-start', cls='pedestrian', kind='straight', ego=(5.0, 10.0), tgt=(8.0,), heading=-90, face='front',
         pct=(50,), start_m=7.5, src=None, what='Starting off; pedestrian crossing from the left (CPFA at 5–10 km/h)'),
    dict(id='CPLA-static', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(0.0,), heading=0,
         face='front', pct=(50,), src=None, what='A pedestrian standing in the road ahead (CPLA geometry, stationary)'),
    dict(id='CPLA-edge', cls='pedestrian', kind='straight', ego=_rng(10, 60, 10), tgt=(0.0, 5.0), heading=0,
         face='front', pct=(0, 10), src=None, what='A pedestrian at the road edge, at the ego\'s right corner (CPLA '
                                                   'geometry, 0 / 10 % of the width)'),
)
TEMPLATES = {t['id']: t for t in NCAP + EXTRA}

# NHTSA rows -> templates and shares (rows not listed: no road user of these classes, left out)
_LTAP = (('CCFtap', 0.5), ('turn-target-LTAP', 0.5))
_SCP = (('CCCscp-near', 0.5), ('CCCscp-far', 0.5))
_CROSS_TURN = (('cross-near-turning', 0.5), ('cross-far-turning', 0.5))
_LANE = (('lane-change-left', 0.5), ('lane-change-right', 0.5))
_CYC_STRAIGHT = (('CBNA', 1 / 3), ('CBFA', 1 / 3), ('CBLA', 1 / 3))
_CYC_TURN = (('CBTAfs', 0.25), ('CBTAfo', 0.25), ('CBTAns', 0.25), ('CBTAno', 0.25))
MAP_SWANSON2019 = {
    'Pedal Cyclist/Maneuver': _CYC_TURN, 'Pedal Cyclist/No Maneuver': _CYC_STRAIGHT,
    'Backing into Vehicle': (('reverse-vehicle', 1.0),), 'Turning/Same Direction': (('CCRb', 1.0),),
    'Parking/Same Direction': (('parking-side', 1.0),), 'Changing Lanes/Same Direction': _LANE,
    'Drifting/Same Direction': _LANE, 'Opposite Direction/Maneuver': (('CCFhol', 1.0),),
    'Opposite Direction/No Maneuver': (('ELK-On-C2C', 0.5), ('CCFhos', 0.5)),
    'Rear-End/Striking Maneuver': (('CCRm', 1.0),), 'Rear-End/Lead Vehicle Accelerating (LVA)': (('CCRm', 1.0),),
    'Rear-End/Lead Vehicle Moving (LVM)': (('CCRm', 1.0),), 'Rear-End/LVD': (('CCRb', 1.0),),
    'Rear-End/LVS': (('CCRs', 1.0),), 'Right Turn Into Path (RTIP)': (('cross-near-turning', 1.0),),
    'Right Turn Across Path (RTAP)': (('cross-far-turning', 1.0),), 'Straight Crossing Paths (SCP)': _SCP,
    'Left Turn Across Path, Lat. Dir. (LTAP/LD)': _CROSS_TURN, 'Left Turn Into Path (LTIP)': (('cross-far-turning',
                                                                                              1.0),),
    'Left Turn Across Path, Opp. Dir. (LTAP/OD)': _LTAP,
}
MAP_NAJM2007 = {
    'Running Red Light': _SCP, 'Running Stop Sign': _SCP,
    'Pedalcyclist Crash With Prior Vehicle Maneuver': _CYC_TURN,
    'Pedalcyclist Crash Without Prior Vehicle Maneuver': _CYC_STRAIGHT,
    'Backing Up Into Another Vehicle': (('reverse-vehicle', 1.0),), 'Vehicle(s) Turning – Same Direction': (('CCRb', 1.0),),
    'Vehicle(s) Parking – Same Direction': (('parking-side', 1.0),),
    'Vehicle(s) Changing Lanes – Same Direction': _LANE, 'Vehicle(s) Drifting – Same Direction': _LANE,
    'Vehicle(s) Making a Maneuver – Opposite Direction': (('CCFhol', 1.0),),
    'Vehicle(s) Not Making a Maneuver – Opposite Direction': (('ELK-On-C2C', 0.5), ('CCFhos', 0.5)),
    'Following Vehicle Making a Maneuver': (('lane-change-left', 1.0),), 'Lead Vehicle Accelerating': (('CCRm', 1.0),),
    'Lead Vehicle Moving at Lower Constant Speed': (('CCRm', 1.0),), 'Lead Vehicle Decelerating': (('CCRb', 1.0),),
    'Lead Vehicle Stopped': (('CCRs', 1.0),), 'LTAP/OD at Signalized Junctions': _LTAP,
    'Vehicle Turning Right at Signalized Junctions': (('cross-far-turning', 1.0),),
    'LTAP/OD at Non-Signalized Junctions': _LTAP, 'Straight Crossing Paths at Non-Signalized Junctions': _SCP,
    'Vehicle(s) Turning at Non-Signalized Junctions': _CROSS_TURN,
}
# the 2016 pedestrian table: (vehicle manoeuvre, pedestrian action) -> templates
MAP_V2P2016 = {
    ('Going Straight', 'Crossing Road'): (('CPNA', 0.5), ('CPFA', 0.5)),
    ('Going Straight', 'In-Road'): (('CPLA', 0.5), ('CPLA-static', 0.5)),
    ('Going Straight', 'Adjacent to Road'): (('CPLA-edge', 1.0),),
    ('Turning Left', 'Crossing Road'): (('CPTAfs', 0.5), ('CPTAfo', 0.5)),
    ('Turning Left', 'In-Road'): (('CPTAfs', 0.5), ('CPTAfo', 0.5)),
    ('Turning Left', 'Adjacent to Road'): (('CPTAfs', 0.5), ('CPTAfo', 0.5)),
    ('Turning Right', 'Crossing Road'): (('CPTAns', 0.5), ('CPTAno', 0.5)),
    ('Turning Right', 'In-Road'): (('CPTAns', 0.5), ('CPTAno', 0.5)),
    ('Turning Right', 'Adjacent to Road'): (('CPTAns', 0.5), ('CPTAno', 0.5)),
    ('Backing Up', 'Crossing Road'): (('CPMRCm', 1.0),), ('Backing Up', 'In-Road'): (('CPMRCs', 1.0),),
    ('Backing Up', 'Adjacent to Road'): (('CPMRCs', 1.0),), ('Backing Up', 'None/Other/Unknown'): (('CPMRCs', 1.0),),
    ('Starting', 'Crossing Road'): (('CPNA-start', 0.5), ('CPFA-start', 0.5)),
    ('Starting', 'In-Road'): (('CPLA-static', 1.0),),
}


# =========================================================================== settings
@dataclass
class RiskSettings:
    weight: str = 'frequency'           # frequency | cost | fatal | fyl | ncap
    dataset: str = 'swanson2019'        # vehicle and cyclist rows: swanson2019 | najm2007 (fyl: najm2007)
    metric: str = 'resolved'            # resolved | seen | redundant
    ttc_min: float = 0.5
    ttc_max: float = 4.0
    ttc_samples: int = 15
    extended: bool = False              # also the protocols' extended-range impact locations (where tabulated)
    cell_m: float = 0.5                 # fine ground map
    area_m: float = 60.0                # the unweighted area figure counts the ground out to here
    map_m: float = 80.0                 # the fine hazard map's half width

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d) -> 'RiskSettings':
        s = RiskSettings()
        d = d if isinstance(d, dict) else {}
        if d.get('weight') in WEIGHTS:
            s.weight = d['weight']
        if d.get('dataset') in DATASETS:
            s.dataset = d['dataset']
        if d.get('metric') in METRICS:
            s.metric = d['metric']
        v = d.get('extended')
        if isinstance(v, bool):
            s.extended = v
        for k, (lo, hi) in LIMITS.items():
            if k in d:
                try:
                    x = float(d[k])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x):
                    x = min(max(x, lo), hi)
                    setattr(s, k, int(round(x)) if k in INTS else x)
        if s.ttc_max <= s.ttc_min:
            s.ttc_max = s.ttc_min + 0.1
        return s

    def effective_dataset(self) -> str:
        return 'najm2007' if self.weight == 'fyl' else self.dataset


def settings_of(project) -> RiskSettings:
    return RiskSettings.from_dict(getattr(getattr(project, 'analysis', None), 'risk_coverage', None) or {})


# =========================================================================== the ego's geometry
@dataclass
class Body:
    x_front: float = 3.6
    x_rear: float = -0.8
    width: float = 1.75

    @property
    def length(self):
        return self.x_front - self.x_rear

    @staticmethod
    def of(project, variant=None) -> 'Body':
        try:
            v = project.vehicle.mesh.vertices
            return Body(float(v[:, 0].max()), float(v[:, 0].min()), float(v[:, 1].max() - v[:, 1].min()))
        except Exception:                                        # noqa: BLE001 — no mesh: a typical small car
            return Body()


def impact_point(body: Body, face: str, pct: float) -> tuple[np.ndarray, np.ndarray]:
    """The impact point on the ego (vehicle frame) and the face's outward normal. front / rear: pct of the width from
    the right edge (0 %) to the left (100 %); left / right: pct of the length from the rear."""
    W = body.width
    if face in ('front', 'rear'):
        y = -W / 2 + pct / 100.0 * W
        return (np.array([body.x_front, y]), np.array([1.0, 0.0])) if face == 'front' else \
            (np.array([body.x_rear, y]), np.array([-1.0, 0.0]))
    x = body.x_rear + pct / 100.0 * body.length
    return (np.array([x, W / 2]), np.array([0.0, 1.0])) if face == 'left' else \
        (np.array([x, -W / 2]), np.array([0.0, -1.0]))


def _half_depth(size, heading_deg, n):
    """Half of a target's extent (length, width) along a direction n, the target heading heading_deg."""
    L, Wd = size
    h = math.radians(heading_deg)
    u = np.array([math.cos(h), math.sin(h)])
    v = np.array([-u[1], u[0]])
    return 0.5 * (L * abs(float(u @ n)) + Wd * abs(float(v @ n)))


def turn_path(side: str, speed_kmh: float, ds=0.05, after_m=40.0):
    """The ego's rear-axle path through a Frontal v1.2 §1.2.4 turn: (s, x, y, heading rad) from the turn's start (s =
    0, at the origin, heading 0) through clothoid α, arc β, clothoid α, then straight. side: 'left' | 'right'."""
    tab = TURN_FAR if side == 'left' else TURN_NEAR
    k = min(tab, key=lambda v: abs(v - speed_kmh))
    a_deg, R, b_deg = tab[k]
    a, b = math.radians(a_deg), math.radians(b_deg)
    L1 = 2.0 * R * a                                    # a clothoid from 0 to 1/R turns L / 2R
    L2 = R * b
    total = 2 * L1 + L2 + after_m
    s = np.arange(0.0, total + ds, ds)
    kap = np.where(s < L1, s / L1 / R, np.where(s < L1 + L2, 1.0 / R,
                                               np.where(s < 2 * L1 + L2, (2 * L1 + L2 - s) / L1 / R, 0.0)))
    sign = 1.0 if side == 'left' else -1.0
    psi = sign * np.concatenate([[0.0], np.cumsum((kap[1:] + kap[:-1]) / 2 * ds)])
    x = np.concatenate([[0.0], np.cumsum((np.cos(psi[1:]) + np.cos(psi[:-1])) / 2 * ds)])
    y = np.concatenate([[0.0], np.cumsum((np.sin(psi[1:]) + np.sin(psi[:-1])) / 2 * ds)])
    return s, x, y, psi


def _pose_on(path, s_q):
    """Rear-axle pose at arc lengths s_q (straight before the path's start)."""
    s, x, y, psi = path
    s_q = np.asarray(s_q, float)
    before = s_q < 0
    xi = np.interp(np.clip(s_q, 0, s[-1]), s, x)
    yi = np.interp(np.clip(s_q, 0, s[-1]), s, y)
    pi = np.interp(np.clip(s_q, 0, s[-1]), s, psi)
    xi = np.where(before, s_q, xi)
    yi = np.where(before, 0.0, yi)
    pi = np.where(before, 0.0, pi)
    return xi, yi, pi


def _to_ego(px, py, ex, ey, epsi):
    c, s = np.cos(epsi), np.sin(epsi)
    dx, dy = px - ex, py - ey
    return c * dx + s * dy, -s * dx + c * dy


# =========================================================================== one template's hazard points
def test_points(t: dict, extended=False) -> list:
    """The template's test grid: [(ego km/h, target km/h, impact %, lateral m/s)], each equally likely."""
    pcts = list(t.get('pct') or (50,))
    if extended and t.get('pct_ext'):
        pcts += list(t['pct_ext'])
    lats = list(t.get('lat') or (0.0,))
    out = []
    if t.get('pairs'):
        pairs = [(float(a), float(b)) for a, b in t['pairs']]
    elif t.get('same_speed'):
        pairs = [(float(v), float(v)) for v in t['ego']]
    elif t.get('dv') is not None:
        pairs = [(float(v), float(v) + float(t['dv'])) for v in t['ego']]
    elif t.get('brake'):
        pairs = [(float(v), float(v)) for v in t['ego']]
    else:
        pairs = [(float(a), float(b)) for a in t['ego'] for b in t['tgt']]
    for a, b in pairs:
        for p in pcts:
            for la in lats:
                out.append((a, b, float(p), float(la)))
    return out


def hazard_points(t: dict, body: Body, taus) -> dict:
    """Every test point × time before the conflict: the target's centre in the ego frame at −τ. Returns {'xy' (n, 2),
    'tau' (n,), 'w' (n,) summing to 1}."""
    taus = np.asarray(taus, float)
    pts = test_points(t)
    xs, ys, ts = [], [], []
    size = TARGET_SIZE[t['cls']]
    for ve_k, vt_k, pct, lat in pts:
        ve, vt = ve_k / 3.6, vt_k / 3.6
        if t['kind'] == 'straight':
            x, y = _straight(t, body, ve, vt, pct, lat, taus, size)
        elif t['kind'] == 'turn':
            x, y = _turn(t, body, ve, vt, pct, taus, size, ve_k)
        else:
            x, y = _turn_target(t, body, ve, vt, pct, taus, size, vt_k)
        xs.append(x)
        ys.append(y)
        ts.append(taus)
    n = len(pts) * len(taus)
    return {'xy': np.column_stack([np.concatenate(xs), np.concatenate(ys)]) if n else np.zeros((0, 2)),
            'tau': np.concatenate(ts) if n else np.zeros(0), 'w': np.full(n, 1.0 / max(n, 1)),
            'n_test_points': len(pts)}


def _straight(t, body, ve, vt, pct, lat, taus, size):
    face = t.get('face', 'front')
    if face == 'side':
        face = 'left' if lat >= 0 else 'right'
    P, nrm = impact_point(body, face, pct)
    h = float(t.get('heading', 0.0))
    if t.get('gap_m') is not None:               # a path beside the ego (dooring): the target's front at the point
        u = np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])
        c = P + nrm * float(t['gap_m']) - u * size[0] / 2
    else:
        c = P + nrm * _half_depth(size, h, nrm)
    if t.get('rear_gap') is not None:            # a path behind the reversing ego
        c = c + nrm * float(t['rear_gap'])
    u = np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])
    prof = t.get('ego_profile', 'const')
    # the ego's displacement at −τ (vehicle frame at the conflict; its heading stays 0)
    if prof == 'parked':
        ex, ey = np.zeros_like(taus), np.zeros_like(taus)
    elif prof == 'reverse':
        ex, ey = ve * taus, np.zeros_like(taus)           # it moves −x, so earlier it was ahead
    elif prof == 'sfs':
        t_move = math.sqrt(2 * SFS_DIST_M / SFS_ACCEL)
        tt = np.clip(t_move - taus, 0.0, None)            # time since it started moving
        ex = -(SFS_DIST_M - 0.5 * SFS_ACCEL * tt ** 2)
        ey = np.zeros_like(taus)
    else:
        ex, ey = -ve * taus, -lat * taus
    if t.get('brake'):
        hw, a = t['brake']
        x_rel = _braking_gap(ve, hw, a, taus)
        tx, ty = c[0] + x_rel, np.full_like(taus, c[1])
        return tx - 0.0 * ex, ty - ey                     # relative already (both move along x)
    d = vt * taus
    if t.get('start_m') is not None and vt > 0:
        d = np.minimum(d, float(t['start_m']))
    tx = c[0] - u[0] * d
    ty = c[1] - u[1] * d
    if t.get('lane_change') and vt > 0:
        off, end_ttc, lengths = t['lane_change']
        L = lengths.get(int(round(vt * 3.6)), max(lengths.values()))
        frac = np.clip((taus - end_ttc) * vt / L, 0.0, 1.0)
        ty = ty + off * frac                              # it came from the lane to the ego's left
    return tx - ex, ty - ey


def _braking_gap(v0, headway, a, taus, dt=0.005):
    """CCRb: both at v0, the lead's rear headway · v0 ahead of the ego's front, the lead braking at a until it stops:
    the gap at τ before the gap closes (the gap stays at its start value before the braking)."""
    g0 = headway * v0
    t = 0.0
    g = g0
    vl = v0
    hist = [(0.0, g0)]
    while g > 0 and t < 60:
        vl = max(vl - a * dt, 0.0)
        g -= (v0 - vl) * dt
        t += dt
        hist.append((t, g))
    tc = t
    tt = np.array([h[0] for h in hist])
    gg = np.array([h[1] for h in hist])
    q = tc - np.asarray(taus, float)
    return np.where(q <= 0, g0, np.interp(np.clip(q, 0, tc), tt, gg))


def _turn(t, body, ve, vt, pct, taus, size, ve_k):
    """The ego on the turn path from its lane (1.75 m right of the main road's centre line); the target on a line
    parallel to the main road at line_y; the conflict where the ego's impact point crosses the line."""
    path = turn_path(t['side'], ve_k)
    s, x, y, psi = path
    y0 = -LANE_OFFSET_M
    P, _ = impact_point(body, 'front', pct)
    # impact point along the path (world: main-road frame, ego lane at y0)
    ix = x + np.cos(psi) * P[0] - np.sin(psi) * P[1]
    iy = y0 + y + np.sin(psi) * P[0] + np.cos(psi) * P[1]
    ly = float(t['line_y'])
    f = (iy - ly) * (1.0 if ly > y0 else -1.0)
    k = int(np.argmax(f >= 0)) if (f >= 0).any() else len(s) - 1
    if k > 0 and f[k - 1] < 0:                    # interpolate the crossing
        r = -f[k - 1] / (f[k] - f[k - 1])
        s_c = s[k - 1] + r * (s[k] - s[k - 1])
        cx = ix[k - 1] + r * (ix[k] - ix[k - 1])
    else:
        s_c, cx = s[k], ix[k]
    h = math.radians(float(t.get('heading', 0.0)))
    u = np.array([math.cos(h), math.sin(h)])
    d = vt * taus
    if t.get('start_m') is not None and vt > 0:
        d = np.minimum(d, float(t['start_m']))
    tx, ty = cx - u[0] * d, ly - u[1] * d
    ex, ey, ep = _pose_on(path, s_c - ve * taus)
    return _to_ego(tx, ty, ex, ey + y0, ep)


def _turn_target(t, body, ve, vt, pct, taus, size, vt_k):
    """The roles of CCFtap swapped: the target turns left across the ego's path from the oncoming lane; the ego drives
    straight. Worked in the turner's frame (its lane at −1.75 m, the ego's at +1.75 m heading −x), then into the
    ego's."""
    path = turn_path(t['side'], vt_k)
    s, x, y, psi = path
    y0 = -LANE_OFFSET_M
    front = 3.6                                       # a typical car's front from its rear axle (m)
    ix = x + np.cos(psi) * front
    iy = y0 + y + np.sin(psi) * front
    ly = float(t['line_y'])
    k = int(np.argmax(iy >= ly)) if (iy >= ly).any() else len(s) - 1
    s_c, cx = s[k], ix[k]
    # the ego's front centre at the conflict point, heading −x (180°)
    erx = cx + body.x_front
    ex = erx + ve * taus
    ey = np.full_like(taus, ly)
    tx, ty, tp = _pose_on(path, s_c - vt * taus)
    cxx = tx + np.cos(tp) * 1.3                       # the target's centre 1.3 m ahead of its rear axle (typical)
    cyy = y0 + ty + np.sin(tp) * 1.3
    return _to_ego(cxx, cyy, ex, ey, np.full_like(taus, math.pi))


# =========================================================================== the weights
def rows_of(dataset: str) -> list:
    """The NHTSA rows of a dataset: [{'name', 'crashes', 'cost_musd', 'fatal', 'fyl', 'row', 'table'}]."""
    out = []
    if dataset == 'najm2007':
        for no, name, cr, cost, fyl in NAJM2007:
            out.append({'name': name, 'no': no, 'crashes': cr, 'cost_musd': cost, 'fatal': None, 'fyl': fyl,
                        'table': f'Najm 2007, scenario {no}: Table 13 (p.37), 14 (p.38), 15 (p.39)'})
    else:
        for no, name, cr, fat, cost, el in SWANSON2019:
            out.append({'name': name, 'no': no, 'crashes': cr, 'cost_musd': cost, 'fatal': fat, 'fyl': None,
                        'equivalent_lives': el, 'table': f'Swanson 2019, Table 12 (p.83)' +
                                                         (f', row {no}' if no is not None else '')})
    return out


def _value(row, weight):
    k = {'frequency': 'crashes', 'cost': 'cost_musd', 'fatal': 'fatal', 'fyl': 'fyl'}[weight]
    v = row.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def weighting(st: RiskSettings) -> dict:
    """Per class: [{'template', 'weight' (share of the class, sums to 1), 'trace': {...}}] and per class the rows left
    out with their share. NCAP: every scenario of the class equally."""
    out = {c: {'items': [], 'left_out': [], 'basis': ''} for c in CLASSES}
    if st.weight == 'ncap':
        for c in CLASSES:
            ids = [s['id'] for s in NCAP if s['cls'] == c]
            for i in ids:
                out[c]['items'].append({'template': i, 'weight': 1.0 / len(ids), 'trace': _ncap_trace(i)})
            out[c]['basis'] = f'{len(ids)} Euro NCAP scenarios of the current Crash Avoidance protocols, equally'
        return out
    ds = st.effective_dataset()
    mp = MAP_NAJM2007 if ds == 'najm2007' else MAP_SWANSON2019
    rows = rows_of(ds)
    raw = {c: [] for c in CLASSES}
    total = {c: 0.0 for c in CLASSES}
    for r in rows:
        v = _value(r, st.weight)
        m = mp.get(r['name'])
        if m is None:
            continue
        for tid, share in m:
            c = TEMPLATES[tid]['cls']
            if v is None:
                continue
            raw[c].append((tid, v * share, r, share))
            total[c] += v * share
    # pedestrians: the 2016 pedestrian table (the weight fyl has none there: its frequencies, said so)
    pw = 'frequency' if st.weight == 'fyl' else st.weight
    pk = {'frequency': 2, 'fatal': 3, 'cost': 4}[pw]
    for row in V2P2016:
        key = (row[0], row[1])
        v = row[pk]
        m = MAP_V2P2016.get(key)
        tr = {'source': 'v2p2016', 'row': f'{row[0]} / {row[1]}', 'value': v,
              'column': {2: 'crashes', 3: 'fatal crashes', 4: 'cost $M'}[pk], 'table': NHTSA_SOURCES['v2p2016']['tables']}
        if m is None or v is None:
            if v:
                out['pedestrian']['left_out'].append(tr)
            continue
        for tid, share in m:
            raw['pedestrian'].append((tid, float(v) * share, {'name': tr['row'], 'table': tr['table']}, share))
            total['pedestrian'] += float(v) * share
    for c in ('vehicle', 'cyclist', 'pedestrian'):
        for tid, w, r, share in raw[c]:
            out[c]['items'].append({'template': tid, 'weight': w / total[c] if total[c] > 0 else 0.0,
                                    'trace': {'source': 'v2p2016' if c == 'pedestrian' else ds, 'row': r['name'],
                                              'table': r['table'], 'value': w / share if share else w,
                                              'share_of_row': share, 'weight': pw if c == 'pedestrian' else st.weight,
                                              'geometry': _ncap_trace(tid)}})
    # left out rows (vehicle and cyclist datasets): the rows without a mapping
    tot_all = sum(_value(r, st.weight) or 0.0 for r in rows)
    # (the pedestrian rows are modelled from the 2016 pedestrian table, not left out)
    lo = [{'row': r['name'], 'value': _value(r, st.weight), 'share': (_value(r, st.weight) or 0.0) / tot_all
           if tot_all else None, 'table': r['table']} for r in rows
          if r['name'] not in mp and _value(r, st.weight) and 'pedestrian' not in r['name'].lower()]
    out['vehicle']['left_out'] = lo
    out['vehicle']['basis'] = f'{NHTSA_SOURCES[ds]["citation"]} ({st.weight})'
    out['cyclist']['basis'] = out['vehicle']['basis']
    out['pedestrian']['basis'] = f'{NHTSA_SOURCES["v2p2016"]["citation"]} ({pw})'
    ids = [s['id'] for s in NCAP if s['cls'] == 'motorcyclist']
    out['motorcyclist']['items'] = [{'template': i, 'weight': 1.0 / len(ids), 'trace': _ncap_trace(i)} for i in ids]
    out['motorcyclist']['basis'] = ('Euro NCAP scenarios equally: the NHTSA light-vehicle typology does not separate '
                                    'motorcyclists')
    return out


def _ncap_trace(tid) -> dict:
    t = TEMPLATES[tid]
    if t.get('src'):
        s = NCAP_SOURCES[t['src']]
        return {'template': tid, 'what': t['what'], 'protocol': f'{s["title"]} v{s["version"]} ({s["date"]})',
                'section': t['sec'], 'page': t['page']}
    return {'template': tid, 'what': t['what'], 'protocol': 'Vantage template (Euro NCAP geometry adapted, stated)'}


# =========================================================================== coverage at points
def _object_classes(project):
    from . import perception as PE
    st = PE.PerceptionSettings.from_dict(getattr(project.analysis, 'perception', None))
    by = {c.name.lower(): c for c in st.object_classes()}
    cyc = by.get('cyclist') or PE.ObjectClass('Cyclist', 1.7, 0.6, 24.0)
    out = {'pedestrian': by.get('pedestrian') or PE.ObjectClass('Pedestrian', 1.7, 0.5, 24.0),
           'cyclist': cyc, 'vehicle': by.get('car') or PE.ObjectClass('Car', 1.5, 1.8, 20.0),
           'motorcyclist': by.get('motorcyclist') or PE.ObjectClass(MOTORCYCLIST_CLASS['name'],
                                                                    MOTORCYCLIST_CLASS['height'],
                                                                    MOTORCYCLIST_CLASS['width'], cyc.min_px)}
    return out


class Coverage:
    """The cameras' perception test at ground points, per class (core.perception.object_pixels)."""

    def __init__(self, project, engine, variant=None):
        from . import perception as PE
        from .lens import resolve
        self.PE = PE
        v = variant or project.variant
        self.cams = [c for c in v.cameras if c.visible]
        self.intr = [resolve(project.module_for(c), project.lens_for(c)) for c in self.cams]
        vg = engine.vehicle_geom(project, v) if engine is not None else None
        self.cubes = [engine.cube(c.pos()) if vg is not None else None for c in self.cams]
        self.classes = _object_classes(project)
        self.body = None
        if vg is not None:
            lo, hi = vg.bbox
            self.body = ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))

    def count(self, xy, cls: str, metric='resolved') -> np.ndarray:
        """Per point: the number of cameras that pass the metric's test (seen, or resolved)."""
        oc = self.classes[cls]
        xy = np.asarray(xy, float).reshape(-1, 2)
        n = np.zeros(len(xy), int)
        for c, intr, cube in zip(self.cams, self.intr, self.cubes):
            h, _, seen = self.PE.object_pixels(c, intr, xy, oc, cube)
            ok = seen if metric == 'seen' else seen & (h >= oc.min_px)
            n += ok.astype(int)
        return n

    def covered(self, xy, cls, metric='resolved') -> np.ndarray:
        k = self.count(xy, cls, 'resolved' if metric == 'redundant' else metric)
        return k >= (2 if metric == 'redundant' else 1)

    def inside_body(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.body is None:
            return np.zeros(len(xy), bool)
        (x0, y0), (x1, y1) = self.body
        return (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)


# =========================================================================== the study
@dataclass
class RiskResult:
    settings: RiskSettings
    variant: str
    grid: RegionGrid
    body: Body
    classes: dict              # class -> figures, regions, top gaps, items
    weights: dict              # weighting(st)
    fine: dict                 # 'x' edges, 'y' edges, per class 'density' (H) and 'covered' (H) of the ground
    seconds: float = 0.0
    notes: list = field(default_factory=list)
    finished_at: float = 0.0


def run(project, engine=None, settings: RiskSettings | None = None, variant=None, progress=None, cancel=None,
        coverage=None) -> RiskResult | None:
    """The hazard densities and the rig's coverage weighted by them (None when cancelled). coverage: an object with
    covered(xy, cls, metric) and inside_body(xy) (tests); by default the cameras' perception test."""
    t0 = time.perf_counter()
    st = settings or settings_of(project)
    stop = cancel or (lambda: False)

    def prog(f):
        if progress is not None:
            progress(min(max(float(f), 0.0), 1.0))
    if variant is None:
        v = project.variant if project is not None else None
    elif isinstance(variant, str):
        v = next((x for x in project.variants if x.uid == variant or x.name == variant), project.variant)
    else:
        v = variant
    if coverage is None:
        from .analysis import Engine
        coverage = Coverage(project, engine or Engine(), v)
    grid = default_grid(project, v) if project is not None else RegionGrid()
    body = Body.of(project, v) if project is not None else Body()
    W = weighting(st)
    taus = st.ttc_min + (np.arange(st.ttc_samples) + 0.5) * (st.ttc_max - st.ttc_min) / st.ttc_samples
    half = float(st.map_m)
    nb = int(round(2 * half / st.cell_m))
    edges = np.linspace(-half, half, nb + 1)
    xe = edges + grid.origin_xy[0]
    ye = edges
    cls_out = {}
    fine = {'x_edges': xe, 'y_edges': ye, 'cell_m': st.cell_m}
    notes = []
    work = sum(len(W[c]['items']) for c in CLASSES) + len(CLASSES)
    done = 0
    for c in CLASSES:
        items = W[c]['items']
        XY, WT, TID, TAU = [], [], [], []
        for it in items:
            if stop():
                return None
            hp = hazard_points(TEMPLATES[it['template']], body, taus)
            XY.append(hp['xy'])
            WT.append(hp['w'] * it['weight'])
            TID += [it['template']] * len(hp['w'])
            TAU.append(hp['tau'])
            it['test_points'] = hp['n_test_points']
            done += 1
            prog(0.6 * done / work)
        if not XY:
            continue
        XY = np.vstack(XY)
        WT = np.concatenate(WT)
        TAU = np.concatenate(TAU)
        TID = np.array(TID)
        inb = coverage.inside_body(XY)
        if inb.any():
            notes.append(f'{CLASS_LABEL[c]}: {inb.mean() * 100:.1f} % of the hazard points fall on the body footprint '
                         '(the target touching it near the conflict): counted as uncovered.')
        cov = coverage.covered(XY, c, st.metric) & ~inb
        tot = float(WT.sum())
        weighted = float((WT * cov).sum() / tot) if tot > 0 else None
        # the unweighted figure: the ground out to area_m, off the body
        gx = (xe[:-1] + xe[1:]) / 2
        gy = (ye[:-1] + ye[1:]) / 2
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        G = np.column_stack([GX.ravel(), GY.ravel()])
        rr = np.hypot(G[:, 0] - grid.origin_xy[0], G[:, 1] - grid.origin_xy[1])
        on = (rr <= st.area_m) & ~coverage.inside_body(G)
        gcov = np.zeros(len(G), bool)
        gcov[on] = coverage.covered(G[on], c, st.metric)
        area = float(gcov[on].mean()) if on.any() else None
        H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=[xe, ye], weights=WT)
        fine[c] = {'density': H / max(tot, 1e-300) / (st.cell_m ** 2), 'covered': gcov.reshape(len(gx), len(gy)),
                   'on': on.reshape(len(gx), len(gy))}
        # regions
        s_i, r_i = grid.assign(XY[:, 0], XY[:, 1])
        gs, gr = grid.assign(G[:, 0], G[:, 1])
        regions = {}
        for s, r, rid, name in grid.regions():
            m = (s_i == s) & (r_i == r)
            wr = float(WT[m].sum())
            gm = on & (gs == s) & (gr == r)
            if wr <= 0 and not gm.any():
                continue
            unc = m & ~cov
            by_t = {}
            for tid, w in zip(TID[unc], WT[unc]):
                by_t[tid] = by_t.get(tid, 0.0) + float(w)
            top = sorted(by_t.items(), key=lambda kv: -kv[1])[:3]
            regions[rid] = {'name': name, 'hazard_share': wr / tot if tot else 0.0,
                            'covered': float((WT[m] * cov[m]).sum() / wr) if wr > 0 else None,
                            'uncovered_share': float(WT[unc].sum() / tot) if tot else 0.0,
                            'area_covered': float(gcov[gm].mean()) if gm.any() else None,
                            'area_m2': grid.area_m2(s, r),
                            'density_per_m2': wr / tot / grid.area_m2(s, r) if tot else 0.0,
                            'top_uncovered': [{'template': k, 'share_of_class': v_ / tot,
                                               'what': TEMPLATES[k]['what']} for k, v_ in top]}
        outside = float(WT[r_i < 0].sum() / tot) if tot else 0.0
        gaps = sorted(((rid, d) for rid, d in regions.items() if d['uncovered_share'] > 0),
                      key=lambda kv: -kv[1]['uncovered_share'])[:5]
        by_template = {}
        for tid in np.unique(TID):
            m = TID == tid
            by_template[str(tid)] = {'share': float(WT[m].sum() / tot), 'covered': float((WT[m] * cov[m]).sum() /
                                                                                       max(WT[m].sum(), 1e-300)),
                                     'what': TEMPLATES[str(tid)]['what']}
        # the time to conflict at which the hazard is first covered (weighted median over test points is costly):
        # coverage per τ instead
        per_tau = []
        for tv in taus:
            m = np.isclose(TAU, tv)
            per_tau.append({'tau_s': float(tv), 'covered': float((WT[m] * cov[m]).sum() / max(WT[m].sum(), 1e-300))})
        cls_out[c] = {'weighted': weighted, 'area': area, 'points': int(len(XY)), 'outside_grid': outside,
                      'regions': regions, 'gaps': [{'region': rid, **d} for rid, d in gaps],
                      'templates': by_template, 'per_tau': per_tau, 'basis': W[c]['basis'],
                      'items': W[c]['items'], 'left_out': W[c]['left_out'], 'text': _gap_text(c, weighted, area,
                                                                                               gaps, st)}
        done += 1
        prog(0.6 + 0.4 * done / work)
    res = RiskResult(st, getattr(v, 'name', ''), grid, body, cls_out, W, fine, time.perf_counter() - t0, notes,
                     time.time())
    prog(1.0)
    return res


def _gap_text(c, weighted, area, gaps, st) -> str:
    if weighted is None:
        return f'{CLASS_LABEL[c]}: no hazard data.'
    s = (f'{CLASS_LABEL[c]}: the rig covers {area * 100:.0f} % of the ground out to {st.area_m:g} m but '
         f'{weighted * 100:.0f} % of the pre-crash {c} exposure ({WEIGHT_LABEL[st.weight].lower()})')
    if gaps:
        rid, d = gaps[0]
        tops = ', '.join(t['template'] for t in d['top_uncovered'][:2])
        s += f'; the largest gap is {d["name"]} ({d["uncovered_share"] * 100:.1f} % of the exposure: {tops})'
    return s + '.'


# =========================================================================== exports
def _r(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if math.isfinite(f) else None


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _r(o, 6)
    return o


def summary(res: RiskResult) -> dict:
    st = res.settings
    return {'schema': SCHEMA, 'variant': res.variant, 'weight': st.weight, 'dataset': st.effective_dataset(),
            'metric': st.metric, 'classes': {c: {'weighted': _r(d['weighted'], 4), 'area': _r(d['area'], 4),
                                                 'largest_gap': (d['gaps'][0]['name'] if d['gaps'] else None),
                                                 'text': d['text']} for c, d in res.classes.items()},
            'seconds': _r(res.seconds, 2)}


def sources_doc() -> dict:
    return {'nhtsa': NHTSA_SOURCES, 'euro_ncap': NCAP_SOURCES,
            'tables': {'najm2007': {'rows': len(NAJM2007), 'totals': NAJM2007_TOTALS},
                       'swanson2019': {'rows': len(SWANSON2019), 'totals': SWANSON2019_TOTALS},
                       'v2p2016': {'rows': len(V2P2016), 'totals': V2P2016_TOTALS}}}


def to_doc(res: RiskResult, project=None) -> dict:
    st = res.settings
    return _clean({
        'schema': SCHEMA, 'project': getattr(project, 'name', ''), 'variant': res.variant, 'frame': FRAME,
        'units': {'position': 'm (vehicle frame)', 'density': 'share of the class\'s exposure per m²',
                  'coverage': 'share', 'time': 's before the conflict'},
        'settings': st.to_dict(), 'weight': {'id': st.weight, 'text': WEIGHT_LABEL[st.weight],
                                            'dataset': st.effective_dataset()},
        'metric': {'id': st.metric, 'text': METRIC_LABEL[st.metric]}, 'grid': res.grid.to_dict(),
        'ego': {'x_front_m': res.body.x_front, 'x_rear_m': res.body.x_rear, 'width_m': res.body.width},
        'sources': sources_doc(), 'method': __doc__.split('Weights (selectable')[0].strip(),
        'assumptions': ASSUMPTIONS, 'limits': LIMITS_TEXT,
        'classes': {c: {k: v for k, v in d.items()} for c, d in res.classes.items()},
        'templates': {k: {kk: vv for kk, vv in t.items() if kk not in ('lane_change',)} | (
            {'source': _ncap_trace(k)}) for k, t in TEMPLATES.items()},
        'notes': list(res.notes), 'seconds': _r(res.seconds, 2)})


def to_csv(res: RiskResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('class', 'region', 'region_name', 'hazard_share', 'covered', 'uncovered_share', 'area_covered',
                'density_per_m2', 'top_uncovered'))
    for c, d in res.classes.items():
        for rid, r in d['regions'].items():
            w.writerow([c, rid, r['name'], _r(r['hazard_share'], 5), _r(r['covered'], 4),
                        _r(r['uncovered_share'], 5), _r(r['area_covered'], 4), _r(r['density_per_m2'], 7),
                        '; '.join(f'{t["template"]} {t["share_of_class"] * 100:.2f} %' for t in r['top_uncovered'])])
    return buf.getvalue()


def weights_csv(res: RiskResult) -> str:
    """Every template weight traced to its table row."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(('class', 'template', 'weight_in_class', 'source', 'row', 'table', 'row_value', 'share_of_row',
                'geometry', 'section', 'page'))
    for c, d in res.weights.items():
        for it in d['items']:
            tr = it['trace']
            g = tr.get('geometry', tr)
            w.writerow([c, it['template'], _r(it['weight'], 6), tr.get('source', 'euro_ncap'), tr.get('row', ''),
                        tr.get('table', ''), tr.get('value', ''), tr.get('share_of_row', ''), g.get('protocol', ''),
                        g.get('section', ''), g.get('page', '')])
    return buf.getvalue()


def map_png(res: RiskResult, cls='pedestrian', size=720) -> bytes:
    """The fine hazard map of a class over the plan view: density (log, orange), uncovered ground (grey hatch), the
    body (outline), the region grid's rings."""
    import cv2
    img = np.full((size, size, 3), 255, np.uint8)
    f = res.fine.get(cls)
    if f is None:
        return cv2.imencode('.png', img)[1].tobytes()
    D = cv2.GaussianBlur(np.asarray(f['density'], np.float64), (0, 0), 1.0)      # smoothed for display only
    nx, ny = D.shape
    # x forward = up, y left = left
    cov = f['covered']
    on = f['on']
    base = np.full((nx, ny, 3), 255, np.uint8)
    base[on & ~cov] = (225, 225, 225)
    pos = D > 0
    if pos.any():
        lv = np.log10(D[pos])
        lo, hi = np.percentile(lv, 5), lv.max()
        t = np.zeros_like(D)
        t[pos] = np.clip((np.log10(D[pos]) - lo) / max(hi - lo, 1e-9), 0.05, 1)
        col = np.stack([40 + 0 * t, 120 - 80 * t, 255 - 30 * t], -1)     # BGR orange to red
        a = (t * 0.85)[..., None]
        base = np.where(pos[..., None], (base * (1 - a) + col * a).astype(np.uint8), base)
    im = cv2.resize(np.ascontiguousarray(base[::-1, ::-1]), (size, size), interpolation=cv2.INTER_NEAREST)
    xe, ye = res.fine['x_edges'], res.fine['y_edges']

    def px(x, y):
        u = (ye[-1] - y) / (ye[-1] - ye[0]) * size
        v = (xe[-1] - x) / (xe[-1] - xe[0]) * size
        return int(round(u)), int(round(v))
    b = res.body
    p0, p1 = px(b.x_front, b.width / 2), px(b.x_rear, -b.width / 2)
    cv2.rectangle(im, p0, p1, (60, 60, 60), 1)
    o = px(*res.grid.origin_xy)
    for e in res.grid.range_edges:
        rpx = int(e / (xe[-1] - xe[0]) * size)
        cv2.circle(im, o, rpx, (180, 180, 180), 1, cv2.LINE_AA)
    d = res.classes.get(cls) or {}
    txt = f'{CLASS_LABEL[cls]}: {(d.get("weighted") or 0) * 100:.0f} % of the exposure covered, ' \
          f'{(d.get("area") or 0) * 100:.0f} % of the ground (density smoothed by 1 cell)'
    cv2.putText(im, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return cv2.imencode('.png', im)[1].tobytes()


def files(res: RiskResult, project=None) -> list:
    out = [('risk_coverage.json', json.dumps(to_doc(res, project), indent=1, ensure_ascii=False).encode('utf-8')),
           ('risk_coverage_regions.csv', to_csv(res).encode('utf-8')),
           ('risk_coverage_weights.csv', weights_csv(res).encode('utf-8'))]
    for c in res.classes:
        out.append((f'risk_coverage_{c}.png', map_png(res, c)))
    return out


def export(res: RiskResult, project, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, data in files(res, project):
        p = os.path.join(out_dir, name)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


def requirement_proposal(res: RiskResult, margin=0.05) -> list:
    """Metric proposals for the requirements scorecard: 'risk-weighted coverage ≥ X % for class C' at the figure this
    rig reaches minus a margin (rounded down to 5 %)."""
    out = []
    for c, d in res.classes.items():
        if d['weighted'] is None:
            continue
        x = max(0.0, math.floor((d['weighted'] - margin) * 20) / 20)
        out.append({'metric': f'risk_coverage.{c}', 'text': f'Risk-weighted coverage ≥ {x * 100:.0f} % for '
                                                             f'{c}s ({WEIGHT_LABEL[res.settings.weight].lower()}, '
                                                             f'{METRIC_LABEL[res.settings.metric].lower()})',
                    'op': '>=', 'value': x, 'unit': 'share', 'measured': _r(d['weighted'], 4),
                    # the same as a requirements scorecard row (core.requirements' metric is in per cent)
                    'item': {'metric': 'risk_coverage.weighted_pct', 'params': {'cls': c}, 'op': '>=',
                             'target': round(x * 100, 1), 'priority': 'must'}})
    return out
