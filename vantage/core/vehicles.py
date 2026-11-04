"""The vehicle library: real vehicles with the dimensions their makers state, each with its source and the date it was
read, and, where one exists under a licence that allows it, a 3D model Vantage can download from its source.

A template body is lofted from its family's side profile (sedan, hatchback, SUV, van, truck) and scaled to the
vehicle's stated length, width, height, wheelbase, overhangs, track, ground clearance and wheel size; its shape between
those figures is the family's, not the vehicle's. A figure the maker does not state is drawn from the family's
proportions and named in the body's note; only stated figures go into the spec check. For the vehicle's real shape,
download its 3D model (Vehicle ▸ Vehicle library ▸ Download 3D model) where one is listed, or import a scan or CAD
model of your own (Vehicle ▸ Replace model).

The 3D models are not part of Vantage: they are fetched from their publishers when asked, kept in the user's cache and
loaded through the import wizard with the vehicle's stated dimensions for its spec check. Each one's licence and the
attribution it asks for are recorded in the project. None of these licences grants a trademark right: the makers'
names and the vehicles' styling remain theirs.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass

CHECKED = '2026-11-04'
LIBRARY_NOTE = ('Dimensions as each maker states them (spec sheets, press kits, brochures, owner\'s manuals), read '
                f'{CHECKED}; each entry names its sources. Bodies are the family\'s profile scaled to those figures.')

# the old generic template keys, kept for projects and command lines that name them
ALIASES = {'sedan': 'lincoln_mkz_2017', 'hatchback': 'renault_zoe_ph1', 'suv': 'lexus_rx450h_2020',
           'van': 'mercedes_sprinter_170_hr', 'truck': 'freightliner_cascadia_126'}

# the family profiles' shares, used only for a figure the maker does not state (named in the body's note)
FAMILY = {
    # rear overhang / (front + rear overhang), ground clearance m, wheel-arch gap
    'sedan': {'rear_share': 0.55, 'clearance': 0.13},
    'hatch': {'rear_share': 0.46, 'clearance': 0.14},
    'suv': {'rear_share': 0.50, 'clearance': 0.18},
    'van': {'rear_share': 0.55, 'clearance': 0.18},
    'truck': {'rear_share': 0.45, 'clearance': 0.30},
}
ICON = {'sedan': 'sedan', 'hatch': 'hatchback', 'suv': 'suv', 'van': 'van', 'truck': 'truck'}


@dataclass(frozen=True)
class Model3D:
    """A 3D model of the vehicle that may be downloaded and used: where it is, its licence, the credit it asks for."""
    page: str                     # the publisher's page
    url: str                      # the file Vantage downloads
    licence: str                  # SPDX-like id: CC0-1.0, BSD-3-Clause, Apache-2.0
    attribution: str              # the line to keep with it
    member: str = ''              # the mesh inside an archive (.zip, .tar.gz)
    fmt: str = ''                 # obj, dae, glb ...
    size_mb: float = 0.0          # the download's size as published (for the user, and a limit)
    exact: bool = True            # the vehicle itself (else a lookalike)
    note: str = ''
    units: str = 'm'              # what the import wizard is told: m, cm, mm
    up: str = ''                  # '+Z', '+Y' ('' = the file's own or the wizard's guess)
    forward: str = ''

    @property
    def filename(self) -> str:
        return os.path.basename(self.member or self.url.split('?')[0]) or 'model'


@dataclass(frozen=True)
class VehicleModel:
    id: str
    maker: str
    name: str
    generation: str
    family: str                   # sedan | hatch | suv | van | truck: the body profile it is drawn with
    length: float = 0.0           # m; 0 = not stated
    width: float = 0.0            # without mirrors
    height: float = 0.0
    wheelbase: float = 0.0
    ground_clearance: float = 0.0
    front_track: float = 0.0
    rear_track: float = 0.0
    front_overhang: float = 0.0
    rear_overhang: float = 0.0
    width_mirrors: float = 0.0
    tyre: str = ''                # e.g. '215/55 R17'
    kerb_kg: float = 0.0
    sources: tuple = ()           # ((what, url), ...)
    note: str = ''
    used_by: str = ''             # the public datasets and programmes that drive it
    cameras: str = '6-cam'
    model3d: Model3D | None = None
    trailer: 'VehicleModel | None' = None
    trailer_gap: float = 0.0

    @property
    def title(self) -> str:
        return f'{self.maker} {self.name}'

    @property
    def url(self) -> str:
        return self.sources[0][1] if self.sources else ''

    def stated(self) -> dict:
        """The figures the maker states, as the spec check reads them (m)."""
        out = {}
        for k, f in (('Length', self.length), ('Width', self.width), ('Height', self.height),
                     ('Wheelbase', self.wheelbase), ('Ground clearance', self.ground_clearance),
                     ('Front track', self.front_track), ('Rear track', self.rear_track),
                     ('Front overhang', self.front_overhang), ('Rear overhang', self.rear_overhang)):
            if f > 0:
                out[k] = f
        return out

    def missing(self) -> list[str]:
        names = {'length': 'length', 'width': 'width', 'height': 'height', 'wheelbase': 'wheelbase',
                 'ground_clearance': 'ground clearance', 'front_track': 'track', 'front_overhang': 'overhangs',
                 'tyre': 'tyre size'}
        return [v for k, v in names.items() if not getattr(self, k)]


def tyre_radius(tyre: str) -> float:
    """The unladen radius (m) of a metric tyre size: 215/55 R17 is 17″ / 2 + 215 mm × 55 %. 0 for another form."""
    m = re.search(r'(\d{3})\s*/\s*(\d{2})\s*Z?R\s*(\d{2}(?:\.\d)?)', tyre or '', re.I)
    if not m:
        return 0.0
    w, a, rim = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return round((rim * 25.4 / 2 + w * a / 100) / 1000, 4)


def tyre_width(tyre: str) -> float:
    m = re.search(r'(\d{3})\s*/\s*\d{2}', tyre or '')
    return float(m.group(1)) / 1000 if m else 0.0


def _s(*pairs):
    return tuple(pairs)


def library() -> list[VehicleModel]:
    """The built-in vehicles, figures as stated by their sources (read 2026-11-04)."""
    return [
        VehicleModel(
            'renault_zoe_ph1', 'Renault', 'ZOE', 'Phase 1 (2012–2019)', 'hatch',
            length=4.084, width=1.730, height=1.562, wheelbase=2.588, ground_clearance=0.120,
            front_track=1.511, rear_track=1.510, front_overhang=0.839, rear_overhang=0.657, width_mirrors=1.945,
            tyre='195/55 R16', kerb_kg=1480.0, used_by='nuScenes (Motional, Boston and Singapore)',
            sources=_s(('Renault Jordan, ZOE dimensions (Internet Archive copy, 8 Feb 2026)',
                        'http://web.archive.org/web/20260208051144/https://renault.jo/cars/zoe/dimensions.html'),
                       ('kerb mass of the Z.E. 40 (Renault table as reproduced by x-engineer.org)',
                        'https://x-engineer.org/?p=1231')),
            note='Ground clearance laden; height unladen. Tyres 185/65 R15 (Life) or 195/55 R16 (Zen, Intens): '
                 'drawn with the 16″.'),
        VehicleModel(
            'vw_passat_b6_variant', 'Volkswagen', 'Passat Variant', 'B6 (2005–2010)', 'suv',
            length=4.774, width=1.820, height=1.517, wheelbase=2.709, front_track=1.552, rear_track=1.551,
            tyre='205/55 R16', kerb_kg=1452.0, used_by='KITTI (Karlsruhe)',
            sources=_s(('Volkswagen press release, Aug 2005 (length 4.77, width 1.82, height 1.52 m)',
                        'https://www.theautochannel.com/news/2005/08/07/139643.html'),
                       ('Passat Variant 1.9 TDI Trendline data (length, height, track, tyre, mass; third party)',
                        'https://cars-data.com/en/volkswagen/passat/b6-2005-wagon/variant-1-9-tdi-105hp-trendline--'
                        '36888/specs'),
                       ('wheelbase: Wikipedia (fallback; no maker figure found)',
                        'https://en.wikipedia.org/wiki/Volkswagen_Passat_(B6)')),
            note='The weakest-sourced entry: Volkswagen\'s own 2005 figures give length, width and height to the '
                 'centimetre; the rest are third-party. An estate: drawn with the long-roof (SUV) profile.'),
        VehicleModel(
            'lincoln_mkz_2017', 'Lincoln', 'MKZ', '2017 model year', 'sedan',
            length=4.925, width=1.864, height=1.476, wheelbase=2.850, front_track=1.582, rear_track=1.575,
            width_mirrors=2.116, tyre='245/45 R18', kerb_kg=1696.0,
            used_by='Baidu Apollo, Autoware, Dataspeed drive-by-wire kits and many university labs',
            sources=_s(('Lincoln 2017 MKZ technical specifications (Internet Archive copy)',
                        'http://web.archive.org/web/20231218222812/https://media.lincoln.com/content/dam/lincolnmedia/'
                        'lna/us/product/2016/17MKZ-TechSpecs.pdf'),),
            note='Inches in the sheet (193.9 × 73.4 × 58.1 in, wheelbase 112.2 in). Kerb mass: 2.0T FWD 1696 kg, '
                 'AWD 1769 kg; 3.0TT 1825–1901 kg. Tyres P245/45R18 or P245/40R19: drawn with the 18″.',
            model3d=Model3D(
                page='https://bitbucket.org/dataspeedinc/dbw_mkz_ros',
                url='https://bitbucket.org/dataspeedinc/dbw_mkz_ros/raw/master/dbw_mkz_description/meshes/mkz_17/'
                    'body.tar.gz',
                licence='BSD-3-Clause', attribution='MKZ 2017 body mesh from dbw_mkz_ros, Copyright (c) 2015-2019, '
                                                    'Dataspeed Inc. All rights reserved. BSD 3-Clause licence.',
                member='body.dae', fmt='dae', size_mb=3.5, units='m', up='+Z',
                note='The body without its wheels (Dataspeed ships them as a separate mesh): set the wheel centres '
                     'in the import wizard. 142,000 triangles.')),
        VehicleModel(
            'toyota_prius_xw50', 'Toyota', 'Prius', '4th generation XW50 (2016 model year)', 'hatch',
            length=4.539, width=1.760, height=1.476, wheelbase=2.700, ground_clearance=0.130,
            front_track=1.529, rear_track=1.539, front_overhang=0.950, rear_overhang=0.889,
            tyre='195/65 R15', kerb_kg=1395.0, used_by='Waymo\'s first fleet (earlier generation); Gazebo and ROS',
            sources=_s(('Toyota, 2016 Prius preliminary product information (as attached on PriusChat)',
                        'https://attachments.priuschat.com/attachment-files/2015/11/'
                        '97312_2016_Prius_Liftback_Product_Information_1.pdf'),),
            note='Track 1.529/1.539 m with 15″ wheels (1.509/1.519 with 17″). Kerb mass 1365–1397 kg; the Prius '
                 'Two 1395 kg.',
            model3d=Model3D(
                page='https://app.gazebosim.org/OpenRobotics/fuel/models/Prius%20Hybrid',
                url='https://fuel.gazebosim.org/1.0/OpenRobotics/models/Prius%20Hybrid.zip',
                licence='CC0-1.0', attribution='Prius Hybrid by Open Robotics (Gazebo Fuel), CC0 1.0',
                member='meshes/Hybrid.obj', fmt='obj', size_mb=1.3, units='cm', up='+Z',
                note='A light model (2,200 triangles) in centimetres, with its wheels.')),
        VehicleModel(
            'chrysler_pacifica_hybrid_2017', 'Chrysler', 'Pacifica Hybrid', '2017 model year', 'suv',
            length=5.176, width=2.022, height=1.777, wheelbase=3.089, ground_clearance=0.130,
            front_track=1.734, rear_track=1.735, front_overhang=0.960, rear_overhang=1.127, width_mirrors=2.297,
            tyre='235/65 R17', kerb_kg=2262.0, used_by='Waymo (Pacifica Hybrid fleet)',
            sources=_s(('FCA US media, 2017 Chrysler Pacifica and Pacifica Hybrid specifications',
                        'https://media.stellantisnorthamerica.com/view-spec.do?id=17213'),),
            note='The Hybrid\'s figures; kerb mass is the EPA curb weight. 235/65R17 (Premium) or 235/60R18 '
                 '(Platinum). A minivan: drawn with the SUV profile.'),
        VehicleModel(
            'hyundai_ioniq5_2021', 'Hyundai', 'IONIQ 5', 'NE, before the 2024 facelift', 'suv',
            length=4.635, width=1.890, height=1.605, wheelbase=3.000, ground_clearance=0.160,
            front_track=1.638, rear_track=1.647, tyre='235/55 R19', kerb_kg=1830.0,
            used_by='Motional (robotaxi base vehicle)',
            sources=_s(('Hyundai Suisse press kit, technical data, 29 Jun 2021',
                        'https://cdn.uc.assets.prezly.com/e605a6ea-6c97-401e-b7d5-fe13309859ec/-/inline/no/'
                        '2021-06-29_Presskit_Hyundai%20IONIQ%205_DE_Technische%20Daten.pdf'),
                       ('tyre sizes: Hyundai Japan specifications (facelift page)', 'https://www.hyundai.com/jp/ioniq5/spec')),
            note='Track 1.638/1.647 m on 19″ wheels (1.628/1.637 on 20″). Kerb mass 1830–2100 kg by version.'),
        VehicleModel(
            'kia_eniro_2019', 'Kia', 'e-Niro', 'DE (2018–2022, Europe)', 'suv',
            length=4.375, width=1.805, height=1.560, wheelbase=2.700, ground_clearance=0.155,
            front_track=1.576, rear_track=1.585, front_overhang=0.885, rear_overhang=0.790,
            tyre='215/55 R17', kerb_kg=1737.0,
            sources=_s(('Kia Spain, e-Niro technical specifications',
                        'https://press.kia.com/content/dam/kiapress/ES/mediagallery/KiaE-Niro/'
                        'Especificaciones%20T%C3%A9cnicas%20Kia%20e-Niro.pdf'),
                       ('Kia Austria press kit, Nov 2018 (tyre, unladen mass)',
                        'https://prod2-press.kia.com/content/dam/kiapress/AT/PDF2018/'
                        'Pressemappe%20Kia%20e-Niro%20November%202018.pdf')),
            note='The two press kits disagree on the track: 1.576/1.585 m (Spain, used) against 1.562/1.572 m '
                 '(Austria). Height 1.570 m with roof rails. Kerb mass 1737–1791 kg (64 kWh, unladen).'),
        VehicleModel(
            'lexus_rx450h_2020', 'Lexus', 'RX 450h', '4th generation AL20, facelift (2020–2022)', 'suv',
            length=4.890, width=1.895, height=1.685, wheelbase=2.790, ground_clearance=0.208,
            front_track=1.640, rear_track=1.630, front_overhang=1.080, rear_overhang=1.020,
            tyre='235/55 R20', kerb_kg=2100.0, used_by='Autoware\'s sample vehicle; Tier IV',
            sources=_s(('Lexus UK, RX technical specification, Jan 2022',
                        'https://media.lexus.co.uk/wp-content/uploads/sites/3/pdf/220112M-RX-Tech-Spec.pdf'),
                       ('Lexus Canada, 2020 RX and RXh specifications (ground clearance)',
                        'https://media.lexus.ca/content/dam/media-lexus/general/unassigned/20145/2020_RX_and_RX_h.pdf')),
            note='Height 1.685 m (UK) against 1.720 m (Canada, with roof rails); wheelbase 2.790 against 2.789 m. '
                 'Kerb mass 2100–2210 kg (UK).',
            model3d=Model3D(
                page='https://github.com/autowarefoundation/sample_vehicle_launch',
                url='https://raw.githubusercontent.com/autowarefoundation/sample_vehicle_launch/main/'
                    'sample_vehicle_description/mesh/lexus.dae',
                licence='Apache-2.0', attribution='lexus.dae from autowarefoundation/sample_vehicle_launch, '
                                                  'Copyright Autoware Foundation contributors, Apache License 2.0',
                fmt='dae', size_mb=15.0, units='m', up='+Y', exact=True,
                note='Its origin is not documented, and it is 1.58 m high against Lexus\'s 1.685 m: the spec check '
                     'shows the difference. 144,000 triangles.')),
        VehicleModel(
            'toyota_sienna_2021', 'Toyota', 'Sienna', '4th generation XL40 (2021 model year)', 'suv',
            length=5.169, width=1.994, height=1.740, wheelbase=3.061, ground_clearance=0.160,
            front_track=1.720, rear_track=1.740, tyre='235/65 R17', kerb_kg=2091.0,
            used_by='Pony.ai, May Mobility, Aurora (Sienna-based fleets)',
            sources=_s(('Toyota, 2021 Sienna product information (Internet Archive copy)',
                        'http://web.archive.org/web/20211026210113id_/https://toyota-cms-media.s3.amazonaws.com/'
                        'wp-content/uploads/2020/10/2021-Toyota-Sienna-Product-Info1.pdf'),
                       ('Toyota pressroom, 2021 Sienna', 'https://pressroom.toyota.com/toyota-sienna-2021/')),
            note='Length 5.169 m (product information, 203.5 in) against 5.174 m (pressroom, 203.7 in). Height '
                 '1.740 m without roof rails, 1.770 m with. Tyres P235/65R17, P235/60R18 or P235/50R20 by grade. '
                 'A minivan: drawn with the SUV profile.'),
        VehicleModel(
            'tesla_model3_2024', 'Tesla', 'Model 3', '2024 facelift ("Highland"), RWD and Long Range', 'sedan',
            length=4.720, width=1.850, height=1.440, wheelbase=2.875, ground_clearance=0.138,
            front_track=1.584, rear_track=1.584, front_overhang=0.868, rear_overhang=0.977, width_mirrors=2.089,
            tyre='235/45 R18',
            sources=_s(('Tesla Model 3 owner\'s manual, Europe, software 2023.44.30, pp. 198–199 (third-party copy)',
                        'https://tesla-info.com/doc/m3/model_3_owners_manual_europe_en_2023_44_30.pdf'),),
            note='The manual calls its dimensions approximate; it states no kerb mass. Width 1.933 m with mirrors '
                 'folded. Tyres 235/45R18 or 235/40R19.'),
        VehicleModel(
            'mercedes_sprinter_170_hr', 'Mercedes-Benz', 'Sprinter 2500 Cargo', 'VS30 (2021), 170″ wheelbase, high '
            'roof, 4×2', 'van',
            length=6.967, width=2.019, height=2.725, wheelbase=4.318, front_track=1.707, rear_track=1.732,
            front_overhang=1.021, width_mirrors=2.344, tyre='245/75 R16', cameras='8-cam',
            used_by='mapping vans and sensor-development vans',
            sources=_s(('Mercedes-Benz Vans USA, 2021 Sprinter brochure, p. 21',
                        'https://assets.mbvans.com/Mercedes-Benz-Vans/Brochures/'
                        '2021-Mercedes-Benz-Sprinter-Vans-Brochure.pdf'),),
            note='Width 2.426 m with the optional 96″ mirrors. The brochure\'s rear overhang (1.621 m) is to the end '
                 'of the frame, not the body: drawn as the body\'s length less the rest. Tyre LT245/75R16 (read from '
                 'the brochure\'s table layout). No ground clearance or kerb mass is stated.'),
        VehicleModel(
            'tata_nexon_ev_2023', 'Tata', 'Nexon.ev', '2023 facelift (30 and 40.5 kWh)', 'suv',
            length=3.994, width=1.811, height=1.616, wheelbase=2.498, ground_clearance=0.190, tyre='215/60 R16',
            sources=_s(('Tata Motors, Nexon.ev specification sheet, 14 Sep 2023',
                        'https://www.tatamotors.com/wp-content/uploads/2023/10/'
                        'press-14sep23-2-specification-sheet-1.pdf'),),
            note='Ground clearance 0.190 m unladen (Long Range; 0.205 m Medium Range). No track, overhangs or '
                 'kerb mass are stated.'),
        VehicleModel(
            'freightliner_cascadia_126', 'Freightliner', 'Cascadia 126 day cab', '6×4 tractor (2024, Australia) '
            'with a 53 ft dry van', 'truck',
            height=3.030, wheelbase=4.425, tyre='11R22.5', kerb_kg=8200.0, cameras='8-cam',
            used_by='autonomous trucking (Aurora, Kodiak, TuSimple ran Cascadia-class tractors)',
            sources=_s(('Freightliner Australia, Cascadia 126 specification sheet, 2024',
                        'https://www.freightliner.com.au/siteassets/documents/cascadia-spec-sheet-126-2024.pdf'),
                       ('Freightliner Body Builder Book (Daimler Truck Australia)',
                        'https://www.daimlertruck.com.au/siteassets/freightliner-bodybuilder-book.pdf')),
            note='Stated: bumper to back of cab 3.220 m, height 3.030 m without aero, wheelbase 4.425 m (to the '
                 'tandem\'s centre, axle spread 1.295 m), tare 8200 kg. Freightliner states no width, overall length, '
                 'overhangs or track for it: those are drawn (the trailer\'s 2.60 m width; overhangs from the '
                 'truck profile) and are not checked. 11R22.5 tyres: their radius is not in the size.',
            trailer=VehicleModel(
                'wabash_duraplate_53', 'Wabash', 'DuraPlate 53 ft dry van', 'Cell Core', 'truck',
                length=16.154, width=2.600, height=4.115,
                sources=_s(('Wabash, DuraPlate Cell Core dry van product sheet',
                            'https://onewabash.com/docs/default-source/wabash-trailer-downloads/'
                            'van-trailers-product-sheets/Dry-van-duraplate-cell-core-tech.pdf'),),
                note='Wabash states no kingpin or axle position, tare or tyres: drawn.'),
            trailer_gap=0.5),
    ]


def by_id() -> dict[str, VehicleModel]:
    return {v.id: v for v in library()}


def resolve(key: str) -> VehicleModel | None:
    """The library vehicle of an id or an old template key ('sedan', 'suv', ...)."""
    lib = by_id()
    return lib.get(key) or lib.get(ALIASES.get(key, ''))


# ---------------------------------------------------------------- bodies
def body_spec(v: VehicleModel):
    """The template body of a library vehicle: its family's profile scaled to the stated figures. Returns a BodySpec
    whose `spec` holds only stated figures and whose `drawn` note names every figure taken from the family."""
    from .bodygen import _generic
    fam = FAMILY[v.family]
    drawn = []
    L, W, H, wb = v.length, v.width, v.height, v.wheelbase
    if v.family == 'truck':
        # a tractor: no overall length stated; its length is drawn from the bumper-to-back-of-cab and the tandem
        W = W or (v.trailer.width if v.trailer else 2.5)
        fo = v.front_overhang or 1.25
        ro = v.rear_overhang or 0.95
        L = L or fo + wb + ro
        drawn += [x for x, ok in (('width', v.width), ('length', v.length), ('front overhang', v.front_overhang),
                                  ('rear overhang', v.rear_overhang)) if not ok]
    else:
        ro = v.rear_overhang
        if not ro:
            if v.front_overhang:
                ro = L - wb - v.front_overhang
            else:
                ro = (L - wb) * fam['rear_share']
                drawn.append('overhangs')
    gc = v.ground_clearance or fam['clearance']
    if not v.ground_clearance:
        drawn.append('ground clearance')
    wr = tyre_radius(v.tyre)
    if not wr:
        wr = {'truck': 0.52}.get(v.family, 0.32)
        drawn.append('wheel radius')
    kind = {'hatch': 'hatch', 'truck': 'box'}.get(v.family, v.family)
    spec = _generic(v.id, v.title, L, W, H, wb, gc, ro, wr, kind, v.cameras)
    if v.front_track:
        # the drawn tyre is 0.185 m × (r / 0.29) wide, its outer face 15 mm beyond track_half: its centre on the
        # stated track, inside the body's width
        spec.track_half = min(v.front_track / 2 - 0.015 + 0.0925 * wr / 0.29, W / 2 - 0.02)
    else:
        drawn.append('track')
    spec.spec = v.stated()
    if v.trailer is not None:
        t = v.trailer
        tl = _generic(t.id, t.title, t.length, t.width, t.height, t.length * 0.62, 0.35, t.length * 0.08, 0.5,
                      'box', v.cameras)
        tl.top = [t.height * 0.98] * len(tl.u)
        tl.glass_u = (2.0, 2.0)
        tl.spec = t.stated()
        spec.second_body = tl
        spec.second_gap = v.trailer_gap
        drawn.append('the trailer\'s axles and kingpin')
    spec.drawn = list(drawn)
    spec.name = v.title
    return spec


# ---------------------------------------------------------------- 3D models
MAX_DOWNLOAD_MB = 64.0


class DownloadError(Exception):
    pass


def cache_dir() -> str:
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'Vantage', 'vehicles')


def _safe_member(names: list[str], want: str) -> str:
    """The archive member `want` (or the only mesh in it), refusing absolute paths and '..'."""
    ok = [n for n in names if not (n.startswith('/') or '..' in n.replace('\\', '/').split('/'))]
    hit = [n for n in ok if n == want or n.endswith('/' + want)]
    if hit:
        return hit[0]
    meshes = [n for n in ok if os.path.splitext(n)[1].lower() in ('.obj', '.dae', '.glb', '.gltf', '.stl', '.ply')]
    if len(meshes) == 1:
        return meshes[0]
    raise DownloadError(f'{want or "a mesh"} is not in the archive')


def _extract(archive: str, member: str, out_dir: str) -> str:
    """Extract the archive's files next to its mesh (textures, .mtl) into out_dir, safely; returns the mesh path."""
    limit = int(MAX_DOWNLOAD_MB * 4 * 1e6)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            name = _safe_member([i.filename for i in infos], member)
            if sum(i.file_size for i in infos) > limit:
                raise DownloadError('the archive unpacks to more than Vantage accepts')
            for i in infos:
                n = i.filename
                if n.startswith('/') or '..' in n.replace('\\', '/').split('/'):
                    continue
                dst = os.path.join(out_dir, *n.split('/'))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(i) as src, open(dst, 'wb') as f:
                    shutil.copyfileobj(src, f)
            return os.path.join(out_dir, *name.split('/'))
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            files = [m for m in t.getmembers() if m.isfile()]
            name = _safe_member([m.name for m in files], member)
            if sum(m.size for m in files) > limit:
                raise DownloadError('the archive unpacks to more than Vantage accepts')
            for m in files:
                n = m.name
                if n.startswith('/') or '..' in n.split('/'):
                    continue
                dst = os.path.join(out_dir, *n.split('/'))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    continue
                with src, open(dst, 'wb') as f:
                    shutil.copyfileobj(src, f)
            return os.path.join(out_dir, *name.split('/'))
    raise DownloadError('the download is not an archive Vantage reads')


def download_model(v: VehicleModel, folder: str | None = None, opener=None, progress=None) -> str:
    """Download the vehicle's 3D model from its publisher into the cache (once) and return the mesh file's path.
    opener(url) -> a file-like response (urllib's by default); progress(fraction) as it arrives."""
    m = v.model3d
    if m is None:
        raise DownloadError(f'{v.title} has no 3D model Vantage may download')
    folder = os.path.join(folder or cache_dir(), v.id)
    os.makedirs(folder, exist_ok=True)
    done = os.path.join(folder, '.mesh')
    if os.path.exists(done):
        path = open(done, encoding='utf-8').read().strip()
        if path and os.path.exists(os.path.join(folder, path)):
            return os.path.join(folder, path)
    name = os.path.basename(m.url.split('?')[0]) or 'download'
    raw = os.path.join(folder, 'download', name)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    op = opener or (lambda url: urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'Vantage vehicle library'}), timeout=60))
    limit = int(MAX_DOWNLOAD_MB * 1e6)
    got, h = 0, hashlib.sha256()
    try:
        with op(m.url) as r, open(raw + '.part', 'wb') as f:
            total = int(getattr(r, 'headers', {}).get('Content-Length', 0) or 0) if hasattr(r, 'headers') else 0
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                got += len(b)
                if got > limit:
                    raise DownloadError(f'the model is larger than {MAX_DOWNLOAD_MB:.0f} MB')
                h.update(b)
                f.write(b)
                if progress and total:
                    progress(min(got / total, 1.0))
    except DownloadError:
        _rm(raw + '.part')
        raise
    except Exception as e:                       # network, HTTP, timeouts: a short message
        _rm(raw + '.part')
        raise DownloadError(f'could not download {m.url}: {e}') from None
    os.replace(raw + '.part', raw)
    mesh_dir = os.path.join(folder, 'mesh')
    shutil.rmtree(mesh_dir, ignore_errors=True)
    os.makedirs(mesh_dir)
    if m.member:
        path = _extract(raw, m.member, mesh_dir)
    else:
        path = os.path.join(mesh_dir, name)
        shutil.copyfile(raw, path)
    with open(os.path.join(folder, 'LICENCE.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{m.attribution}\nLicence: {m.licence}\nSource: {m.page}\nFile: {m.url}\nSHA-256: {h.hexdigest()}\n'
                f'Downloaded by Vantage for {v.title}.\n')
    with open(done, 'w', encoding='utf-8') as f:
        f.write(os.path.relpath(path, folder))
    return path


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def credit(v: VehicleModel) -> str:
    """The design-log line for a project whose body is the vehicle's downloaded model."""
    m = v.model3d
    return f'{v.title} 3D model: {m.attribution}. Licence {m.licence}; source {m.page}.' if m else ''
