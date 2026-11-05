"""Regions around the vehicle: the sector × range grid every perception figure is reported on, so a rig's detection,
its learned model, its tolerance study and its risk weighting all name the same places ("left · 40–60 m").

Sectors are measured about the vehicle's vertical axis from straight ahead, positive to the left (ISO 8855: x forward,
y left), from the reference point below; eight sectors of 45° by default, the first centred on straight ahead. Range is
the horizontal distance from the same point. The reference point is the middle of the vehicle's footprint along x (the
body mesh's extent: default_grid; or length and rear overhang: RegionGrid.for_vehicle), on the centre line, else the
rear-axle centre (the vehicle frame's origin): a rig's regions are then the same whichever camera sees them, and two
rigs on the same vehicle are compared on the same ground. Region ids are 'S<sector>R<range bin>'.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'vantage.regions/1'
SECTOR_NAMES_8 = ('front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right', 'right', 'front-right')
SECTOR_NAMES_4 = ('front', 'left', 'rear', 'right')
DEFAULT_RANGE_EDGES = (10.0, 20.0, 40.0, 60.0, 80.0)          # bins 0–10, 10–20, 20–40, 40–60, 60–80, 80+ m


def _fmt(x):
    return f'{x:g}'


@dataclass
class RegionGrid:
    sectors: int = 8                                       # 4 or 8 (named), or any n ≥ 1 (numbered by bearing)
    range_edges: tuple = DEFAULT_RANGE_EDGES               # inner edges in metres, increasing; the last bin is open
    max_range_m: float = 0.0                               # > 0: objects beyond are outside every region
    origin_xy: tuple = (0.0, 0.0)                          # the reference point in the vehicle frame (m)
    extra: dict = field(default_factory=dict)

    @staticmethod
    def for_vehicle(length_m=None, rear_overhang_m=None, **kw) -> 'RegionGrid':
        """The grid about the footprint's middle: x = length / 2 − rear overhang (ahead of the rear axle)."""
        g = RegionGrid(**kw)
        if length_m and rear_overhang_m is not None:
            g.origin_xy = (float(length_m) / 2.0 - float(rear_overhang_m), 0.0)
        return g

    def to_dict(self):
        return {'schema': SCHEMA, 'sectors': int(self.sectors), 'range_edges': [float(e) for e in self.range_edges],
                'max_range_m': float(self.max_range_m), 'origin_xy': [float(v) for v in self.origin_xy],
                'frame': 'vehicle (ISO 8855, rear axle)', 'bearing': 'degrees from straight ahead, positive left'}

    @staticmethod
    def from_dict(d) -> 'RegionGrid':
        d = d if isinstance(d, dict) else {}
        g = RegionGrid()
        try:
            n = int(d.get('sectors', g.sectors))
            g.sectors = min(max(n, 1), 36)
        except (TypeError, ValueError):
            pass
        try:
            e = sorted({float(x) for x in d.get('range_edges', g.range_edges) if 0 < float(x) < 10000})
            g.range_edges = tuple(e) if e else g.range_edges
        except (TypeError, ValueError):
            pass
        try:
            g.max_range_m = max(0.0, float(d.get('max_range_m', 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            o = d.get('origin_xy', g.origin_xy)
            g.origin_xy = (float(o[0]), float(o[1]))
        except (TypeError, ValueError, IndexError):
            pass
        return g

    # ---- names
    def sector_names(self) -> list:
        if self.sectors == 8:
            return list(SECTOR_NAMES_8)
        if self.sectors == 4:
            return list(SECTOR_NAMES_4)
        w = 360.0 / self.sectors
        return [f'{_fmt(round(i * w, 1))}°' for i in range(self.sectors)]

    def range_names(self, unit='m') -> list:
        e = [0.0] + [float(x) for x in self.range_edges]
        out = [f'{_fmt(e[i])}–{_fmt(e[i + 1])} {unit}' for i in range(len(e) - 1)]
        last = f'{_fmt(e[-1])}–{_fmt(self.max_range_m)} {unit}' if self.max_range_m > e[-1] else f'{_fmt(e[-1])}+ {unit}'
        return out + [last]

    @property
    def n_ranges(self) -> int:
        return len(self.range_edges) + 1

    @property
    def n_regions(self) -> int:
        return self.sectors * self.n_ranges

    def region_id(self, s: int, r: int) -> str:
        """A stable id: 'S<sector>R<range bin>' (the export key; region_name gives the words)."""
        return f'S{int(s)}R{int(r)}'

    def region_name(self, s: int, r: int) -> str:
        return f'{self.sector_names()[int(s)]} · {self.range_names()[int(r)]}'

    def parse_id(self, rid: str):
        try:
            a, b = rid[1:].split('R')
            s, r = int(a), int(b)
        except (ValueError, AttributeError):
            return None
        return (s, r) if 0 <= s < self.sectors and 0 <= r < self.n_ranges else None

    def regions(self):
        """Every (sector, range bin, id, name), sectors outer."""
        return [(s, r, self.region_id(s, r), self.region_name(s, r))
                for s in range(self.sectors) for r in range(self.n_ranges)]

    # ---- assignment
    def bearing_deg(self, x, y):
        x = np.asarray(x, float) - self.origin_xy[0]
        y = np.asarray(y, float) - self.origin_xy[1]
        return np.degrees(np.arctan2(y, x))

    def range_m(self, x, y):
        x = np.asarray(x, float) - self.origin_xy[0]
        y = np.asarray(y, float) - self.origin_xy[1]
        return np.hypot(x, y)

    def sector_of(self, x, y):
        """Sector index of points (vehicle frame, m); sector 0 is centred on straight ahead."""
        w = 360.0 / self.sectors
        b = (self.bearing_deg(x, y) + w / 2.0) % 360.0
        return np.minimum((b // w).astype(int), self.sectors - 1)

    def range_bin_of(self, x, y):
        """Range-bin index; −1 beyond max_range_m (when set) or for non-finite points."""
        r = self.range_m(x, y)
        k = np.searchsorted(np.asarray(self.range_edges, float), r, side='right').astype(int)
        bad = ~np.isfinite(r)
        if self.max_range_m > 0:
            bad |= r > self.max_range_m
        return np.where(bad, -1, k)

    def assign(self, x, y):
        """(sector, range bin) of points; range bin −1 means outside every region."""
        return self.sector_of(x, y), self.range_bin_of(x, y)

    def centre_xy(self, s: int, r: int):
        """A representative point of a region (its bearing's middle, its range bin's middle; the open bin at its
        edge + 10 m): what a model is evaluated at when it needs one point per region."""
        w = 360.0 / self.sectors
        b = np.radians(s * w)
        e = [0.0] + [float(x) for x in self.range_edges]
        rr = (e[r] + e[r + 1]) / 2.0 if r + 1 < len(e) else (
            (e[-1] + self.max_range_m) / 2.0 if self.max_range_m > e[-1] else e[-1] + 10.0)
        return (self.origin_xy[0] + rr * np.cos(b), self.origin_xy[1] + rr * np.sin(b))

    def area_m2(self, s: int, r: int) -> float:
        """Ground area of a region (m²); the open last bin is counted to max_range_m, or 20 m beyond its edge."""
        e = [0.0] + [float(x) for x in self.range_edges]
        r0 = e[r]
        r1 = e[r + 1] if r + 1 < len(e) else (self.max_range_m if self.max_range_m > e[-1] else e[-1] + 20.0)
        return float(np.pi * (r1 ** 2 - r0 ** 2) / self.sectors)


def default_grid(project=None, variant=None) -> RegionGrid:
    """The grid about the middle of the project's vehicle footprint (its mesh's extent along x, in the vehicle frame),
    else about the rear axle."""
    try:
        v = project.vehicle.mesh.vertices
        x0, x1 = float(v[:, 0].min()), float(v[:, 0].max())
        if x1 - x0 > 0.5:
            return RegionGrid(origin_xy=((x0 + x1) / 2.0, 0.0))
    except Exception:
        pass
    return RegionGrid()
