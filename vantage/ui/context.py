"""Application context shared by every view: store, analysis, tools, overlays, preferences."""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, QSettings

from .store import ProjectStore, AnalysisController

OVERLAYS = [('frustums', 'Frustums', 'frustum', True), ('footprints', 'Footprints', 'coverage', True),
            ('camlight', 'Camera light', 'lens', False), ('shadows', 'Shadows and lighting', 'shading', True),
            ('occlusion', 'Occlusion rays', 'occlusion', False), ('seams', 'Seams', 'seam', True),
            ('roi', 'ROI', 'roi', True), ('targets', 'Calibration targets', 'checker', False),
            ('dimensions', 'Dimensions', 'measure', False), ('grid', 'Grid', 'snap-grid', True),
            ('labels', 'Labels', 'pin', True),                   # camera tags and target names on plates in 3D
            ('volume', 'Blind volume', 'blind', False), ('guidelines', 'Parking guidelines', 'axle', False),
            ('sensors', 'Lidar rings and radar fields', 'sensor', True),
            ('lidarpts', 'Lidar points', 'density', False),      # one simulated sweep per lidar
            ('gnsssky', 'GNSS sky', 'sun', False),               # the sky each antenna sees, as a dome over it
            ('brackets', 'Mount brackets', 'bracket', True),     # generated brackets on the vehicle
            ('critical', 'Critical targets', 'target', False),   # Analysis ▸ Critical targets, in the plan and 3D
            ('harness', 'Cable harness', 'cable', False),        # the routed cables, in the plan and 3D
            ('wipers', 'Wiper sweep', 'layers', False),          # Analysis ▸ Materials: the wipers on the windscreen
            ('clocktree', 'Timestamp error (lidar points × 50)', 'sync', False)]   # Analysis ▸ Clock tree and time sync
HEATMAPS = [('none', 'None'), ('coverage', 'Coverage'), ('density', 'Density'), ('redundancy', 'Redundancy'),
            ('stitched', 'Objects')]       # objects standing in the stitched bird's-eye view
TOOLS = [('select', 'Select', 'select', 'V'), ('move', 'Move', 'move', 'G'), ('rotate', 'Rotate', 'rotate', 'R'),
         ('place', 'Place camera on surface', 'place', 'C'), ('measure', 'Measure', 'measure', 'M')]


class AppContext(QObject):
    overlays_changed = pyqtSignal()
    tool_changed = pyqtSignal(str)
    cursor_world = pyqtSignal(object)       # (x, y, z) or None
    shading_changed = pyqtSignal(str)
    focus_request = pyqtSignal(object)      # ('frame', bbox) etc.
    message = pyqtSignal(str, int)
    library_pick = pyqtSignal(str, str)     # module id, lens id

    def __init__(self, store: ProjectStore):
        super().__init__()
        self.store = store
        self.analysis = AnalysisController(store)
        self.settings = QSettings('Vantage', 'Vantage')
        self.overlays = {k: bool(self.settings.value(f'overlay/{k}', d, type=bool)) for k, _, _, d in OVERLAYS}
        self.heatmap = 'none'
        self.tool = 'select'
        self.shading = 'solid'
        self.snap = 'surface'               # surface | grid | off
        self.place_lens = ('sturdecam31', 'h05r2')
        self.nav_style = self.settings.value('nav/style', 'blender')
        self.photo_shading = False
        self.isolate: set = set()
        self.window = None
        from .volumeview import VolumeController
        self.volume = VolumeController(self)    # the 3D blind volume (its page and the 3D overlay)
        store.changed.connect(self._after_edit)
        store.project_replaced.connect(self._after_edit)

    def _after_edit(self, *_):
        """Keep view state pointing at things that exist: an edit can delete cameras or switch the variant."""
        uids = {c.uid for c in self.store.variant.cameras}
        if self.isolate and not self.isolate <= uids:
            self.isolate = self.isolate & uids
            self.overlays_changed.emit()
        dead = [u for u in self.store.sel_cams if u not in uids]
        if dead:
            self.store.select_cams([u for u in self.store.sel_cams if u in uids])
        p = self.store.project
        mid, lid = self.place_lens
        if lid not in p.lenses or mid not in p.modules:
            m = next(iter(p.modules), mid)
            lens = next((k for k, l in p.lenses.items() if l.module_id in ('', m)), next(iter(p.lenses), lid))
            self.place_lens = (m, lens)

    def set_overlay(self, key, on):
        self.overlays[key] = on
        self.settings.setValue(f'overlay/{key}', on)
        self.overlays_changed.emit()

    def set_heatmap(self, key):
        self.heatmap = key
        self.overlays_changed.emit()

    def set_tool(self, tool):
        if tool != self.tool:
            self.tool = tool
            self.tool_changed.emit(tool)

    def set_shading(self, s):
        self.shading = s
        self.shading_changed.emit(s)
        self.overlays_changed.emit()

    def msg(self, text, ms=3000):
        self.message.emit(text, ms)
