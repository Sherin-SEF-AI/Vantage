"""Application context shared by every view: store, analysis, tools, overlays, preferences."""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, QSettings

from .store import ProjectStore, AnalysisController

OVERLAYS = [('frustums', 'Frustums', 'frustum', True), ('footprints', 'Footprints', 'coverage', True),
            ('occlusion', 'Occlusion rays', 'occlusion', False), ('seams', 'Seams', 'seam', True),
            ('roi', 'ROI', 'roi', True), ('targets', 'Calibration targets', 'checker', False),
            ('dimensions', 'Dimensions', 'measure', False), ('grid', 'Grid', 'snap-grid', True)]
HEATMAPS = [('none', 'None'), ('coverage', 'Coverage'), ('density', 'Density'), ('redundancy', 'Redundancy')]
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
