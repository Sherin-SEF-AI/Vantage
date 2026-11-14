"""The watchdog's local status page, read-only, on the standard library's HTTP server.

GET /             a page for a browser on the car's network (refreshes itself every 2 s)
GET /status.json  the current status (schema vantage.edge_status/1)
GET /alerts.json  every alert raised since the start (active and cleared)
GET /metrics      Prometheus text exposition (vantage_edge_* gauges per camera)
GET /healthz      200 "ok" while the watchdog runs (for systemd or a supervisor)
Nothing can be changed through it. It binds 127.0.0.1 by default; bind the car's network address to reach it from a
laptop on that network (there is no authentication: keep it off networks you do not control).
"""
from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE_COLOUR = {'ok': '#199e70', 'warning': '#c98500', 'fault': '#d0433a', 'starting': '#888', 'off': '#888'}


def metrics(st: dict) -> str:
    out = ['# HELP vantage_edge_fps Measured frame rate per camera', '# TYPE vantage_edge_fps gauge']
    lines = {'fps': [], 'dropped_share': [], 'age_s': [], 'frozen_frames': [], 'mean': [], 'knock': [], 'ok': []}
    for tag, c in (st.get('cameras') or {}).items():
        lab = f'{{rig="{st.get("rig", "")}",camera="{tag}"}}'
        for k in ('fps', 'dropped_share', 'age_s', 'frozen_frames', 'mean'):
            v = c.get(k)
            if isinstance(v, (int, float)) and v is not None:
                lines[k].append(f'vantage_edge_{k}{lab} {float(v):g}')
        kn = (c.get('knock') or {}).get('state')
        if kn is not None:
            lines['knock'].append(f'vantage_edge_knock{lab} {1 if kn == "knock" else 0}')
        lines['ok'].append(f'vantage_edge_camera_ok{lab} {1 if c.get("status") == "ok" else 0}')
    out += lines['fps']
    for k, help_ in (('dropped_share', 'Share of frames dropped in the window'), ('age_s', 'Age of the newest frame'),
                     ('frozen_frames', 'Identical consecutive frames'), ('mean', 'Mean grey level')):
        out += [f'# HELP vantage_edge_{k} {help_}', f'# TYPE vantage_edge_{k} gauge'] + lines[k]
    out += ['# HELP vantage_edge_knock 1 when the camera has been knocked', '# TYPE vantage_edge_knock gauge']
    out += lines['knock']
    out += ['# HELP vantage_edge_camera_ok 1 when the camera passes every check', '# TYPE vantage_edge_camera_ok gauge']
    out += lines['ok']
    cnt = st.get('counts') or {}
    out += ['# HELP vantage_edge_alerts Alerts raised now', '# TYPE vantage_edge_alerts gauge',
            f'vantage_edge_alerts{{severity="fault"}} {cnt.get("faults", 0)}',
            f'vantage_edge_alerts{{severity="warning"}} {cnt.get("warnings", 0)}']
    p = st.get('process') or {}
    if p.get('rss_mb') is not None:
        out += ['# HELP vantage_edge_rss_megabytes Resident memory of the watchdog',
                '# TYPE vantage_edge_rss_megabytes gauge', f'vantage_edge_rss_megabytes {p["rss_mb"]:g}']
    return '\n'.join(out) + '\n'


def page(st: dict) -> str:
    e = html.escape
    rows = []
    for tag, c in (st.get('cameras') or {}).items():
        col = STATE_COLOUR.get(c.get('status'), '#888')
        k = c.get('knock') or {}
        rows.append(
            f'<tr><td><b>{e(tag)}</b></td><td style="color:{col}">{e(str(c.get("status", "")))}</td>'
            f'<td>{(c.get("fps") or 0):.1f} / {c.get("nominal_fps", 0):g}</td>'
            f'<td>{100 * (c.get("dropped_share") or 0):.1f} %</td><td>{c.get("frozen_frames", 0)}</td>'
            f'<td>{(c.get("mean") or 0):.0f}</td><td>{e(str(k.get("state", "")))}'
            + (f' {k.get("size_deg"):.2f}°' if k.get('size_deg') else '') + '</td>'
            f'<td>{e("; ".join(c.get("reasons") or []))}</td></tr>')
    al = ''.join(f'<li><b>{e(a["severity"])}</b> {e(a["camera"])} · {e(a["check"])}: {e(a["message"])}</li>'
                 for a in st.get('alerts') or []) or '<li>none</li>'
    p = st.get('process') or {}
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{e(st.get("rig", "rig"))} · Vantage watchdog</title>'
            '<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fafaf8;color:#222}'
            'table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}'
            '@media(prefers-color-scheme:dark){body{background:#1b1b1a;color:#ddd}td,th{border-color:#333}}</style>'
            f'</head><body><h2>{e(st.get("rig", "rig"))}</h2>'
            f'<p>{st.get("clock_s", 0):.0f} s · CPU {p.get("cpu_percent_of_one_core", 0)} % of one core · '
            f'{(p.get("rss_mb") or 0):.0f} MB</p><h3>Alerts</h3><ul>{al}</ul>'
            '<table><tr><th>Camera</th><th>State</th><th>fps</th><th>Dropped</th><th>Frozen</th><th>Mean</th>'
            f'<th>Knock</th><th>Reasons</th></tr>{"".join(rows)}</table>'
            '<p><a href="status.json">status.json</a> · <a href="alerts.json">alerts.json</a> · '
            '<a href="metrics">metrics</a></p></body></html>')


def serve(wd, bind='127.0.0.1', port=8088) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            st = wd.status()
            if path in ('/', '/index.html'):
                self._send(200, page(st), 'text/html; charset=utf-8')
            elif path == '/status.json':
                self._send(200, json.dumps(st, default=str), 'application/json')
            elif path == '/alerts.json':
                self._send(200, json.dumps({'rig': wd.cfg['rig'], 'alerts': wd.book.to_list()}, default=str),
                           'application/json')
            elif path == '/metrics':
                self._send(200, metrics(st), 'text/plain; version=0.0.4')
            elif path == '/healthz':
                self._send(200 if not wd.stop_ev.is_set() else 503, 'ok\n' if not wd.stop_ev.is_set() else
                           'stopping\n', 'text/plain')
            else:
                self._send(404, 'not found\n', 'text/plain')

        def do_POST(self):
            self._send(405, 'read-only\n', 'text/plain')

    srv = ThreadingHTTPServer((bind, int(port)), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name='edge-http', daemon=True).start()
    return srv
