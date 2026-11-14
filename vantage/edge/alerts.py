"""Alerts: which conditions are raised, and who is told.

AlertBook debounces each (camera, check) condition: it is raised once it has been seen continuously for raise_s, and
cleared once it has been absent for clear_s, so a single late frame or one noisy estimate does not page anyone. Each
raise and clear is an event with a running number (no random ids), the time on the watchdog's clock (the recording's
time in a dry run) and the wall-clock time.

Notifier sends every event to the configured channels, none of which needs a cloud service: a rotating log file, an
MQTT broker (vantage.edge.mqtt), a webhook (an HTTP POST of the event as JSON, on its own thread so a slow server never
holds the watchdog up) and a GPIO output line for a buzzer or a lamp (vantage.edge.gpio), which is on while any fault is
raised. The line only signals: nothing the watchdog drives commands the vehicle.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import logging
import logging.handlers
import os
import queue
import re
import threading
import time
import urllib.request

SEVERITY_RANK = {'warning': 1, 'fault': 2}


def wall() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


class AlertBook:
    def __init__(self, raise_s=1.0, clear_s=3.0, keep=500):
        self.raise_s, self.clear_s = float(raise_s), float(clear_s)
        self.cond = {}                  # (camera, check) -> {'since', 'last_seen', 'severity', 'message'}
        self.active = {}                # (camera, check) -> alert dict
        self.history = collections.deque(maxlen=keep)
        self.counter = 0

    def update(self, now: float, seen: dict) -> list:
        """seen: {(camera, check): (severity, message)} of the conditions present now. Returns the events (raised and
        cleared alerts) this update produced."""
        events = []
        for key, (sev, msg) in seen.items():
            c = self.cond.get(key)
            if c is None:
                c = self.cond[key] = {'since': now, 'severity': sev, 'message': msg}
            c['last_seen'], c['message'] = now, msg
            if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(c['severity'], 0):
                c['severity'] = sev
            a = self.active.get(key)
            if a is None and now - c['since'] >= self.raise_s:
                self.counter += 1
                a = {'id': self.counter, 'camera': key[0], 'check': key[1], 'severity': c['severity'],
                     'message': msg, 'raised_t': now, 'raised_wall': wall(), 'cleared_t': None, 'active': True}
                self.active[key] = a
                self.history.append(a)
                events.append(dict(a, event='raised'))
            elif a is not None:
                a['message'] = msg
                if SEVERITY_RANK.get(sev, 0) > SEVERITY_RANK.get(a['severity'], 0):
                    a['severity'] = sev
                    events.append(dict(a, event='escalated'))
        for key in list(self.cond):
            if key in seen:
                continue
            gone = now - self.cond[key]['last_seen']
            if key in self.active:
                if gone >= self.clear_s:                 # a raised alert clears once absent for clear_s
                    self.cond.pop(key)
                    a = self.active.pop(key)
                    a['active'], a['cleared_t'], a['cleared_wall'] = False, now, wall()
                    events.append(dict(a, event='cleared'))
            elif gone >= min(self.clear_s, 0.5):         # a blip that was never raised is forgotten
                self.cond.pop(key)
        return events

    def any_fault(self) -> bool:
        return any(a['severity'] == 'fault' for a in self.active.values())

    def to_list(self, active_only=False) -> list:
        src = self.active.values() if active_only else self.history
        return [dict(a) for a in src]


class Notifier:
    """Fans each alert event out to the configured channels; every channel's failure is logged, never raised."""

    def __init__(self, cfg: dict, rig: str):
        al = cfg
        self.rig = rig
        self.log = logging.getLogger(f'vantage.edge.{re.sub(r"[^A-Za-z0-9_]", "_", rig)}')
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        for h in list(self.log.handlers):
            self.log.removeHandler(h)
            h.close()
        path = al['log'].get('path') or ''
        self.log_path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            h = logging.handlers.RotatingFileHandler(path, maxBytes=int(al['log']['max_mb'] * 1e6),
                                                     backupCount=int(al['log']['keep']), encoding='utf-8')
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.log.addHandler(h)
        self.mqtt = None
        m = al['mqtt']
        self.topic = (m.get('topic') or 'vantage/edge').rstrip('/')
        if m.get('host'):
            from .mqtt import MqttPublisher
            cid = 'vantage-edge-' + re.sub(r'[^A-Za-z0-9_-]', '-', rig)[:40]
            self.mqtt = MqttPublisher(m['host'], m.get('port', 1883), cid, m.get('user', ''),
                                      m.get('keepalive_s', 60), log=self.log.warning)
        self.webhook = al['webhook'].get('url') or ''
        self.webhook_timeout = float(al['webhook'].get('timeout_s', 3.0))
        self.webhook_sent = 0
        self.webhook_failed = 0
        self.webhook_error = ''
        self._wq = queue.Queue(maxsize=200)
        if self.webhook:
            threading.Thread(target=self._webhook_run, name='edge-webhook', daemon=True).start()
        from .gpio import make
        self.gpio = make(al['gpio'], log=self.log.warning)
        self.events_sent = 0

    def info(self, msg):
        self.log.info(msg)

    def send(self, events: list, any_fault: bool):
        for e in events:
            self.events_sent += 1
            word = {'raised': 'ALERT', 'cleared': 'CLEAR', 'escalated': 'ALERT'}.get(e['event'], e['event'])
            line = f'{word} {e["camera"]} {e["check"]} {e["severity"]}: {e["message"]}'
            (self.log.warning if e['event'] != 'cleared' else self.log.info)(line)
            doc = dict(e, rig=self.rig)
            payload = json.dumps(doc, default=str)
            if self.mqtt is not None:
                self.mqtt.publish(f'{self.topic}/alerts', payload)
            if self.webhook:
                try:
                    self._wq.put_nowait(payload)
                except queue.Full:
                    self.webhook_failed += 1
        if self.gpio is not None:
            self.gpio.set_alarm(any_fault)

    def status(self, doc: dict):
        """The periodic status (retained, so a client that connects later gets the latest)."""
        if self.mqtt is not None:
            self.mqtt.publish(f'{self.topic}/status', json.dumps(doc, default=str), retain=True)

    def _webhook_run(self):
        while True:
            payload = self._wq.get()
            if payload is None:
                return
            try:
                req = urllib.request.Request(self.webhook, data=payload.encode('utf-8'), method='POST',
                                             headers={'Content-Type': 'application/json',
                                                      'User-Agent': 'vantage-edge'})
                with urllib.request.urlopen(req, timeout=self.webhook_timeout) as r:
                    r.read()
                self.webhook_sent += 1
            except Exception as e:                       # noqa: BLE001 - a notification never stops the watchdog
                self.webhook_failed += 1
                self.webhook_error = str(e)
                self.log.warning(f'webhook {self.webhook}: {e}')
            finally:
                self._wq.task_done()

    def channels(self) -> dict:
        return {'log': self.log_path or None,
                'mqtt': None if self.mqtt is None else {'broker': f'{self.mqtt.host}:{self.mqtt.port}',
                                                        'connected': self.mqtt.connected, 'sent': self.mqtt.sent,
                                                        'dropped': self.mqtt.dropped, 'error': self.mqtt.error},
                'webhook': None if not self.webhook else {'url': self.webhook, 'sent': self.webhook_sent,
                                                          'failed': self.webhook_failed, 'error': self.webhook_error},
                'gpio': None if self.gpio is None else self.gpio.describe()}

    def flush(self, timeout=3.0):
        t = time.monotonic() + timeout
        if self.webhook:
            while self._wq.unfinished_tasks and time.monotonic() < t:
                time.sleep(0.02)
        if self.mqtt is not None:
            self.mqtt.flush(max(0.1, t - time.monotonic()))

    def close(self):
        self.flush(2.0)
        if self.webhook:
            try:
                self._wq.put_nowait(None)
            except queue.Full:
                pass
        if self.mqtt is not None:
            self.mqtt.close()
        if self.gpio is not None:
            self.gpio.close()
        for h in list(self.log.handlers):
            h.flush()
            h.close()
            self.log.removeHandler(h)
