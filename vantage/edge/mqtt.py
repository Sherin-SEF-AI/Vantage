"""A minimal MQTT 3.1.1 publisher (QoS 0) on the standard library: CONNECT, CONNACK, PUBLISH, PINGREQ and DISCONNECT
(OASIS MQTT Version 3.1.1, 29 October 2014, sections 3.1-3.3, 3.12 and 3.14). Enough to notify a broker on the car's
network or the depot's (Mosquitto, EMQX, a fleet's own) without a third-party client library. No TLS: put the broker on
the car's own network or tunnel it; the password, when one is needed, comes from the environment variable
VANTAGE_EDGE_MQTT_PASSWORD so it never sits in the configuration file.

Messages are queued (at most QUEUE_MAX; the oldest is dropped when the broker is unreachable for long) and sent by one
background thread, which connects on demand and reconnects with a growing back-off (1, 2, 4 … 60 s).
"""
from __future__ import annotations

import collections
import os
import socket
import struct
import threading
import time

QUEUE_MAX = 200


def _varlen(n: int) -> bytes:
    """MQTT's remaining-length encoding (1-4 bytes, 7 bits each, high bit = more)."""
    out = bytearray()
    while True:
        b, n = n % 128, n // 128
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _s(text: str) -> bytes:
    b = text.encode('utf-8')
    return struct.pack('>H', len(b)) + b


def connect_packet(client_id: str, keepalive: int = 60, user: str = '', password: str = '') -> bytes:
    flags = 0x02                                       # clean session
    payload = _s(client_id)
    if user:
        flags |= 0x80
        payload += _s(user)
        if password:
            flags |= 0x40
            payload += _s(password)
    var = _s('MQTT') + bytes([4, flags]) + struct.pack('>H', int(keepalive))
    body = var + payload
    return bytes([0x10]) + _varlen(len(body)) + body


def publish_packet(topic: str, payload: bytes, retain: bool = False) -> bytes:
    body = _s(topic) + payload                         # QoS 0: no packet identifier
    return bytes([0x30 | (0x01 if retain else 0)]) + _varlen(len(body)) + body


PINGREQ = b'\xc0\x00'
DISCONNECT = b'\xe0\x00'


def read_packet(sock) -> tuple:
    """(first byte, body) of one packet from a socket (used by the tests' broker and for the CONNACK)."""
    def need(n):
        buf = b''
        while len(buf) < n:
            c = sock.recv(n - len(buf))
            if not c:
                raise ConnectionError('closed')
            buf += c
        return buf
    h = need(1)[0]
    mult, n = 1, 0
    while True:
        b = need(1)[0]
        n += (b & 0x7F) * mult
        if not b & 0x80:
            break
        mult *= 128
    return h, need(n) if n else b''


class MqttPublisher:
    def __init__(self, host: str, port: int = 1883, client_id: str = 'vantage-edge', user: str = '',
                 keepalive: int = 60, log=None):
        self.host, self.port, self.client_id, self.user = host, int(port), client_id, user
        self.keepalive = max(5, int(keepalive))
        self.password = os.environ.get('VANTAGE_EDGE_MQTT_PASSWORD', '')
        self.q = collections.deque(maxlen=QUEUE_MAX)
        self.cv = threading.Condition()
        self.sock = None
        self.stop_ev = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.error = ''
        self.connected = False
        self.log = log or (lambda m: None)
        self.thread = threading.Thread(target=self._run, name='edge-mqtt', daemon=True)
        self.thread.start()

    def publish(self, topic: str, payload: bytes | str, retain: bool = False):
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        with self.cv:
            if len(self.q) == self.q.maxlen:
                self.dropped += 1
            self.q.append((topic, payload, retain))
            self.cv.notify()

    def _connect(self):
        s = socket.create_connection((self.host, self.port), timeout=5.0)
        s.sendall(connect_packet(self.client_id, self.keepalive, self.user, self.password))
        h, body = read_packet(s)
        if h >> 4 != 2 or len(body) < 2 or body[1] != 0:
            s.close()
            raise ConnectionError(f'the broker refused the connection (CONNACK {body.hex()})')
        s.settimeout(5.0)
        self.sock, self.connected, self.error = s, True, ''

    def _drop(self, err):
        self.error = str(err)
        self.connected = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _run(self):
        backoff, last_tx = 1.0, time.monotonic()
        while not self.stop_ev.is_set():
            with self.cv:
                if not self.q:
                    self.cv.wait(timeout=max(1.0, self.keepalive / 2))
                item = self.q[0] if self.q else None
            if self.stop_ev.is_set():
                break
            try:
                if self.sock is None:
                    if item is None:
                        continue
                    self._connect()
                    backoff = 1.0
                if item is not None:
                    self.sock.sendall(publish_packet(*item))
                    with self.cv:
                        if self.q and self.q[0] is item:
                            self.q.popleft()
                    self.sent += 1
                    last_tx = time.monotonic()
                elif time.monotonic() - last_tx > self.keepalive / 2:
                    self.sock.sendall(PINGREQ)
                    last_tx = time.monotonic()
            except (OSError, ConnectionError) as e:
                self._drop(e)
                self.log(f'MQTT {self.host}:{self.port}: {e}; retry in {backoff:.0f} s')
                self.stop_ev.wait(backoff)
                backoff = min(60.0, backoff * 2)
        if self.sock is not None:
            try:
                self.sock.sendall(DISCONNECT)
            except OSError:
                pass
            self._drop('stopped')

    def flush(self, timeout=3.0) -> bool:
        t = time.monotonic() + timeout
        while self.q and time.monotonic() < t:
            time.sleep(0.02)
        return not self.q

    def close(self):
        self.flush(1.0)
        self.stop_ev.set()
        with self.cv:
            self.cv.notify()
        self.thread.join(timeout=2.0)
