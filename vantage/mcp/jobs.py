"""Background jobs for the MCP server: any tool run in the background (start_job), read (get_job), stopped
(cancel_job) and listed (list_jobs).

A job is queued and run by one worker thread, one job after another, so jobs never race each other on the project;
ordinary tool calls go on meanwhile. A job runs as the tool would run when called (in the window, or on the file),
with a cancel event that cancel_job sets: a batch run's process is stopped at once, a study stops at its next check.
Jobs live as long as the MCP server; a finished job keeps its result until the server stops (the last MAX_KEPT)."""
from __future__ import annotations

import queue
import threading
import itertools
import time

MAX_KEPT = 50
_IDS = itertools.count(1)                 # job ids: numbered per server (no random numbers in Vantage)
STATES = ('queued', 'running', 'done', 'failed', 'cancelled')


class Job:
    def __init__(self, tool, args, label=''):
        self.id = f'job-{next(_IDS)}'
        self.tool, self.args, self.label = tool, dict(args or {}), label or tool
        self.state = 'queued'
        self.created, self.started, self.finished = time.time(), None, None
        self.result = None                       # the tool's MCP result (content, isError, structuredContent)
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.note = ''

    def summary(self) -> dict:
        now = time.time()
        d = {'job_id': self.id, 'tool': self.tool, 'label': self.label, 'state': self.state,
             'queued_s': round((self.started or self.finished or now) - self.created, 1)}
        if self.started:
            d['running_s' if self.state == 'running' else 'ran_s'] = round((self.finished or now) - self.started, 1)
        if self.result is not None:
            d['is_error'] = bool(self.result.get('isError'))
        if self.note:
            d['note'] = self.note
        return d


def _stopped(res) -> bool:
    """Whether a tool result says the tool was stopped: cancelled: true, a batch run's exit code -15, or an error
    that mentions the cancel."""
    if not isinstance(res, dict):
        return False
    sc = res.get('structuredContent')
    if isinstance(sc, dict) and (sc.get('cancelled') is True or sc.get('exit_code') == -15):
        return True
    if not res.get('isError'):
        return False
    text = ' '.join(str(c.get('text', '')) for c in res.get('content') or [] if isinstance(c, dict))
    return 'cancel' in text.lower() or 'stopped' in text.lower()


class Jobs:
    """run(tool, args, cancel) -> an MCP tool result; on_done(job) is told when a job ends (for notifications)."""

    def __init__(self, run, on_done=None):
        self.run, self.on_done = run, on_done
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._worker = None

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name='vantage-mcp-jobs', daemon=True)
            self._worker.start()

    def _loop(self):
        while True:
            job = self._q.get()
            if job.state == 'cancelled':           # cancelled while queued: cancel() has already ended it
                continue
            job.state, job.started = 'running', time.time()
            try:
                res = self.run(job.tool, job.args, job.cancel)
                job.result = res
                if job.cancel.is_set() and _stopped(res):
                    job.state = 'cancelled'
                else:                               # it ran to its end, even if a cancel came too late to stop it
                    job.state = 'failed' if res.get('isError') else 'done'
                    if job.cancel.is_set():
                        job.note = 'finished before the cancel could stop it'
            except Exception as e:                  # noqa: BLE001 — a job never takes the server down
                job.result = {'content': [{'type': 'text', 'text': f'{type(e).__name__}: {e}'}], 'isError': True}
                job.state = 'failed'
            job.finished = time.time()
            job.done.set()
            if self.on_done is not None:
                try:
                    self.on_done(job)
                except Exception:                   # noqa: BLE001
                    pass

    def start(self, tool, args=None, label='') -> dict:
        job = Job(tool, args, label)
        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            while len(self.order) > MAX_KEPT:
                old = self.jobs.get(self.order[0])
                if old is not None and old.state in ('queued', 'running'):
                    break
                self.jobs.pop(self.order.pop(0), None)
        self._ensure_worker()
        self._q.put(job)
        ahead = sum(1 for j in self.jobs.values() if j.state in ('queued', 'running') and j is not job)
        return {**job.summary(), 'jobs_ahead': ahead,
                'note': 'get_job with this job_id reads its progress and result (wait_s waits for it); cancel_job '
                        'stops it'}

    def get(self, job_id, wait_s=0.0):
        job = self.jobs.get(str(job_id))
        if job is None:
            return None
        if wait_s and not job.done.is_set():
            job.done.wait(max(0.0, min(float(wait_s), 300.0)))
        return job

    def cancel(self, job_id) -> dict:
        job = self.jobs.get(str(job_id))
        if job is None:
            return {'error': f'no job {job_id!r} (list_jobs lists them)'}
        if job.state in ('done', 'failed', 'cancelled'):
            return {**job.summary(), 'note': f'the job had already ended ({job.state})'}
        job.cancel.set()
        if job.state == 'queued':
            job.state, job.finished = 'cancelled', time.time()
            job.done.set()                          # it never runs: a get_job waiting on it returns now
            if self.on_done is not None:
                try:
                    self.on_done(job)
                except Exception:                   # noqa: BLE001
                    pass
        return {**job.summary(), 'note': 'cancelling' if job.state == 'running' else 'cancelled before it started'}

    def list(self) -> list:
        return [self.jobs[i].summary() for i in reversed(self.order) if i in self.jobs]

    def ids(self) -> list:
        return list(reversed(self.order))
