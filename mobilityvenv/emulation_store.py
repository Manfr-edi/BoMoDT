"""Persistent control state shared by Django and an independent worker process."""
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

from libraries.constants import SUMO_PATH
from libraries.utils.executionTiming import connect as timing_connection

ACTIVE = ('starting', 'running', 'pausing', 'paused', 'stopping', 'draining')


@contextmanager
def connect():
    with timing_connection() as db:
        db.row_factory = __import__('sqlite3').Row
        db.execute('''CREATE TABLE IF NOT EXISTS emulation_sessions (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, command TEXT NOT NULL,
            config TEXT NOT NULL, progress TEXT NOT NULL, created_at REAL NOT NULL,
            heartbeat REAL NOT NULL, pid INTEGER, error TEXT)''')
        db.execute('''CREATE TABLE IF NOT EXISTS emulation_jobs (
            session_id TEXT NOT NULL, day TEXT NOT NULL, hour INTEGER NOT NULL,
            status TEXT NOT NULL, sample_count INTEGER NOT NULL,
            macro_seconds REAL, simulation_seconds REAL, macro_latency_sum REAL,
            simulation_latency_sum REAL, error TEXT,
            PRIMARY KEY (session_id, day, hour))''')
        yield db


def session_folder(session_id):
    return Path(SUMO_PATH) / 'emulations' / str(UUID(session_id))


def unpack(row):
    if row is None:
        return None
    result = dict(row)
    result['config'] = json.loads(result['config'])
    result['progress'] = json.loads(result['progress'])
    return result


def get_session(session_id=None):
    with connect() as db:
        # A worker heartbeat is independent of individual network/SUMO calls.
        db.execute("UPDATE emulation_sessions SET status='interrupted', command='stop', error=? "
                   "WHERE status IN ('starting','running','pausing','paused','stopping','draining') AND heartbeat < ?",
                   ('Worker heartbeat was lost. Start a new emulation.', time.time() - 90))
        if session_id:
            row = db.execute('SELECT * FROM emulation_sessions WHERE id=?', (session_id,)).fetchone()
        else:
            row = db.execute('SELECT * FROM emulation_sessions ORDER BY created_at DESC LIMIT 1').fetchone()
    return unpack(row)


def update_session(session_id, *, status=None, error=None, progress=None, heartbeat=False):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current = db.execute('SELECT * FROM emulation_sessions WHERE id=?', (session_id,)).fetchone()
        if current is None:
            raise ValueError('Emulation session not found.')
        state = json.loads(current['progress'])
        if progress:
            state.update(progress)
        db.execute('UPDATE emulation_sessions SET status=?,error=?,progress=?,heartbeat=? WHERE id=?',
                   (status or current['status'], error if error is not None else current['error'],
                    json.dumps(state), time.time() if heartbeat else current['heartbeat'], session_id))


def create_session(config, launch=True):
    get_session()  # expire a genuinely orphaned worker before acquiring the start lock
    session_id = str(uuid4())
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        active = db.execute("SELECT id FROM emulation_sessions WHERE status IN ('starting','running','pausing','paused','stopping','draining')").fetchone()
        if active:
            raise ValueError('An emulation is already active. Pause, stop or finish it first.')
        now = time.time()
        db.execute('INSERT INTO emulation_sessions VALUES (?,?,?,?,?,?,?,?,?)',
                   (session_id, 'starting', 'run', json.dumps(config), json.dumps({
                       'attempted': 0, 'succeeded': 0, 'failed': 0, 'total': config['total'],
                       'stage': 'Preparing sensors', 'sends_done': False}), now, now, None, None))
    if launch:
        folder = session_folder(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with (folder / 'worker.log').open('ab') as output:
                process = subprocess.Popen([sys.executable, '-m', 'mobilityvenv.emulation_worker', session_id],
                    cwd=Path(__file__).resolve().parents[1], stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with connect() as db:
                db.execute('UPDATE emulation_sessions SET pid=? WHERE id=?', (process.pid, session_id))
        except Exception as exc:
            update_session(session_id, status='failed', error=str(exc))
            raise
    return get_session(session_id)


def control(session_id, action):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM emulation_sessions WHERE id=?', (session_id,)).fetchone()
        if row is None or row['status'] not in ACTIVE:
            raise ValueError('This emulation is no longer active.')
        if action == 'pause' and row['command'] == 'run':
            command, status = 'pause', 'pausing'
        elif action == 'resume' and row['command'] == 'pause':
            command = 'run'
            status = 'draining' if json.loads(row['progress']).get('sends_done') else 'running'
        elif action == 'stop':
            command, status = 'stop', 'stopping'
        else:
            raise ValueError('Invalid action for the current emulation state.')
        db.execute('UPDATE emulation_sessions SET command=?,status=? WHERE id=?', (command, status, session_id))


def recent_sessions():
    with connect() as db:
        return [unpack(row) for row in db.execute('SELECT * FROM emulation_sessions ORDER BY created_at DESC LIMIT 20')]


def add_job(session_id, day, hour, sample_count):
    with connect() as db:
        db.execute('INSERT INTO emulation_jobs VALUES (?,?,?,?,?,NULL,NULL,NULL,NULL,NULL)',
                   (session_id, day, hour, 'queued', sample_count))


def update_job(session_id, day, hour, **values):
    allowed = {'status', 'macro_seconds', 'simulation_seconds', 'macro_latency_sum', 'simulation_latency_sum', 'error'}
    if not values or set(values) - allowed:
        raise ValueError('Invalid job update')
    with connect() as db:
        db.execute('UPDATE emulation_jobs SET ' + ','.join(key + '=?' for key in values)
                   + ' WHERE session_id=? AND day=? AND hour=?', (*values.values(), session_id, day, hour))


def summary(session_id):
    with connect() as db:
        rows = db.execute('SELECT entity_id,sent_at_ms,rendered_at_ms FROM measurements WHERE session_id=?', (session_id,)).fetchall()
        jobs = [dict(row) for row in db.execute('SELECT * FROM emulation_jobs WHERE session_id=? ORDER BY day,hour', (session_id,))]
    rendered = [(row['rendered_at_ms'] - row['sent_at_ms']) / 1000 for row in rows if row['rendered_at_ms'] is not None]
    sensors = {}
    for row in rows:
        sensor = sensors.setdefault(row['entity_id'], {'sent': 0, 'rendered': 0, 'sum_seconds': 0})
        sensor['sent'] += 1
        if row['rendered_at_ms'] is not None:
            sensor['rendered'] += 1
            sensor['sum_seconds'] += (row['rendered_at_ms'] - row['sent_at_ms']) / 1000
    for sensor in sensors.values():
        total_seconds = sensor.pop('sum_seconds')
        sensor['mean_seconds'] = total_seconds / sensor['rendered'] if sensor['rendered'] else None
    def phase(name):
        measured = [job for job in jobs if job[name + '_seconds'] is not None]
        count = sum(job['sample_count'] for job in measured)
        total = sum(job[name + '_seconds'] for job in measured)
        latency = sum(job[name + '_latency_sum'] or 0 for job in measured)
        return {'windows': len(measured), 'samples': count, 'total_seconds': total,
                'mean_compute_seconds_per_sample': total / count if count else None,
                'mean_end_to_end_seconds': latency / count if count else None}
    return {'baseline': {'count': len(rendered), 'sent_count': len(rows),
                'mean_seconds': sum(rendered) / len(rendered) if rendered else None,
                'min_seconds': min(rendered) if rendered else None,
                'max_seconds': max(rendered) if rendered else None},
            'sensors': sensors, 'macromodeling': phase('macro'), 'simulation': phase('simulation'), 'jobs': jobs}
