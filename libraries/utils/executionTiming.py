"""Wall-clock phase durations and correlated, first-browser-frame latency samples."""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from libraries.constants import SUMO_PATH


@contextmanager
def connect():
    path = Path(SUMO_PATH) / 'telemetry' / 'baseline.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        with connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS measurements (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                observation_date TEXT NOT NULL, timeslot TEXT NOT NULL,
                sent_at_ms REAL NOT NULL, rendered_at_ms REAL)''')
            yield connection
    finally:
        connection.close()


def record_send(session_id, entity_id, observation_date, timeslot, sent_at_ms, measurement_id=None):
    measurement_id = measurement_id or str(uuid4())
    with connect() as db:
        db.execute('INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, NULL)',
                   (measurement_id, session_id, entity_id, str(observation_date)[:10], timeslot, sent_at_ms))
    return measurement_id


def record_render(measurement_ids, rendered_at_ms):
    # Use the browser frame timestamp, not the time its acknowledgement arrived.
    if not isinstance(rendered_at_ms, (int, float)) or not 0 < rendered_at_ms <= time.time() * 1000 + 5000:
        raise ValueError('Invalid browser timestamp')
    with connect() as db:
        for measurement_id in measurement_ids:
            db.execute('''UPDATE measurements SET rendered_at_ms = ?
                WHERE id = ? AND (rendered_at_ms IS NULL OR rendered_at_ms > ?)
                AND sent_at_ms <= ?''',
                       (rendered_at_ms, measurement_id, rendered_at_ms, rendered_at_ms))
        placeholders = ','.join('?' for _ in measurement_ids)
        return [row[0] for row in db.execute(
            f'SELECT id FROM measurements WHERE rendered_at_ms IS NOT NULL AND id IN ({placeholders})',
            measurement_ids)] if measurement_ids else []


def baseline_summary(observation_date, timeslot, session_id=None):
    with connect() as db:
        # A baseline belongs to an emulation session, not to the later SUMO run.
        session = (session_id,) if session_id else db.execute('''SELECT session_id FROM measurements
            WHERE observation_date = ? AND CAST(substr(timeslot, 1, 2) AS INTEGER) >= ?
            AND CAST(substr(timeslot, 1, 2) AS INTEGER) < ? ORDER BY sent_at_ms DESC LIMIT 1''',
                             (observation_date, *timeslot)).fetchone()
        if session is None:
            return {'count': 0, 'sent_count': 0, 'mean_seconds': None}
        rows = db.execute('''SELECT sent_at_ms, rendered_at_ms FROM measurements
            WHERE session_id = ? AND observation_date = ?
            AND CAST(substr(timeslot, 1, 2) AS INTEGER) >= ?
            AND CAST(substr(timeslot, 1, 2) AS INTEGER) < ?''',
                          (session[0], observation_date, *timeslot)).fetchall()
    durations = sorted((end - start) / 1000 for start, end in rows if end is not None)
    return {'session_id': session[0], 'count': len(durations), 'sent_count': len(rows),
            'mean_seconds': sum(durations) / len(durations) if durations else None,
            'min_seconds': min(durations) if durations else None,
            'max_seconds': max(durations) if durations else None}


class ExecutionTiming:
    def __init__(self):
        self.seconds = {'macromodeling': 0.0, 'simulation': 0.0}
        self.slots = []

    def save(self, folder, date, timeslot):
        result = {'schema_version': 1, 'unit': 'seconds', 'date': date, 'timeslot': timeslot,
                  'phases': self.seconds, 'slots': self.slots,
                  'baseline': baseline_summary(date, timeslot),
                  'baseline_endpoint': 'first browser frame after the Grafana flow panel commits',
                  'simulation_scope': 'scenario preparation, route generation, SUMO startup and complete execution',
                  'excluded': ['intentional sleeps', 'result evaluation', 'input CSV and network loading']}
        target = Path(folder) / 'execution_times.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        temporary.replace(target)


def read_execution_times(folder):
    try:
        result = json.loads((Path(folder) / 'execution_times.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # Rendering can happen after SUMO finishes, when the user opens Results.
    # Keep the original emulation session rather than silently switching runs.
    session_id = result.get('baseline', {}).get('session_id')
    if session_id:
        result['baseline'] = baseline_summary(result['date'], result['timeslot'], session_id)
    return result
