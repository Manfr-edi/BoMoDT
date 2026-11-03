"""Deterministic selection of the same historical measurements used by option 2."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from libraries.constants import REAL_TRAFFIC_FLOW_DATA_MVENV_PATH


def load_data():
    files = sorted(Path(REAL_TRAFFIC_FLOW_DATA_MVENV_PATH).glob('*.csv'))
    if not files:
        raise ValueError('No emulation CSV files are available.')
    frame = pd.concat([pd.read_csv(path, sep=';') for path in files], ignore_index=True)
    frame['sensor_id'] = frame['ID_univoco_stazione_spira'].map(lambda value: 'TL' + str(int(value)))
    frame['data'] = pd.to_datetime(frame['data'], errors='raise').dt.strftime('%Y-%m-%d')
    return frame.drop_duplicates(['data', 'sensor_id']).sort_values(['data', 'sensor_id'])


def select_data(config, frame=None):
    frame = load_data() if frame is None else frame
    selected = frame[frame['data'].between(config['start_date'], config['end_date'])]
    if config.get('sensor_ids'):
        selected = selected[selected['sensor_id'].isin(config['sensor_ids'])]
    if selected.empty:
        raise ValueError('No measurements match the selected dates and sensors.')
    available = set(selected['data'])
    first, last = date.fromisoformat(config['start_date']), date.fromisoformat(config['end_date'])
    missing = [(first + timedelta(days=offset)).isoformat()
               for offset in range((last - first).days + 1)
               if (first + timedelta(days=offset)).isoformat() not in available]
    if missing and not config.get('skip_missing_days', True):
        raise ValueError('No measurements for: ' + ', '.join(missing[:10]))
    return selected, missing


def dispatch_interval(config, event_count):
    if config['pacing'] == 'day':
        return config['day_seconds'] / event_count
    return config['interval_seconds']
