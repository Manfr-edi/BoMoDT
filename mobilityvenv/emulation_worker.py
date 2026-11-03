
"""Run city dispatch and a serial analysis queue outside Django's request process."""
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

from libraries.constants import CONTAINER_ENV_FILE_PATH, SUMO_PATH, SUMO_NET_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.utils.generalUtils import loadEnvVar
from libraries.utils.executionTiming import connect as timing_connection
from mobilityvenv import emulation_store as store
from mobilityvenv.emulation_data import select_data, dispatch_interval


class Control:
    def __init__(self, session_id):
        self.session_id = session_id
        self.stop = threading.Event()
        self.run = threading.Event()
        self.run.set()
        self.closed = threading.Event()
        self.pause_seconds = 0.0
        self.pause_started = None
        self.lock = threading.Lock()
        self.waits = threading.local()

    def heartbeat(self):
        while not self.closed.is_set():
            try:
                session = store.get_session(self.session_id)
                command = session['command']
                with self.lock:
                    if command == 'pause' and self.pause_started is None:
                        self.pause_started = time.monotonic()
                    elif command != 'pause' and self.pause_started is not None:
                        self.pause_seconds += time.monotonic() - self.pause_started
                        self.pause_started = None
                if command == 'stop' or session['status'] not in store.ACTIVE:
                    self.stop.set()
                    self.run.set()
                elif command == 'pause':
                    self.run.clear()
                else:
                    self.run.set()
                store.update_session(self.session_id, heartbeat=True)
            except Exception:
                traceback.print_exc()
                self.stop.set()
                self.run.set()
            self.closed.wait(0.25)

    def checkpoint(self):
        if self.stop.is_set():
            raise InterruptedError('Emulation stopped')
        if not self.run.is_set():
            store.update_session(self.session_id, status='paused')
        waiting_started = time.perf_counter()
        while not self.run.wait(0.1):
            if self.stop.is_set():
                raise InterruptedError('Emulation stopped')
        if self.stop.is_set():
            raise InterruptedError('Emulation stopped')
        self.waits.seconds = getattr(self.waits, 'seconds', 0.0) + time.perf_counter() - waiting_started

    def work_time(self):
        return time.perf_counter() - getattr(self.waits, 'seconds', 0.0)

    def active_time(self):
        with self.lock:
            pause = self.pause_seconds
            if self.pause_started is not None:
                pause += time.monotonic() - self.pause_started
        return time.monotonic() - pause

    def wait_until(self, deadline):
        while True:
            self.checkpoint()
            remaining = deadline - self.active_time()
            if remaining <= 0:
                return
            self.stop.wait(min(remaining, 0.1))


def ensure_subscriptions(agent, env):
    from libraries.classes.SubscriptionManager import QuantumLeapManager
    from libraries.classes.Broker import Broker
    response = requests.get(f"http://localhost:{env['ORIONLD_PORT']}/ngsi-ld/v1/subscriptions",
                            headers={'NGSILD-Tenant': 'openiot'}, params={'limit': 1000}, timeout=15)
    response.raise_for_status()
    existing = response.json()
    broker = Broker(pn=env['ORIONLD_PORT'], pnt=None, host='localhost', fiwareservice='openiot')
    connection = broker.createConnection()
    agent.cbReference, agent.cbConnection = broker, connection
    manager = QuantumLeapManager('fiware-quantumleap', env['ORIONLD_PORT'], env['QUANTUMLEAP_PORT'])
    for entity_type in ('Device', 'RoadSegment', 'trafficflowobserved'):
        description = f'BoMoDT emulation timing: {entity_type}'
        if not any(item.get('description') == description for item in existing):
            manager.createQuantumLeapSubscription(connection, entity_type, 'trafficFlow', description)


def input_for_job(processed, successful, day, hour, folder):
    """Restrict model/scenario input to values actually sent successfully this hour."""
    sensor_ids = {int(row['ID_univoco_stazione_spira']) for row in successful}
    selected = processed[(processed['data'] == day) & processed['ID_univoco_stazione_spira'].isin(sensor_ids)].copy()
    if selected.empty or set(selected['ID_univoco_stazione_spira']) != sensor_ids:
        raise ValueError('Some transmitted sensors have no SUMO edge mapping for this date.')
    slot = f'{hour:02d}:00-{hour + 1:02d}:00'
    values = {int(row['ID_univoco_stazione_spira']): row[slot] for row in successful}
    selected[slot] = selected['ID_univoco_stazione_spira'].map(values)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'transmitted_measurements.csv'
    selected.to_csv(path, sep=';', index=False)
    return path


def analyze(session_id, config, work, control):
    from libraries.classes.SumoSimulator import Simulator
    from libraries.classes.TrafficModeler import TrafficModeler
    from libraries.classes.Planner import ScenarioGenerator
    from libraries.utils.preprocessingUtils import generateEdgeDataFile
    while True:
        job = work.get()
        if job is None:
            return
        day, hour, source, markers, job_folder = job
        try:
            control.checkpoint()
            store.update_job(session_id, day, hour, status='macromodeling')
            slot = f'{hour:02d}:00-{hour + 1:02d}:00'
            simulator = Simulator(str(Path(SUMO_PATH) / 'standalone'), str(job_folder / 'command_log.txt'))
            simulator.control_callback = control.checkpoint
            model = TrafficModeler(simulator, str(source), SUMO_NET_PATH, day, slot,
                                   config['macromodel'], outputRoot=str(job_folder))
            macro_finished = time.time() * 1000
            model.saveTrafficData(str(job_folder / 'model.csv'))
            with timing_connection() as db:
                sent_times = [row[0] for marker in markers for row in db.execute(
                    'SELECT sent_at_ms FROM measurements WHERE id=?', (marker,))]
            macro_latency = sum((macro_finished - sent) / 1000 for sent in sent_times)
            store.update_job(session_id, day, hour, macro_seconds=model.macromodeling_seconds,
                             macro_latency_sum=macro_latency)
            if config['pipeline'] == 'simulation':
                control.checkpoint()
                store.update_job(session_id, day, hour, status='simulation')
                simulation_started = control.work_time()
                params = {
                    'Krauss': {'sigma': str(config['sigma']), 'sigmaStep': str(config['sigma_step'])},
                    'IDM': {'delta': str(config['delta']), 'stepping': str(config['stepping'])},
                    'W99': {'cc1': str(config['cc1']), 'cc2': str(config['cc2'])},
                }[config['car_following_model']]
                output, _ = model.vTypeGeneration(config['car_following_model'], str(config['tau']), params)
                Path(output, 'output').mkdir(parents=True, exist_ok=True)
                model.saveTrafficData(str(Path(output) / 'model.csv'))
                edge_file = job_folder / 'edgedata.xml'
                generateEdgeDataFile(str(source), day, slot, output_file=str(edge_file))
                generator = ScenarioGenerator(str(Path(SUMO_PATH) / 'standalone'), simulator)
                generator.routesRoot = str(job_folder / 'routes')
                control.checkpoint()
                generator.generateRoute(str(edge_file), slot, totalCount=config['total_count'])
                simulator.changeRouteFilePath(str(Path(generator.routesRoot) / slot.replace(':', '-')))
                control.checkpoint()
                simulator.start(activeGui=config.get('sumo_gui', False))
                elapsed = control.work_time() - simulation_started
                simulation_finished = time.time() * 1000
                store.update_job(session_id, day, hour, simulation_seconds=elapsed,
                    simulation_latency_sum=sum((simulation_finished - sent) / 1000 for sent in sent_times))
            store.update_job(session_id, day, hour, status='completed')
        except InterruptedError:
            store.update_job(session_id, day, hour, status='cancelled')
        except Exception as exc:
            traceback.print_exc()
            store.update_job(session_id, day, hour, status='failed', error=str(exc))
        finally:
            work.task_done()


def run(session_id):
    from libraries.classes.Agent import Agent
    from mobilityvenv.MobilityVirtualEnvironment import setupPhysicalSystem
    session = store.get_session(session_id)
    config = session['config']
    control = Control(session_id)
    watcher = threading.Thread(target=control.heartbeat, daemon=True)
    watcher.start()
    work = queue.Queue()
    analyzer = None
    attempted = succeeded = failed = 0
    started = control.active_time()
    try:
        env = loadEnvVar(CONTAINER_ENV_FILE_PATH)
        agent = Agent('web-emulation', 'localhost', env['ORIONLD_PORT'], env['IOTA_SOUTH_PORT'],
                      env['IOTA_NORTH_PORT'], 'openiot', '/')
        agent.timing_session_id = session_id
        selected, missing = select_data(config)
        ensure_subscriptions(agent, env)
        roads, _ = setupPhysicalSystem(agent, selected, control.checkpoint)
        sensors = {str(sensor.devicePartialID): sensor for road in roads.values() for sensor in road.sensors}
        processed = pd.read_csv(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, sep=';') if config['pipeline'] != 'baseline' else None
        if config['pipeline'] != 'baseline':
            analyzer = threading.Thread(target=analyze, args=(session_id, config, work, control), daemon=True)
            analyzer.start()
        store.update_session(session_id, status='running', progress={'stage': 'Sending measurements', 'missing_days': missing})
        started = control.active_time()
        for day, daily in selected.groupby('data', sort=True):
            count = len(daily) * (config['end_hour'] - config['start_hour'])
            interval = dispatch_interval(config, count)
            day_started = control.active_time()
            index = 0
            for hour in range(config['start_hour'], config['end_hour']):
                successful, markers = [], []
                for row in daily.to_dict('records'):
                    control.wait_until(day_started + index * interval)
                    lag = max(0, control.active_time() - day_started - index * interval)
                    slot = f'{hour:02d}:00-{hour + 1:02d}:00'
                    store.update_session(session_id, progress={'day': day, 'hour': hour,
                        'sensor': row['sensor_id'], 'schedule_lag_seconds': lag})
                    attempted += 1
                    index += 1
                    try:
                        sensor = sensors[row['sensor_id']]
                        coordinates = [float(part.strip()) for part in row['geopoint'].split(',')]
                        agent.measurementSending(day, slot, int(row[slot]), coordinates, str(row['direzione']),
                                                 str(row['codZone']), 'trafficFlow', sensor.apiKey, sensor.devicePartialID)
                        succeeded += 1
                        successful.append(row)
                        markers.append(agent.last_measurement_id)
                    except Exception as exc:
                        failed += 1
                        store.update_session(session_id, progress={'last_send_error': str(exc)})
                        if not config['continue_on_error']:
                            raise
                    finally:
                        store.update_session(session_id, progress={'attempted': attempted, 'succeeded': succeeded,
                            'failed': failed, 'active_seconds': control.active_time() - started})
                if analyzer and successful:
                    folder = store.session_folder(session_id) / day / f'{hour:02d}'
                    try:
                        source = input_for_job(processed, successful, day, hour, folder)
                        store.add_job(session_id, day, hour, len(markers))
                        work.put((day, hour, source, markers, folder))
                    except Exception as exc:
                        store.add_job(session_id, day, hour, len(markers))
                        store.update_job(session_id, day, hour, status='failed', error=str(exc))
                control.checkpoint()
            # Preserve the full selected day duration, rather than front-loading a burst.
            if config['pacing'] == 'day':
                control.wait_until(day_started + config['day_seconds'])
        store.update_session(session_id, status='draining', progress={'sends_done': True,
            'stage': 'Waiting for queued calculations', 'active_seconds': control.active_time() - started})
        if analyzer:
            work.put(None)
            analyzer.join()
        control.checkpoint()
        failed_jobs = sum(job['status'] == 'failed' for job in store.summary(session_id)['jobs'])
        store.update_session(session_id, status='completed_with_errors' if failed_jobs else 'completed',
            progress={'stage': 'Completed', 'failed_jobs': failed_jobs})
    except InterruptedError:
        control.stop.set()
        control.run.set()
        if analyzer:
            work.put(None)
            analyzer.join()
        store.update_session(session_id, status='stopped', progress={'stage': 'Stopped'})
    except Exception as exc:
        traceback.print_exc()
        control.stop.set()
        control.run.set()
        if analyzer:
            work.put(None)
            analyzer.join()
        store.update_session(session_id, status='failed', error=str(exc), progress={'stage': 'Failed'})
    finally:
        control.closed.set()
        watcher.join(timeout=2)
        snapshot = {'session': store.get_session(session_id), 'timings': store.summary(session_id)}
        folder = store.session_folder(session_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'execution_times.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run(sys.argv[1])
