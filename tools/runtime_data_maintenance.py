"""Check the selected network and explicitly reset local timing/session history."""
import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libraries.constants import SUMO_NET_PATH, SUMO_DETECTORS_ADD_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH


def check_network():
    import sumolib
    network = sumolib.net.readNet(SUMO_NET_PATH)
    with open(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, encoding='utf-8-sig', newline='') as source:
        measured_edges = {row['edge_id'] for row in csv.DictReader(source, delimiter=';')}
    missing_edges = sorted(edge for edge in measured_edges if not network.hasEdge(edge))
    missing_lanes = []
    detectors = list(ET.parse(SUMO_DETECTORS_ADD_FILE_PATH).getroot().iter('inductionLoop'))
    for detector in detectors:
        try:
            network.getLane(detector.attrib['lane'])
        except (KeyError, ValueError):
            missing_lanes.append(detector.attrib['lane'])
    print(json.dumps({'network': SUMO_NET_PATH, 'network_edges': len(network.getEdges()),
        'measured_edges': len(measured_edges), 'missing_measured_edges': missing_edges,
        'detectors': len(detectors), 'missing_detector_lanes': missing_lanes}, indent=2))


def reset_telemetry():
    from mobilityvenv.emulation_store import connect
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        for table in ('emulation_jobs', 'measurements', 'emulation_sessions'):
            count = db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
            db.execute(f'DELETE FROM {table}')
            print(f'{table}: removed {count} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-telemetry', action='store_true')
    arguments = parser.parse_args()
    check_network()
    if arguments.reset_telemetry:
        reset_telemetry()
