#!/usr/bin/env python3
"""
generate_e2_from_net.py

Usage:
    python generate_e2_from_net.py full.net.xml [output_add.xml]

Genera un file .add.xml con e2Detector per ogni TLS trovato in full.net.xml.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from libraries import constants

NS = {}  # net.xml normally not namespaced; if yours has namespace adjust here.


def find_lane_element(root, lane_id):
    # lanes are <lane id="..."> usually inside <edge> elements
    for lane in root.findall(".//lane", NS):
        if lane.get("id") == lane_id:
            return lane
    return None


def get_lane_length(root, lane_id):
    lane = find_lane_element(root, lane_id)
    if lane is not None:
        l = lane.get("length")
        try:
            return float(l)
        except Exception:
            return None
    return None


def find_junction_by_id(root, jid):
    for j in root.findall(".//junction", NS):
        if j.get("id") == jid:
            return j
    return None


def find_first_to_lane_for_fromlane(root, lane_id):
    """
    Find a to-lane that is actually connected to the from-lane based on the <connection>.
    Returns an id of type 'edge_laneIndex' only if the connection actually exists.
    """
    if "_" not in lane_id:
        return None

    from_edge, from_lane_idx = lane_id.rsplit("_", 1)

    # Verifica che from_lane_idx sia un numero
    try:
        from_lane_idx = int(from_lane_idx)
    except ValueError:
        return None

    # Scansiona tutte le connection
    for conn in root.findall(".//connection", NS):
        if conn.get("from") == from_edge and conn.get("fromLane") is not None:
            try:
                if int(conn.get("fromLane")) == from_lane_idx:
                    # connessione valida → costruiamo la to-lane reale
                    to_edge = conn.get("to")
                    to_lane = conn.get("toLane")
                    if to_edge is not None and to_lane is not None:
                        return f"{to_edge}_{to_lane}"
            except:
                continue

    # no valid connection → no error, simply no to-lane
    return None


def main(netfile: str,
         output: str = "e2_detectors.add.xml",
         desired_length: float = 35.0,
         offset_from_end: float = 5.0,
         min_length: float = 10.0,
         prefix: str = "e2") -> int:
    """
    Generate E2 detectors for each TLS found in the netfile

    Args:
        :param netfile: path of the SUMO .net.xml file
        :param output: file path of .add.xml output file
        :param desired_length: desired detector length in meters (default 35m)
        :param offset_from_end: distance from lane end to detector end in meters (default 5m)
        :param min_length: minimum detector length for very short lanes (default 10m)
        :param prefix: id prefix for detectors
    :return:
        number of generated detectors
    """
    netpath = Path(netfile)
    if not netpath.exists():
        raise FileNotFoundError(f"Network file not found: {netpath}")

    tree = ET.parse(str(netfile))
    root = tree.getroot()

    # create root for add file
    add_root = ET.Element("additional")

    # find all tlLogic elements
    tl_logics = root.findall(".//tlLogic", NS)
    if not tl_logics:
        print("Nessun tlLogic trovato nel file .net.xml. Controlla se i TLS sono definiti in un file addizionale.")

    for tl in tl_logics:
        tls_id = tl.get("id")
        if tls_id is None:
            continue
        # find junction with same id (typical)
        junction = find_junction_by_id(root, tls_id)
        inc_lanes = []
        if junction is not None:
            inc = junction.get("incLanes") or ""
            inc_lanes = [s.strip() for s in inc.split() if s.strip()]
        else:
            # fallback: try to infer incoming lanes by scanning connections that reference this tls (rare)
            for conn in root.findall(".//connection", NS):
                if conn.get("tl") and conn.get("tl") == tls_id:
                    # try to reconstruct lane id
                    from_edge = conn.get("from")
                    from_lane_idx = conn.get("fromLane")
                    if from_edge and from_lane_idx is not None:
                        inc_lanes.append(f"{from_edge}_{from_lane_idx}")

        # deduplicate
        inc_lanes = sorted(set(inc_lanes))

        for lane_id in inc_lanes:
            lane_len = get_lane_length(root, lane_id)

            if lane_len is None:
                # If we can't determine lane length, use conservative defaults
                detector_length = min_length
                pos = 0.0
            else:
                # Calculate detector end position (offset_from_end meters before lane end)
                detector_end = lane_len - offset_from_end

                # Calculate available space for detector
                available_length = detector_end

                # Determine actual detector length
                if available_length >= desired_length:
                    # Plenty of space - use desired length
                    detector_length = desired_length
                elif available_length >= min_length:
                    # Limited space - use what's available
                    detector_length = available_length
                else:
                    # Very short lane - use minimum length and adjust position
                    detector_length = min_length
                    detector_end = lane_len  # Place at lane end

                # Calculate detector start position
                pos = max(0.0, detector_end - detector_length)

                # Round for cleaner output
                pos = round(pos, 3)
                detector_length = round(detector_length, 3)

            # try to find a "to" lane to measure jam for specific link
            to_lane = find_first_to_lane_for_fromlane(root, lane_id)

            det_id = f"e2_{tls_id}_{lane_id}".replace(":", "_").replace("-", "_")
            e2 = ET.Element("e2Detector", {
                "id": det_id,
                "lane": lane_id,
                "pos": str(pos),
                "length": str(detector_length),
                "tl": tls_id,
                "file": "../output/e2_global_output.xml"
            })

            add_root.append(e2)

    # pretty print (simple)
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for e in elem:
                indent(e, level + 1)
            if not e.tail or not e.tail.strip():
                e.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(add_root)
    add_tree = ET.ElementTree(add_root)
    add_tree.write(str(output), encoding="utf-8", xml_declaration=True)
    print(f"Additional file generated: {output} (containing {len(add_root)} e2Detector)")

    return len(add_root)


if __name__ == "__main__":
    number = main(
        netfile=constants.SUMO_NETWORK_PATH + '/full.net.xml',
        output=constants.SUMO_NETWORK_PATH + '/e2Detector.xml',
        desired_length=35.0,  # Try for 35m detectors
        offset_from_end=0.0,  # End 0m before intersection
        min_length=10.0  # Minimum 10m for short lanes
    )