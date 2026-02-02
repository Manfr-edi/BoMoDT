import xml.etree.ElementTree as ET
import json
import csv
from collections import defaultdict
from libraries import constants


# --------------------------------------------------------
#  UTILITY: Point-in-polygon (ray casting)
# --------------------------------------------------------
def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        if (y1 > y) != (y2 > y):
            xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside

    return inside


# --------------------------------------------------------
#  LOAD TAZ POLYGONS
# --------------------------------------------------------
def load_taz_polygons(taz_file):
    tree = ET.parse(taz_file)
    root = tree.getroot()

    taz_polys = {}
    for taz in root.findall(".//taz"):
        tid = taz.get("id")
        shape = taz.get("shape")
        if not shape:
            continue
        coords = []
        for p in shape.split():
            xs, ys = p.split(",")
            coords.append((float(xs), float(ys)))
        taz_polys[tid] = coords

    print(f"[OK] Loaded {len(taz_polys)} TAZ polygons")
    return taz_polys


# --------------------------------------------------------
#  LOAD TLS IDs
# --------------------------------------------------------
def load_tls_ids(tls_file):
    tree = ET.parse(tls_file)
    root = tree.getroot()
    tls_ids = {tl.get("id") for tl in root.findall(".//tlLogic")}
    print(f"[OK] Loaded {len(tls_ids)} TLS IDs")
    return tls_ids


# --------------------------------------------------------
#  LOAD JUNCTION POSITIONS FROM NETWORK
# --------------------------------------------------------
def load_junction_positions(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()

    jpos = {}
    for j in root.findall(".//junction"):
        jid = j.get("id")
        x, y = j.get("x"), j.get("y")
        if x and y:
            jpos[jid] = (float(x), float(y))

    print(f"[OK] Loaded {len(jpos)} junction positions")
    return jpos


# --------------------------------------------------------
#  LOAD CONNECTIONS: tl → list of nodes involved
# --------------------------------------------------------
def load_tls_connections(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()

    # map edge → toNode
    edge_to_node = {}
    for e in root.findall(".//edge"):
        eid = e.get("id")
        to = e.get("to")
        if eid and to:
            edge_to_node[eid] = to

    # map tls → list of real nodes
    tls_nodes = defaultdict(list)

    for c in root.findall(".//connection"):
        tl = c.get("tl")
        if not tl:
            continue

        frm = c.get("from")
        if frm in edge_to_node:
            real_node = edge_to_node[frm]
            tls_nodes[tl].append(real_node)

    print(f"[OK] Loaded connections for {len(tls_nodes)} TLS")
    return tls_nodes


# --------------------------------------------------------
#  MAP TLS TO TAZ (Robust Version)
# --------------------------------------------------------
def assign_taz_for_tls(tl_id, taz_polys, node_positions):
    # compute TAZ membership for each node
    votes = defaultdict(int)

    for (x, y) in node_positions:
        assigned = None
        for taz_id, poly in taz_polys.items():
            if point_in_poly(x, y, poly):
                votes[taz_id] += 1
                assigned = True
                break
        if not assigned:
            votes["NONE"] += 1

    if not votes:
        return None

    # pick TAZ with highest votes
    best_taz = max(votes, key=votes.get)

    if best_taz == "NONE":
        return None

    return best_taz


def map_tls_to_taz(tls_ids, taz_polys, junction_positions, tls_nodes_mapping):
    tls_to_taz = {}

    for tl in tls_ids:

        # 1) CASE A: junction exists with same ID → simple TLS
        if tl in junction_positions:
            x, y = junction_positions[tl]
            assigned = None
            for taz_id, poly in taz_polys.items():
                if point_in_poly(x, y, poly):
                    assigned = taz_id
                    break
            tls_to_taz[tl] = assigned
            continue

        # 2) CASE B: joinedS or virtual TLS → group of real nodes
        node_ids = tls_nodes_mapping.get(tl, [])

        if not node_ids:
            tls_to_taz[tl] = None
            continue

        # convert node ids → node positions
        node_positions = [
            junction_positions[nid]
            for nid in node_ids
            if nid in junction_positions
        ]

        if not node_positions:
            tls_to_taz[tl] = None
            continue

        # assign based on majority vote
        assigned = assign_taz_for_tls(tl, taz_polys, node_positions)
        tls_to_taz[tl] = assigned

    return tls_to_taz


# --------------------------------------------------------
#  MAIN
# --------------------------------------------------------
def main(
        taz_file=constants.SUMO_NETWORK_PATH + "/output_taz.add.xml",
        tls_file=constants.SUMO_NETWORK_PATH + "/optimized_tls.add.xml",
        net_file=constants.SUMO_NETWORK_PATH + "/full.net.xml",
        out_csv=constants.REAL_WORLD_DATA_PATH + "/tls_to_taz.csv",
        out_json=constants.REAL_WORLD_DATA_PATH + "/taz_to_tls.json"
):

    print("\n--- MAPPING TLS TO TAZ (ROBUST VERSION) ---\n")

    taz_polys = load_taz_polygons(taz_file)
    tls_ids = load_tls_ids(tls_file)
    junction_positions = load_junction_positions(net_file)
    tls_nodes_mapping = load_tls_connections(net_file)

    tls_to_taz = map_tls_to_taz(
        tls_ids,
        taz_polys,
        junction_positions,
        tls_nodes_mapping
    )

    # reverse mapping
    taz_to_tls = defaultdict(list)
    for tl, taz in tls_to_taz.items():
        if taz is not None:
            taz_to_tls[taz].append(tl)

    # save CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tl_id", "taz_id"])
        for tl, taz in tls_to_taz.items():
            w.writerow([tl, taz])

    # save JSON
    with open(out_json, "w") as f:
        json.dump(taz_to_tls, f, indent=2)

    print(f"\n[OK] Saved {out_csv}")
    print(f"[OK] Saved {out_json}\n")


if __name__ == "__main__":
    main()
