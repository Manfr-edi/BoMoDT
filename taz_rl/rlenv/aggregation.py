import pandas as pd
import numpy as np

def aggregate_tls_metrics(detector_records):
    veh_total = sum(r["nVeh"] for r in detector_records)

    if veh_total > 0:
        mean_speed = sum(r["meanSpeed"] * r["nVeh"] for r in detector_records) / veh_total
    else:
        mean_speed = 0.0

    mean_occupancy = sum(r["occupancy"] for r in detector_records) / len(detector_records)
    max_jam = max(r["maxJamLength"] for r in detector_records)

    # proxy semplice di pressione
    pressure = veh_total * (1 - mean_speed)

    return {
        "veh_total": veh_total,
        "mean_speed": mean_speed,
        "mean_occupancy": mean_occupancy,
        "max_jam": max_jam,
        "pressure": pressure
    }


def aggregate_taz_metrics(tls_metrics):
    veh_totals = [m["veh_total"] for m in tls_metrics]
    occupancies = [m["mean_occupancy"]/100 for m in tls_metrics]
    speeds = [m["mean_speed"] for m in tls_metrics]

    total_veh = sum(veh_totals)

    if total_veh > 0:
        mean_speed = sum(speed * veh for speed, veh in zip(speeds,veh_totals)) / total_veh
    else:
        mean_speed = 0.0

    return {
        "veh_total": total_veh,
        "mean_occupancy": sum(occupancies) / len(occupancies),
        "max_occupancy": max(occupancies),
        "std_occupancy": pd.Series(occupancies).std(),
        "mean_speed": mean_speed,
        "critical_ratio": sum(o > 0.7 for o in occupancies) / len(occupancies)
    }


