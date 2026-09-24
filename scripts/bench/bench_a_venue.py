"""A. Venue / grid / pathfinding scale + load timings."""
import json
import time

import numpy as np
from bench_common import DATA_DIR, venue, build_setlist, fmt

from backend.venue.loader import load_venue
from backend.sim.pathfinding import FlowFieldCache
from backend.sim.micro import _precompute_walls

out = {}

# --- venue load time (median of 5) ---
times = []
for _ in range(5):
    t0 = time.perf_counter()
    load_venue("hard_summer_2025", DATA_DIR)
    times.append(time.perf_counter() - t0)
v = venue()
out["venue_load_median_s"] = float(np.median(times))

rows, cols = v.grid_shape
total_cells = rows * cols
walkable = int(v.occupancy.sum())
cell_area = v.cell_m ** 2

out.update({
    "grid_rows": rows,
    "grid_cols": cols,
    "grid_total_cells": total_cells,
    "grid_walkable_cells": walkable,
    "cell_m": v.cell_m,
    "walkable_area_m2": walkable * cell_area,
    "walkable_area_acres": walkable * cell_area / 4046.86,
    "extent_x_m": cols * v.cell_m,
    "extent_y_m": rows * v.cell_m,
    "n_stages": len(v.stages),
    "n_gates": len(v.gates),
    "n_facilities": len(v.facilities),
    "geojson_features": len(v.geojson["features"]),
})

# --- wall distance field ---
occ = v.occupancy.astype(bool)
t0 = time.perf_counter()
_precompute_walls(occ, v.cell_m)
out["wall_field_s"] = time.perf_counter() - t0

# --- flow field precompute (Dijkstra + flow vectors) per destination ---
amenity_types = {"restroom", "water", "bar"}
n_amenities = sum(1 for f in v.facilities if f.get("facility_type") in amenity_types)
n_dest = len(v.stages) + len(v.gates) + n_amenities
out["n_flow_destinations"] = n_dest

cache = FlowFieldCache(occ, v.cell_m, v.origin_m)
per = []
for s in v.stages:
    t0 = time.perf_counter()
    cache.get_flow(f"stage_{s['id']}", (s["pos_m"][0], s["pos_m"][1]))
    per.append(time.perf_counter() - t0)
out["flow_field_per_dest_median_s"] = float(np.median(per))
out["flow_field_all_dests_projected_s"] = float(np.median(per)) * n_dest
out["dijkstra_cells_per_s"] = walkable / float(np.median(per))

# --- setlist fixture facts ---
sl = build_setlist(v)
out["setlist_sets"] = len(sl)
out["setlist_stages"] = len({e["stage"] for e in sl})

print(json.dumps(out, indent=2))
with open("results_a.json", "w") as f:
    json.dump(out, f, indent=2)
