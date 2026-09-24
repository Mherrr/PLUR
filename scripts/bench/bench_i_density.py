"""I. Is reported hotspot density a property of the crowd, or of n_agents?

Runs the SAME schedule at two agent counts and measures peak density two ways:
  (a) nearest-cell binning -- what festival.py does today
  (b) Gaussian-kernel smoothing -- the standard pedestrian-dynamics estimator
      (Steffen & Seyfried), where each agent's represented people are spread
      over a measurement radius instead of dumped into one cell.
If (a) moves with n_agents and (b) doesn't, (a) is a sampling artifact and no
single calibration constant can fix it.
"""
import json
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from bench_common import venue, build_setlist, synth_draw  # noqa: E402
from backend.sim.festival import run_festival  # noqa: E402

TICKETS = 75000
SIGMA_M = 2.0  # measurement radius for the smoothed estimator

v = venue()
# shortened day so the sweep is quick; density behaviour is unchanged
sl = [e for e in build_setlist(v) if e["start"] in ("15:00", "16:00", "17:00")]
draw = synth_draw(sl)
print(f"{len(sl)} sets, {len({e['stage'] for e in sl})} stages", flush=True)

occ = v.occupancy.astype(bool)
rows, cols = v.grid_shape
ox, oy = v.origin_m
cell_area = v.cell_m ** 2
wall_dist_m = distance_transform_edt(occ) * v.cell_m
# same spirit as festival.py's exclusions: ignore wall-adjacent artifacts
valid = occ & (wall_dist_m >= 5.0)

results = []
for n_agents in (2000, 8000):
    res = run_festival(venue=v, setlist=sl, draw=draw, tickets_sold=TICKETS,
                       n_agents=n_agents, affinity={})
    scale = TICKETS / n_agents

    peak_nearest = np.zeros((rows, cols))
    peak_smooth = np.zeros((rows, cols))
    for fr in res["frames"]:
        if not fr["agents"]:
            continue
        a = np.asarray(fr["agents"], dtype=float)
        x, y = v.to_utm(a[:, 0], a[:, 1])
        gi = np.clip(((y - oy) / v.cell_m).astype(int), 0, rows - 1)
        gj = np.clip(((x - ox) / v.cell_m).astype(int), 0, cols - 1)

        counts = np.zeros((rows, cols))
        np.add.at(counts, (gi, gj), 1.0)

        # (a) nearest-cell: all of an agent's people land in one cell
        np.maximum(peak_nearest, counts * scale / cell_area, out=peak_nearest)

        # (b) smoothed: spread each agent's people over SIGMA_M
        people = gaussian_filter(counts * scale, sigma=SIGMA_M / v.cell_m,
                                 mode="constant")
        np.maximum(peak_smooth, people / cell_area, out=peak_smooth)

    row = {
        "n_agents": n_agents,
        "people_per_agent": scale,
        "quantum_p_m2": scale / cell_area,
        "peak_nearest_raw": float(peak_nearest[valid].max()),
        "peak_nearest_displayed_0.56x": float(peak_nearest[valid].max() * 0.56),
        "peak_smoothed": float(peak_smooth[valid].max()),
        "cells_over_6_nearest": int((peak_nearest[valid] >= 6).sum()),
        "cells_over_6_smoothed": int((peak_smooth[valid] >= 6).sum()),
        "hotspots_reported": len(res["hotspots"]),
        "hotspot_densities": sorted({h["peak_density"] for h in res["hotspots"]}),
    }
    results.append(row)
    print(json.dumps(row, indent=2), flush=True)

a, b = results
print("\n--- ratio when agent count goes 2000 -> 8000 (4x) ---")
print(f"nearest-cell peak : {a['peak_nearest_raw']:.1f} -> {b['peak_nearest_raw']:.1f} "
      f"({b['peak_nearest_raw']/a['peak_nearest_raw']:.2f}x)")
print(f"smoothed peak     : {a['peak_smoothed']:.1f} -> {b['peak_smoothed']:.1f} "
      f"({b['peak_smoothed']/a['peak_smoothed']:.2f}x)")
with open("results_i.json", "w") as f:
    json.dump({"sigma_m": SIGMA_M, "runs": results}, f, indent=2)
