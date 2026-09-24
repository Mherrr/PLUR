"""J. After-fix verification: density estimator and optimizer move set.

Density: the same schedule at two agent counts should now report densities that
converge rather than tracking the agent-count slider.

Optimizer: relocations into empty slots should beat a swap-only search, and a
uniform draw vector (the no-API-key case) should no longer leave it inert.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import venue, build_setlist, synth_draw  # noqa: E402

from backend.sim.festival import run_festival, DENSITY_SIGMA_M  # noqa: E402
from backend.sim.macro import MacroModel  # noqa: E402
from backend.optimize.schedule import (  # noqa: E402
    ScheduleOptimizer, _score_schedule, _slot_grid, _empty_slots,
)

TICKETS, CAP = 75000, 80000
v = venue()
out = {"density_sigma_m": DENSITY_SIGMA_M}

# ---------------- 1. density estimator ----------------
short = [e for e in build_setlist(v) if e["start"] in ("15:00", "16:00", "17:00")]
short_draw = synth_draw(short)
dens = []
for n_agents in (2000, 8000):
    res = run_festival(venue=v, setlist=short, draw=short_draw,
                       tickets_sold=TICKETS, n_agents=n_agents, affinity={})
    hs = [h["peak_density"] for h in res["hotspots"]]
    dens.append({
        "n_agents": n_agents,
        "people_per_agent": TICKETS / n_agents,
        "hotspots": len(res["hotspots"]),
        "hotspot_densities": sorted(hs),
        "max_hotspot_density": max(hs) if hs else 0.0,
        "levels": sorted({h["level"] for h in res["hotspots"]}),
    })
    print(json.dumps(dens[-1], indent=2), flush=True)
out["density_runs"] = dens
if dens[0]["max_hotspot_density"] and dens[1]["max_hotspot_density"]:
    out["density_ratio_2k_to_8k"] = (
        dens[1]["max_hotspot_density"] / dens[0]["max_hotspot_density"]
    )
    print(f"\ndensity ratio 2k->8k: {out['density_ratio_2k_to_8k']:.2f}x "
          f"(was 0.33x with nearest-cell binning)\n", flush=True)

# ---------------- 2. optimizer ----------------
macro = MacroModel()
opt = ScheduleOptimizer()
full = build_setlist(v)
draw = synth_draw(full)

grid = _slot_grid(full, v.stages)
out["grid_slots"] = len(grid)
out["filled_slots"] = len(full)
out["empty_slots_now_reachable"] = len(_empty_slots(full, grid))
print(f"grid {len(grid)} slots, {len(full)} filled, "
      f"{out['empty_slots_now_reachable']} empty (previously unreachable)", flush=True)

for n_iters in (100, 300):
    t0 = time.perf_counter()
    res = opt.optimize(setlist=full, draw=draw, affinity={}, stages=v.stages,
                       headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                       macro_model=macro, n_iterations=n_iters, n_jobs=1)
    wall = time.perf_counter() - t0
    row = {
        "n_iterations": n_iters,
        "wall_s": wall,
        "risk_before": res["risk_before"],
        "risk_after": res["risk_after"],
        "risk_reduction_pct": (1 - res["risk_after"] / res["risk_before"]) * 100,
        "n_changes": len(res["changes"]),
    }
    out[f"opt_{n_iters}"] = row
    print(json.dumps(row, indent=2), flush=True)

# ---------------- 3. uniform draw (no API key) ----------------
uniform = {e["artist"]: 0.5 for e in full}
res = opt.optimize(setlist=full, draw=uniform, affinity={}, stages=v.stages,
                   headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                   macro_model=macro, n_iterations=100, n_jobs=1)
out["uniform_draw_raw"] = {
    "risk_before": res["risk_before"],
    "risk_after": res["risk_after"],
    "risk_reduction_pct": (1 - res["risk_after"] / res["risk_before"]) * 100,
}
print("uniform draw straight into optimizer:",
      json.dumps(out["uniform_draw_raw"], indent=2), flush=True)

# through the API's guard, which is what actually runs now
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.main import _ensure_varied_draw  # noqa: E402

guarded = _ensure_varied_draw(uniform, full)
res = opt.optimize(setlist=full, draw=guarded, affinity={}, stages=v.stages,
                   headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                   macro_model=macro, n_iterations=100, n_jobs=1)
out["uniform_draw_guarded"] = {
    "distinct_draw_values": len(set(guarded.values())),
    "risk_before": res["risk_before"],
    "risk_after": res["risk_after"],
    "risk_reduction_pct": (1 - res["risk_after"] / res["risk_before"]) * 100,
    "n_changes": len(res["changes"]),
}
print("uniform draw through _ensure_varied_draw:",
      json.dumps(out["uniform_draw_guarded"], indent=2), flush=True)

# ---------------- 4. headliner lock still honoured ----------------
locked = sorted(draw, key=draw.get, reverse=True)[:5]
res = opt.optimize(setlist=full, draw=draw, affinity={}, stages=v.stages,
                   headliners=locked, tickets_sold=TICKETS, max_capacity=CAP,
                   macro_model=macro, n_iterations=100, n_jobs=1)
moved = {c["artist"] for c in res["changes"]}
out["headliner_lock_respected"] = len(moved & set(locked)) == 0
out["locked_moved"] = sorted(moved & set(locked))
print("headliner lock respected:", out["headliner_lock_respected"], flush=True)

# ---------------- 5. no duplicate slot assignments ----------------
slots = [(e["stage"], e["start"]) for e in res["proposed_schedule"]]
out["no_slot_collisions"] = len(slots) == len(set(slots))
out["artists_preserved"] = (
    sorted(e["artist"] for e in res["proposed_schedule"])
    == sorted(e["artist"] for e in full)
)
print("no slot collisions:", out["no_slot_collisions"],
      "| artists preserved:", out["artists_preserved"], flush=True)

with open("results_j.json", "w") as f:
    json.dump(out, f, indent=2)
