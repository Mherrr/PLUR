"""D. Macro model throughput + schedule optimizer risk reduction."""
import json
import time

import numpy as np
from bench_common import venue, build_setlist, synth_draw

from backend.sim.macro import MacroModel
from backend.optimize.schedule import ScheduleOptimizer, _score_schedule

TICKETS = 75000
CAP = 80000

v = venue()
sl = build_setlist(v)
draw = synth_draw(sl)
macro = MacroModel()
out = {"n_sets": len(sl), "n_stages": len({e["stage"] for e in sl})}

# ---- macro model throughput (this is what the optimizer scores against) ----
macro.run(setlist=sl, draw=draw, affinity={}, stages=v.stages,
          tickets_sold=TICKETS, max_capacity=CAP)
reps = 30
t0 = time.perf_counter()
for _ in range(reps):
    macro.run(setlist=sl, draw=draw, affinity={}, stages=v.stages,
              tickets_sold=TICKETS, max_capacity=CAP)
macro_s = (time.perf_counter() - t0) / reps
out["macro_run_s"] = macro_s
out["macro_runs_per_s"] = 1 / macro_s

mr = macro.run(setlist=sl, draw=draw, affinity={}, stages=v.stages,
               tickets_sold=TICKETS, max_capacity=CAP)
out["macro_time_bins"] = len(mr["attendance"])
out["macro_risk_windows"] = len(mr["risk_windows"])

# ---- single candidate score ----
t0 = time.perf_counter()
for _ in range(reps):
    _score_schedule(sl, draw, {}, v.stages, TICKETS, CAP, macro)
score_s = (time.perf_counter() - t0) / reps
out["candidate_score_s"] = score_s
out["candidate_scores_per_s_serial"] = 1 / score_s

# ---- optimizer, as the API calls it (n_iterations=100, n_jobs=4) ----
opt = ScheduleOptimizer()
for n_jobs in (1, 4):
    t0 = time.perf_counter()
    res = opt.optimize(
        setlist=sl, draw=draw, affinity={}, stages=v.stages,
        headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
        macro_model=macro, n_iterations=100, n_jobs=n_jobs,
    )
    wall = time.perf_counter() - t0
    swappable = len(sl)
    pairs_per_iter = min(20, swappable * (swappable - 1) // 2)
    candidates = 100 * pairs_per_iter
    row = {
        "n_jobs": n_jobs,
        "wall_clock_s": wall,
        "iterations": 100,
        "candidates_evaluated": candidates,
        "candidates_per_s": candidates / wall,
        "risk_before": res["risk_before"],
        "risk_after": res["risk_after"],
        "risk_reduction_pct": (1 - res["risk_after"] / res["risk_before"]) * 100,
        "n_changes": len(res["changes"]),
    }
    out[f"optimizer_n_jobs_{n_jobs}"] = row
    print(json.dumps(row, indent=2), flush=True)

out["optimizer_thread_speedup_x"] = (
    out["optimizer_n_jobs_1"]["wall_clock_s"] / out["optimizer_n_jobs_4"]["wall_clock_s"]
)

# ---- headliner lock respected? ----
locked = sorted(draw, key=draw.get, reverse=True)[:5]
res = opt.optimize(
    setlist=sl, draw=draw, affinity={}, stages=v.stages,
    headliners=locked, tickets_sold=TICKETS, max_capacity=CAP,
    macro_model=macro, n_iterations=50, n_jobs=4,
)
moved = {c["artist"] for c in res["changes"]}
out["headliners_locked"] = locked
out["locked_artists_moved"] = sorted(moved & set(locked))
out["headliner_lock_respected"] = len(moved & set(locked)) == 0

print(json.dumps(out, indent=2))
with open("results_d.json", "w") as f:
    json.dump(out, f, indent=2)
