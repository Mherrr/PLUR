"""E. Follow-ups: (1) is joblib threading helping at all? (2) how much risk
does the optimizer remove from a genuinely badly-balanced schedule?"""
import copy
import json
import time

import numpy as np
from bench_common import venue, build_setlist, synth_draw

from backend.sim.macro import MacroModel
from backend.optimize.schedule import ScheduleOptimizer, _score_schedule

TICKETS, CAP = 75000, 80000
v = venue()
macro = MacroModel()
opt = ScheduleOptimizer()
out = {}

sl = build_setlist(v)
draw = synth_draw(sl)

# ---------- 1. joblib n_jobs sweep (local, non-Dask path) ----------
sweep = []
for n_jobs in (1, 2, 4, 8):
    t0 = time.perf_counter()
    opt.optimize(setlist=sl, draw=draw, affinity={}, stages=v.stages,
                 headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                 macro_model=macro, n_iterations=40, n_jobs=n_jobs)
    wall = time.perf_counter() - t0
    sweep.append({"n_jobs": n_jobs, "wall_s": wall,
                  "candidates_per_s": 40 * 20 / wall})
    print(f"  n_jobs={n_jobs}: {wall:.2f}s", flush=True)
out["n_jobs_sweep_40_iters"] = sweep
best = min(sweep, key=lambda r: r["wall_s"])
worst_shipped = next(r for r in sweep if r["n_jobs"] == 4)
out["best_n_jobs"] = best["n_jobs"]
out["shipped_n_jobs4_penalty_x"] = worst_shipped["wall_s"] / best["wall_s"]

# ---------- 2. optimizer on a deliberately bad schedule ----------
# Stack the five highest-draw acts into the SAME time slot, and put them on
# the two stages the adjacency term says are closest together. This is the
# failure mode a human scheduler actually creates.
stage_ids = [s["id"] for s in v.stages]
pos = {s["id"]: np.array(s["pos_m"]) for s in v.stages}
pairs = [(a, b, float(np.linalg.norm(pos[a] - pos[b])))
         for i, a in enumerate(stage_ids) for b in stage_ids[i + 1:]]
closest = min(pairs, key=lambda p: p[2])
out["closest_stage_pair"] = {"a": closest[0], "b": closest[1],
                             "distance_m": closest[2]}

bad = copy.deepcopy(sl)
top = sorted(draw, key=draw.get, reverse=True)[:6]
# force all top acts concurrent at 23:00 on the two adjacent stages
for i, artist in enumerate(top):
    for e in bad:
        if e["artist"] == artist:
            e["stage"] = closest[0] if i % 2 == 0 else closest[1]
            e["start"], e["end"] = "23:00", "00:00"
# displace whoever already held those slots to spread-out later slots
seen = set()
for e in bad:
    key = (e["stage"], e["start"])
    if e["artist"] in top:
        continue
    while key in seen or (e["start"] == "23:00" and e["stage"] in (closest[0], closest[1])):
        h = (int(e["start"][:2]) + 1) % 24
        e["start"], e["end"] = f"{h:02d}:00", f"{(h + 1) % 24:02d}:00"
        key = (e["stage"], e["start"])
    seen.add(key)

bad_score = _score_schedule(bad, draw, {}, v.stages, TICKETS, CAP, macro)
balanced_score = _score_schedule(sl, draw, {}, v.stages, TICKETS, CAP, macro)
out["balanced_schedule_risk"] = balanced_score
out["bad_schedule_risk"] = bad_score
out["bad_vs_balanced_x"] = bad_score / balanced_score

for n_iters in (100, 300):
    t0 = time.perf_counter()
    res = opt.optimize(setlist=bad, draw=draw, affinity={}, stages=v.stages,
                       headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                       macro_model=macro, n_iterations=n_iters, n_jobs=1)
    wall = time.perf_counter() - t0
    row = {
        "n_iterations": n_iters,
        "wall_s": wall,
        "candidates_evaluated": n_iters * 20,
        "candidates_per_s": n_iters * 20 / wall,
        "risk_before": res["risk_before"],
        "risk_after": res["risk_after"],
        "risk_reduction_pct": (1 - res["risk_after"] / res["risk_before"]) * 100,
        "n_changes": len(res["changes"]),
    }
    out[f"bad_schedule_opt_{n_iters}"] = row
    print(json.dumps(row, indent=2), flush=True)

# ---------- 3. peak stage load, before vs after ----------
def peak_load(schedule):
    r = macro.run(setlist=schedule, draw=draw, affinity={}, stages=v.stages,
                  tickets_sold=TICKETS, max_capacity=CAP)
    caps = r["stage_safe_capacity"]
    worst = 0.0
    for sid, pops in r["stage_pop"].items():
        cap = caps.get(sid, 1.0)
        for e in pops:
            worst = max(worst, e["pop"] / cap)
    return worst

res300 = opt.optimize(setlist=bad, draw=draw, affinity={}, stages=v.stages,
                      headliners=[], tickets_sold=TICKETS, max_capacity=CAP,
                      macro_model=macro, n_iterations=300, n_jobs=1)
out["peak_stage_load_ratio_before"] = peak_load(bad)
out["peak_stage_load_ratio_after"] = peak_load(res300["proposed_schedule"])

print(json.dumps(out, indent=2))
with open("results_e.json", "w") as f:
    json.dump(out, f, indent=2)
