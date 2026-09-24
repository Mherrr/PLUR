"""C. End-to-end full-day festival simulation, swept over agent count."""
import json
import time

import numpy as np
from bench_common import venue, build_setlist, synth_draw

from backend.sim.festival import run_festival

TICKETS = 75000
DT = 0.5
BIN_MIN = 2.0
STEPS_PER_BIN = int(BIN_MIN * 60 / DT)

v = venue()
sl = build_setlist(v)
draw = synth_draw(sl)

print(f"venue grid {v.grid_shape}, {len(sl)} sets on {len({e['stage'] for e in sl})} stages")
print(f"steps per {BIN_MIN}-min bin: {STEPS_PER_BIN} (dt={DT}s)")

runs = []
for n_agents in (1000, 2000, 5000, 8000):
    t0 = time.perf_counter()
    res = run_festival(
        venue=v,
        setlist=sl,
        draw=draw,
        tickets_sold=TICKETS,
        n_agents=n_agents,
        dt=DT,
        sim_bin_minutes=BIN_MIN,
        affinity={},
    )
    wall = time.perf_counter() - t0

    frames = res["frames"]
    n_active = [f["n_active"] for f in frames]
    t_mins = [f["t_min"] for f in frames]
    event_span_min = (max(t_mins) - min(t_mins)) if t_mins else 0
    agent_steps = sum(n_active) * STEPS_PER_BIN
    payload = len(json.dumps(res, separators=(",", ":")).encode())

    row = {
        "n_agents": n_agents,
        "people_represented": TICKETS,
        "people_per_agent": TICKETS / n_agents,
        "wall_clock_s": wall,
        "frames": len(frames),
        "event_span_min": event_span_min,
        "event_span_h": event_span_min / 60,
        "peak_concurrent_agents": max(n_active) if n_active else 0,
        "peak_concurrent_people": (max(n_active) if n_active else 0) * TICKETS / n_agents,
        "agent_physics_steps": agent_steps,
        "agent_steps_per_s": agent_steps / wall,
        "sim_min_per_wall_s": event_span_min / wall,
        "realtime_factor": (event_span_min * 60) / wall,
        "hotspots": len(res["hotspots"]),
        "payload_bytes": payload,
        "payload_mb": payload / 1e6,
        "agent_coords_emitted": sum(len(f["agents"]) for f in frames),
    }
    runs.append(row)
    print(json.dumps(row, indent=2), flush=True)
    with open("results_c.json", "w") as f:
        json.dump({"runs": runs, "steps_per_bin": STEPS_PER_BIN,
                   "dt_s": DT, "bin_minutes": BIN_MIN,
                   "n_sets": len(sl)}, f, indent=2)

print("DONE")
