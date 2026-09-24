"""F. End-to-end API test through the live FastAPI server + Redis.

Also verifies the venue-resolution fix: a project uploaded with a MUTATED
geojson (one stage deleted) must be simulated and optimized against that
mutated map, not the bundled HARD Summer venue.
"""
import copy
import json
import time

import requests
from bench_common import REPO, build_setlist, synth_draw, venue

BASE = "http://127.0.0.1:8000"
out = {}


def timed(method, path, **kw):
    t0 = time.perf_counter()
    r = requests.request(method, BASE + path, timeout=1800, **kw)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


# ---- health ----
h, dt = timed("GET", "/health")
out["health"] = h
out["health_latency_ms"] = dt * 1000
print("health:", h)

v, dt = timed("GET", "/venues")
out["venues_latency_ms"] = dt * 1000
out["venues_listed"] = len(v)

vd, dt = timed("GET", "/venues/hard_summer_2025")
out["venue_detail_latency_ms"] = dt * 1000
out["venue_detail_grid"] = vd["grid"]

# ---- create a project whose map DIFFERS from the bundled venue ----
gj = json.loads((REPO / "backend" / "data" / "venues" / "hard_summer_2025" /
                 "venue.geojson").read_text())
meta = json.loads((REPO / "backend" / "data" / "venues" / "hard_summer_2025" /
                   "meta.json").read_text())

mutated = copy.deepcopy(gj)
before = [f for f in mutated["features"] if f["properties"].get("type") == "stage"]
# drop the Pink Stage -> project has 4 stages, bundled venue has 5
mutated["features"] = [
    f for f in mutated["features"]
    if not (f["properties"].get("type") == "stage"
            and f["properties"].get("stage_id") == "pink")
]
after = [f for f in mutated["features"] if f["properties"].get("type") == "stage"]
out["bundled_stage_count"] = len(before)
out["project_stage_count"] = len(after)

proj, dt = timed("POST", "/projects", json={
    "name": "BENCH Mutated Venue",
    "geojson": mutated,
    "meta": meta,
    "artists": [],
    "setlist": [],
})
pid = proj["id"]
out["create_project_latency_ms"] = dt * 1000
out["project_id"] = pid
print("project:", pid)

# ---- setlist restricted to the 4 surviving stages ----
vobj = venue()
surviving = [f["properties"]["stage_id"] for f in after]
sl = [e for e in build_setlist(vobj) if e["stage"] in surviving]
out["setlist_sets"] = len(sl)

# ---- simulate through the API ----
sim, dt = timed("POST", "/simulate_festival", json={
    "venue_id": "hard_summer_2025",
    "project_id": pid,
    "setlist": sl,
    "sliders": {"max_capacity": 80000, "tickets_sold": 75000, "n_agents": 2000},
    "barriers": [],
    "density_orange": 4.0,
    "density_red": 6.0,
})
out["simulate_latency_s"] = dt
out["simulate_frames"] = sim["n_frames"]
out["simulate_hotspots"] = len(sim["hotspots"])
out["hotspot_levels"] = sorted({hs.get("level") for hs in sim["hotspots"]})
out["hotspot_sample"] = sim["hotspots"][:3]
print(f"sim: {sim['n_frames']} frames, {len(sim['hotspots'])} hotspots in {dt:.1f}s")

# ---- saved sim round-trips through Redis ----
saved, dt = timed("GET", f"/projects/{pid}/sim")
out["get_saved_sim_latency_ms"] = dt * 1000
out["saved_sim_frames"] = len(saved["frames"])
out["redis_roundtrip_ok"] = len(saved["frames"]) == sim["n_frames"]

# ---- optimize through the API with project_id (the fix under test) ----
opt, dt = timed("POST", "/optimize_schedule", json={
    "venue_id": "hard_summer_2025",
    "project_id": pid,
    "setlist": sl,
    "headliners": [],
    "sliders": {"max_capacity": 80000, "tickets_sold": 75000},
})
out["optimize_latency_s"] = dt
out["optimize_risk_before"] = opt["risk_before"]
out["optimize_risk_after"] = opt["risk_after"]
out["optimize_reduction_pct"] = (1 - opt["risk_after"] / opt["risk_before"]) * 100
out["optimize_changes"] = len(opt["changes"])
out["optimize_stages_in_result"] = sorted({e["stage"] for e in opt["proposed_schedule"]})
out["optimize_used_project_map"] = (
    "pink" not in out["optimize_stages_in_result"]
)
print(f"optimize: {dt:.1f}s, stages in result {out['optimize_stages_in_result']}")

# ---- safety briefing (no API key -> placeholder path) ----
peak = max((hs["peak_density"] for hs in sim["hotspots"]), default=0.0)
brief, dt = timed("POST", "/safety_briefing", json={
    "venue_id": "hard_summer_2025",
    "project_id": pid,
    "setlist": sl,
    "sliders": {"max_capacity": 80000, "tickets_sold": 75000},
    "peak_density": peak,
    "hotspots": sim["hotspots"],
    "amenities": [],
})
out["briefing_latency_ms"] = dt * 1000
out["briefing_peak_density_sent"] = peak
out["briefing_nonzero_peak_density"] = peak > 0
out["briefing_text"] = brief["briefing"][:400]

# ---- demand scores endpoint ----
names = [e["artist"] for e in sl][:10]
dem, dt = timed("POST", "/demand/scores", json={"artists": names})
out["demand_latency_ms"] = dt * 1000
out["demand_returned"] = len(dem["draw"])

# ---- cleanup ----
requests.delete(BASE + f"/projects/{pid}", timeout=60)

print(json.dumps(out, indent=2))
with open("results_f.json", "w") as f:
    json.dump(out, f, indent=2)
