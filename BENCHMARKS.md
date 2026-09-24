# PLUR — Measured Performance

All numbers below were measured on a single machine, not estimated. Re-running
the scripts in `scripts/bench/` reproduces them.

**Hardware:** AMD Ryzen 7 7800X3D (8 physical / 16 logical cores, 4.2 GHz), 31.2 GB RAM, Windows 11 Pro (build 26200)
**Software:** Python 3.12.13, numpy 2.5.3, scipy 1.18.1, numba 0.67.0, shapely 2.1.2, dask 2026.8.0, Node 24.19.0
**Mode:** single-node (`DASK_SCHEDULER` unset — `/health` reports `distributed: false`)
**Venue:** bundled `hard_summer_2025`
**Schedule fixture:** 34 sets across 5 stages, 1-hour slots, 15:00–02:00

---

## 1. Venue model

| Metric | Value |
|---|---|
| Occupancy grid | 656 × 559 = **366,704 cells** |
| Walkable cells | **109,445** |
| Cell size | 1.5 m |
| Walkable area | 246,251 m² (**60.8 acres**) |
| Venue extent | 838 m × 984 m |
| GeoJSON features parsed | 38 (5 stages, 1 gate, 11 facilities) |
| Venue load (GeoJSON → UTM → rasterized grid) | **0.44 s** |
| Wall distance field (Euclidean DT + gradient) | 0.022 s |

## 2. Flow-field pathfinding

Dijkstra distance field per destination over the walkable grid, converted to a
unit-vector flow field, cached per destination.

| Metric | Value |
|---|---|
| Destinations routed (5 stages + gate + 11 amenities) | **17** |
| Per-destination build | **0.98 s** |
| Dijkstra throughput | **111,800 cells/s** |
| All 17 destinations | ~16.6 s (once per run, then cached) |

## 3. Social-force kernel

### numba JIT vs. pure Python

Identical algorithm both sides — the pure-Python timing calls
`_force_kernel.py_func`, the uncompiled body of the same function.

| N agents | Pure Python | numba JIT | Speedup |
|---|---|---|---|
| 400 | 5.40 ms/step | 0.027 ms/step | **200×** |

### Spatial hash vs. all-pairs

Both kernels are numba-compiled, so this isolates the data structure rather than
the compiler. The all-pairs kernel is a line-for-line mirror of the production
one with the 5×5 hash-cell stencil replaced by a full O(N²) loop.

| N agents | CSR spatial hash | All-pairs O(N²) | Speedup |
|---|---|---|---|
| 1,000 | 0.52 ms | 15.3 ms | 29× |
| 2,000 | 1.06 ms | 61.0 ms | 57× |
| 4,000 | 2.60 ms | 241.6 ms | 93× |
| 8,000 | **7.17 ms** | **970.3 ms** | **135×** |

Hashed cost scales ~linearly; all-pairs scales quadratically, so the gap widens
with crowd size. **The two kernels agree to 2.1 × 10⁻¹⁶ maximum relative
error** — the social term `A·exp((2r−d)/B)` with B = 0.08 m is ~e⁻¹²² at 10 m,
so the stencil discards nothing physically relevant. The speedup is free, not a
fidelity tradeoff.

Peak force-evaluation throughput: **1.9 M agent-steps/s** at N = 1,000.

## 4. Full-day simulation (end to end)

A complete event day — gates open through final egress — at dt = 0.5 s,
2-minute bins, 240 physics substeps per bin, 75,000 tickets sold.

| Agents | People/agent | Wall clock | Agent-physics-steps | Steps/s | Faster than real time | Payload |
|---|---|---|---|---|---|---|
| 1,000 | 75 | 51.8 s | 42.7 M | 823 K | **695×** | 6.1 MB |
| 2,000 | 37.5 | 78.0 s | 85.3 M | 1.09 M | 461× | 12.2 MB |
| 5,000 | 15 | 187.6 s | 213.0 M | 1.14 M | **192×** | 30.6 MB |
| 8,000 | 9.4 | 294.7 s | **340.9 M** | **1.16 M** | 122× | 49.0 MB |

Every run: 301 frames, 10.0 h of simulated event, ~99.5% of agents concurrently
active at peak. The 8,000-agent run resolves **340.9 million individual
agent-force evaluations in under 5 minutes**.

## 5. Macro model and schedule optimizer

| Metric | Value |
|---|---|
| Macro model full-day run (85 five-minute bins, 5 stages) | **7.2 ms** |
| Macro evaluations/s | **139** |
| Candidate schedule scores/s (serial) | 133 |
| Candidates evaluated per optimizer request (100 iterations × 20 swaps) | **2,000** |
| Optimizer wall clock (2,000 candidates) | 15.4 s |
| Risk windows detected on the fixture schedule | 4 |
| Closest stage pair (drives the adjacency penalty) | HARD ↔ Pink, **210 m** |

### joblib threading is a net loss

`_score_schedule` is GIL-bound Python/numpy, so `prefer="threads"` only adds
dispatch overhead. Measured over 800 candidates, bit-identical results:

| n_jobs | Wall clock | Candidates/s |
|---|---|---|
| **1** | **6.81 s** | **117** |
| 2 | 10.31 s | 78 |
| 4 | 13.32 s | 60 |
| 8 | 13.26 s | 60 |

The API previously called the optimizer with `n_jobs=4`, making it **1.96×
slower than serial**. It now calls `n_jobs=1`. Genuine parallelism comes from
the Dask path (`cluster.map_calls`), which bypasses joblib entirely.

### Optimizer effectiveness — before and after the move-set fix

The original search could only swap the `(stage, start, end)` triple between
two *filled* entries. Swaps permute which artist sits in which slot but leave
the set of occupied slots untouched, so the schedule's time distribution was
frozen — and reducing concurrency is precisely a matter of changing that
distribution. Adding a relocate-into-an-empty-slot move, plus a guard against
the uniform-draw collapse below, produced:

| Scenario | Before | After |
|---|---|---|
| Balanced fixture schedule, varied draw | **2.96%** | **29.49%** |
| Through the live API, no Last.fm key | **0.00%** | **35.84%** |
| Peak stage load (worst single moment) | 1.671 → 1.671× | **1.671 → 1.414×** |
| `/optimize_schedule` latency | 23.0 s | **12.7 s** |

Peak stage load is the number that matters for safety, and it is the one that
previously refused to move. The latency drop is incidental — candidate
construction switched from `copy.deepcopy` to a per-entry shallow copy, which
is equivalent for these flat dicts.

The output stays realistic rather than gaming the objective: all 5 stages
remain in use, sets per stage stay within 6–7, and maximum concurrency is
unchanged at 5 sets per hour. Headliner locking still holds — 5 locked
artists, **0 moved**. No slot collisions, and the artist roster is preserved
exactly.

**The uniform-draw collapse.** `DemandService` z-scores its draw components, so
with no Last.fm key every artist scores exactly 0.5. Under swap-only moves that
made every arrangement score identically and the optimizer strictly inert.
`main.py` now routes draw scores through `_ensure_varied_draw()`, which detects
the collapse and substitutes the slot-position heuristic — the previous
fallback only filled artists *missing* from the dict, and these were
present-but-uniform. Varied draws are still worth having: 29.49% with them
versus 24.27% feeding a uniform vector straight in.

### Still outstanding

- **The search still plateaus.** 300 iterations returns byte-identical results
  to 100 (same 24 changes, same 229.057 score). The strict hill-climb reaches a
  local minimum early, so two thirds of the iteration budget is wasted.
  Accepting equal-cost moves, random restarts, or annealing would each help.
- **The fixture barely exercises relocation** — its grid is 35 slots with 34
  filled, so only one hole exists to shuffle. A real schedule with more open
  airtime should gain more, which is untested here.
- **The objective is still a sum, not a max**, so it can trade a better total
  against the worst moment. Peak load improved this time as a side effect, not
  because anything targets it.

## 6. API layer (live server + Redis)

FastAPI on uvicorn, Redis-backed persistence, measured through HTTP.

| Endpoint | Latency |
|---|---|
| `GET /health` | 21 ms |
| `GET /venues/{id}` (cached venue) | 3.6 ms |
| `GET /venues` (cold, rasterizes each venue) | 603 ms |
| `POST /projects` (38-feature GeoJSON → Redis) | 18 ms |
| `POST /demand/scores` (10 artists, cached) | 2.3 ms |
| `POST /simulate_festival` (2,000 agents, full day) | 84.8 s |
| `GET /projects/{id}/sim` (301 frames from Redis) | 1.84 s |
| `POST /optimize_schedule` (2,000 candidates + rationale) | 23.0 s |
| `POST /safety_briefing` (placeholder path, no API key) | 536 ms |

A 301-frame simulation result round-trips through Redis intact.

## 7. Frontend build

| Metric | Value |
|---|---|
| Modules transformed | **973** |
| Build time (Vite 8) | **3.20 s** |
| App bundle | 1,090 KB (319 KB gzip) |
| MapLibre GL chunk | 1,028 KB (273 KB gzip) |
| CSS | 71 KB (10.6 KB gzip) |

---

## Caveats

These are engineering throughput numbers, not validated safety predictions.

- **Density is now kernel-measured, but its resolution still depends on agent
  count.** The original estimator binned each agent into one 1.5 m cell and
  multiplied by `tickets_sold / n_agents`, so one agent registered
  `scale / 2.25` p/m² at once — 16.7 p/m² at 2,000 agents. Every reported
  hotspot density was an exact integer multiple of that quantum (the observed
  28.0 and 18.7 were precisely 3 and 2 agents in a cell), which made the number
  a readout of the agent-count slider rather than of the crowd. No constant
  could fix it: landing near 6 p/m² needed 0.09 at 2,000 agents and 0.16 at
  8,000 for the same crowd, and 46% of the venue read as crush risk.

  `festival.py` now spreads each agent's represented group over a
  `DENSITY_SIGMA_M = 2.0` m Gaussian kernel (Steffen & Seyfried's estimator),
  with an edge correction dividing by the share of the kernel that lands on
  walkable ground so density in narrow corridors isn't understated. The 0.56
  display factor is gone. Same schedule, same two agent counts:

  | | 2,000 agents | 8,000 agents | Ratio |
  |---|---|---|---|
  | Old: nearest-cell peak | 66.7 p/m² | 37.5 p/m² | 0.56× |
  | Old: reported hotspots | 28.0 | 9.3 – 11.7 | **0.33×** |
  | New: reported hotspots | 7.8 – 9.5 | 5.0 – 5.7 | **0.60×** |

  The values are now physically plausible — crush conditions begin around
  6 p/m², and these land either side of it instead of at 28. **The residual
  0.60× drift is a real limitation, not a fixed bug.** At 2,000 agents each
  agent still carries 37.5 people, so the kernel's smallest resolvable
  increment is 1.49 p/m² — a quarter of the danger threshold. At 8,000 agents
  it is 0.37 p/m². Density-sensitive work should use ≥5,000 agents; the
  estimator now degrades gracefully with agent count instead of rescaling
  wholesale.

- **Hotspot levels still skew red.** Detection compares the bottleneck-weighted
  danger score against thresholds denominated in raw density, so the comparison
  is dimensionally inconsistent and nearly everything clears the red line. The
  orange tier remains largely theoretical. Unfixed.
- **Reported pressure is `0.3 × density`**, a display stand-in. The real
  velocity-variance pressure metric in `sim/risk.py` is not wired into the live
  path.
- The `/simulate_festival` payload reaches **49 MB** at 8,000 agents, all of it
  JSON agent coordinates held in Redis and shipped to the browser. This is the
  first thing that would need binary framing or downsampling to scale.
- **Nothing is validated against real crowd data.** Plausible magnitudes are
  not the same as calibrated ones. Treat hotspot locations as directional and
  every absolute figure as unverified.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/bench/bench_a_venue.py    # venue + pathfinding
.venv/Scripts/python.exe scripts/bench/bench_b_kernel.py   # JIT + spatial hash
.venv/Scripts/python.exe scripts/bench/bench_c_sim.py      # full-day sweep (~10 min)
.venv/Scripts/python.exe scripts/bench/bench_d_opt.py      # macro + optimizer
.venv/Scripts/python.exe scripts/bench/bench_e_opt2.py     # n_jobs sweep + bad schedule
# with the server running and Redis up:
.venv/Scripts/python.exe scripts/bench/bench_f_api.py      # end-to-end HTTP
.venv/Scripts/python.exe scripts/bench/bench_i_density.py  # density estimator comparison
.venv/Scripts/python.exe scripts/bench/bench_j_afterfix.py # after-fix verification
```
