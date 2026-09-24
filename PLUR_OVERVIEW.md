# PLUR — Project Overview (context brief)

*This document exists to give an assistant enough accurate detail about PLUR to write applications, cover letters, outreach emails, and resume bullets without inventing anything. Everything below was verified against the source code, not the README or design docs. Claims marked "aspirational" were planned but are not in the shipped code — do not present them as built.*

---

## One-line version

PLUR (Predictive Large-scale User Routing) is a crowd-crush prediction and mitigation tool for multi-stage music festivals: it runs an agent-based social-force simulation of a full event day over a real venue footprint, flags where and when dangerous crowd density forms, and lets event-safety planners intervene by drawing barriers, moving amenities, or handing the set-times schedule to an optimizer that rearranges the lineup to reduce risk.

## Paragraph version

PLUR takes a festival venue as a GeoJSON footprint and a lineup as a stage-by-time grid, and answers the question event-safety teams cannot currently answer before an event: *where will people get crushed, and what should we change?* It projects the venue into UTM and rasterizes it to a walkable occupancy grid, pulls artist popularity and fan-affinity data from Last.fm and Ticketmaster to estimate how the crowd will split across concurrent sets, then simulates the entire day — gates open to final egress — with thousands of individual agents driven by a numba-compiled Helbing–Molnar social-force model over flow-field pathfinding. It tracks peak density per grid cell, weights it by corridor narrowness, excludes the places crowds are supposed to be dense, and surfaces the remaining hazards as clustered hotspots on an interactive deck.gl map. Planners can then draw and rotate physical barriers directly on the map, drag restrooms and water stations and bars to new positions, and re-run; or submit the schedule to a parallel local-search optimizer that permutes unlocked artists across slots, scoring thousands of candidate lineups against a fast analytic crowd model. An Anthropic Claude agent writes the plain-text rationale for each schedule change and a full pre-event safety briefing with specific amenity-repositioning advice.

---

## Context and constraints

- **Format:** two-day hackathon build (commits span 2026-06-20 to 2026-06-21), targeting three tracks — Ddoski's Lab, Anthropic, and Most Technical.
- **Team:** 3 contributors, 26 commits on `main`. Work was split into four declared workstreams with dedicated per-workstream agent instructions committed into the repo (`.claude/commands/{sim,data,frontend,infra}.md`): simulation engine, data/ML, frontend, and integration/infrastructure.
- **Test venue:** HARD Summer 2025, the Hollywood Park grounds wrapping around SoFi Stadium in Inglewood, CA. 80,000 capacity, 5 mapped stages, one gate. The venue was hand-traced from the official festival map, including back-of-house zones, VIP areas, vendor structures, and — critically — the north/south barrier pair whose single gap is the venue's real crush-risk bottleneck.
- **Real-world framing:** deliberately positioned as a "decision-support prototype, not a certified life-safety system." Every generated briefing ends on that disclaimer, hardcoded into the prompt.

---

## Technical architecture (all verified in code)

### Backend — Python 3.12, FastAPI

**Geospatial pipeline** (`backend/venue/loader.py`)
GeoJSON FeatureCollection in WGS84 -> typed feature extraction -> shapely union of walkable polygons minus obstacle polygons -> pyproj reprojection to UTM Zone 11N (EPSG:32611) -> vectorized `shapely.contains` rasterization onto a 1.5 m boolean occupancy grid. All simulation math runs in meters; lon/lat conversion happens only at the API boundary.

**Demand modeling** (`backend/demand/service.py`)
Composite artist draw score, z-scored and blended:
`0.39 * z(sqrt(listeners * playcount)) + 0.33 * z(1/us_regional_rank) + 0.28 * z(ticketmaster_venue_capacity)`, normalized to [0,1]. Last.fm `artist.getSimilar` match scores build a symmetric pairwise fan-affinity matrix that drives inter-stage migration. B2B and collaboration sets ("DJ Seinfeld B2B Luuk van Dijk") are regex-split, fetched per member, and aggregated — listeners summed, tags unioned, affinities maxed. Every API response is SHA-hashed and cached to JSON so the demo never touches a live endpoint. Graceful degradation: with no API key, draw falls back to a schedule-position heuristic capped below headliner territory.

**Macroscopic crowd model** (`backend/sim/macro.py`)
Analytic per-5-minute-bin model of the whole day. Arrival inflow as a draw-weighted sum of Gaussians centered on set start times, integrated to a CDF against tickets sold. Concurrent-set audience split by softmax over *effective* draw, where effective draw includes an **affinity-bleed** term: high-affinity acts on stages within 400 m leak up to 15% of share to each other, softening winner-takes-all. Set-end migration redistributes crowd 60% on destination draw / 40% on affinity to the act just watched, decayed by inter-stage distance. Produces per-stage population curves and merged risk windows where a stage exceeds 120% of safe capacity.

**Agent simulation** (`backend/sim/festival.py`, `micro.py`, `pathfinding.py`) — the technical centerpiece
- **Helbing–Molnar social-force model** with granular contact, JIT-compiled via `numba.njit`: driving force with tau = 0.5 s relaxation, exponential agent-agent repulsion `A*exp((2r-d)/B)` at A = 2000 N / B = 0.08 m, body-compression force `k*overlap` and tangential sliding friction `kappa*overlap*dv_t` on physical contact, plus wall repulsion sampled from a precomputed Euclidean distance transform and its gradient.
- **CSR spatial hash** for neighbor lookup on 2 m cells — argsort into a compressed sparse row layout, rebuilt every physics step, scanned over a 5x5 stencil. Turns the O(N^2) pairwise force problem into O(N).
- **Flow-field pathfinding**: per-destination Dijkstra distance fields over the occupancy grid with a wall-proximity cost multiplier (`1 + 4/(d_wall+1)`) so agents route down corridor centers instead of hugging walls; converted to unit-vector flow fields, cached per destination (each stage, the gate, every amenity), sampled per agent with angular noise. Blocked moves decompose into axis-wise retries so agents slide along walls rather than bouncing off them.
- **Behavioral layer**: logistic arrival curve from gates-open to music-end; first stage sampled by softmax over opening-act draws; set-end migration weighted by draw and affinity; amenity detours with per-facility dwell times (45 s water / 90 s restroom / 4 min bar), queue-length-sensitive and capped at 15 min, with idling agents dropped to 0.02 m/s so they queue in place and still exert crowd pressure; a 21+ eligibility flag gating bar visits; progressive egress-to-gate routing after the last set.
- Runs in 2-minute bins at dt = 0.5 s (240 physics steps per bin), up to 8000 agents, each scaled to represent `tickets_sold / n_agents` real attendees.

**Hotspot extraction**
Peak per-cell density tracked across the entire run, then: exclusion masks for wall-adjacent rasterization artifacts, stage-front audience zones (60 m), and the gate apron (30 m); a bottleneck weight of `clip(3 - d_wall/10, 1, 3)` so a 5 m corridor scores 3x an open field at identical density; thresholding against a user-set red density; density-weighted centroid clustering at 30 m followed by iterative merging until no two hotspots sit within 100 m.

**Schedule optimizer** (`backend/optimize/schedule.py`)
Stochastic local search over slot permutations — each move swaps the `(stage, start, end)` triple between two entries. 100 iterations x 20 candidate swaps scored in parallel = up to ~2000 macro-model evaluations per request. Locked slots and named headliners are immovable; auto-filled entries carry 3x sampling weight so the optimizer prefers moving machine-placed acts over human-placed ones. The objective combines a **cubic** per-stage load penalty `(pop/safe_cap)^3` — chosen over a threshold-gated quadratic specifically so that 95%-full isn't treated as free — with a **corridor adjacency penalty** `ra * rb * exp(-d/200) * 0.35` across every stage pair within 500 m loaded in the same time bin, capturing transit pinch points the per-stage model can't see.

**Distributed execution** (`backend/cluster.py`)
Dual-mode dispatcher: with `DASK_SCHEDULER` set it connects a `dask.distributed.Client` and fans work out; unset, empty, or unreachable, it falls back transparently to in-process execution with `joblib.Parallel`. The documented deployment target is an ESXi cluster of 5-6 Ubuntu 26.04 VMs at 8 vCPUs each (~44 cores, bounded by the per-host licensing cap), one coordinator running FastAPI plus the Dask scheduler with a Tailscale tunnel to the demo laptop, the rest running `dask worker --nworkers 8 --nthreads 1`, provisioned by building one golden VM with a locked conda environment and cloning it with `ovftool`.

**Important nuance for accurate claims:** the cluster parallelizes the *schedule optimizer* (independent candidate scores are embarrassingly parallel). A single day-long simulation is sequential in time and cannot be split across cores — distributing it just offloads the whole run to one worker to keep the API responsive. Say "distributed candidate search across a Dask cluster," not "distributed simulation."

**Claude integration** (`backend/agent/claude.py`)
Two prompts against the Anthropic Messages API. The schedule rationale takes the change list and risk delta and returns one line per move with a plain-language justification. The safety briefing is seeded with macro risk windows, the full schedule, peak density, and live amenity coordinates, and returns a four-section plain-text briefing (executive summary, risk assessment, recommendations, amenity placement) ending on a fixed non-certification disclaimer. Both are prompt-constrained to plain text — no markdown, no bullets — because output renders into fixed-width UI panels. Both degrade to deterministic placeholder text without an API key and append the exception inline rather than failing the request.

**Persistence** (`backend/store/projects.py`)
Redis via `redis.asyncio`. Projects stored as JSON blobs with a sorted-set index for recency ordering; simulation results cached separately under `sim:{id}` with compact separators, since a full run is thousands of frames of agent coordinates. List responses strip the GeoJSON payload; a walkable-polygon thumbnail is extracted at creation time for card rendering.

### Frontend — React 19 + Vite + deck.gl

- **deck.gl 9 / MapLibre GL 5** rendering stack over a vector-only dark basemap at 45-degree pitch. No tile provider and no Mapbox token — venue geometry is drawn entirely from the project's own GeoJSON across `PolygonLayer` (walkable area, obstacles, VIP zones, back-of-house, user barriers), `ScatterplotLayer` (agents colored by speed, gates, amenities, hotspot rings), `TextLayer` (stage labels placed at back-of-house centroids), and `HeatmapLayer` (crowd density).
- **Direct-manipulation map editing** built by hand on deck.gl's pick/drag events: click to place a barrier, then drag to move, drag corner handles to resize (with correct rotation-aware local-space math around the opposite corner), or drag a handle to rotate. Amenities are draggable dots whose positions feed both the simulation's attractor set and the Claude briefing prompt.
- **Three-step project wizard**: drag-and-drop GeoJSON upload with a hand-rolled SVG preview and auto-detected stage list; festival metadata and artist roster entry with multi-line paste and `.txt` roster upload; and a drag-and-drop stage/time schedule grid.
- **Auto-fill** calls the backend for Last.fm draw scores and places unassigned artists by ascending draw into slots by ascending start time — openers early, headliners last — interleaving across stages rather than filling one stage at a time. Slots are lockable by double-click, and auto-filled vs. optimizer-placed vs. manual entries are visually distinguished.
- **Playback**: frame scrubbing and variable-speed playback of the simulation timeline, with layer toggles for heatmap / individual agents / hotspots.
- **Landing page**: a custom canvas particle animation that samples the rendered text "PLUR" to pixel targets and eases thousands of particles from random scatter into the wordmark, then transitions into a continuous wave, with a blue-to-purple gradient. Plays once per hard refresh, skipped on SPA navigation.

---

## Skills this project demonstrates

**Scientific computing / numerical simulation**
Implementing a published pedestrian-dynamics model (Helbing–Molnar social force with granular contact) from the physics rather than from a library. Numba JIT compilation of the force kernel. Spatial data structures for N-body-style problems (CSR spatial hashing). Graph search over rasterized geometry (Dijkstra distance fields, flow-field navigation with cost shaping). Scipy signal/morphology work — Euclidean distance transforms, uniform filtering for local velocity moments, connected-component labeling. Numerical stability and integration choices (velocity clamping, wall-sliding collision response, fixed-timestep substepping).

**Geospatial engineering**
Coordinate-system discipline — WGS84 to UTM projection with pyproj, all physics in meters, conversion only at I/O boundaries. Shapely 2.x vectorized geometry operations, polygon boolean algebra, rasterization of vector geometry to occupancy grids. Runtime modification of the navigable environment (user-drawn barriers rasterized into the grid between runs).

**Optimization / search**
Stochastic local search with weighted neighborhood sampling and constraint handling (locked slots, headliner protection). Objective-function design informed by domain reasoning — the choice of a cubic penalty over a gated quadratic, and the addition of a distance-decayed pairwise adjacency term, are both deliberate modeling decisions with stated rationale.

**Distributed systems / infrastructure**
Dask distributed scheduler/worker topology on self-provisioned ESXi VMs. Golden-image build-and-clone provisioning with `ovftool`, pinned conda environments for cross-node version consistency, Tailscale tunneling. Graceful degradation design — the same code path runs identically on one laptop or 44 cores, with the distributed layer as a pure throughput enhancement rather than a correctness dependency. Understanding *what is actually parallelizable*: recognizing that a time-sequential simulation cannot be split, while candidate-schedule scoring is embarrassingly parallel.

**Full-stack web**
FastAPI with Pydantic request models, CORS, lifecycle hooks, and a Vite dev proxy. React 19 with hooks, router-driven multi-page state, and non-trivial interaction state machines (drag/resize/rotate, drag-and-drop scheduling, playback loops). WebGL data visualization with deck.gl. Redis-backed persistence with payload-shaping for large binary-ish results.

**Third-party data integration**
Multi-source API integration (Last.fm, Ticketmaster Discovery) with content-hashed disk caching, offline-first demo design, and defensive parsing of inconsistent response shapes. Feature engineering across heterogeneous signals into a single normalized composite score.

**LLM application engineering**
Anthropic Messages API integration with structured, domain-grounded prompts; output-format constraints driven by the rendering target; graceful fallback to deterministic text when unavailable; failure surfacing rather than silent failure. Safety-conscious prompt design — the non-certification disclaimer is enforced in the prompt, not left to the model's discretion.

**Agentic development workflow**
The repo is itself instrumented for AI-assisted development: a `CLAUDE.md` with architectural decisions marked "do not re-litigate," four committed per-workstream slash-command skill files scoping each engineer's agent to their domain, and a design doc plus decisions log treated as source of truth over code comments.

**Domain modeling and judgment**
Translating a real safety problem into tractable computation: the two-tier decision to use a cheap analytic model for search and an expensive agent model for validation; the recognition that raw density is the wrong hotspot signal (stage fronts are *supposed* to be dense) leading to exclusion masks and bottleneck weighting; honest framing of the tool's limits.

---

## Concrete details worth citing

All figures below are **measured**, not estimated — benchmarked on an AMD Ryzen 7 7800X3D (8 cores), 31 GB RAM, single-node mode. Methodology and reproduction scripts are in the repo's `BENCHMARKS.md` and `scripts/bench/`.

| Fact | Measured value |
|---|---|
| Full event day, 8,000 agents | **340.9 M agent-force evaluations in 295 s — 122× faster than real time** |
| Full event day, 5,000 agents (demo default) | 213.0 M evaluations in 188 s — 192× real time |
| Peak kernel throughput | **1.16 M agent-physics-steps/s** |
| Spatial hash vs. all-pairs O(N²) | **135× faster at 8,000 agents** (29× at 1,000), agreeing to 2 × 10⁻¹⁶ relative error |
| numba JIT vs. same algorithm in pure Python | **200×** |
| Occupancy grid | 656 × 559 = 366,704 cells, 109,445 walkable, 1.5 m resolution, 60.8 acres |
| Venue load (GeoJSON → UTM → rasterized grid) | 0.44 s |
| Flow-field pathfinding | 17 destinations, 0.98 s each, 111,800 Dijkstra cells/s |
| Macro crowd model | 7.2 ms per full-day run → 139 evaluations/s |
| Optimizer throughput | 2,000 candidate schedules per request in 15.4 s |
| Optimizer effectiveness | **~3% risk reduction** (see limitations — this one is weak) |
| Scale represented | 80,000-capacity venue, 75,000 tickets, 9-75 real people per agent |
| Physics resolution | dt = 0.5 s, 240 substeps per 2-minute bin, 301 frames/run |
| Frontend build | 973 modules in 3.20 s; 319 KB gzipped app bundle |
| Cluster target (provisioned, not benchmarked) | 5-6 ESXi VMs, ~44 cores, Dask scheduler + workers |
| Codebase | ~7,000 lines across Python backend and React frontend |
| Build time | 2 days, 3 contributors |

---

## Honest limitations — do not overclaim

- **No validation against ground truth.** The simulation has never been calibrated against real crowd data from HARD Summer or any other event. Hotspot densities carry an empirical 0.56 scaling factor, and reported "pressure" is a derived `0.3 * density` display value, not the velocity-variance pressure metric that exists in the codebase but is not wired to any route. Describe outputs as plausible and directionally useful, never as accurate predictions.
- **Heuristic, not globally optimal.** Barrier and amenity placements are drawn by the user; the schedule optimizer is local search with no optimality guarantee. The project documentation itself insists on the word "recommended."
- **Some planned components are unwired.** A fuller risk analyzer (`sim/risk.py`, with real pressure math and zone classification) and an automated mitigation planner (`optimize/mitigation.py`, which suggests barrier segments and staff positions) are implemented but unreachable from the API. The original "two-tier macro-seeds-micro" architecture was simplified during the build into a single full-day agent run, with the macro model retained for optimizer scoring and risk windows.
- **Prototype-grade edges.** Unpinned dependencies and a committed Redis dump remain. (A round of bug fixes landed after the hackathon: the safety briefing's peak-density field mismatch, the dead orange-threshold slider, the optimizer and briefing routes ignoring a user-uploaded project's own GeoJSON, and a stale Claude model id are all fixed.)
- **The "44 cores across 7 VMs" figure is a deployment target from the infrastructure plan, not a constant in the code.** The code reads a scheduler address from an environment variable and reports whatever worker count it finds. If asked about scale, describe the topology as provisioned infrastructure and be ready to say it also runs on one laptop.

---

## Suggested framings

### Resume bullets (all numbers measured)

Pick 2-3; they overlap deliberately so you can match the role.

**Simulation / scientific computing**
> Built the simulation engine for a festival crowd-crush prediction tool — a numba-JIT Helbing–Molnár social-force model with granular contact over Dijkstra-derived flow-field navigation — resolving **340 million agent-force evaluations in under 5 minutes** to simulate a full 10-hour event day for an 80,000-capacity venue at **122× real time**.

**Performance engineering** (the strongest single bullet)
> Replaced O(N²) pairwise force computation with a CSR spatial hash rebuilt every physics step, cutting per-step cost **135× at 8,000 agents** (970 ms → 7.2 ms) while proving equivalence to 2 × 10⁻¹⁶ relative error; numba JIT compilation of the force kernel contributed a further **200×** over the equivalent pure-Python implementation.

**Profiling / measurement rigor**
> Benchmarked the full pipeline and corrected a threading misconfiguration in which `joblib`'s thread backend made GIL-bound candidate scoring **1.96× slower than serial execution**, and documented that the schedule optimizer plateaus after ~100 of its 300 search iterations — replacing unverified performance claims in the README with a reproducible benchmark suite.

**Geospatial engineering**
> Built the venue pipeline converting hand-traced GeoJSON footprints to simulation-ready occupancy grids — shapely polygon boolean algebra, pyproj WGS84→UTM reprojection, vectorized rasterization to a **366,704-cell grid at 1.5 m resolution** — in 0.44 s, with user-drawn barriers rasterized into the navigable environment between runs.

**Distributed systems / infra**
> Designed a dual-mode Dask execution layer that fans schedule-optimization candidate scoring across a self-provisioned ESXi cluster (~44 cores, golden-image VM cloning via `ovftool`) or degrades transparently to in-process execution, so the same code path runs identically on one laptop or the full cluster with no correctness dependency on the distributed tier.

**Full-stack / applied**
> Shipped an end-to-end crowd-safety planning tool in a two-day hackathon: FastAPI + Redis backend, React/deck.gl WebGL map with direct-manipulation barrier editing (drag, resize, rotate) and drag-and-drop schedule building, Last.fm/Ticketmaster demand modeling with a z-scored composite draw index, and Claude-generated pre-event safety briefings.

**If asked "what would you do differently"**
> The optimizer is the weak link — a strict hill-climb that plateaus at ~3% risk reduction. I'd add restarts and equal-cost move acceptance, wire in the velocity-variance pressure metric already written but unconnected, and calibrate hotspot densities against real event data, since the current 0.56 scaling factor produces physically implausible peaks.

**Outreach framing, research contexts**
> PLUR is an attempt to make pedestrian-dynamics simulation usable as a *planning* interface rather than an offline analysis artifact. The interesting problem wasn't the social-force model itself — it was closing the loop: making a full-day agent simulation cheap enough to re-run after every intervention, and making the intervention surface (barriers, amenity placement, the lineup itself) something a safety planner could actually manipulate. The signal-extraction problem turned out to be as hard as the physics: raw density flags the front of every stage, which is exactly where crowds belong, so hazard detection required masking expected-dense regions and weighting by corridor geometry.

**What I'd build next, if asked**
Validation against real event data; wiring the velocity-variance pressure metric into the live path since pressure, not density alone, is what actually kills people in crowd disasters; automated mitigation suggestion rather than manual barrier placement; and calibrating the demand model against published billing tiers.
