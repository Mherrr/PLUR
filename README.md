# PLUR — Predictive Large-scale User Routing

Crowd-crush prediction and mitigation tooling for multi-stage music festivals. PLUR simulates how an audience moves through a venue over a full event day, finds where and when dangerous density forms, and gives event-ops teams interactive controls — barriers, amenity placement, and the set-times grid itself — to reduce risk before the gates open.

Built over a two-day hackathon (June 2026) for the Ddoski's Lab + Anthropic + Most Technical tracks. Bundled test venue: **HARD Summer 2025, Hollywood Park / SoFi Stadium grounds, Inglewood CA** (5 mapped stages, 80k capacity).

> **Disclaimer:** PLUR is a planning and decision-support prototype. It is not a validated or certified life-safety system. Every recommendation it produces must be reviewed by qualified event-safety professionals.

**Measured on one 8-core desktop:** a full 10-hour event day with 8,000 agents resolves **340.9 M agent-force evaluations in 295 s** (122× faster than real time). The CSR spatial hash beats an all-pairs kernel by **135×** at 8,000 agents while agreeing to 2 × 10⁻¹⁶ relative error, and numba JIT gives **200×** over the same algorithm in pure Python. Full methodology, reproduction steps, and the less flattering numbers are in [PLUR_BENCHMARKS.md](PLUR_BENCHMARKS.md).

---

## What it does

**Model a venue** — Upload a GeoJSON footprint (walkable boundary, obstacles, stages, gates, facilities). The loader reprojects it to UTM, subtracts obstacles, and rasterizes it to a walkable occupancy grid.

**Simulate a full day** — An agent-based social-force simulation runs the entire event: attendees arrive through the gate on a logistic arrival curve, pick stages by artist draw, migrate when sets end, detour to restrooms/water/bars, and egress to the gate when the music stops.

**Find hotspots** — Peak per-cell density is tracked across the whole run, weighted by corridor narrowness, masked against expected crowding (stage fronts, gate apron, walls), then clustered into a short list of flagged locations.

**Intervene** — Draw barriers directly on the map and drag, resize, or rotate them; they are rasterized into the occupancy grid on the next run. Drag restrooms, water stations, and bars to new positions and re-run.

**Optimize the schedule** — Submit the set-times grid to a local-search optimizer that permutes unlocked artists across stage/time slots, scoring thousands of candidates against the macroscopic crowd model. Locked slots are never moved.

**Brief** — Generate a plain-text, Claude-authored pre-event safety briefing: risk windows, stage-by-stage assessment, ops recommendations, and specific amenity-repositioning advice based on current facility coordinates.

---

## How it works

### 1. Venue to occupancy grid (`backend/venue/loader.py`)

A GeoJSON `FeatureCollection` in WGS84 is parsed into typed features. Walkable polygons are unioned and obstacle polygons subtracted, the result is reprojected to UTM (EPSG:32611 for the bundled venue), and rasterized with `shapely.contains` on a vectorized point grid into a boolean occupancy array. Default cell size is 1.5 m for HARD Summer. All downstream simulation math is in meters; conversion back to lon/lat happens only at the response boundary.

Bars are deliberately excluded from the obstacle set — they are movable amenity points, not static geometry.

### 2. Demand model (`backend/demand/service.py`)

Per-artist signals are pulled from **Last.fm** (`artist.getInfo`, `artist.getSimilar`, `artist.getTopTags`, `geo.getTopArtists`) and **Ticketmaster Discovery** (venue capacity as a live-demand proxy). Every response is content-hashed and cached to `backend/data/cache/*.json`; the demo never calls a live API.

A composite draw score is a z-scored blend:

```
draw = 0.39*z(sqrt(listeners * playcount)) + 0.33*z(1/us_rank) + 0.28*z(tm_venue_capacity)
```

normalized to [0, 1]. `artist.getSimilar` match scores build a symmetric pairwise **affinity matrix** that drives inter-stage migration.

B2B and collaboration sets ("DJ Seinfeld B2B Luuk van Dijk", "X CLUB. B2B ATRIP") are split on ` b2b ` / ` x `, fetched per member, then aggregated — listeners and playcounts summed, tags unioned, affinities maxed.

When no API key is configured, artists fall back to a schedule-position heuristic: later slots infer higher billing, capped at 0.75 so inferred acts never outrank artists with real data.

### 3. Macroscopic model (`backend/sim/macro.py`)

An analytic, per-5-minute-bin model of the whole day. Arrival inflow is a draw-weighted sum of Gaussians centered on set start times, integrated to a CDF against tickets sold. Audience is split across concurrent sets by a softmax over *effective* draw, where effective draw includes an **affinity bleed** term — high-affinity acts on stages within 400 m leak up to 15% of share to each other, softening winner-takes-all. When a set ends, an egress migration term redistributes crowd weighted 60% by destination draw, 40% by affinity to the act that just finished, decayed by inter-stage distance.

Stage safe capacity is `min(6000 m2 * 4 p/m2, max_capacity/n_stages * 0.7)`. Bins exceeding 120% of safe capacity are merged into **risk windows**.

This model is cheap enough to evaluate thousands of times, which is what makes the schedule optimizer viable. It is used by `/optimize_schedule` and `/safety_briefing`.

### 4. Agent simulation (`backend/sim/festival.py`, `micro.py`, `pathfinding.py`)

The microscopic layer is a **Helbing–Molnar social-force model** with granular contact, compiled with `numba.njit`:

- Driving force toward a destination, relaxation time tau = 0.5 s
- Exponential agent-agent repulsion `A*exp((2r - d)/B)`, A = 2000 N, B = 0.08 m
- Body compression `k*overlap` and tangential sliding friction `kappa*overlap*dv_t` on contact
- Wall repulsion from a precomputed Euclidean distance transform and its gradient

Neighbor lookup is a **CSR spatial hash** on 2 m cells, rebuilt each physics step, scanned over a 5x5 cell stencil — O(N) rather than O(N^2).

Navigation is **flow-field pathfinding**: a Dijkstra distance field is computed per destination (each stage, the gate, each amenity) over the occupancy grid, using a wall-proximity cost multiplier (`1 + 4/(d_wall + 1)`) so paths prefer corridor centers over hugging walls. Distance fields are converted to unit-vector flow fields, cached per destination, and sampled per agent per bin with plus/minus 15 degrees of angular noise. Blocked moves slide along walls (axis-decomposed retry) rather than bouncing.

Agent behavior over the day:

- **Arrival** — logistic curve from gates-open (music start minus 60 min) to music end, spawned at walkable cells near the gate point
- **First stage** — sampled from a softmax over the draws of the opening batch
- **Migration** — when a set ends, the next stage is sampled 60% on draw / 40% on affinity to the act just watched
- **Amenities** — 25% of arrivals detour to a restroom or water station before their first stage; each agent has 0-1 between-set stops (15% chance per set change); ~22% are marked 21+ and have a 5% chance per set change of a bar visit. Dwell is 45 s (water) / 90 s (restroom) / 4 min (bar), plus a small queue-length bonus, hard-capped at 15 min. Idling agents have their desired speed dropped to 0.02 m/s so they queue in place and still exert crowd pressure.
- **Egress** — after music end, agents progressively reroute to the gate and despawn within 25 m of it

Time is advanced in 2-minute bins, each stepping the physics at dt = 0.5 s (240 steps per bin). Agent count is capped server-side at 8000; each agent represents `tickets_sold / n_agents` real people, and that scale factor is applied when accumulating density.

### 5. Hotspot detection

Peak instantaneous density (people/m2) is tracked per cell across every bin. Flagging then applies:

- **Exclusions** — cells within 5 m of a wall (rasterization artifacts), within 60 m of any stage (that is the audience, not a hazard), within 30 m of the gate, and all non-walkable cells
- **Bottleneck weighting** — `clip(3 - d_wall/10, 1, 3)`, so a 5 m corridor scores 3x an open field at the same density
- **Threshold** — weighted score above the user's red-density setting (default 6.0)
- **Clustering** — density-weighted centroids within 30 m, then iterative merging until no two centroids are within 100 m, capped at 8

Reported hotspot density is scaled by a 0.56 calibration factor to compensate for grid-cell point-sampling bias; `peak_pressure` is reported as a derived `0.3 * density` display value.

### 6. Schedule optimizer (`backend/optimize/schedule.py`)

Stochastic local search over slot permutations. Each move swaps the `(stage, start, end)` triple between two eligible entries. Per iteration it samples 20 candidate swaps, scores them all in parallel, and accepts the best if it beats the incumbent — 100 iterations, so up to ~2000 macro-model evaluations per request.

Eligibility: `locked=True` entries and named headliners are never moved; auto-filled entries (`manual=False`) get 3x sampling weight, so the optimizer prefers rearranging what the machine placed over what a human placed.

The objective sums two terms across all stages and time bins:

- **Cubic load penalty** `(pop/safe_cap)^3` — cubic rather than threshold-gated, so 95%-full is not treated as free
- **Corridor adjacency penalty** `ra * rb * exp(-d/200) * 0.35` for every stage pair within 500 m loaded in the same bin — pinch points between two simultaneously-packed nearby stages that the per-stage macro model cannot see

"Headliners last" is enforced upstream in the UI's auto-fill, which sorts unplaced artists by ascending draw into slots sorted by ascending start time.

### 7. Claude agent (`backend/agent/claude.py`)

Two prompts against the Anthropic Messages API, both constrained to plain text (no markdown) because the output renders into fixed-width panels:

- **Schedule rationale** — one line per move, artist and move followed by why it helps, 150 words max
- **Safety briefing** — executive summary, risk assessment, recommendations, and amenity placement advice, seeded with macro risk windows, the full schedule, and live facility coordinates, 250 words max, ending on a fixed non-certification disclaimer

Both degrade to deterministic placeholder text when `ANTHROPIC_API_KEY` is absent, and append the error inline rather than failing the request when the API call throws.

---

## Distributed execution

`backend/cluster.py` is a dual-mode dispatcher. If `DASK_SCHEDULER` is set it connects a `dask.distributed.Client`; if it is unset, empty, or the connection fails, everything runs in-process on the coordinator. The deployment target is an ESXi cluster of 5-6 Ubuntu VMs at 8 vCPUs each (about 44 cores, the ESXi per-host licensing cap), one coordinator running FastAPI plus the Dask scheduler, the rest running `dask worker --nworkers 8 --nthreads 1`.

What the cluster actually buys you:

| Workload | Distributed | Local |
|---|---|---|
| `/simulate_festival` | one `submit()` — the whole run is a **single task offloaded to one worker**; the cluster does not parallelize a single sim | runs in-process |
| `/optimize_schedule` | `map_calls()` fans the 20 candidate scores per iteration across workers | serial — see below |

The local path runs candidate scoring **serially on purpose**. `joblib` is configured with `prefer="threads"`, and `_score_schedule` is GIL-bound Python/numpy, so threads only add dispatch overhead: 6.81 s at `n_jobs=1` versus 13.32 s at `n_jobs=4` for the same 800 candidates, with bit-identical results. Real parallelism requires the Dask path, which bypasses joblib.

A single day-long simulation is sequential in time, so it cannot be split across cores. The cluster is a *throughput* win for candidate-schedule search, and keeps the sim off the coordinator so the API stays responsive. Everything runs correctly on one machine; `GET /health` reports the live worker count.

Workers must have the repo cloned and on `PYTHONPATH` — `Client.upload_file()` breaks the package-relative imports in `backend/`.

---

## Architecture

```
Browser
  React 19 + deck.gl 9 + MapLibre GL 5 (vector-only dark basemap, no tile provider, no token)
        <->  /api -> Vite dev proxy -> :8000
Backend - FastAPI, Python 3.12
  |-- venue/loader.py      GeoJSON -> UTM -> occupancy grid (shapely, pyproj)
  |-- demand/service.py    Last.fm + Ticketmaster -> draw scores + affinity matrix (JSON-cached)
  |-- sim/macro.py         5-min-bin share-of-audience timeline + risk windows
  |-- sim/pathfinding.py   Dijkstra distance fields -> cached flow fields
  |-- sim/micro.py         numba social-force kernel + CSR spatial hash
  |-- sim/festival.py      full-day agent loop, amenity behavior, hotspot extraction
  |-- optimize/schedule.py local search over slot permutations
  |-- agent/claude.py      Claude rationale + safety briefing
  |-- cluster.py           Dask dual-mode dispatch
  `-- store/projects.py    Redis project + sim-result persistence
```

### Not wired up

These files exist and are functional in isolation, but no route reaches them:

- `backend/sim/risk.py` — a fuller risk analyzer with real pressure math (density times variance of local velocity, via `uniform_filter` over binned velocity moments), connected-component zone classification, and hotspot extraction. `festival.py` computes its own simpler hotspots inline instead.
- `backend/optimize/mitigation.py` — `MitigationPlanner`, which suggests barrier segments at chokepoints and staff positions from a risk field. Barriers are currently placed by hand in the UI.
- `MicroSim` class in `backend/sim/micro.py` — the windowed-simulation wrapper from the original two-tier design. `festival.py` imports its kernel and constants directly and runs the full day instead.

---

## Setup

### Prerequisites

- Python 3.12 — **not 3.13+**; numba has no wheels for newer versions yet. If your system Python is newer, `uv python install 3.12` then `uv venv --python 3.12 .venv` gets you a clean interpreter without touching it.
- Node.js 18+ (verified on 24.19)
- Redis on `localhost:6379`. On Windows, the winget `Redis.Redis` package is the 3.0.504 legacy port, which predates the `HELLO` command that redis-py 5+ uses to negotiate RESP3 — either pin `redis==4.6.0` locally or run a current server under WSL, Docker, or Memurai.
- Last.fm API key — free, key-only, at [last.fm/api](https://www.last.fm/api)
- Anthropic API key — for `/optimize_schedule` rationale and `/safety_briefing`
- Ticketmaster Discovery key (optional) — free tier, 5k req/day

### Environment

Create a `.env` **at the repository root** (`backend/main.py` loads it from its parent directory):

```bash
LASTFM_API_KEY=...
TICKETMASTER_API_KEY=...           # optional - capacity term drops out if unset
ANTHROPIC_API_KEY=...              # optional - agent falls back to placeholder text
REDIS_URL=redis://localhost:6379   # default
DASK_SCHEDULER=                    # empty = local mode; tcp://host:8786 to use a cluster
```

### Backend

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run from the REPOSITORY ROOT - backend/ uses package-relative imports
uvicorn backend.main:app --reload --port 8000
```

Interactive API docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. `/api/*` proxies to `http://localhost:8000` (override with `VITE_API_TARGET`).

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check and version |
| `GET` | `/health` | Status plus `distributed` flag and live Dask worker count |
| `GET` | `/venues` | List bundled venues with bbox and stage points |
| `GET` | `/venues/{venue_id}` | Full GeoJSON, grid metadata, stages, gates, facilities |
| `GET` | `/projects` | List saved projects (id, name, thumbnail, artist count) |
| `POST` | `/projects` | Create a project from a GeoJSON upload |
| `GET` | `/projects/{id}` | Full project including GeoJSON |
| `PUT` | `/projects/{id}` | Partial update — name, artists, setlist, meta |
| `DELETE` | `/projects/{id}` | Delete project and its cached sim |
| `GET` | `/projects/{id}/sim` | Last saved simulation result |
| `POST` | `/simulate_festival` | Run the full-day agent sim; returns frames and hotspots |
| `POST` | `/optimize_schedule` | Local-search optimizer plus Claude rationale |
| `POST` | `/safety_briefing` | Claude safety briefing from macro risk windows |

All three simulation routes accept an optional `project_id`. When set and that project has a stored GeoJSON, the venue is loaded from the project; otherwise `venue_id` selects a bundled venue.
| `POST` | `/demand/scores` | Draw scores for a list of artist names (used by auto-fill) |

### `POST /simulate_festival`

```json
{
  "venue_id": "hard_summer_2025",
  "project_id": "abc12345",
  "setlist": [{ "artist": "Mau P", "stage": "hard", "start": "22:00", "end": "23:00" }],
  "sliders": { "max_capacity": 80000, "tickets_sold": 60000, "n_agents": 5000 },
  "barriers": [[[-118.34, 33.95], [-118.34, 33.95]]],
  "density_orange": 4.0,
  "density_red": 6.0
}
```

`n_agents` is clamped to 8000. `barriers` are lon/lat polygon rings, rasterized out of the occupancy grid before the run. Hotspots are flagged from `density_orange` upward and each carries `level: "orange" | "red"` depending on whether it crosses `density_red`. The result is written back to Redis under `sim:{project_id}`.

Response frames are `{ t_min, agents: [[lon, lat, vx, vy], ...], n_active, n_arrived, n_exited, scale }`.

### `POST /optimize_schedule`

```json
{
  "venue_id": "hard_summer_2025",
  "setlist": [{ "artist": "...", "stage": "...", "start": "HH:MM", "end": "HH:MM", "locked": false, "manual": true }],
  "headliners": ["Kali Uchis"],
  "sliders": { "max_capacity": 80000, "tickets_sold": 60000 }
}
```

Returns `{ proposed_schedule, risk_before, risk_after, changes[], rationale }`. Risk scores are objective-function units, not densities — only their ratio is meaningful.

### `POST /safety_briefing`

Takes `setlist`, `sliders`, `peak_density`, `hotspots[]`, and `amenities[]` (`{ id, name, facility_type, lat, lon }`). Returns `{ briefing }` as plain text.

---

## Using the app

**New Project** is a three-step wizard: upload a venue GeoJSON (drag-and-drop, with a live SVG preview and detected stage list), enter festival name, date, hours, capacity, expected attendance and the artist roster (paste-a-list or `.txt` upload — `hsd1.txt` is a ready HARD Summer Day 1 roster), then fill the stage/time grid by drag-and-drop or **Auto-fill**, which orders unplaced artists by Last.fm draw so openers land early and headliners land late.

**Project view** puts the deck.gl map behind a left control panel:

- **Parameters** — hard capacity and a tickets-sold slider that flags over-capacity
- **Risk Thresholds** — red density (sent to the backend as the hotspot cutoff)
- **Schedule** — reopen the set-times grid; double-click a slot to lock it against the optimizer, and locked artists are passed as headliners
- **Barriers** — click *Place Barrier* then click the map; select to drag, resize from corner handles, or rotate. Barriers persist to the project and become obstacles on the next run.
- **Amenities** — restrooms, water stations, and bars as draggable dots; positions feed both the simulation's attractor set and the Claude briefing
- **Optimize** — runs the scheduler and shows risk before/after, a change list, and the Claude rationale, with accept-or-discard
- **Safety Briefing** — generates the Claude pre-event briefing
- **Layers** — toggle heatmap, individual agents (colored by speed), and hotspot rings

The timeline bar scrubs and plays back frames at variable speed.

---

## Venue data format

```
backend/data/venues/<venue_id>/
|-- venue.geojson   # FeatureCollection, WGS84
`-- meta.json       # name, origin_lonlat, utm_epsg, grid_cell_m, max_capacity, event_hours
```

Feature `properties.type` values:

| `type` | `subtype` | Behavior |
|---|---|---|
| `walkable` | — | Defines where agents can exist |
| `obstacle` | `structure`, `terrain`, `barrier`, `stage_boh`, `vendor` | Subtracted from the walkable area |
| `obstacle` | `bar` | **Not** an obstacle — treated as a movable amenity |
| `vip_area` | — | Subtracted (GA agents route around it) |
| `gate_area` | — | Rendered only |
| `stage` | Point | Agent destination during sets |
| `gate` | Point | Spawn and exit point |
| `facility` | Point, `facility_type` = `restroom` / `water` / `bar` | Secondary attractor with dwell time |

The bundled HARD Summer geojson has 5 stage points (HARD, HARDER, Green, Purple, Pink), one gate, 11 facilities, and a north/south barrier pair whose gap is the venue's designed crush-risk bottleneck. `meta.json` also lists two smaller stages (BeatBox art car, Locals Only truck) that have no geometry yet; only mapped stage features reach the simulation.

---

## Risk thresholds

| Density | Level |
|---|---|
| under 3 p/m2 | Green — comfortable |
| 3-4 p/m2 | Yellow — busy |
| 4-6 p/m2 | Orange — caution |
| 6+ p/m2 | Red — crush risk |

Both the orange and red thresholds are sent to the backend. Hotspots are detected from the orange threshold upward and tagged `orange` or `red`; the map rings them in the matching color. The green/yellow constants live in the unwired `sim/risk.py`.

---

## Tech stack

| Layer | |
|---|---|
| Frontend | React 19, Vite 8, React Router 7, deck.gl 9 (`ScatterplotLayer`, `PolygonLayer`, `PathLayer`, `TextLayer`, `HeatmapLayer`), MapLibre GL 5 via react-map-gl |
| Backend | Python 3.12, FastAPI, uvicorn, Pydantic |
| Numerics | numpy, scipy (`distance_transform_edt`, `uniform_filter`, `label`), numba JIT |
| GIS | shapely 2.x vectorized ops, pyproj |
| Parallelism | Dask distributed (cluster), joblib (local) |
| Storage | Redis via `redis.asyncio` |
| AI | Anthropic Messages API |

---

## Known gaps

Measured rather than guessed — see [PLUR_BENCHMARKS.md](PLUR_BENCHMARKS.md) § Caveats for the data behind these.

- **The optimizer's search still plateaus.** 300 iterations returns byte-identical results to 100, so the strict hill-climb finds its local minimum early and two thirds of the budget is wasted. Equal-cost move acceptance, restarts, or annealing would each help. (It is no longer *weak* — relocation moves took it from ~3% to ~30% risk reduction — but it is still leaving improvement on the table.)
- **The objective is a sum, not a max**, so it can trade a better total against the worst single moment. Peak stage load did improve (1.671 → 1.414× safe capacity) but as a side effect, not because anything targets it.
- **Density resolution scales with agent count.** The Gaussian estimator is sound, but at 2,000 agents each agent still carries 37.5 people, making the smallest resolvable increment 1.49 p/m² — a quarter of the danger threshold. Use ≥5,000 agents for density-sensitive work. Peak densities read 0.60× lower at 8,000 agents than at 2,000 on the same schedule.
- **Reported pressure is a `0.3 × density` stand-in.** The real velocity-variance pressure metric in `sim/risk.py` is still unwired.
- **Hotspot levels skew red.** Detection compares the bottleneck-weighted danger score against thresholds denominated in raw density — dimensionally inconsistent, so nearly everything clears the red line and the orange tier stays theoretical.
- **Sim payloads get large** — 49 MB of JSON agent coordinates at 8,000 agents, held in Redis and shipped to the browser. Binary framing or downsampling is the first thing needed to scale.
- The map has no satellite or street basemap — it renders venue geometry over a flat dark background. Adding a raster tile source to the MapLibre style is a small change.
- Nothing is calibrated against real crowd data. Treat every output as directional, not predictive.
- `requirements.txt` is unpinned. Dask requires identical library versions across every node, so pin before deploying to a cluster.
