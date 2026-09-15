from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent.claude import PLURAgent
from .cluster import init_client, is_distributed, worker_count, submit, shutdown
from .demand.service import DemandService
from .optimize.schedule import ScheduleOptimizer
from .sim.macro import MacroModel
from .sim.festival import run_festival
from .store.projects import ProjectStore
from .venue.loader import VenueGrid, load_venue, load_venue_from_geojson

_DATA_DIR = Path(__file__).parent / "data"


def _slot_position_draw(artist: str, setlist: list[dict]) -> float:
    """Infer draw score from schedule position when API data is unavailable.

    Slots that start later in the day are assumed to be higher-billed acts.
    Returns a value in [0.1, 0.75] — capped below headliner territory so
    inferred acts never outrank artists with real streaming data.
    """
    def _t(s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    times = [_t(e["start"]) for e in setlist if e.get("start")]
    if not times:
        return 0.2
    t_min, t_max = min(times), max(times)
    artist_start = next((_t(e["start"]) for e in setlist if e["artist"] == artist and e.get("start")), None)
    if artist_start is None or t_max == t_min:
        return 0.2
    # Linear map: earliest slot → 0.10, latest slot → 0.75
    return round(0.10 + 0.65 * (artist_start - t_min) / (t_max - t_min), 3)
_venue_cache: dict[str, VenueGrid] = {}

app = FastAPI(
    title="PLUR",
    description="Predictive Large-scale User Routing — crowd-crush prediction & mitigation for music festivals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_demand_svc = DemandService(_DATA_DIR / "cache")
_macro_model = MacroModel()
_scheduler = ScheduleOptimizer()
_agent = PLURAgent()
_project_store = ProjectStore(os.getenv("REDIS_URL", "redis://localhost:6379"))


@app.on_event("startup")
async def _startup():
    init_client()


@app.on_event("shutdown")
async def _shutdown():
    shutdown()


# ---------- request models ----------

class SetlistEntry(BaseModel):
    artist: str
    stage: str
    start: str  # "HH:MM"
    end: str
    locked: bool = False   # user locked — optimizer must not move
    manual: bool = True    # False = auto-filled — optimizer should prefer these


class DemandScoresRequest(BaseModel):
    artists: list[str]


class SimSliders(BaseModel):
    max_capacity: int = 80000
    tickets_sold: int = 75000
    arrival_steepness: float = 1.0
    n_agents: int = 5000


class OptimizeRequest(BaseModel):
    venue_id: str = "hard_summer_2025"
    project_id: str = ""
    setlist: list[SetlistEntry]
    headliners: list[str] = []
    sliders: SimSliders = SimSliders()


class FestivalSimRequest(BaseModel):
    venue_id: str = "hard_summer_2025"
    project_id: str = ""
    setlist: list[SetlistEntry] = []
    sliders: SimSliders = SimSliders()
    barriers: list[list[list[float]]] = []
    density_orange: float = 4.0
    density_red: float = 6.0


class SafetyBriefingRequest(BaseModel):
    venue_id: str = "hard_summer_2025"
    project_id: str = ""
    setlist: list[SetlistEntry] = []
    sliders: SimSliders = SimSliders()
    peak_density: float = 0.0
    hotspots: list[dict] = []
    amenities: list[dict] = []


class CreateProjectRequest(BaseModel):
    name: str
    geojson: dict
    meta: dict = {}
    artists: list[str] = []
    setlist: list[dict] = []


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    artists: list[str] | None = None
    setlist: list[dict] | None = None
    meta: dict | None = None


# ---------- helpers ----------

def _get_venue(venue_id: str) -> VenueGrid:
    if venue_id not in _venue_cache:
        try:
            _venue_cache[venue_id] = load_venue(venue_id, _DATA_DIR)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Venue '{venue_id}' not found: {exc}")
    return _venue_cache[venue_id]


def _setlist_dicts(entries: list[SetlistEntry]) -> list[dict]:
    return [e.model_dump() for e in entries]


async def _resolve_venue(project_id: str, venue_id: str) -> VenueGrid:
    """Prefer the project's own uploaded GeoJSON; fall back to a bundled venue.

    Every route that reasons about stage geometry must go through this — using
    the bundled venue for a user-uploaded project silently scores the wrong map.
    """
    if project_id:
        project = await _project_store.get(project_id)
        if project and project.get("geojson"):
            try:
                return load_venue_from_geojson(
                    project["geojson"],
                    project.get("meta", {}),
                    venue_id=project_id,
                )
            except Exception:
                pass
    return _get_venue(venue_id)


# ---------- routes ----------

@app.get("/")
async def root():
    return {"status": "ok", "project": "PLUR", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "distributed": is_distributed(),
        "dask_workers": worker_count(),
    }


@app.get("/venues")
async def list_venues():
    venues_dir = _DATA_DIR / "venues"
    result = []
    if not venues_dir.exists():
        return result
    for folder in sorted(venues_dir.iterdir()):
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        try:
            venue = _get_venue(folder.name)
            bbox = list(venue.bbox_lonlat)
            stages_out = [{"id": s["id"], "name": s["name"], "lonlat": s["lonlat"]} for s in venue.stages]
        except Exception:
            bbox = []
            stages_out = []
        venue_id = meta.get("id", folder.name)
        location = meta.get("location") or meta.get("venue") or meta.get("address", "")
        result.append({
            "id": venue_id,
            "name": meta.get("name", folder.name),
            "location": location,
            "bbox_lonlat": bbox,
            "stages": stages_out,
        })
    return result


@app.get("/venues/{venue_id}")
async def get_venue(venue_id: str):
    venue = _get_venue(venue_id)
    return {
        "id": venue_id,
        "meta": venue.meta,
        "geojson": venue.geojson,
        "grid": {
            "rows": venue.grid_shape[0],
            "cols": venue.grid_shape[1],
            "cell_m": venue.cell_m,
            "origin_m": list(venue.origin_m),
        },
        "stages": venue.stages,
        "gates": venue.gates,
        "facilities": venue.facilities,
        "bbox_lonlat": list(venue.bbox_lonlat),
    }


@app.post("/demand/scores")
async def get_demand_scores(req: DemandScoresRequest):
    """Return draw scores for a list of artist names — used by the auto-fill UI."""
    if not req.artists:
        return {"draw": {}}
    # Synthetic setlist: time/stage are irrelevant for draw scoring
    synthetic = [
        {"artist": a, "stage": "s0", "start": "12:00", "end": "13:00", "locked": False, "manual": True}
        for a in req.artists
    ]
    demand = _demand_svc.compute(synthetic)
    return {"draw": demand.get("draw", {})}


@app.post("/simulate_festival")
async def simulate_festival(req: FestivalSimRequest):
    venue = await _resolve_venue(req.project_id, req.venue_id)
    setlist = _setlist_dicts(req.setlist)

    draw: dict[str, float] = {}
    affinity: dict[str, dict[str, float]] = {}
    try:
        demand = _demand_svc.compute(setlist)
        draw = demand.get("draw", {})
        affinity = demand.get("affinity", {})
    except Exception:
        pass
    for entry in setlist:
        if entry["artist"] not in draw:
            draw[entry["artist"]] = _slot_position_draw(entry["artist"], setlist)

    n_agents = min(req.sliders.n_agents, 8000)
    result = submit(
        run_festival,
        venue=venue,
        setlist=setlist,
        draw=draw,
        tickets_sold=req.sliders.tickets_sold,
        n_agents=n_agents,
        extra_obstacles=req.barriers if req.barriers else None,
        density_orange=req.density_orange,
        density_red=req.density_red,
        affinity=affinity,
    )
    response = {
        "frames": result["frames"],
        "hotspots": result["hotspots"],
        "n_frames": len(result["frames"]),
    }

    if req.project_id:
        try:
            await _project_store.save_sim(req.project_id, response)
        except Exception:
            pass

    return response


@app.post("/optimize_schedule")
async def optimize_schedule(req: OptimizeRequest):
    venue = await _resolve_venue(req.project_id, req.venue_id)
    setlist = _setlist_dicts(req.setlist)
    demand = _demand_svc.compute(setlist)

    result = _scheduler.optimize(
        setlist=setlist,
        draw=demand["draw"],
        affinity=demand["affinity"],
        stages=venue.stages,
        headliners=req.headliners,
        tickets_sold=req.sliders.tickets_sold,
        max_capacity=req.sliders.max_capacity,
        macro_model=_macro_model,
        n_iterations=100,
        n_jobs=4,
    )

    rationale = _agent.generate_rationale(
        changes=result["changes"],
        risk_before=result["risk_before"],
        risk_after=result["risk_after"],
        venue_name=venue.meta.get("name", req.venue_id),
    )
    result["rationale"] = rationale
    return result


@app.post("/safety_briefing")
async def safety_briefing(req: SafetyBriefingRequest):
    venue = await _resolve_venue(req.project_id, req.venue_id)
    setlist = _setlist_dicts(req.setlist)

    demand = _demand_svc.compute(setlist)
    draw = demand.get("draw", {})
    affinity = demand.get("affinity", {})

    macro_result = _macro_model.run(
        setlist=setlist,
        draw=draw,
        affinity=affinity,
        stages=venue.stages,
        tickets_sold=req.sliders.tickets_sold,
        max_capacity=req.sliders.max_capacity,
    )

    briefing = _agent.generate_safety_briefing(
        venue_name=venue.meta.get("name", req.venue_id),
        risk_windows=macro_result.get("risk_windows", []),
        peak_density=req.peak_density,
        schedule=setlist,
        amenities=req.amenities or [],
    )
    return {"briefing": briefing}


# ---------- project routes ----------

@app.get("/projects")
async def list_projects():
    return await _project_store.list_all()


@app.post("/projects")
async def create_project(req: CreateProjectRequest):
    return await _project_store.create(
        req.name, req.geojson, req.meta, req.artists, req.setlist
    )


@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = await _project_store.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.put("/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest):
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    project = await _project_store.update(project_id, updates)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    await _project_store.delete(project_id)
    return {"ok": True}


@app.get("/projects/{project_id}/sim")
async def get_sim(project_id: str):
    sim = await _project_store.get_sim(project_id)
    if not sim:
        return {"frames": [], "hotspots": []}
    return sim
