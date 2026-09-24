"""Shared fixtures for PLUR benchmarks."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.venue.loader import load_venue  # noqa: E402

DATA_DIR = REPO / "backend" / "data"


def venue():
    return load_venue("hard_summer_2025", DATA_DIR)


def roster():
    names = [
        line.strip()
        for line in (REPO / "hsd1.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names


def build_setlist(v, artists=None, start_hour=15, n_slots=11):
    """Realistic HARD Summer day: 1-hour slots, artists spread across stages.

    Fills earliest slots first across all stages (so sets run concurrently),
    which is what the real schedule looks like.
    """
    artists = artists or roster()
    stage_ids = [s["id"] for s in v.stages]
    n_stages = len(stage_ids)
    setlist = []
    for i, artist in enumerate(artists):
        slot = i // n_stages
        if slot >= n_slots:
            break
        stage = stage_ids[i % n_stages]
        h0 = (start_hour + slot) % 24
        h1 = (start_hour + slot + 1) % 24
        setlist.append({
            "artist": artist,
            "stage": stage,
            "start": f"{h0:02d}:00",
            "end": f"{h1:02d}:00",
            "locked": False,
            "manual": True,
        })
    return setlist


def synth_draw(setlist):
    """Deterministic draw scores matching main.py's no-API fallback:
    later slot = higher billing, capped below headliner territory.
    """
    def mins(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    times = [mins(e["start"]) for e in setlist]
    t_min, t_max = min(times), max(times)
    draw = {}
    for e in setlist:
        t = mins(e["start"])
        if t_max == t_min:
            draw[e["artist"]] = 0.2
        else:
            draw[e["artist"]] = round(0.10 + 0.65 * (t - t_min) / (t_max - t_min), 3)
    return draw


def fmt(n):
    return f"{n:,.0f}" if abs(n) >= 1000 else f"{n:,.3g}"
