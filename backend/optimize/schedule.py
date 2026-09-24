from __future__ import annotations

import copy
import random
import numpy as np
from joblib import Parallel, delayed

from ..cluster import is_distributed, map_calls
from ..sim.macro import MacroModel


def _score_schedule(
    setlist: list[dict],
    draw: dict[str, float],
    affinity: dict[str, dict[str, float]],
    stages: list[dict],
    tickets_sold: int,
    max_capacity: int,
    macro: MacroModel,
) -> float:
    result = macro.run(
        setlist=setlist,
        draw=draw,
        affinity=affinity,
        stages=stages,
        tickets_sold=tickets_sold,
        max_capacity=max_capacity,
    )
    stage_pop = result["stage_pop"]
    safe_caps = result["stage_safe_capacity"]
    fallback_cap = max_capacity / max(len(stage_pop), 1)

    # Stage positions for adjacency penalty
    stage_pos: dict[str, np.ndarray] = {
        st["id"]: np.array(st.get("pos_m", [0.0, 0.0])) for st in stages
    }

    # Primary term: smooth cubic load penalty.
    # Cubic (not threshold-gated quadratic) means sub-cap crowding still
    # contributes — the optimizer can't treat 95%-full as free.
    total = 0.0
    for sid, pops in stage_pop.items():
        cap = safe_caps.get(sid, fallback_cap)
        for entry in pops:
            ratio = entry["pop"] / max(cap, 1.0)
            total += ratio ** 3

    # Corridor adjacency penalty: when two nearby stages are both heavily
    # loaded in the same time bin, fans in transit between them create
    # pinch points that the macro model doesn't see. Penalise the product
    # of their load ratios, decayed by distance.
    stage_ids = list(stage_pop.keys())
    for i, sid_a in enumerate(stage_ids):
        for j, sid_b in enumerate(stage_ids):
            if j <= i:
                continue
            pos_a = stage_pos.get(sid_a)
            pos_b = stage_pos.get(sid_b)
            if pos_a is None or pos_b is None:
                continue
            dist = float(np.linalg.norm(pos_a - pos_b))
            if dist > 500.0:
                continue
            proximity = np.exp(-dist / 200.0)
            cap_a = safe_caps.get(sid_a, fallback_cap)
            cap_b = safe_caps.get(sid_b, fallback_cap)
            for pa, pb in zip(stage_pop[sid_a], stage_pop[sid_b]):
                ra = pa["pop"] / max(cap_a, 1.0)
                rb = pb["pop"] / max(cap_b, 1.0)
                total += ra * rb * proximity * 0.35

    return total


def _copy_setlist(setlist: list[dict]) -> list[dict]:
    # Entries are flat dicts of scalars, so a shallow copy per entry is
    # equivalent to deepcopy here and materially cheaper — this runs once per
    # candidate, thousands of times per request.
    return [dict(e) for e in setlist]


def _swap_slots(setlist: list[dict], i: int, j: int) -> list[dict]:
    new_sl = _copy_setlist(setlist)
    new_sl[i]["stage"], new_sl[j]["stage"] = new_sl[j]["stage"], new_sl[i]["stage"]
    new_sl[i]["start"], new_sl[j]["start"] = new_sl[j]["start"], new_sl[i]["start"]
    new_sl[i]["end"], new_sl[j]["end"] = new_sl[j]["end"], new_sl[i]["end"]
    return new_sl


def _move_to_slot(setlist: list[dict], i: int, slot: tuple[str, str, str]) -> list[dict]:
    """Relocate one act into an empty (stage, start, end) slot."""
    new_sl = _copy_setlist(setlist)
    new_sl[i]["stage"], new_sl[i]["start"], new_sl[i]["end"] = slot
    return new_sl


def _slot_grid(setlist: list[dict], stages: list[dict]) -> list[tuple[str, str, str]]:
    """Every (stage, start, end) the schedule grid offers, filled or not."""
    times = sorted({(e["start"], e["end"]) for e in setlist})
    return [(st["id"], start, end) for st in stages for start, end in times]


def _empty_slots(
    setlist: list[dict], grid: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    occupied = {(e["stage"], e["start"], e["end"]) for e in setlist}
    return [slot for slot in grid if slot not in occupied]


class ScheduleOptimizer:
    def optimize(
        self,
        setlist: list[dict],
        draw: dict[str, float],
        affinity: dict[str, dict[str, float]],
        stages: list[dict],
        headliners: list[str],
        tickets_sold: int,
        max_capacity: int,
        macro_model: MacroModel | None = None,
        n_iterations: int = 200,
        n_jobs: int = 8,
    ) -> dict:
        macro = macro_model or MacroModel()
        headliner_set = set(headliners)

        current = copy.deepcopy(setlist)
        risk_before = _score_schedule(
            current, draw, affinity, stages, tickets_sold, max_capacity, macro
        )
        best = copy.deepcopy(current)
        best_score = risk_before

        # Swap eligibility rules (in priority order):
        #   locked=True  → never swap (user-pinned or legacy headliner)
        #   manual=False → auto-filled; optimizer prefers these (3x weight)
        #   manual=True  → user-placed but not locked; can be moved (1x weight)
        swappable = [
            i for i, e in enumerate(current)
            if not e.get("locked", False) and e["artist"] not in headliner_set
        ]

        if len(swappable) < 2:
            return {
                "proposed_schedule": current,
                "risk_before": risk_before,
                "risk_after": risk_before,
                "changes": [],
                "rationale": "",
            }

        # Build weight array: auto-filled entries are 3x more likely to be chosen
        sw_arr = np.array(swappable)
        sw_weights = np.array([
            3.0 if not current[i].get("manual", True) else 1.0
            for i in swappable
        ], dtype=float)
        sw_weights /= sw_weights.sum()

        rng_np = np.random.default_rng(42)
        pairs_per_iter = min(20, len(swappable) * (len(swappable) - 1) // 2)

        def _sample_pair() -> tuple[int, int]:
            # Weighted sample of 2 distinct indices
            idx_a = int(rng_np.choice(len(swappable), p=sw_weights))
            w2 = sw_weights.copy()
            w2[idx_a] = 0.0
            if w2.sum() == 0:
                return swappable[0], swappable[1]
            w2 /= w2.sum()
            idx_b = int(rng_np.choice(len(swappable), p=w2))
            return swappable[idx_a], swappable[idx_b]

        # A swap-only move set can only ever permute the acts already placed,
        # so every empty slot in the grid is unreachable — which rules out the
        # most natural fix for an overloaded hour, moving an act into open
        # airtime. Relocations make that reachable.
        grid = _slot_grid(current, stages)
        RELOCATE_FRAC = 0.4

        for _ in range(n_iterations):
            empties = _empty_slots(best, grid)
            candidates: list[list[dict]] = []
            for _ in range(pairs_per_iter):
                if empties and rng_np.random() < RELOCATE_FRAC:
                    idx = swappable[int(rng_np.choice(len(swappable), p=sw_weights))]
                    slot = empties[int(rng_np.integers(len(empties)))]
                    candidates.append(_move_to_slot(best, idx, slot))
                else:
                    i, j = _sample_pair()
                    candidates.append(_swap_slots(best, i, j))

            if is_distributed():
                arg_lists = [
                    (sl, draw, affinity, stages, tickets_sold, max_capacity, macro)
                    for sl in candidates
                ]
                scores = map_calls(_score_schedule, arg_lists)
            else:
                scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_score_schedule)(
                        sl, draw, affinity, stages, tickets_sold, max_capacity, macro
                    )
                    for sl in candidates
                )

            min_idx = int(np.argmin(scores))
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best = candidates[min_idx]

        changes = _compute_changes(setlist, best)

        return {
            "proposed_schedule": best,
            "risk_before": float(risk_before),
            "risk_after": float(best_score),
            "changes": changes,
            "rationale": "",
        }


def _compute_changes(original: list[dict], proposed: list[dict]) -> list[dict]:
    orig_map = {e["artist"]: e for e in original}
    changes = []
    for entry in proposed:
        a = entry["artist"]
        if a not in orig_map:
            continue
        orig = orig_map[a]
        if orig["stage"] != entry["stage"] or orig["start"] != entry["start"]:
            changes.append({
                "artist": a,
                "from_stage": orig["stage"],
                "from_time": orig["start"],
                "to_stage": entry["stage"],
                "to_time": entry["start"],
            })
    return changes
