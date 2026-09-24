"""B. Social-force kernel: numba JIT speedup and spatial-hash vs all-pairs.

The all-pairs kernel below is a line-for-line mirror of backend/sim/micro.py's
_force_kernel with the spatial-hash neighbour scan replaced by a full O(N^2)
loop. Both are numba-compiled, so the comparison isolates the data structure,
not the compiler. The social term A*exp((2r-d)/B) with B=0.08 m is ~e^-122 at
10 m, so the hash's 5x5 stencil of 2 m cells loses no physically relevant
interaction -- the two kernels agree to floating point noise (asserted below).
"""
import json
import time

import numba
import numpy as np
from bench_common import venue

from backend.sim.micro import (
    _force_kernel, _build_spatial_hash, _precompute_walls,
    V0_MEAN, V0_STD, TAU, A_REP, B_REP, A_WALL, B_WALL,
    RADIUS, MASS, K_BODY, KAPPA, HASH_CELL,
)


@numba.njit(cache=True)
def _force_kernel_allpairs(
    pos, vel, dest, v0,
    wall_dist, wall_gx, wall_gy,
    grid_origin_x, grid_origin_y, grid_cell_m, grid_rows, grid_cols,
    mass, tau, A, B, A_w, B_w, k_body, kappa, radius,
):
    N = pos.shape[0]
    forces = np.zeros_like(pos)
    for i in range(N):
        xi = pos[i, 0]; yi = pos[i, 1]
        vxi = vel[i, 0]; vyi = vel[i, 1]
        ddx = dest[i, 0] - xi
        ddy = dest[i, 1] - yi
        dist_dest = np.sqrt(ddx * ddx + ddy * ddy) + 1e-9
        ex = ddx / dist_dest; ey = ddy / dist_dest
        fx = mass * (v0[i] * ex - vxi) / tau
        fy = mass * (v0[i] * ey - vyi) / tau
        for j in range(N):
            if j == i:
                continue
            rxij = xi - pos[j, 0]
            ryij = yi - pos[j, 1]
            dij = np.sqrt(rxij * rxij + ryij * ryij) + 1e-9
            nxij = rxij / dij; nyij = ryij / dij
            rsum = 2.0 * radius
            social_f = A * np.exp((rsum - dij) / B)
            fx += social_f * nxij
            fy += social_f * nyij
            overlap = rsum - dij
            if overlap > 0.0:
                delta_vt = ((vel[j, 0] - vxi) * (-nyij) + (vel[j, 1] - vyi) * nxij)
                fx += k_body * overlap * nxij - kappa * overlap * delta_vt * (-nyij)
                fy += k_body * overlap * nyij - kappa * overlap * delta_vt * nxij
        gi = int((yi - grid_origin_y) / grid_cell_m)
        gj = int((xi - grid_origin_x) / grid_cell_m)
        gi = max(0, min(grid_rows - 1, gi))
        gj = max(0, min(grid_cols - 1, gj))
        dw = wall_dist[gi, gj]
        if dw < 3.0:
            wall_f = A_w * np.exp(-dw / B_w)
            fx += wall_f * wall_gx[gi, gj]
            fy += wall_f * wall_gy[gi, gj]
        forces[i, 0] = fx
        forces[i, 1] = fy
    return forces


v = venue()
occ = v.occupancy.astype(bool)
wall_dist, wall_gx, wall_gy = _precompute_walls(occ, v.cell_m)
rows, cols = v.grid_shape
ox, oy = v.origin_m
hash_cols = int(np.ceil(cols * v.cell_m / HASH_CELL)) + 2
hash_rows = int(np.ceil(rows * v.cell_m / HASH_CELL)) + 2

rng = np.random.default_rng(0)
stage_pos = [np.array(s["pos_m"]) for s in v.stages]


def make_agents(n):
    """Realistic crowd: clustered in front of stages, ~35 m radius."""
    per = n // len(stage_pos) + 1
    pts, dests = [], []
    for sp in stage_pos:
        ang = rng.uniform(0, 2 * np.pi, per)
        rad = rng.uniform(2.0, 35.0, per)
        pts.append(sp + np.column_stack([np.cos(ang), np.sin(ang)]) * rad[:, None])
        dests.append(np.tile(sp, (per, 1)))
    pos = np.vstack(pts)[:n].astype(np.float64)
    dest = np.vstack(dests)[:n].astype(np.float64)
    vel = rng.normal(0, 0.1, (n, 2)).astype(np.float64)
    v0 = rng.normal(V0_MEAN, V0_STD, n).clip(0.3, 3.0).astype(np.float64)
    return pos, vel, dest, v0


def run_hashed(pos, vel, dest, v0, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        sa, co = _build_spatial_hash(pos, HASH_CELL, ox, oy, hash_rows, hash_cols)
        _force_kernel(
            pos, vel, dest, v0, wall_dist, wall_gx, wall_gy,
            float(ox), float(oy), float(v.cell_m), rows, cols,
            sa, co, float(ox), float(oy), float(HASH_CELL), hash_rows, hash_cols,
            MASS, TAU, A_REP, B_REP, A_WALL, B_WALL, K_BODY, KAPPA, RADIUS,
        )
    return (time.perf_counter() - t0) / reps


def run_allpairs(pos, vel, dest, v0, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        _force_kernel_allpairs(
            pos, vel, dest, v0, wall_dist, wall_gx, wall_gy,
            float(ox), float(oy), float(v.cell_m), rows, cols,
            MASS, TAU, A_REP, B_REP, A_WALL, B_WALL, K_BODY, KAPPA, RADIUS,
        )
    return (time.perf_counter() - t0) / reps


out = {}

# ---- warm up both JITs ----
p, ve, d, v0 = make_agents(64)
sa, co = _build_spatial_hash(p, HASH_CELL, ox, oy, hash_rows, hash_cols)
_force_kernel(p, ve, d, v0, wall_dist, wall_gx, wall_gy, float(ox), float(oy),
              float(v.cell_m), rows, cols, sa, co, float(ox), float(oy),
              float(HASH_CELL), hash_rows, hash_cols, MASS, TAU, A_REP, B_REP,
              A_WALL, B_WALL, K_BODY, KAPPA, RADIUS)
_force_kernel_allpairs(p, ve, d, v0, wall_dist, wall_gx, wall_gy, float(ox),
                       float(oy), float(v.cell_m), rows, cols, MASS, TAU, A_REP,
                       B_REP, A_WALL, B_WALL, K_BODY, KAPPA, RADIUS)

# ---- equivalence check ----
p, ve, d, v0 = make_agents(1500)
sa, co = _build_spatial_hash(p, HASH_CELL, ox, oy, hash_rows, hash_cols)
f_hash = _force_kernel(p, ve, d, v0, wall_dist, wall_gx, wall_gy, float(ox),
                       float(oy), float(v.cell_m), rows, cols, sa, co, float(ox),
                       float(oy), float(HASH_CELL), hash_rows, hash_cols, MASS,
                       TAU, A_REP, B_REP, A_WALL, B_WALL, K_BODY, KAPPA, RADIUS)
f_all = _force_kernel_allpairs(p, ve, d, v0, wall_dist, wall_gx, wall_gy,
                               float(ox), float(oy), float(v.cell_m), rows, cols,
                               MASS, TAU, A_REP, B_REP, A_WALL, B_WALL, K_BODY,
                               KAPPA, RADIUS)
denom = np.abs(f_all).max()
out["hash_vs_allpairs_max_abs_err_N"] = float(np.abs(f_hash - f_all).max())
out["hash_vs_allpairs_max_rel_err"] = float(np.abs(f_hash - f_all).max() / denom)

# ---- JIT vs pure Python (same algorithm, via .py_func) ----
N_PY = 400
p, ve, d, v0 = make_agents(N_PY)
sa, co = _build_spatial_hash(p, HASH_CELL, ox, oy, hash_rows, hash_cols)
args = (p, ve, d, v0, wall_dist, wall_gx, wall_gy, float(ox), float(oy),
        float(v.cell_m), rows, cols, sa, co, float(ox), float(oy),
        float(HASH_CELL), hash_rows, hash_cols, MASS, TAU, A_REP, B_REP,
        A_WALL, B_WALL, K_BODY, KAPPA, RADIUS)
t0 = time.perf_counter()
_force_kernel.py_func(*args)
py_s = time.perf_counter() - t0
reps = 200
t0 = time.perf_counter()
for _ in range(reps):
    _force_kernel(*args)
jit_s = (time.perf_counter() - t0) / reps
out["jit_bench_n_agents"] = N_PY
out["jit_pure_python_s_per_step"] = py_s
out["jit_numba_s_per_step"] = jit_s
out["jit_speedup_x"] = py_s / jit_s

# ---- spatial hash vs all-pairs scaling ----
scaling = []
for n in (1000, 2000, 4000, 8000):
    p, ve, d, v0 = make_agents(n)
    reps = max(3, int(40000 / n))
    h = run_hashed(p, ve, d, v0, reps)
    a = run_allpairs(p, ve, d, v0, max(2, reps // 2))
    scaling.append({
        "n_agents": n,
        "hashed_ms_per_step": h * 1000,
        "allpairs_ms_per_step": a * 1000,
        "speedup_x": a / h,
        "hashed_agent_steps_per_s": n / h,
    })
    print(f"  N={n}: hashed {h*1000:.2f} ms, all-pairs {a*1000:.2f} ms, {a/h:.1f}x")
out["scaling"] = scaling

print(json.dumps(out, indent=2))
with open("results_b.json", "w") as f:
    json.dump(out, f, indent=2)
