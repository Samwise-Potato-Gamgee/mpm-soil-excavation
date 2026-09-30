#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_pour_run.py -- Step 6d-test2 full pouring run (one strain basis: P0 or P1d).

Reuses the emission mechanism of scripts/mpm_pour_feasibility.py: 559 layers x 81 particles are
revived one layer per emission event (one batch: host flag update + one notify_model_changed +
_reset_history_kernel + position/velocity kernel for both states).  Layer k is emitted at step 4k
at z = 0.35 m with velocity (0, 0, -1.0) m/s; the last layer (k = 558) is emitted at step 2232.

Pour phase: steps 0 .. 2235.  Settle phase: from step 2236 the mean speed of ACTIVE particles is
reduced on the GPU every 10 steps; the run stops early when it has stayed below 1e-3 m/s for 40
consecutive checks (400 steps) and the step is >= 2236 + 400 = 2636.  Hard maximum 3436 steps.

Fixed scene: dry sand phi = 34 deg, rho = 1700, E = 3e7, nu = 0.3, yield_pressure = 5e6,
yield_stress = 0, dilatancy = 0, viscosity = 0; dense grid, apic, S2 collider, Q1 velocity,
max_iterations 50, tolerance 1e-4, voxel 0.015 m, ppc 2, spacing 0.0075 m, dt = 0.0025 s, ground
top z = 0, friction 0.5, half-extent 0.6 m.  Inactive pool particles use the compact parking block
(36 x 36 x 35, lower corner (0.70, -0.135, 0.40)).

Writes logs/<prefix>.npz and prints exactly one final line RUN_DONE or RUN_FAILED <reason>.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

from mpm_common import ACTIVE, GRAVITY, SceneAConfig, SoilParams, SolverParams

# ---- fixed design ---------------------------------------------------------------------------------
VOXEL = 0.015
PPC = 2
SPACING = VOXEL / PPC                 # 0.0075 m
DT = 0.0025
GROUND_HALF = 0.6
N_LAYERS = 559
N_PER_LAYER = 81
N_POOL = N_LAYERS * N_PER_LAYER       # 45,279
PARTICLE_MASS = 1700.0 * SPACING ** 3
EMIT_EVERY = 4
Z_EMIT = 0.35
V_EMIT = -1.0
POUR_STEPS = 2236                     # steps 0 .. 2235 (last emission at 2232)
SETTLE_MIN_STEP = POUR_STEPS + 400    # 2636
HARD_MAX_STEPS = 3436
CHECK_EVERY = 10
REST_THRESH = 1.0e-3
REST_CHECKS = 40
SNAP_STEPS = (600, 1200, 1800, 2236)


# --------------------------------------------------------------------------------------------------
# GPU kernels (copied from mpm_pour_feasibility.py; reduction kernel added)
# --------------------------------------------------------------------------------------------------
@wp.kernel
def _reset_history_kernel(
    indices: wp.array[wp.int32],
    initial_Jp: wp.array[wp.float32],
    elastic_strain: wp.array[wp.mat33],
    particle_transform: wp.array[wp.mat33],
    qd_grad: wp.array[wp.mat33],
    stress: wp.array[wp.mat33],
    Jp: wp.array[wp.float32],
):
    i = wp.tid()
    idx = indices[i]
    ident = wp.identity(n=3, dtype=float)
    elastic_strain[idx] = ident
    particle_transform[idx] = ident
    qd_grad[idx] = wp.mat33(0.0)
    stress[idx] = wp.mat33(0.0)
    Jp[idx] = initial_Jp[idx]


@wp.kernel
def _set_pos_vel_kernel(
    indices: wp.array[wp.int32],
    new_pos: wp.array[wp.vec3],
    new_vel: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    idx = indices[i]
    q[idx] = new_pos[i]
    qd[idx] = new_vel[i]


@wp.kernel
def _active_stats_kernel(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    stats: wp.array[wp.float32],
):
    """stats = [sum |v|, max |v|, max z] over ACTIVE particles (atomic reduction)."""
    i = wp.tid()
    if (flags[i] & ACTIVE) != 0:
        s = wp.length(qd[i])
        wp.atomic_add(stats, 0, s)
        wp.atomic_max(stats, 1, s)
        wp.atomic_max(stats, 2, q[i][2])


# --------------------------------------------------------------------------------------------------
# Pool / scene construction
# --------------------------------------------------------------------------------------------------
def pool_positions() -> np.ndarray:
    """Compact parking block (36 x 36 x 35), lower corner (0.70, -0.135, 0.40)."""
    pos = np.empty((N_POOL, 3), dtype=np.float32)
    for i in range(N_POOL):
        a = i % 36
        b = (i // 36) % 36
        c = i // 1296
        pos[i, 0] = 0.70 + a * SPACING
        pos[i, 1] = -0.135 + b * SPACING
        pos[i, 2] = 0.40 + c * SPACING
    return pos


def build_scene(strain_basis: str):
    cfg = SceneAConfig(
        diameter=0.15,
        voxel_override=VOXEL,
        ground_half_override=GROUND_HALF,
        ppc=PPC,
        ground_friction=0.5,
    )
    soil = SoilParams(friction_angle_deg=34.0)
    solver_params = SolverParams(
        grid_type="dense",
        transfer_scheme="apic",
        integration_scheme="pic",
        collider_basis="S2",
        velocity_basis="Q1",
        strain_basis=strain_basis,
        max_iterations=50,
        tolerance=1.0e-4,
    )

    positions = pool_positions()
    mass = cfg.particle_mass
    radius = cfg.particle_radius

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    flags = np.zeros(N_POOL, dtype=np.int32)
    # Layer 0 pre-activated so the dense grid can be built (all-inactive bbox = +-inf).
    flags[0:N_PER_LAYER] = ACTIVE

    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros((N_POOL, 3), dtype=np.float32).tolist(),
        mass=[mass] * N_POOL,
        radius=[radius] * N_POOL,
        flags=flags.tolist(),
    )
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
        hx=GROUND_HALF,
        hy=GROUND_HALF,
        hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
    )

    model = builder.finalize()
    model.set_gravity(wp.vec3(*GRAVITY))
    for attr, value in soil.model_attributes().items():
        getattr(model.mpm, attr).fill_(value)

    solver = SolverImplicitMPM(model, config=solver_params.make_config(cfg.voxel))
    return model, solver, cfg


def layer_positions(k: int) -> np.ndarray:
    pos = np.empty((N_PER_LAYER, 3), dtype=np.float32)
    for j in range(N_PER_LAYER):
        pos[j, 0] = ((j % 9) - 4) * SPACING
        pos[j, 1] = ((j // 9) - 4) * SPACING
        pos[j, 2] = Z_EMIT
    return pos


def emit(solver, model, state_a, state_b, device, k: int) -> int:
    idx = np.arange(k * N_PER_LAYER, (k + 1) * N_PER_LAYER, dtype=np.int32)
    flags = model.particle_flags.numpy()
    flags[idx] = flags[idx] | ACTIVE
    model.particle_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    idx_wp = wp.array(idx, dtype=wp.int32, device=device)
    npos = wp.array(layer_positions(k), dtype=wp.vec3, device=device)
    nvel_np = np.zeros((N_PER_LAYER, 3), dtype=np.float32)
    nvel_np[:, 2] = V_EMIT
    nvel = wp.array(nvel_np, dtype=wp.vec3, device=device)

    for st in (state_a, state_b):
        wp.launch(
            _reset_history_kernel,
            dim=N_PER_LAYER,
            inputs=[idx_wp, model.mpm.particle_Jp, st.mpm.particle_elastic_strain,
                    st.mpm.particle_transform, st.mpm.particle_qd_grad, st.mpm.particle_stress,
                    st.mpm.particle_Jp],
            device=device,
        )
        wp.launch(
            _set_pos_vel_kernel,
            dim=N_PER_LAYER,
            inputs=[idx_wp, npos, nvel, st.particle_q, st.particle_qd],
            device=device,
        )

    n_active = int(np.count_nonzero(flags & ACTIVE))
    expected = N_PER_LAYER * (k + 1)
    if n_active != expected:
        raise RuntimeError(f"active count {n_active} != expected {expected} after emission {k}")
    return n_active


def active_stats(state, model, device) -> tuple[float, float, float]:
    """Return (sum |v|, max |v|, max z) over ACTIVE particles via one GPU reduction."""
    stats = wp.array(np.array([0.0, -1.0e30, -1.0e30], dtype=np.float32), dtype=float, device=device)
    wp.launch(
        _active_stats_kernel,
        dim=int(model.particle_count),
        inputs=[state.particle_q, state.particle_qd, model.particle_flags, stats],
        device=device,
    )
    wp.synchronize()
    vals = stats.numpy()
    return float(vals[0]), float(vals[1]), float(vals[2])


# --------------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--strain-basis", required=True, choices=["P0", "P1d"])
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    model, solver, cfg = build_scene(args.strain_basis)
    device = model.device
    state_0 = model.state()
    state_1 = model.state()

    step_time_ms = np.zeros(HARD_MAX_STEPS, dtype=np.float64)
    active_counts = np.zeros(HARD_MAX_STEPS, dtype=np.int64)
    check_steps: list[int] = []
    check_mean_speed: list[float] = []
    check_max_speed: list[float] = []
    check_max_z: list[float] = []
    snaps: dict[int, np.ndarray] = {}

    n_active = int(np.count_nonzero(model.particle_flags.numpy() & ACTIVE))
    if n_active != N_PER_LAYER:
        raise RuntimeError(f"initial active count {n_active} != {N_PER_LAYER}")

    print(f"[pour] strain_basis={args.strain_basis} N_pool={N_POOL} voxel={cfg.voxel} "
          f"spacing={cfg.spacing} dt={DT} pour_steps={POUR_STEPS} hard_max={HARD_MAX_STEPS}", flush=True)

    total = 0
    stop_reason = "max_steps"
    rest_count = 0
    for s in range(HARD_MAX_STEPS):
        # pour phase: emit layer k = s//4 while k < N_LAYERS
        if s % EMIT_EVERY == 0 and (s // EMIT_EVERY) < N_LAYERS:
            n_active = emit(solver, model, state_0, state_1, device, s // EMIT_EVERY)
        active_counts[s] = n_active

        t0 = time.perf_counter()
        solver.step(state_0, state_1, None, None, DT)
        solver.project_outside(state_1, state_1, DT)
        wp.synchronize()
        step_time_ms[s] = (time.perf_counter() - t0) * 1.0e3
        state_0, state_1 = state_1, state_0
        total = s + 1

        if total in SNAP_STEPS:
            q = state_0.particle_q.numpy()
            mask = (model.particle_flags.numpy() & ACTIVE) != 0
            snaps[total] = q[mask].astype(np.float32).copy()

        if total % CHECK_EVERY == 0:
            ssum, smax, zmax = active_stats(state_0, model, device)
            mean_speed = ssum / max(n_active, 1)
            check_steps.append(total)
            check_mean_speed.append(mean_speed)
            check_max_speed.append(smax)
            check_max_z.append(zmax)
            if total >= POUR_STEPS:
                if mean_speed < REST_THRESH:
                    rest_count += 1
                else:
                    rest_count = 0
                if rest_count >= REST_CHECKS and total >= SETTLE_MIN_STEP:
                    stop_reason = "settled"
                    break

    # ---- end-of-run readouts ---------------------------------------------------------------------
    final_q = state_0.particle_q.numpy().astype(np.float32)
    final_v = state_0.particle_qd.numpy().astype(np.float32)
    final_stress = state_0.mpm.particle_stress.numpy()
    active_mask = (model.particle_flags.numpy() & ACTIVE) != 0
    active_end = int(np.count_nonzero(active_mask))
    n_below = int(np.count_nonzero(active_mask & (final_q[:, 2] < -1.0e-3)))
    n_outside = int(np.count_nonzero(active_mask & ((np.abs(final_q[:, 0]) > GROUND_HALF) |
                                                    (np.abs(final_q[:, 1]) > GROUND_HALF))))

    nan_flag = False
    for arr in (final_q.astype(np.float64), final_v.astype(np.float64),
                np.asarray(final_stress, dtype=np.float64)):
        if not np.all(np.isfinite(arr)):
            nan_flag = True

    q_end = state_0.particle_q.numpy()
    snaps[total] = q_end[active_mask].astype(np.float32).copy()

    st = step_time_ms[:total]
    quarters = np.array_split(st, 4)
    med_q = [float(np.median(qq)) if qq.size else float("nan") for qq in quarters]
    while len(med_q) < 4:
        med_q.append(float("nan"))

    out_path = os.path.join(log_dir, args.prefix + ".npz")
    np.savez_compressed(
        out_path,
        strain_basis=args.strain_basis,
        dt=np.float64(DT),
        voxel=np.float64(VOXEL),
        spacing=np.float64(SPACING),
        ground_half=np.float64(GROUND_HALF),
        particle_mass=np.float64(PARTICLE_MASS),
        n_pool=np.int64(N_POOL),
        n_layers=np.int64(N_LAYERS),
        n_per_layer=np.int64(N_PER_LAYER),
        total_steps=np.int64(total),
        stop_reason=stop_reason,
        step_time_ms=st,
        active_count=active_counts[:total],
        check_steps=np.asarray(check_steps, dtype=np.int64),
        check_mean_speed=np.asarray(check_mean_speed, dtype=np.float64),
        check_max_speed=np.asarray(check_max_speed, dtype=np.float64),
        check_max_z=np.asarray(check_max_z, dtype=np.float64),
        snap_steps=np.asarray(sorted(snaps.keys()), dtype=np.int64),
        snap_q_600=snaps.get(600, np.zeros((0, 3), dtype=np.float32)),
        snap_q_1200=snaps.get(1200, np.zeros((0, 3), dtype=np.float32)),
        snap_q_1800=snaps.get(1800, np.zeros((0, 3), dtype=np.float32)),
        snap_q_2236=snaps.get(2236, np.zeros((0, 3), dtype=np.float32)),
        snap_q_end=snaps.get(total, np.zeros((0, 3), dtype=np.float32)),
        final_q=final_q,
        final_v=final_v,
        active_mask=active_mask,
        nan=np.bool_(nan_flag),
        n_below=np.int64(n_below),
        n_outside=np.int64(n_outside),
        active_end=np.int64(active_end),
    )

    print(f"[pour] wrote {out_path}", flush=True)
    print(f"[POUR] sbasis={args.strain_basis} steps={total} stop={stop_reason} nan={nan_flag} "
          f"med_q1={med_q[0]:.3f} med_q2={med_q[1]:.3f} med_q3={med_q[2]:.3f} med_q4={med_q[3]:.3f} "
          f"active_end={active_end} n_below={n_below} n_outside={n_outside}", flush=True)


if __name__ == "__main__":
    wp.init()
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RUN_FAILED " + traceback.format_exc().strip().splitlines()[-1])
        sys.exit(1)
    print("RUN_DONE")
