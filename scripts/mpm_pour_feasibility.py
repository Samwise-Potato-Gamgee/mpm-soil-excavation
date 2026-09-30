#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_pour_feasibility.py -- Step 6d-test2 pour feasibility run (one configuration).

Tests the two riskiest assumptions of the planned pouring repose test:
  (1) revive/emission works with strain basis P1d (particle-backed stress warmstart), and
  (2) a large, mostly inactive pool does not make the solver steps slow.

Scene (fixed): dry sand (phi = 34 deg, rho = 1700, E = 3e7, nu = 0.3, yield_pressure = 5e6,
yield_stress = 0, dilatancy = 0, viscosity = 0), dense grid, apic transfer, collider basis S2,
velocity basis Q1, max_iterations 50, tolerance 1e-4, voxel 0.015 m, ppc 2, spacing 0.0075 m,
dt = 0.0025 s, ground box top at z = 0, friction 0.5, half-extent 0.6 m.

Pool: 45,279 particles = 559 layers x 81 (9 x 9, spacing 0.0075, centred on x = y = 0).  Layer
k = i // 81.  Emission event k happens at step 4*k (steps 0, 4, ..., 296; 75 events in 300 steps):
layer k is revived at z = 0.35 m, centred on x = y = 0, with velocity (0, 0, -1.0) m/s.  One batch
per event: host flag update + one notify_model_changed + _reset_history_kernel + position/velocity
kernel for BOTH states.  Layer 0 is activated before solver construction so the dense grid always
has at least one active particle (the solver builds its grid at construction; an all-inactive pool
gives a +-inf bounding box).

Parking of the inactive pool: "tall" = a vertical column above the emitter (layer k at
z = 0.35 + (k+1)*0.0075); "compact" = a 36 x 36 x 35 block beside the emitter, lower corner
(0.70, -0.135, 0.40).  Positions of inactive particles are ignored until they are revived.

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

from mpm_common import (
    ACTIVE,
    GRAVITY,
    SceneAConfig,
    SoilParams,
    SolverParams,
)

# ---- fixed design (do not change) -----------------------------------------------------------------
VOXEL = 0.015
PPC = 2
SPACING = VOXEL / PPC                 # 0.0075 m
DT = 0.0025
GROUND_HALF = 0.6
N_LAYERS = 559
N_PER_LAYER = 81                      # 9 x 9 lattice
N_POOL = N_LAYERS * N_PER_LAYER       # 45,279
EMIT_EVERY = 4                        # steps between emission events
N_EMIT_MAX = 75                       # events within 300 steps
Z_EMIT = 0.35
V_EMIT = -1.0


# --------------------------------------------------------------------------------------------------
# GPU kernels (copied from scripts/mpm_kill_test.py; position/velocity kernel gained a velocity arg)
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
    """Reset per-particle MPM history to identity/initial/zero for the listed indices only."""
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
    """Overwrite position and velocity for the listed indices."""
    i = wp.tid()
    idx = indices[i]
    q[idx] = new_pos[i]
    qd[idx] = new_vel[i]


# --------------------------------------------------------------------------------------------------
# Pool / scene construction
# --------------------------------------------------------------------------------------------------
def pool_positions(parking: str) -> np.ndarray:
    """Initial (parking) positions of all pool particles; emission overwrites the revived layer."""
    pos = np.empty((N_POOL, 3), dtype=np.float32)
    for i in range(N_POOL):
        ix = (i % N_PER_LAYER) % 9
        iy = (i % N_PER_LAYER) // 9
        if parking == "tall":
            k = i // N_PER_LAYER
            pos[i, 0] = (ix - 4) * SPACING
            pos[i, 1] = (iy - 4) * SPACING
            pos[i, 2] = Z_EMIT + (k + 1) * SPACING
        elif parking == "compact":
            a = i % 36
            b = (i // 36) % 36
            c = i // 1296
            pos[i, 0] = 0.70 + a * SPACING
            pos[i, 1] = -0.135 + b * SPACING
            pos[i, 2] = 0.40 + c * SPACING
        else:
            raise ValueError(f"unknown parking {parking!r}")
    return pos


def build_scene(strain_basis: str, parking: str):
    """Build the pool model + solver; layer 0 is pre-activated so the dense grid has a bbox."""
    cfg = SceneAConfig(
        diameter=0.15,
        voxel_override=VOXEL,
        ground_half_override=GROUND_HALF,
        ppc=PPC,
        ground_friction=0.5,
    )
    soil = SoilParams(friction_angle_deg=34.0)  # all other defaults as required
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

    positions = pool_positions(parking)
    mass = cfg.particle_mass
    radius = cfg.particle_radius

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)

    flags = np.zeros(N_POOL, dtype=np.int32)
    # Layer 0 is pre-activated so the dense grid has a finite bbox: with ALL particles inactive the
    # solver constructor raises ValueError (dense bbox = +-inf -> negative grid resolution).  This
    # was verified with a probe run; the first emission re-activates layer 0 with a notify.
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
    return model, solver, cfg, soil, solver_params


def layer_positions(k: int) -> np.ndarray:
    """Emission position (z = Z_EMIT, centred on x = y = 0) of layer k."""
    pos = np.empty((N_PER_LAYER, 3), dtype=np.float32)
    for j in range(N_PER_LAYER):
        ix = j % 9
        iy = j // 9
        pos[j, 0] = (ix - 4) * SPACING
        pos[j, 1] = (iy - 4) * SPACING
        pos[j, 2] = Z_EMIT
    return pos


def emit(solver, model, state_a, state_b, device, k: int) -> int:
    """Revive layer k in one batch; return the new active count."""
    idx = np.arange(k * N_PER_LAYER, (k + 1) * N_PER_LAYER, dtype=np.int32)

    # one host flag update ...
    flags = model.particle_flags.numpy()
    flags[idx] = flags[idx] | ACTIVE
    model.particle_flags.assign(flags)
    # ... one notify ...
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    idx_wp = wp.array(idx, dtype=wp.int32, device=device)
    npos = wp.array(layer_positions(k), dtype=wp.vec3, device=device)
    nvel_np = np.zeros((N_PER_LAYER, 3), dtype=np.float32)
    nvel_np[:, 2] = V_EMIT
    nvel = wp.array(nvel_np, dtype=wp.vec3, device=device)

    # ... one history-reset kernel + one position/velocity kernel per state
    for st in (state_a, state_b):
        wp.launch(
            _reset_history_kernel,
            dim=N_PER_LAYER,
            inputs=[
                idx_wp,
                model.mpm.particle_Jp,
                st.mpm.particle_elastic_strain,
                st.mpm.particle_transform,
                st.mpm.particle_qd_grad,
                st.mpm.particle_stress,
                st.mpm.particle_Jp,
            ],
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


# --------------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--strain-basis", required=True, choices=["P0", "P1d"])
    ap.add_argument("--parking", required=True, choices=["tall", "compact"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    steps = int(args.steps)

    model, solver, cfg, soil, solver_params = build_scene(args.strain_basis, args.parking)
    device = model.device
    state_0 = model.state()
    state_1 = model.state()

    step_times_ms = np.zeros(steps, dtype=np.float64)
    emission_times_ms: list[float] = []
    active_counts = np.zeros(steps, dtype=np.int64)
    layer0_mean_z = np.full(steps, np.nan, dtype=np.float64)
    stress_layer0_step40 = float("nan")
    pos15_step60 = np.zeros((15, N_PER_LAYER, 3), dtype=np.float32)
    n_active = int(np.count_nonzero(model.particle_flags.numpy() & ACTIVE))
    if n_active != N_PER_LAYER:
        raise RuntimeError(f"initial active count {n_active} != {N_PER_LAYER}")

    print(f"[pour] strain_basis={args.strain_basis} parking={args.parking} N_pool={N_POOL} "
          f"voxel={cfg.voxel} spacing={cfg.spacing} dt={DT} steps={steps}", flush=True)

    for s in range(steps):
        if s % EMIT_EVERY == 0 and (s // EMIT_EVERY) < N_EMIT_MAX:
            k = s // EMIT_EVERY
            t_e = time.perf_counter()
            n_active = emit(solver, model, state_0, state_1, device, k)
            wp.synchronize()
            emission_times_ms.append((time.perf_counter() - t_e) * 1.0e3)
        active_counts[s] = n_active

        # pre-step snapshot (t = s*DT since layer 0 was emitted at s = 0)
        q_host = state_0.particle_q.numpy()
        layer0_mean_z[s] = float(q_host[0:N_PER_LAYER, 2].mean())
        if s == 40:
            stress_layer0_step40 = float(np.abs(state_0.mpm.particle_stress.numpy()[0:N_PER_LAYER]).max())
        if s == 60:
            pos15_step60 = q_host[0:15 * N_PER_LAYER].reshape(15, N_PER_LAYER, 3).astype(np.float32).copy()

        t0 = time.perf_counter()
        solver.step(state_0, state_1, None, None, DT)
        solver.project_outside(state_1, state_1, DT)
        wp.synchronize()
        step_times_ms[s] = (time.perf_counter() - t0) * 1.0e3
        state_0, state_1 = state_1, state_0

    # ---- end-of-run readouts ---------------------------------------------------------------------
    final_q = state_0.particle_q.numpy().astype(np.float32)
    final_v = state_0.particle_qd.numpy().astype(np.float32)
    final_stress = state_0.mpm.particle_stress.numpy()
    active_mask = (model.particle_flags.numpy() & ACTIVE) != 0
    active_end = int(np.count_nonzero(active_mask))
    n_below = int(np.count_nonzero(active_mask & (final_q[:, 2] < -1.0e-3)))

    nan_flag = False
    for arr in (final_q.astype(np.float64), final_v.astype(np.float64), np.asarray(final_stress, dtype=np.float64)):
        if not np.all(np.isfinite(arr)):
            nan_flag = True

    aq = final_q[active_mask].astype(np.float64)
    z_max_end = float(aq[:, 2].max()) if aq.shape[0] else float("nan")
    runout_end = float(np.sqrt(aq[:, 0] ** 2 + aq[:, 1] ** 2).max()) if aq.shape[0] else float("nan")
    n_outside_end = int(np.count_nonzero((np.abs(aq[:, 0]) > GROUND_HALF) | (np.abs(aq[:, 1]) > GROUND_HALF)))

    st = step_times_ms
    med_0_49 = float(np.median(st[:50])) if st.size > 0 else float("nan")
    med_50_end = float(np.median(st[50:])) if st.size > 50 else float("nan")
    med_emit = float(np.median(emission_times_ms)) if emission_times_ms else float("nan")

    out_path = os.path.join(log_dir, args.prefix + ".npz")
    np.savez_compressed(
        out_path,
        strain_basis=args.strain_basis,
        parking=args.parking,
        steps=np.int64(steps),
        dt=np.float64(DT),
        voxel=np.float64(VOXEL),
        spacing=np.float64(SPACING),
        ground_half=np.float64(GROUND_HALF),
        n_pool=np.int64(N_POOL),
        n_layers=np.int64(N_LAYERS),
        n_per_layer=np.int64(N_PER_LAYER),
        n_emissions=np.int64(len(emission_times_ms)),
        n_emit_max=np.int64(N_EMIT_MAX),
        step_time_ms=st,
        emission_time_ms=np.asarray(emission_times_ms, dtype=np.float64),
        active_count=active_counts,
        layer0_mean_z=layer0_mean_z,
        stress_layer0_step40=np.float64(stress_layer0_step40),
        pos15_step60=pos15_step60,
        final_q=final_q,
        final_v=final_v,
        active_mask=active_mask,
        nan=np.bool_(nan_flag),
        n_below=np.int64(n_below),
        active_end=np.int64(active_end),
        z_max_end=np.float64(z_max_end),
        runout_end=np.float64(runout_end),
        n_outside_end=np.int64(n_outside_end),
    )

    print(f"[pour] wrote {out_path}", flush=True)
    print(f"[FEAS] sbasis={args.strain_basis} parking={args.parking} steps={steps} nan={nan_flag} "
          f"med_step_0_49_ms={med_0_49:.4f} med_step_50_299_ms={med_50_end:.4f} "
          f"med_emit_ms={med_emit:.4f} active_end={active_end} n_below={n_below}", flush=True)


if __name__ == "__main__":
    wp.init()
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RUN_FAILED " + traceback.format_exc().strip().splitlines()[-1])
        sys.exit(1)
    print("RUN_DONE")
