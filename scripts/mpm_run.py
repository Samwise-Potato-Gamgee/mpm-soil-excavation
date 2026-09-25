#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_run.py -- runner for the local MPM soil simulation (Phase 3, part 1: Scene A).

Builds Scene A (`mpm_scene.build_scene_a`), steps it for a given duration with one solver step per
`dt` (plus `project_outside`), records per-step metrics and position/velocity snapshots, writes an
`.npz` under `PROJECT/logs/`, and prints exactly one final line `RUN_DONE` or `RUN_FAILED <reason>`.

The GPU work is Newton/Warp; the recording is numpy on the host.  Run it only through the project
conda env, e.g.:

    conda run --no-capture-output --prefix PROJECT/envs/mpm python PROJECT/scripts/mpm_run.py \
        --scene A --diameter 0.15 --duration 3.0 --out-prefix sceneA_D015

The runner never retunes solver parameters; it reports whatever the physics produces.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import warp as wp

from mpm_common import SceneAConfig, SoilParams, SolverParams
from mpm_scene import build_scene_a


def snapshot_schedule(duration: float) -> list[float]:
    """Snapshot times [s]: the design-note list, extended if the run is longer than 3 s."""
    base = [0.0, 0.25, 0.5, 1.0, 2.0, 2.5, 3.0]
    extra = [3.5, 4.0, 5.0, 6.0]
    times = [t for t in base if t <= duration + 1e-9]
    if duration > 3.0 + 1e-9:
        times += [t for t in extra if 3.0 < t <= duration + 1e-9]
    return times


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="A", choices=["A"], help="scene to run (only A in this step)")
    p.add_argument("--diameter", "-D", type=float, required=True, help="nozzle diameter D [m]")
    p.add_argument("--duration", type=float, default=3.0, help="simulated seconds")
    p.add_argument("--frames", type=int, default=None, help="alternative to --duration: number of steps")
    p.add_argument("--voxel-over-D", type=float, default=5.0, help="resolution D/voxel (>=5)")
    p.add_argument("--ppc", type=int, default=2, help="particles per cell per axis")
    p.add_argument("--ground-friction", type=float, default=0.5,
                   help="ground collider Coulomb friction coefficient (default 0.5 = current behaviour)")
    p.add_argument("--soil-phi-deg", type=float, default=34.0,
                   help="soil internal friction angle [deg]; mu = tan(phi) (default 34.0 = current behaviour)")
    p.add_argument("--height-factor", type=float, default=2.5,
                   help="column height = height_factor * D (default 2.5 = current behaviour)")
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--tolerance", type=float, default=1.0e-4)
    p.add_argument("--grid-type", default="dense", choices=["dense", "sparse"])
    p.add_argument("--seed", type=int, default=0, help="recorded seed (Scene A has no stochastic elements)")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--log-dir", default=None, help="default PROJECT/logs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    # Seed is recorded for provenance; Scene A uses a deterministic lattice and no randomness.
    np.random.seed(args.seed)

    soil = SoilParams(friction_angle_deg=args.soil_phi_deg)
    solver_params = SolverParams(grid_type=args.grid_type, max_iterations=args.max_iterations, tolerance=args.tolerance)
    cfg = SceneAConfig(
        diameter=args.diameter,
        voxel_over_D=args.voxel_over_D,
        ppc=args.ppc,
        ground_friction=args.ground_friction,
        block_height_factor=args.height_factor,
        duration=args.duration,
    )
    if args.frames is not None:
        cfg.duration = args.frames * cfg.dt

    dt = cfg.dt
    n_steps = int(round(cfg.duration / dt))

    print(f"[run] scene=A D={cfg.diameter} voxel={cfg.voxel:.6f} spacing={cfg.spacing:.6f} "
          f"ppc={cfg.ppc} dt={dt:.6f} n_steps={n_steps} N_expected={cfg.particle_count}", flush=True)
    print(f"[run] soil={soil.name} rho={soil.density} phi={soil.friction_angle_deg} mu={soil.friction:.6f} "
          f"E={soil.young_modulus} nu={soil.poisson_ratio} yield_pressure={soil.yield_pressure} "
          f"yield_stress={soil.yield_stress} tensile_yield_ratio={soil.tensile_yield_ratio}", flush=True)
    print(f"[run] solver grid_type={solver_params.grid_type} transfer={solver_params.transfer_scheme} "
          f"integration={solver_params.integration_scheme} solver={solver_params.solver} "
          f"warmstart={solver_params.warmstart_mode} strain_basis={solver_params.strain_basis} "
          f"collider_basis={solver_params.collider_basis} velocity_basis={solver_params.velocity_basis} "
          f"max_iter={solver_params.max_iterations} tol={solver_params.tolerance}", flush=True)

    scene = build_scene_a(cfg, soil, solver_params)
    model = scene.model
    gravity = model.gravity.numpy()[0].astype(np.float64)  # use the value actually stored in the model

    if model.particle_count != cfg.particle_count:
        raise RuntimeError(f"particle count {model.particle_count} != expected {cfg.particle_count}")

    # ---- recording containers -------------------------------------------------------------------
    times: list[float] = []
    masses: list[float] = []
    kes: list[float] = []
    mean_speeds: list[float] = []
    max_speeds: list[float] = []
    min_zs: list[float] = []
    coms: list[np.ndarray] = []
    com_vels: list[np.ndarray] = []
    ground_forces: list[np.ndarray] = []
    step_times: list[float] = []
    snap_t: list[float] = []
    snap_q: list[np.ndarray] = []
    snap_v: list[np.ndarray] = []

    def record(t: float, step_time: float | None = None) -> None:
        """Append one row of metrics (and optionally a step time)."""
        m = scene.metrics()
        gw = scene.ground_wrench(dt)
        times.append(float(t))
        masses.append(m["mass"])
        kes.append(m["ke"])
        mean_speeds.append(m["mean_speed"])
        max_speeds.append(m["max_speed"])
        min_zs.append(m["min_z"])
        coms.append(m["com"])
        com_vels.append(m["com_vel"])
        ground_forces.append(gw)
        if step_time is not None:
            step_times.append(float(step_time))

    def take_snapshot(t: float) -> None:
        q, v = scene.snapshot()
        snap_t.append(float(t))
        snap_q.append(q.astype(np.float32))
        snap_v.append(v.astype(np.float32))

    # Initial state (t = 0) before any step.
    record(0.0)
    take_snapshot(0.0)

    schedule = snapshot_schedule(cfg.duration)
    snap_i = 1  # snapshot 0 already taken
    for i in range(n_steps):
        t0 = time.perf_counter()
        scene.step(dt)
        st = time.perf_counter() - t0
        t = (i + 1) * dt
        record(t, st)
        while snap_i < len(schedule) and t + 1e-9 >= schedule[snap_i]:
            take_snapshot(schedule[snap_i])
            snap_i += 1

    # ---- save -----------------------------------------------------------------------------------
    out_path = os.path.join(log_dir, args.out_prefix + ".npz")
    np.savez_compressed(
        out_path,
        # per-step arrays (length n_steps + 1)
        t=np.asarray(times),
        mass=np.asarray(masses),
        ke=np.asarray(kes),
        mean_speed=np.asarray(mean_speeds),
        max_speed=np.asarray(max_speeds),
        min_z=np.asarray(min_zs),
        com=np.asarray(coms),
        com_vel=np.asarray(com_vels),
        ground_force=np.asarray(ground_forces),
        # step times (length n_steps)
        step_time=np.asarray(step_times),
        # snapshots (length len(schedule))
        snap_t=np.asarray(snap_t),
        snap_q=np.asarray(snap_q),
        snap_v=np.asarray(snap_v),
        # metadata
        scene=args.scene,
        diameter=np.float64(cfg.diameter),
        voxel=np.float64(cfg.voxel),
        spacing=np.float64(cfg.spacing),
        ppc=np.int64(cfg.ppc),
        dt=np.float64(dt),
        duration=np.float64(cfg.duration),
        n_steps=np.int64(n_steps),
        n_particles=np.int64(model.particle_count),
        total_mass=np.float64(scene.total_mass),
        gravity=gravity,
        ground_friction=np.float64(cfg.ground_friction),
        ground_half_xy=np.float64(cfg.ground_half_xy),
        ground_half_z=np.float64(cfg.ground_half_z),
        block_base_factor=np.float64(cfg.block_base_factor),
        block_height_factor=np.float64(cfg.block_height_factor),
        start_gap_spacing=np.float64(cfg.start_gap_spacing),
        soil_name=soil.name,
        soil_density=np.float64(soil.density),
        soil_friction_angle_deg=np.float64(soil.friction_angle_deg),
        soil_friction=np.float64(soil.friction),
        soil_young_modulus=np.float64(soil.young_modulus),
        soil_poisson_ratio=np.float64(soil.poisson_ratio),
        soil_yield_pressure=np.float64(soil.yield_pressure),
        soil_yield_stress=np.float64(soil.yield_stress),
        soil_tensile_yield_ratio=np.float64(soil.tensile_yield_ratio),
        seed=np.int64(args.seed),
    )
    print(f"[run] wrote {out_path}", flush=True)
    print(f"[run] N={model.particle_count} total_mass={scene.total_mass:.6f} kg "
          f"first_step_ms={1e3*step_times[0]:.1f} median_step_ms={1e3*float(np.median(step_times[1:])):.2f} "
          f"ms_per_sim_s={1e3*float(np.median(step_times[1:]))/dt:.0f}", flush=True)


if __name__ == "__main__":
    wp.init()
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("RUN_FAILED " + traceback.format_exc().strip().splitlines()[-1])
        sys.exit(1)
    print("RUN_DONE")
