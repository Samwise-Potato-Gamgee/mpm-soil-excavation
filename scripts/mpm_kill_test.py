#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_kill_test.py -- runtime particle "kill"/"revive" experiment for Newton's implicit MPM solver.

This is a COPY-AND-ADAPT of Newton's own example
``newton/examples/mpm/example_mpm_granular.py`` (particle generation + solver step loop) and of
``newton/examples/mpm/example_mpm_twoway_coupling.py`` (the collider-impulse -> net wrench reduction).
It does NOT modify anything inside the conda env.

What it tests (selected with --test):
  t1  Deadness: kill a top slab at --kill-frame, measure the max per-particle displacement of the
      killed particles from frame kill+1 to the end.  --mode inplace (leave them where they are) or
      park (teleport them to --park-xyz, e.g. 100 m away, zero velocity).
  t2  Equivalence: one invocation per --t2-case:
        K = all particles built, slab killed at frame 0 before the first step
        R = reference model built WITHOUT the slab particles (same survivor order)
        C = control, all particles active, no kill
      Saves survivors' final positions/velocities and the per-frame net wrench on the box collider.
      The analyze script compares K vs R vs C.
  t3  Revival: kill in place at --kill-frame, then at --revive-frame reactivate the slab, reset ONLY
      the revived particles' MPM history with a small kernel, write new positions/zero velocity, and
      run on.  Checks revived motion and that history was reset to identity/initial/zero.
  t4  Memory/grid: pick --grid-type sparse or dense and --mode none|inplace|park; saves step times.
      Intended for the full 226,981-particle scene.
  t5  Flag-update cost: median time of one notify_model_changed(MODEL_PROPERTIES) over 20 calls, step
      time with/without notify each step, and ONE in-place flag edit WITHOUT notify (expected ignored).

Tolerances used by the analyze script (also printed in the results):
  T1 dead tolerance      : max killed displacement < 1e-6 m
  T2 equivalence         : max|K-R| <= 5% of max|K-C| (and of max|R-C|)
  T3 revival motion      : revived displacement > 1e-3 m after revival
  T3 history match       : max |history - expected| < 1e-5
  T5 ignored-without-notify: mean displacement of flag-cleared-but-not-notified particles > 1e-4 m

Every run writes PROJECT/logs/<out-prefix>.npz and prints exactly one final line RUN_DONE or
RUN_FAILED <reason>.
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

# --------------------------------------------------------------------------------------------------
# Tolerances (also reported in the saved results)
# --------------------------------------------------------------------------------------------------
T1_DEAD_TOL = 1e-6          # m
T2_EQUIV_REL_TOL = 0.05     # fraction of the K-vs-C difference
T3_MOVE_TOL = 1e-3          # m
T3_HIST_TOL = 1e-5
T5_IGNORE_TOL = 1e-4        # m

ACTIVE = int(newton.ParticleFlags.ACTIVE)


# --------------------------------------------------------------------------------------------------
# Warp kernels
# --------------------------------------------------------------------------------------------------
@wp.kernel
def _set_active_kernel(flags: wp.array[wp.int32], indices: wp.array[wp.int32], value: wp.int32):
    """Set or clear the ACTIVE bit for the listed particle indices."""
    i = wp.tid()
    idx = indices[i]
    if value == 1:
        flags[idx] = flags[idx] | int(newton.ParticleFlags.ACTIVE)
    else:
        flags[idx] = flags[idx] & ~int(newton.ParticleFlags.ACTIVE)


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
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    """Overwrite position and zero velocity for the listed indices."""
    i = wp.tid()
    idx = indices[i]
    q[idx] = new_pos[i]
    qd[idx] = wp.vec3(0.0)


@wp.kernel
def _reduce_wrench_kernel(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    center: wp.vec3,
    dt: float,
    out: wp.array[wp.spatial_vector],
):
    """Sum per-node collider impulses into one net force+torque about ``center``.

    Copied from example_mpm_twoway_coupling.py's compute_body_forces (single-collider variant):
    F = sum(impulse)/dt,  tau = sum(r x impulse)/dt.
    """
    i = wp.tid()
    f = impulses[i] / dt
    r = positions[i] - center
    wp.atomic_add(out, 0, wp.spatial_vector(f, wp.cross(r, f)))


# --------------------------------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------------------------------
def make_particles(lo, hi, voxel_size, ppc, density, slab_z, exclude_slab):
    """Deterministic lattice ordered z-major/y/x so the top slab is a contiguous tail block.

    Returns (positions, slab_indices, cell) where slab_indices are indices into the FULL ordering.
    With ``exclude_slab=True`` the slab is omitted but the survivor order is unchanged.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    res = np.ceil(ppc * (hi - lo) / voxel_size).astype(int)
    cell = (hi - lo) / res
    pos = []
    slab = []
    # res+1 grid points per axis (same convention as ModelBuilder.add_particle_grid), so the
    # full scene is 61**3 = 226,981 particles, matching example_mpm_granular defaults.
    for k in range(res[2] + 1):
        for j in range(res[1] + 1):
            for i in range(res[0] + 1):
                p = lo + np.array([i, j, k], dtype=float) * cell
                is_slab = slab_z is not None and p[2] >= slab_z
                if exclude_slab and is_slab:
                    continue
                pos.append(p)
                if is_slab:
                    slab.append(len(pos) - 1)
    return np.asarray(pos, dtype=np.float32), np.asarray(slab, dtype=np.int64), cell


class Experiment:
    """Owns one model + solver + two states and provides step/kill/park/wrench helpers."""

    def __init__(self, args):
        self.args = args
        self.device = wp.get_device()  # cuda:0 by default

        # ---- geometry / particle set -----------------------------------------------------------------
        if args.cube:
            # Small cube-collider scene: particles fall onto a single static box (no ground plane),
            # so every collider impulse belongs to the box and the net wrench is unambiguous.
            lo, hi = (-0.35, -0.35, 0.30), (0.35, 0.35, 0.90)
            slab_z = 0.75
            density = 1000.0
        elif args.full:
            # Full granular scene (same numbers as example_mpm_granular defaults).
            lo, hi = (-1.0, -1.0, 1.5), (1.0, 1.0, 3.5)
            slab_z = 2.5
            density = 1000.0
        else:
            lo, hi = args.lo, args.hi
            slab_z = args.slab_z
            density = args.density

        exclude = args.test == "t2" and args.t2_case == "R"
        positions, slab_idx, cell = make_particles(lo, hi, args.voxel_size, args.ppc, density, slab_z, exclude)
        self.positions = positions
        self.cell = cell
        # For R the slab was excluded, so no slab indices exist; survivors are all particles.
        self.slab_full_idx = None if exclude else slab_idx

        mass = density * float(np.prod(cell)) if density > 0.0 else 0.0
        radius = 0.5 * float(np.max(cell))
        n = positions.shape[0]
        self.n_particles = n

        # ---- build model -----------------------------------------------------------------------------
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)  # must precede particles
        vel = np.zeros((n, 3), dtype=np.float32)
        builder.add_particles(
            pos=positions.tolist(),
            vel=vel.tolist(),
            mass=[mass] * n,
            radius=[radius] * n,
            flags=[ACTIVE] * n,
        )

        if args.cube:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
                hx=0.35,
                hy=0.35,
                hz=0.15,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
            )
            self.cube_center = wp.vec3(0.0, 0.0, 0.15)
        else:
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.set_gravity(wp.vec3(0.0, 0.0, -10.0))

        # ---- solver ----------------------------------------------------------------------------------
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = args.voxel_size
        cfg.grid_type = args.grid_type
        cfg.max_iterations = args.max_iterations
        cfg.tolerance = args.tolerance
        self.solver = SolverImplicitMPM(self.model, config=cfg)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_dt = 1.0 / args.fps

        # survivor indices = everything except the killed slab (same order as R's particles)
        if self.slab_full_idx is not None:
            self.surv_idx = np.setdiff1d(np.arange(n), self.slab_full_idx, assume_unique=False)
        else:
            self.surv_idx = np.arange(n)
        self.surv_idx = np.asarray(self.surv_idx, dtype=np.int64)

    # --------------------------------------------------------------------------------------------------
    def step(self, n=1):
        for _ in range(n):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def set_active(self, indices, active):
        """Edit model.particle_flags in place and refresh the solver's copied masks."""
        flags = self.model.particle_flags.numpy()
        if active:
            flags[indices] = flags[indices] | ACTIVE
        else:
            flags[indices] = flags[indices] & ~ACTIVE
        self.model.particle_flags.assign(flags)
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def set_active_no_notify(self, indices, active):
        """Edit model.particle_flags in place WITHOUT notifying the solver (T5 negative control)."""
        flags = self.model.particle_flags.numpy()
        if active:
            flags[indices] = flags[indices] | ACTIVE
        else:
            flags[indices] = flags[indices] & ~ACTIVE
        self.model.particle_flags.assign(flags)

    def park(self, indices, xyz):
        """Teleport particles to a finite far-away point with zero velocity (both states)."""
        for st in (self.state_0, self.state_1):
            q = st.particle_q.numpy()
            q[indices] = np.asarray(xyz, dtype=np.float32)
            st.particle_q.assign(q)
            qd = st.particle_qd.numpy()
            qd[indices] = 0.0
            st.particle_qd.assign(qd)

    def reset_history(self, indices):
        """Reset only the listed particles' MPM history (small kernel), both states."""
        idx = wp.array(np.asarray(indices, dtype=np.int32), dtype=wp.int32, device=self.device)
        for st in (self.state_0, self.state_1):
            wp.launch(
                _reset_history_kernel,
                dim=int(idx.shape[0]),
                inputs=[
                    idx,
                    self.model.mpm.particle_Jp,
                    st.mpm.particle_elastic_strain,
                    st.mpm.particle_transform,
                    st.mpm.particle_qd_grad,
                    st.mpm.particle_stress,
                    st.mpm.particle_Jp,
                ],
                device=self.device,
            )

    def set_positions(self, indices, new_pos):
        idx = wp.array(np.asarray(indices, dtype=np.int32), dtype=wp.int32, device=self.device)
        npos = wp.array(np.asarray(new_pos, dtype=np.float32), dtype=wp.vec3, device=self.device)
        for st in (self.state_0, self.state_1):
            wp.launch(
                _set_pos_vel_kernel,
                dim=int(idx.shape[0]),
                inputs=[idx, npos, st.particle_q, st.particle_qd],
                device=self.device,
            )

    def wrench(self, state, center, dt):
        """Return the net [fx,fy,fz,tx,ty,tz] on the (single) collider."""
        impulses, positions, _cids = self.solver.collect_collider_impulses(state)
        out = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)
        wp.launch(
            _reduce_wrench_kernel,
            dim=int(impulses.shape[0]),
            inputs=[impulses, positions, center, float(dt)],
            outputs=[out],
            device=self.device,
        )
        wp.synchronize()
        return out.numpy()[0].astype(float)

    def active_cell_count(self):
        """Best-effort active/grid cell count from the solver scratchpad (may be unavailable)."""
        try:
            grid = self.solver._scratchpad.grid
            if hasattr(grid, "get_voxel_count"):
                return int(grid.get_voxel_count())
            if hasattr(grid, "cell_count"):
                return int(grid.cell_count())
        except Exception:
            pass
        return -1


# --------------------------------------------------------------------------------------------------
# Individual tests
# --------------------------------------------------------------------------------------------------
def run_t1(args, exp):
    kill = exp.slab_full_idx
    n = exp.n_particles
    q0 = exp.state_0.particle_q.numpy().copy()
    for _ in range(args.kill_frame):
        exp.step()
    exp.set_active(kill, False)
    if args.mode == "park":
        exp.park(kill, args.park_xyz)
    q_kill = exp.state_0.particle_q.numpy().copy()
    per_particle_max = np.zeros(len(kill), dtype=np.float64)
    for _ in range(args.frames - args.kill_frame):
        exp.step()
        d = np.linalg.norm(exp.state_0.particle_q.numpy()[kill] - q_kill[kill], axis=1)
        per_particle_max = np.maximum(per_particle_max, d)
    return {
        "killed_idx": kill,
        "n_particles": n,
        "n_killed": len(kill),
        "per_particle_max_disp": per_particle_max,
        "max_disp": float(per_particle_max.max()) if len(kill) else 0.0,
        "mean_disp": float(per_particle_max.mean()) if len(kill) else 0.0,
    }


def run_t2(args, exp):
    case = args.t2_case
    kill = exp.slab_full_idx
    if case == "K":
        exp.set_active(kill, False)
    # C and R do nothing special; R was already built without the slab.
    force_hist = np.zeros((args.frames, 6), dtype=np.float64)
    for f in range(args.frames):
        exp.step()
        if args.cube:
            force_hist[f] = exp.wrench(exp.state_0, exp.cube_center, exp.sim_dt)
    q = exp.state_0.particle_q.numpy().copy()
    v = exp.state_0.particle_qd.numpy().copy()
    if case in ("K", "C"):
        q_out, v_out = q[exp.surv_idx], v[exp.surv_idx]
    else:  # R
        q_out, v_out = q, v
    return {
        "case": case,
        "n_particles": exp.n_particles,
        "n_killed": 0 if kill is None else len(kill),
        "surv_idx": exp.surv_idx,
        "q": q_out,
        "v": v_out,
        "force_hist": force_hist,
    }


def run_t3(args, exp):
    kill = exp.slab_full_idx
    # settle, then kill in place
    for _ in range(args.kill_frame):
        exp.step()
    exp.set_active(kill, False)
    for _ in range(args.revive_frame - args.kill_frame):
        exp.step()
    # revive: reactivate, reset ONLY revived history, re-spawn above, zero velocity
    exp.set_active(kill, True)
    exp.reset_history(kill)
    # capture history right after reset
    ident = np.eye(3, dtype=np.float32)
    es = exp.state_0.mpm.particle_elastic_strain.numpy()[kill]
    tr = exp.state_0.mpm.particle_transform.numpy()[kill]
    jp = exp.state_0.mpm.particle_Jp.numpy()[kill]
    jp0 = exp.model.mpm.particle_Jp.numpy()[kill]
    st = exp.state_0.mpm.particle_stress.numpy()[kill]
    qg = exp.state_0.mpm.particle_qd_grad.numpy()[kill]
    hist_err = max(
        float(np.abs(es - ident).max()),
        float(np.abs(tr - ident).max()),
        float(np.abs(jp - jp0).max()),
        float(np.abs(st).max()),
        float(np.abs(qg).max()),
    )
    new_pos = exp.state_0.particle_q.numpy()[kill].copy()
    new_pos[:, 2] += 0.8  # lift the slab back above the pile
    exp.set_positions(kill, new_pos)
    q_revive = exp.state_0.particle_q.numpy()[kill].copy()
    per_particle_max = np.zeros(len(kill), dtype=np.float64)
    for _ in range(args.frames - args.revive_frame):
        exp.step()
        d = np.linalg.norm(exp.state_0.particle_q.numpy()[kill] - q_revive, axis=1)
        per_particle_max = np.maximum(per_particle_max, d)
    return {
        "killed_idx": kill,
        "n_particles": exp.n_particles,
        "n_killed": len(kill),
        "history_err": hist_err,
        "revived_max_disp": float(per_particle_max.max()) if len(kill) else 0.0,
        "revived_mean_disp": float(per_particle_max.mean()) if len(kill) else 0.0,
    }


def run_t4(args, exp):
    kill = exp.slab_full_idx
    step_times = []
    err = ""
    if args.mode == "inplace":
        exp.set_active(kill, False)
    elif args.mode == "park":
        exp.set_active(kill, False)
        exp.park(kill, args.park_xyz)
    try:
        for _ in range(args.frames):
            t0 = time.perf_counter()
            exp.step()
            wp.synchronize()
            step_times.append(time.perf_counter() - t0)
        if hasattr(exp.solver, "check_sparse_grid_rebuild_status"):
            exp.solver.check_sparse_grid_rebuild_status()
    except Exception as exc:  # keep the measured numbers, report the error
        err = f"{type(exc).__name__}: {exc}"
    return {
        "n_particles": exp.n_particles,
        "n_killed": 0 if kill is None else len(kill),
        "step_times": np.asarray(step_times, dtype=np.float64),
        "active_cells": exp.active_cell_count(),
        "error": err,
    }


def run_t5(args, exp):
    # warm up a couple of steps
    exp.step(3)
    # 1) median cost of one notify_model_changed call
    notify_times = []
    for _ in range(20):
        wp.synchronize()
        t0 = time.perf_counter()
        exp.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        wp.synchronize()
        notify_times.append(time.perf_counter() - t0)
    notify_times = np.asarray(notify_times, dtype=np.float64)

    def timed_steps(n, notify_each):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            exp.step()
            if notify_each:
                exp.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            wp.synchronize()
            ts.append(time.perf_counter() - t0)
        return np.asarray(ts, dtype=np.float64)

    step_normal = timed_steps(10, False)
    step_notify = timed_steps(10, True)

    # 3) edit flags in place WITHOUT notify -> expected to be ignored
    kill = exp.slab_full_idx
    exp.set_active_no_notify(kill, False)
    q_before = exp.state_0.particle_q.numpy()[kill].copy()
    exp.step()
    q_after = exp.state_0.particle_q.numpy()[kill].copy()
    no_notify_disp = np.linalg.norm(q_after - q_before, axis=1)

    return {
        "n_particles": exp.n_particles,
        "n_killed": len(kill),
        "notify_times": notify_times,
        "notify_median_s": float(np.median(notify_times)),
        "step_normal": step_normal,
        "step_notify": step_notify,
        "step_normal_mean": float(step_normal.mean()),
        "step_notify_mean": float(step_notify.mean()),
        "no_notify_mean_disp": float(no_notify_disp.mean()),
        "no_notify_max_disp": float(no_notify_disp.max()),
    }


# --------------------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, choices=["t1", "t2", "t3", "t4", "t5"])
    p.add_argument("--mode", default="inplace", choices=["none", "inplace", "park"],
                   help="kill mode: none=control, inplace=freeze where they are, park=teleport far away")
    p.add_argument("--grid-type", default="dense", choices=["sparse", "dense"],
                   help="MPM grid type (fixed is intentionally excluded: it cannot follow particles)")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--voxel-size", type=float, default=0.1)
    p.add_argument("--ppc", type=int, default=2, help="particles per cell along each axis")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--tolerance", type=float, default=1.0e-4)
    p.add_argument("--kill-frame", type=int, default=20)
    p.add_argument("--revive-frame", type=int, default=40)
    p.add_argument("--park-xyz", type=float, nargs=3, default=[100.0, 100.0, 100.0])
    p.add_argument("--slab-z", type=float, default=0.75, help="z threshold selecting the killed top slab")
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--lo", type=float, nargs=3, default=[-0.35, -0.35, 0.30])
    p.add_argument("--hi", type=float, nargs=3, default=[0.35, 0.35, 0.90])
    p.add_argument("--cube", action="store_true", help="cube-collider scene (forces); small default region")
    p.add_argument("--full", action="store_true", help="full 226,981-particle granular scene")
    p.add_argument("--t2-case", default="K", choices=["K", "R", "C"])
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--log-dir", default=None)
    return p


def main():
    args = build_parser().parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    # --cube or --full select built-in regions; otherwise use --lo/--hi/--slab-z.
    if args.full:
        pass  # Experiment handles it
    elif not args.cube:
        pass

    exp = Experiment(args)
    funcs = {"t1": run_t1, "t2": run_t2, "t3": run_t3, "t4": run_t4, "t5": run_t5}
    data = funcs[args.test](args, exp)

    data["test"] = args.test
    data["mode"] = args.mode
    data["grid_type"] = args.grid_type
    data["frames"] = args.frames
    data["voxel_size"] = args.voxel_size
    data["tolerances"] = np.array([T1_DEAD_TOL, T2_EQUIV_REL_TOL, T3_MOVE_TOL, T3_HIST_TOL, T5_IGNORE_TOL])
    out = os.path.join(log_dir, args.out_prefix + ".npz")
    np.savez_compressed(out, **data)
    print(f"[RESULT] wrote {out}")
    print(f"[RESULT] test={args.test} grid={args.grid_type} mode={args.mode} "
          f"n_particles={exp.n_particles} n_killed={data.get('n_killed', 0)}")
    for k in ("max_disp", "mean_disp", "history_err", "revived_max_disp",
              "notify_median_s", "step_normal_mean", "step_notify_mean",
              "no_notify_mean_disp", "active_cells", "error"):
        if k in data:
            print(f"[RESULT] {k}={data[k]}")


if __name__ == "__main__":
    wp.init()
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(f"RUN_FAILED {traceback.format_exc().strip().splitlines()[-1]}")
        sys.exit(1)
    print("RUN_DONE")
