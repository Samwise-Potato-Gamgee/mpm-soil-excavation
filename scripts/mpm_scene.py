#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_scene.py -- builds ONE Newton model + `SolverImplicitMPM` for the local MPM soil simulation and
exposes a small scene API used by `mpm_run.py`.

Phase 3, part 1 implements **Scene A**: a dry-sand column on a thick static ground box, no nozzle,
no suction, no particle killing.  The class is intentionally generic (`MPMScene`) so that a nozzle and
Scene B can be added in a later step without changing the runner or the recording code.

What is built here:
  - particles from `mpm_common.block_lattice` (`add_particles` with mass = density * spacing^3,
    radius = 0.5 * spacing);
  - a thick static ground box (body = -1, top face at z = 0, friction 0.5);
  - the dry-sand `mpm:*` attributes on the finalized model, *before* the solver is constructed so the
    solver's cached extrema read the intended values;
  - `SolverImplicitMPM` with the options copied from the Step-4 working setup (`SolverParams`).

Ground reaction: static shapes are auto-discovered as a single MPM collider
(`implicit_mpm_model.py:531-546`), so `solver.collect_collider_impulses(state)` returns per-node
impulses for the ground; the net force is `sum impulse / dt` over nodes whose
`collider_body_index == -1` (design-note section 10; findings B7).  No separate ground body is needed.
"""

from __future__ import annotations

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
    block_lattice,
    center_of_mass,
    center_of_mass_velocity,
    kinetic_energy,
    reduce_collider_wrench_kernel,
)


class MPMScene:
    """Owns one model, one solver and two states, and provides step/readout helpers."""

    def __init__(self, model: newton.Model, solver: SolverImplicitMPM, solver_params: SolverParams):
        self.model = model
        self.solver = solver
        self.solver_params = solver_params
        self.device = model.device
        self.state_0 = model.state()  # ping-pong states; state_0 is the current state
        self.state_1 = model.state()
        # Cached host copies of the per-particle mass (fixed in Scene A) for the numpy metrics.
        self.mass = model.particle_mass.numpy().astype(np.float64)
        self.total_mass = float(self.mass.sum())
        # Per-node collider -> body mapping (static ground maps to -1).
        self.collider_body_index = solver.collider_body_index
        # Reusable output buffer for the wrench reduction (one float6: force + torque).
        self._wrench_out = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)

    # ---- simulation -----------------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """One solver step + collider projection, exactly as in `scripts/mpm_kill_test.py`."""
        self.solver.step(self.state_0, self.state_1, None, None, dt)
        self.solver.project_outside(self.state_1, self.state_1, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    # ---- readout --------------------------------------------------------------------------------
    def collider_wrench(self, select_body: int = -1, center: wp.vec3 | None = None, dt: float = 1.0) -> np.ndarray:
        """Net force+torque [N, N*m] on the collider owned by `select_body` (default -1 = static/ground).

        Returns a length-6 array [fx, fy, fz, tx, ty, tz] in world coordinates.
        """
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state_0)
        if center is None:
            center = wp.vec3(0.0, 0.0, 0.0)
        self._wrench_out.zero_()
        wp.launch(
            reduce_collider_wrench_kernel,
            dim=int(impulses.shape[0]),
            inputs=[
                impulses,
                positions,
                collider_ids,
                self.collider_body_index,
                int(select_body),
                center,
                float(dt),
            ],
            outputs=[self._wrench_out],
            device=self.device,
        )
        wp.synchronize()
        return self._wrench_out.numpy()[0].astype(np.float64)

    def ground_wrench(self, dt: float) -> np.ndarray:
        """Convenience wrapper: net force+torque on the static ground collider."""
        return self.collider_wrench(select_body=-1, center=wp.vec3(0.0, 0.0, 0.0), dt=dt)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of (positions, velocities) as float64 numpy arrays."""
        q = self.state_0.particle_q.numpy().astype(np.float64)
        v = self.state_0.particle_qd.numpy().astype(np.float64)
        return q, v

    def metrics(self) -> dict:
        """Per-step scalar metrics computed on the host from the current state arrays."""
        q = self.state_0.particle_q.numpy().astype(np.float64)
        v = self.state_0.particle_qd.numpy().astype(np.float64)
        speed = np.linalg.norm(v, axis=1)
        return {
            "mass": float(self.mass.sum()),
            "ke": kinetic_energy(self.mass, v),
            "mean_speed": float(speed.mean()),
            "max_speed": float(speed.max()),
            "min_z": float(q[:, 2].min()),
            "com": center_of_mass(self.mass, q),
            "com_vel": center_of_mass_velocity(self.mass, v),
        }


def build_scene_a(cfg: SceneAConfig, soil: SoilParams, solver_params: SolverParams) -> MPMScene:
    """Build Scene A: dry-sand block on a thick static ground box.

    Order matters: soil `mpm:*` attributes are written on the finalized model *before* the solver is
    constructed, so `ImplicitMPMModel.__init__` caches the correct material extrema.
    """
    positions = block_lattice(cfg)
    n = positions.shape[0]
    mass = cfg.particle_mass
    radius = cfg.particle_radius

    builder = newton.ModelBuilder()
    # Custom `mpm:*` attributes must be registered before particles are added.
    SolverImplicitMPM.register_custom_attributes(builder)

    # Soil block: all particles active (Scene A does not kill particles).
    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros((n, 3), dtype=np.float32).tolist(),
        mass=[mass] * n,
        radius=[radius] * n,
        flags=[ACTIVE] * n,
    )

    # Thick static ground box, top face at z = 0 (centre at z = -half_z).
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -cfg.ground_half_z), wp.quat_identity()),
        hx=cfg.ground_half_xy,
        hy=cfg.ground_half_xy,
        hz=cfg.ground_half_z,
        cfg=newton.ModelBuilder.ShapeConfig(mu=cfg.ground_friction, density=0.0),
    )

    model = builder.finalize()
    model.set_gravity(wp.vec3(*GRAVITY))

    # Apply the soil material to every particle (whole array; Scene A has one soil).
    for attr, value in soil.model_attributes().items():
        getattr(model.mpm, attr).fill_(value)

    # Construct the solver *after* the material attributes are set.
    solver = SolverImplicitMPM(model, config=solver_params.make_config(cfg.voxel))

    return MPMScene(model, solver, solver_params)
