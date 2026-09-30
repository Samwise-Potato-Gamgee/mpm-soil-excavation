#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_common.py -- shared parameters, lattice generation, GPU kernels and small helpers for the
local MPM soil simulation (Phase 3, Scene A).

This module is deliberately GPU-light: it only defines dataclasses and *definitions* of Warp kernels
plus pure-numpy helpers.  Importing it does not run a simulation.  `mpm_run.py` builds the scene and
steps it; `mpm_scene.py` owns the Newton model/solver; `mpm_accept.py` is CPU-only.

Units are SI (m, kg, s, Pa) throughout.  z is up; gravity is (0, 0, -9.81) m/s^2.

Newton `mpm:*` attribute semantics (read from
`PROJECT/envs/mpm/lib/python3.12/site-packages/newton/_src/solvers/implicit_mpm/`):
  - `friction`      : Coulomb/Drucker-Prager friction coefficient (dimensionless); mu = tan(phi).
  - `young_modulus` : elastic Young's modulus E [Pa].
  - `poisson_ratio` : Poisson's ratio nu [-].
  - `damping`       : elastic damping relaxation time [s].
  - `yield_pressure`: pressure scale of the Drucker-Prager yield surface [Pa]
                      (used as `yield_pressure * hardening_law` in `get_yield_parameters`).
  - `yield_stress`  : deviatoric yield stress / cohesion [Pa].
  - `tensile_yield_ratio`: tensile-yield ratio relative to the yield pressure [-].
  - `hardening`, `hardening_rate`, `softening_rate`, `dilatancy`, `viscosity` [- / Pa*s].
See `implicit_mpm_solver_kernels.py:get_yield_parameters` and the multi-material example.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

# Particle is active (Newton flag used for the fixed pool; Scene A keeps every particle active).
ACTIVE = int(newton.ParticleFlags.ACTIVE)
# Gravity is set on the model and re-read from it for every expected value, never hard-coded twice.
GRAVITY = (0.0, 0.0, -9.81)


# --------------------------------------------------------------------------------------------------
# GPU kernels
# --------------------------------------------------------------------------------------------------
@wp.kernel
def reduce_collider_wrench_kernel(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    collider_ids: wp.array[wp.int32],
    collider_body_index: wp.array[wp.int32],
    select_body: wp.int32,
    center: wp.vec3,
    dt: float,
    out: wp.array[wp.spatial_vector],
):
    """Sum per-node collider impulses into one net force+torque for a selected collider owner.

    Copied from `example_mpm_twoway_coupling.py:compute_body_forces` (see the Step-3 findings, B7):
    F = sum(impulse)/dt and tau = sum(r x impulse)/dt about `center`.  Only nodes whose collider maps
    to `select_body` are included; `select_body = -1` selects static colliders (the ground in Scene A).
    """
    i = wp.tid()
    cid = collider_ids[i]
    # Guard against a collider id outside the mapping array (defensive; should not happen).
    if cid < 0 or cid >= collider_body_index.shape[0]:
        return
    if collider_body_index[cid] != select_body:
        return
    f = impulses[i] / dt
    r = positions[i] - center
    wp.atomic_add(out, 0, wp.spatial_vector(f, wp.cross(r, f)))


# --------------------------------------------------------------------------------------------------
# Parameter dataclasses
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class SoilParams:
    """Per-particle soil material, mapped onto Newton's `mpm:*` custom attributes.

    Every value is a *starting* literature value for dry sand (see design-note section 4) and must be
    calibrated; none of it is measured.
    """

    name: str = "dry_sand"
    density: float = 1700.0            # bulk density [kg/m^3] (PSU "Some Useful Numbers")
    friction_angle_deg: float = 34.0   # internal friction angle phi [deg]
    young_modulus: float = 3.0e7       # E [Pa]
    poisson_ratio: float = 0.3         # nu [-]
    yield_pressure: float = 5.0e6      # Drucker-Prager pressure scale [Pa]
    yield_stress: float = 0.0          # deviatoric yield stress / cohesion [Pa]
    tensile_yield_ratio: float = 0.0   # tensile yield ratio [-]
    hardening: float = 0.0             # isotropic hardening factor [-]
    dilatancy: float = 0.0             # dilatancy factor [-]
    viscosity: float = 0.0             # viscosity [Pa*s]
    damping: float = 0.0               # elastic damping relaxation time [s]

    @property
    def friction(self) -> float:
        """Coulomb friction coefficient mu = tan(phi) (design-note section 4)."""
        return math.tan(math.radians(self.friction_angle_deg))

    def model_attributes(self) -> dict[str, float]:
        """Map to Newton `mpm:*` attribute names (order/names from `register_custom_attributes`)."""
        return {
            "friction": self.friction,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "yield_pressure": self.yield_pressure,
            "yield_stress": self.yield_stress,
            "tensile_yield_ratio": self.tensile_yield_ratio,
            "hardening": self.hardening,
            "dilatancy": self.dilatancy,
            "viscosity": self.viscosity,
            "damping": self.damping,
        }


@dataclasses.dataclass
class SolverParams:
    """SolverImplicitMPM options.  Defaults equal Newton's Config defaults and match the Step-4
    working setup in `scripts/mpm_kill_test.py` (which set only voxel_size/grid_type/max_iterations/
    tolerance; the rest are the library defaults).  They are set explicitly here for transparency."""

    grid_type: str = "dense"
    transfer_scheme: str = "apic"
    integration_scheme: str = "pic"
    solver: str = "auto"
    warmstart_mode: str = "auto"
    strain_basis: str = "P0"
    collider_basis: str = "S2"
    velocity_basis: str = "Q1"
    grid_padding: int = 0
    max_iterations: int = 50
    tolerance: float = 1.0e-4
    critical_fraction: float = 0.0
    air_drag: float = 1.0

    def make_config(self, voxel_size: float) -> SolverImplicitMPM.Config:
        """Build a `SolverImplicitMPM.Config` from these parameters."""
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = float(voxel_size)
        cfg.grid_type = self.grid_type
        cfg.transfer_scheme = self.transfer_scheme
        cfg.integration_scheme = self.integration_scheme
        cfg.solver = self.solver
        cfg.warmstart_mode = self.warmstart_mode
        cfg.strain_basis = self.strain_basis
        cfg.collider_basis = self.collider_basis
        cfg.velocity_basis = self.velocity_basis
        cfg.grid_padding = self.grid_padding
        cfg.max_iterations = self.max_iterations
        cfg.tolerance = self.tolerance
        cfg.critical_fraction = self.critical_fraction
        cfg.air_drag = self.air_drag
        return cfg


@dataclasses.dataclass
class SceneAConfig:
    """Geometry and timing for Scene A (column collapse on a ground box).

    All lengths scale with the nozzle diameter D, so the study over D = 0.10/0.15/0.20 m keeps the same
    resolution (D/voxel) and the same lattice counts (design-note sections 2.1 and 10).
    """

    diameter: float = 0.15                 # D [m]
    voxel_over_D: float = 5.0              # D / voxel ; >= 5 rule of thumb (UNVERIFIED)
    voxel_override: float | None = None    # Step 6c: explicit voxel size [m]; None = D/voxel_over_D
    ground_half_override: float | None = None  # Step 6c: explicit ground half-extent [m]; None = 5D
    ppc: int = 2                           # particles per cell PER AXIS
    block_base_factor: float = 2.0         # soil block base = 2D x 2D
    block_height_factor: float = 2.5       # soil block height = 2.5D
    start_gap_spacing: float = 0.5         # bottom particle centre = 0.5 * spacing above z=0
    ground_half_factor: float = 5.0        # ground half-extent = 5D in x and y
    ground_half_z: float = 0.5             # ground box half-height [m]
    ground_friction: float = 0.5           # labelled assumption (design-note section 10)
    duration: float = 3.0                  # simulated seconds
    duration_restart: float = 6.0          # used only if the soil is not at rest by t=2.5 s
    fps_reference: float = 60.0            # dt rule: dt = (voxel/0.1 m) * (1/60 s)
    reference_voxel: float = 0.1           # reference voxel for the dt rule [m]
    snapshot_times: tuple = (0.0, 0.25, 0.5, 1.0, 2.0, 2.5, 3.0)

    # ---- derived geometry / timing --------------------------------------------------------------
    @property
    def voxel(self) -> float:
        """Grid voxel size [m].  An explicit ``voxel_override`` wins verbatim (no D/voxel routing)."""
        if self.voxel_override is not None:
            return float(self.voxel_override)
        return self.diameter / self.voxel_over_D

    @property
    def spacing(self) -> float:
        """Particle spacing [m] = voxel / ppc (particles per cell per axis)."""
        return self.voxel / self.ppc

    @property
    def dt(self) -> float:
        """Timestep [s] from the labelled rule dt = (voxel/0.1 m) * (1/60 s) (UNVERIFIED)."""
        return (self.voxel / self.reference_voxel) / self.fps_reference

    @property
    def n_intervals(self) -> tuple[int, int, int]:
        """Lattice intervals (nx, ny, nz); counts are intervals+1 per axis."""
        nx = int(round(self.block_base_factor * self.diameter / self.spacing))
        nz = int(round(self.block_height_factor * self.diameter / self.spacing))
        return nx, nx, nz

    @property
    def particle_count(self) -> int:
        """Expected number of particles (21*21*26 = 11,466 for the default Scene A)."""
        nx, ny, nz = self.n_intervals
        return (nx + 1) * (ny + 1) * (nz + 1)

    @property
    def particle_mass(self) -> float:
        """Mass per particle [kg] = density * spacing^3."""
        return 1700.0 * self.spacing ** 3

    @property
    def particle_radius(self) -> float:
        """Particle radius [m] = 0.5 * spacing, so the MPM particle volume 8 r^3 = spacing^3."""
        return 0.5 * self.spacing

    @property
    def ground_half_xy(self) -> float:
        """Ground box half-extent in x and y [m] = 5D, or ``ground_half_override`` verbatim."""
        if self.ground_half_override is not None:
            return float(self.ground_half_override)
        return self.ground_half_factor * self.diameter


# --------------------------------------------------------------------------------------------------
# Lattice generation and pure-numpy metrics
# --------------------------------------------------------------------------------------------------
def block_lattice(cfg: SceneAConfig) -> np.ndarray:
    """Generate the Scene A soil block lattice, ordered z-major then y then x.

    x and y span the full base `2D` symmetrically about 0; z starts at `0.5 * spacing` above the
    ground (so no particle centre is inside the ground) and spans `2.5D`.  Returns an (N, 3) float32
    array.  With the defaults this is 21 x 21 x 26 = 11,466 particles for every D.
    """
    s = cfg.spacing
    nx, ny, nz = cfg.n_intervals
    half = cfg.block_base_factor * cfg.diameter / 2.0  # base 2D -> half extent D
    z0 = cfg.start_gap_spacing * s
    pos = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float32)
    idx = 0
    for k in range(nz + 1):
        z = z0 + k * s
        for j in range(ny + 1):
            y = -half + j * s
            for i in range(nx + 1):
                x = -half + i * s
                pos[idx, 0] = x
                pos[idx, 1] = y
                pos[idx, 2] = z
                idx += 1
    return pos


def kinetic_energy(mass: np.ndarray, vel: np.ndarray) -> float:
    """Total kinetic energy [J] = 0.5 * sum_i m_i |v_i|^2."""
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def center_of_mass(mass: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mass-weighted centre of mass position [m], shape (3,)."""
    return np.sum(mass[:, None] * q, axis=0) / np.sum(mass)


def center_of_mass_velocity(mass: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mass-weighted centre-of-mass velocity [m/s], shape (3,)."""
    return np.sum(mass[:, None] * v, axis=0) / np.sum(mass)
