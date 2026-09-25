# mpm-soil-excavation

Local 3D material point method (MPM) soil simulation around a suction excavator nozzle, built on
Newton 1.6 (`SolverImplicitMPM`) and NVIDIA Warp. The plan is to reuse a fixed pool of particles and
to remove particles near the nozzle inlet with a simplified suction model; neither is implemented yet
(see Status). The simulator is intended to generate data for a later learned interaction predictor.
This is a Master's project and is work in progress.

## Status

Current state: **Phase 3, Scene A** — a dry-sand column collapsing on a static ground box, with the
ground reaction force read out from the MPM collider impulses — plus the earlier particle
kill/revive tests. Also implemented: the CPU-side acceptance checks, the momentum-balance check of the
force readout, and the repose study over ground friction, soil friction angle and column height.

Not implemented yet: the nozzle/nozzle-bed scene (Scene B), suction and particle removal at the inlet,
buried objects and multiple soils, and the learned interaction predictor.

## Setup

Create a conda environment inside the project folder and install Newton with its examples:

```bash
conda create --prefix ./envs/mpm python=3.12 -y
conda run --no-capture-output --prefix ./envs/mpm pip install "newton[examples]"
```

Tested versions (from `conda run --prefix ./envs/mpm pip list`):

| Package | Version |
|---|---|
| newton | 1.6.0 |
| warp-lang | 1.17.0 |
| numpy | 2.5.3 |

Tested on Ubuntu 26.04 with an NVIDIA RTX 4060 Ti (8 GB VRAM) and driver 580.x. All runs are headless
(no viewer window is opened).

## Scripts

All scripts live in `scripts/`.

| File | Purpose |
|---|---|
| `mpm_common.py` | Parameter dataclasses (soil, scene, solver), lattice generation, the GPU collider-wrench reduction kernel and small numpy helpers. |
| `mpm_scene.py` | Builds one Newton model plus `SolverImplicitMPM` for Scene A (a dry-sand column on a thick static ground box). |
| `mpm_run.py` | Runs a scene, records per-step metrics and position/velocity snapshots, and writes `logs/<out-prefix>.npz`. |
| `mpm_accept.py` | CPU-only acceptance tests for Scene A; the `--repose` option runs the repose study over ground friction, soil friction angle and column height. |
| `mpm_momentum_check.py` | CPU-only momentum-balance check of the ground force readout, using the `.npz` files from `mpm_run.py`. |
| `mpm_kill_test.py` | Runtime particle "kill"/"revive" experiments for the implicit MPM solver (a copy-and-adapt of Newton's granular and two-way-coupling examples). |
| `mpm_kill_analyze.py` | CPU-only analysis of the kill/revive `.npz` results produced by `mpm_kill_test.py`. |

Commands (run from the project root):

```bash
# Scene A (column collapse), then the acceptance tests
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_run.py \
    --scene A --diameter 0.15 --duration 3.0 --seed 0 --out-prefix sceneA_D015
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_accept.py

# Force-readout momentum check (CPU only)
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_momentum_check.py

# Repose study (CPU only; reads the 2 s D=0.15 runs named g<ground friction>_p<phi>_h<height factor>,
# e.g. g20_p34_h25, plus the regr_D015 regression run). Build one with, for example:
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_run.py \
    --scene A --diameter 0.15 --duration 2.0 --ground-friction 2.0 --soil-phi-deg 34 \
    --height-factor 2.5 --seed 0 --out-prefix g20_p34_h25
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_accept.py --repose

# Particle kill/revive experiments, then the analysis
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_kill_test.py \
    --test t1 --mode inplace --grid-type dense --frames 60 --kill-frame 20 --cube --out-prefix t1_inplace
conda run --no-capture-output --prefix ./envs/mpm python scripts/mpm_kill_analyze.py
```

Outputs are written to `logs/`, which is not tracked. The scripts refer to the project directory as
`PROJECT/` in their docstrings and do not contain hard-coded absolute paths; replace `PROJECT/` (and
`./envs/mpm` if the environment lives elsewhere) with the local path when copying the examples. The
docstrings that use the `PROJECT/` placeholder are in `mpm_run.py`, `mpm_accept.py`, `mpm_common.py`,
`mpm_kill_test.py` and `mpm_kill_analyze.py`.

## Notes

- Only one GPU job is run at a time (the GPU is shared with the desktop).
- The first launch compiles the Warp kernels (about 30 s); later launches are faster.
- AI reference notes, the work log and result files are intentionally not part of the repository.
