#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_momentum_check.py -- CPU-only momentum-balance check of the Scene A ground force readout.

Reads the `.npz` files produced by `mpm_run.py` and tests whether the ground reaction recorded with
`solver.collect_collider_impulses` is trustworthy DURING motion, not only at rest.

Physics / sign convention (checked against the Newton source and the data):
  - At rest the soil pushes the ground DOWN, so the recorded z-force is negative: `Fz ~ M*g_z` with
    `g_z = -9.81`.  Fz is therefore the force ON THE GROUND, and the force on the SOIL is `-Fz`.
    Newton's law for the soil, `M a = M g - Fz`, gives the corrected momentum residual
        r = Fz - M (g - a)          [corrected sign]
    whereas the Step-6 diagnostic used the wrong sign:
        r_old = Fz - M (g + a).     Note r_old = r + 2 M a, which is why the old diagnostic blew up
                                    during the collapse (large a) but was ~0 at rest.

Source facts (read-only):
  - `collect_collider_impulses` returns `-cell_volume * state.impulse_field.dof_values`
    (`solver_implicit_mpm.py:2192`); the docstring calls them "Impulse values in world units", so they
    are per-step IMPULSES (N*s); dividing by `dt` gives force.
  - `free_velocity` adds `air_drag` to the nodal mass (`1/(pmass + drag)`,
    `implicit_mpm_solver_kernels.py:200-201`): an unrecorded numerical drag (`Config.air_drag`=1.0).
  - `project_outside` is a separate post-step position/velocity correction (`solver_implicit_mpm.py:2205`),
    not part of the recorded impulse field.

The script never imports warp or newton; it uses numpy only.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

DIAMETERS = {0.10: "sceneA_D010", 0.15: "sceneA_D015", 0.20: "sceneA_D020"}
ACCEL_METHODS = ("vel_forward", "vel_backward", "vel_central", "pos_central")
LAGS = (-2, -1, 0, 1, 2)


def load(log_dir: str, prefix: str):
    path = os.path.join(log_dir, prefix + ".npz")
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def first_rest_time(t: np.ndarray, mean_speed: np.ndarray, thresh: float) -> float:
    """First time from which the mean speed stays below `thresh` (returns t[-1] if never)."""
    bad = np.nonzero(~(mean_speed < thresh))[0]
    if len(bad) == 0:
        return float(t[0])
    if bad[-1] + 1 >= len(t):
        return float(t[-1])
    return float(t[bad[-1] + 1])


def accel_series(com: np.ndarray, com_vel: np.ndarray, dt: float, method: str) -> np.ndarray:
    """Return the z-acceleration series with NaN where a finite difference is not defined."""
    n = com.shape[0]
    a = np.full(n, np.nan)
    if method == "vel_forward":
        a[:-1] = (com_vel[1:, 2] - com_vel[:-1, 2]) / dt
    elif method == "vel_backward":
        a[1:] = (com_vel[1:, 2] - com_vel[:-1, 2]) / dt
    elif method == "vel_central":
        a[1:-1] = (com_vel[2:, 2] - com_vel[:-2, 2]) / (2.0 * dt)
    elif method == "pos_central":
        a[1:-1] = (com[2:, 2] - 2.0 * com[1:-1, 2] + com[:-2, 2]) / (dt * dt)
    else:
        raise ValueError(method)
    return a


def stats(residual: np.ndarray, weight: float) -> tuple[float, float, float]:
    """(RMS, max|.|, RMS/(M|g|)); NaN for an empty/undefined residual."""
    r = residual[np.isfinite(residual)]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan")
    rms = float(np.sqrt(np.mean(r * r)))
    return rms, float(np.abs(r).max()), rms / weight


def analyse(log_dir: str, D: float, prefix: str) -> dict:
    d = load(log_dir, prefix)
    if d is None:
        print(f"[momentum] {prefix}.npz missing")
        return {}
    t = d["t"].astype(np.float64)
    dt = float(d["dt"])
    M = float(np.mean(d["mass"]))
    g_z = float(d["gravity"][2])
    weight = M * abs(g_z)
    Fz = d["ground_force"][:, 2].astype(np.float64)
    com = d["com"].astype(np.float64)
    com_vel = d["com_vel"].astype(np.float64)
    rest_t = first_rest_time(t, d["mean_speed"].astype(np.float64), 1.0e-3)
    n = len(t)

    print("=" * 116)
    print(f"Part A momentum balance -- D={D:.2f} m  (prefix {prefix})")
    print(f"  dt={dt:.6f} s  n_rows={n}  M={M:.4f} kg  g_z={g_z:.5f}  M|g|={weight:.3f} N  rest_t={rest_t:.3f} s")
    print("  corrected r = Fz[k+s] - M*(g_z - a_z[k]) ;  old r = Fz[k+s] - M*(g_z + a_z[k])")
    print("-" * 116)

    best = None
    rows = []
    for method in ACCEL_METHODS:
        a = accel_series(com, com_vel, dt, method)
        for s in LAGS:
            ks = [k for k in range(n) if np.isfinite(a[k]) and 0 <= k + s < n]
            if len(ks) < 5:
                continue
            ks = np.asarray(ks)
            r_corr = Fz[ks + s] - M * (g_z - a[ks])
            r_old = Fz[ks + s] - M * (g_z + a[ks])
            collapse = t[ks] < rest_t
            settled = ~collapse
            cc = stats(np.where(collapse, r_corr, np.nan), weight)
            cs = stats(np.where(settled, r_corr, np.nan), weight)
            oc = stats(np.where(collapse, r_old, np.nan), weight)
            os_ = stats(np.where(settled, r_old, np.nan), weight)
            corr = float(np.corrcoef(Fz[ks + s], M * (g_z - a[ks]))[0, 1]) if np.std(Fz[ks + s]) > 0 else float("nan")
            rows.append({"method": method, "lag": s, "cc": cc, "cs": cs, "oc": oc, "os": os_, "corr": corr})
            score = (cs[2] if np.isfinite(cs[2]) else 1e9, cc[2] if np.isfinite(cc[2]) else 1e9)
            if best is None or score < best["score"]:
                best = {"method": method, "lag": s, "score": score, "corr": corr,
                        "cc": cc, "cs": cs, "oc": oc, "os": os_}

    # Corrected-sign table
    print("CORRECTED sign  r = Fz[k+s] - M*(g_z - a_z[k])")
    print(f"  {'method':<14}{'lag':>4} | {'rms_coll':>10}{'max_coll':>10}{'ratio':>8} | "
          f"{'rms_set':>10}{'max_set':>10}{'ratio':>8} | {'corr':>8}")
    for r in rows:
        print(f"  {r['method']:<14}{r['lag']:>4} | {r['cc'][0]:>10.2f}{r['cc'][1]:>10.2f}{r['cc'][2]:>8.4f} | "
              f"{r['cs'][0]:>10.2f}{r['cs'][1]:>10.2f}{r['cs'][2]:>8.5f} | {r['corr']:>8.4f}")
    print("OLD sign (Step-6)  r = Fz[k+s] - M*(g_z + a_z[k])")
    print(f"  {'method':<14}{'lag':>4} | {'rms_coll':>10}{'max_coll':>10}{'ratio':>8} | "
          f"{'rms_set':>10}{'max_set':>10}{'ratio':>8} |")
    for r in rows:
        print(f"  {r['method']:<14}{r['lag']:>4} | {r['oc'][0]:>10.2f}{r['oc'][1]:>10.2f}{r['oc'][2]:>8.4f} | "
              f"{r['os'][0]:>10.2f}{r['os'][1]:>10.2f}{r['os'][2]:>8.5f} |")
    print("-" * 116)
    b = best
    print(f"  best: corrected sign, method={b['method']}, lag={b['lag']}; "
          f"collapse rms={b['cc'][0]:.3f} N ({100*b['cc'][2]:.4f}% of M|g|), "
          f"max={b['cc'][1]:.3f} N; settled rms={b['cs'][0]:.4f} N, max={b['cs'][1]:.4f} N; corr={b['corr']:.5f}")
    wrong = (b["oc"][0] / b["cc"][0]) if (np.isfinite(b["oc"][0]) and b["cc"][0] > 0) else float("nan")
    print(f"  old-sign collapse rms={b['oc'][0]:.3f} N ({100*b['oc'][2]:.3f}% of M|g|) -> "
          f"{wrong:.0f}x larger than corrected")
    return {"D": D, "prefix": prefix, "best": b, "weight": weight, "rest_t": rest_t}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--diameters", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))

    results = [analyse(log_dir, D, DIAMETERS[D]) for D in args.diameters]
    results = [r for r in results if r]

    print()
    print("=" * 116)
    print("Summary (best combination per D, corrected sign)")
    print(f"{'D (m)':>6} {'method':>14} {'lag':>4} {'coll rms (N)':>13} {'% M|g|':>9} {'coll max (N)':>13} "
          f"{'set rms (N)':>12} {'% M|g|':>9} {'corr':>8}")
    verdicts = []
    for r in results:
        b = r["best"]
        verdict = "TRUSTWORTHY" if (b["cc"][2] < 0.05 and b["cs"][2] < 0.05) else "USABLE-WITH-CORRECTION"
        verdicts.append(verdict)
        print(f"{r['D']:>6.2f} {b['method']:>14} {b['lag']:>4} {b['cc'][0]:>13.3f} {100*b['cc'][2]:>9.4f} "
              f"{b['cc'][1]:>13.3f} {b['cs'][0]:>12.4f} {100*b['cs'][2]:>9.5f} {b['corr']:>8.4f}")
    print("=" * 116)
    print("VERDICT: " + ("TRUSTWORTHY" if all(v == "TRUSTWORTHY" for v in verdicts) else str(verdicts)))
    print("  Lesson: the Step-6 transient of hundreds of N was the WRONG SIGN in the diagnostic "
          "(r_old = r + 2Ma), not an error in the readout.")


if __name__ == "__main__":
    main()
