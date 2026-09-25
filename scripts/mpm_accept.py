#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_accept.py -- CPU-only acceptance checks for Phase 3, Scene A (column collapse).

Reads the `.npz` files produced by `mpm_run.py` from `PROJECT/logs/` and prints a verdict table.
No Warp, no Newton, no GPU: only numpy and the standard library.

Thresholds are STARTING VALUES (design-note section 10); every raw number is printed next to the
verdict.  Tests 8-10 of the design note belong to Scene B and are not implemented here.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

# Starting thresholds (design-note section 10; must match the note).
REST_DISP_TOL = 1.0e-2          # m, max particle displacement over the rest window
REST_SPEED_TOL = 1.0e-3         # m/s, "settled" mean speed
REPOSE_TOL_DEG = 8.0            # deg, allowed difference from the friction angle
MASS_REL_TOL = 1.0e-6           # relative, total active mass drift
WEIGHT_REL_TOL = 0.02           # relative, ground reaction vs weight
REPRO_TOL = 1.0e-4              # m, max particle position difference between identical runs
STEP_MS_TOL = 15.0              # ms, median step time (excluding the first step)
VRAM_GB_TOL = 6.0               # GB, peak process VRAM minus idle

PREFIXES = {0.10: "sceneA_D010", 0.15: "sceneA_D015", 0.20: "sceneA_D020"}


def load(log_dir: str, prefix: str, suffix: str = ""):
    """Load an npz result file, or return None if it is missing."""
    path = os.path.join(log_dir, prefix + suffix + ".npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def snap_index(snap_t: np.ndarray, target: float) -> int:
    """Index of the snapshot nearest to `target` (snap_t is sorted)."""
    return int(np.argmin(np.abs(snap_t - target)))


def max_disp_between(d, t0: float, t1: float) -> tuple[float, float, float]:
    """Max per-particle displacement between the snapshots nearest to t0 and t1 (metres)."""
    i0 = snap_index(d["snap_t"], t0)
    i1 = snap_index(d["snap_t"], t1)
    dq = np.linalg.norm(d["snap_q"][i1].astype(np.float64) - d["snap_q"][i0].astype(np.float64), axis=1)
    return float(dq.max()), float(d["snap_t"][i0]), float(d["snap_t"][i1])


def first_rest_time(t: np.ndarray, mean_speed: np.ndarray, thresh: float) -> float | None:
    """First time from which the mean speed stays below `thresh` (None if it never does)."""
    ok = mean_speed < thresh
    # Find the last index that is NOT below threshold; rest starts right after it.
    bad = np.nonzero(~ok)[0]
    if len(bad) == 0:
        return float(t[0])
    if bad[-1] + 1 >= len(t):
        return None
    return float(t[bad[-1] + 1])


def flank_angles(q: np.ndarray, voxel: float) -> dict:
    """Estimate pile flank angles from the final particle cloud.

    Method (design-note section 10): take a thin slab around the mid plane, build the upper envelope
    z_max in bins of one voxel, then fit a straight line to each flank between 20% and 80% of the
    pile's height range.  Returns angles in degrees for the y=0 and x=0 planes.
    """
    out = {}
    for axis, slab_axis, bin_axis in (("y0", 1, 0), ("x0", 0, 1)):
        slab = q[np.abs(q[:, slab_axis]) < voxel]
        if slab.shape[0] < 4:
            out[axis] = {"left": None, "right": None, "mean": None, "note": "too few particles in slab"}
            continue
        coord = slab[:, bin_axis]
        z = slab[:, 2]
        edges = np.arange(coord.min() - voxel, coord.max() + 2.0 * voxel, voxel)
        centers, zmax = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (coord >= a) & (coord < b)
            if np.any(m):
                centers.append(0.5 * (a + b))
                zmax.append(float(z[m].max()))
        centers = np.asarray(centers)
        zmax = np.asarray(zmax)
        if centers.size < 4:
            out[axis] = {"left": None, "right": None, "mean": None, "note": "too few bins"}
            continue
        peak = int(np.argmax(zmax))
        z_peak = float(zmax[peak])
        lo, hi = 0.2 * z_peak, 0.8 * z_peak

        def fit(side_mask):
            sel = side_mask & (zmax >= lo) & (zmax <= hi)
            if np.count_nonzero(sel) < 2:
                return None
            a, _b = np.polyfit(centers[sel], zmax[sel], 1)
            return math.degrees(math.atan(abs(float(a))))

        left = fit(np.arange(centers.size) < peak)
        right = fit(np.arange(centers.size) > peak)
        vals = [v for v in (left, right) if v is not None]
        out[axis] = {
            "left": left,
            "right": right,
            "mean": float(np.mean(vals)) if vals else None,
            "note": "",
        }
    return out


def vram_peak_minus_idle(log_dir: str, prefix: str) -> tuple[float | None, float | None]:
    """Peak VRAM (MiB) minus the idle value recorded just before that run.

    Prefers a per-run `<prefix>_idle.txt` (written immediately before the run), falling back to the
    shared `idle_vram.txt`.
    """
    run_idle_path = os.path.join(log_dir, prefix + "_idle.txt")
    idle_path = os.path.join(log_dir, "idle_vram.txt")
    csv_path = os.path.join(log_dir, prefix + "_vram.csv")
    if not os.path.exists(csv_path):
        return None, None
    if os.path.exists(run_idle_path):
        idle = float(open(run_idle_path).read().strip())
    elif os.path.exists(idle_path):
        idle = float(open(idle_path).read().strip())
    else:
        idle = 0.0
    vals = [float(x) for x in open(csv_path).read().split() if x.strip()]
    if not vals:
        return None, None
    return max(vals) - idle, max(vals)


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


# --------------------------------------------------------------------------------------------------
# Step-6b Part B: repose study over ground friction, soil phi and column height
# --------------------------------------------------------------------------------------------------
REPOSE_RUNS = [
    "regr_D015", "g09_p34_h25", "g20_p34_h25",
    "g20_p20_h25", "g20_p27_h25", "g20_p40_h25", "g20_p45_h25",
    "g20_p34_h50", "g20_p34_h80", "g100_p34_h25",
]


def run_characteristics(log_dir: str, prefix: str, t_target: float = 2.0):
    """Load one Part-B run and compute its final-pile characteristics (or None if missing)."""
    d = load(log_dir, prefix)
    if d is None:
        return None
    voxel = float(d["voxel"])
    D = float(d["diameter"])
    i = snap_index(d["snap_t"], t_target)          # snapshot nearest t_target (final for 2 s runs)
    q0 = d["snap_q"][0].astype(np.float64)
    qf = d["snap_q"][i].astype(np.float64)
    ang = flank_angles(qf, voxel)
    means = [ang[k]["mean"] for k in ("y0", "x0") if ang[k]["mean"] is not None]
    mean_ang = float(np.mean(means)) if means else None
    runout = float(np.sqrt(qf[:, 0] ** 2 + qf[:, 1] ** 2).max() / D)  # radial extent / initial half width
    h0 = float(q0[:, 2].max())
    hf = float(qf[:, 2].max())
    height_ratio = hf / h0 if h0 > 0 else float("nan")
    rest_t = first_rest_time(d["t"], d["mean_speed"], REST_SPEED_TOL)
    top = q0[:, 2] >= h0 - voxel                    # initial top layer (about one voxel thick)
    top_dz = float(np.abs(qf[top, 2] - q0[top, 2]).max()) if np.any(top) else float("nan")
    return {
        "prefix": prefix, "D": D, "ground_mu": float(d["ground_friction"]),
        "phi": float(d["soil_friction_angle_deg"]), "h": float(d["block_height_factor"]),
        "N": int(d["n_particles"]), "angle": mean_ang, "runout": runout,
        "height_ratio": height_ratio, "rest_t": rest_t, "top_dz": top_dz,
        "snap_t": float(d["snap_t"][i]),
    }


def repose_study(log_dir: str) -> None:
    """Print the Step-6b Part-B tables and write logs/phase3_6b_table.txt."""
    rows = []
    for p in REPOSE_RUNS:
        r = run_characteristics(log_dir, p)
        if r is None:
            print(f"[repose] missing {p}.npz")
        else:
            rows.append(r)

    def s(x, nd=2):
        return "n/a" if x is None else f"{x:.{nd}f}"

    out = []
    out.append("run              D    gmu   phi  hfac       N  snap_t  angle_deg  runout/D  h_ratio  rest_t    top_dz")
    for r in rows:
        out.append(f"{r['prefix']:<16}{r['D']:>5.2f}{r['ground_mu']:>6.2f}{r['phi']:>6.1f}{r['h']:>6.1f}"
                   f"{r['N']:>8d}{r['snap_t']:>8.2f}{s(r['angle']):>11}{s(r['runout'],3):>10}{s(r['height_ratio'],3):>9}"
                   f"{s(r['rest_t'],3):>8}{s(r['top_dz'],4):>10}")

    # Set 1: ground friction (phi=34, h=2.5).
    out.append("")
    out.append("Set 1 - ground friction (phi=34, h=2.5):")
    for r in rows:
        if r["prefix"] in ("regr_D015", "g09_p34_h25", "g20_p34_h25"):
            out.append(f"  ground_mu={r['ground_mu']:.2f}  angle={s(r['angle'])} deg  runout/D={s(r['runout'],3)}")

    # Set 2: soil phi (ground friction 2.0, h=2.5) + linear fit measured = a*phi + b.
    pts = [(r["phi"], r["angle"]) for r in rows
           if r["prefix"] in ("g20_p20_h25", "g20_p27_h25", "g20_p34_h25", "g20_p40_h25", "g20_p45_h25")
           and r["angle"] is not None]
    out.append("")
    out.append("Set 2 - soil phi (ground_mu=2.0, h=2.5):")
    for phi, ang in sorted(pts):
        out.append(f"  phi={phi:.1f}  angle={ang:.2f} deg")
    if len(pts) >= 2:
        phi_a = np.array([p[0] for p in pts])
        ang_a = np.array([p[1] for p in pts])
        a, b = np.polyfit(phi_a, ang_a, 1)
        resid = ang_a - (a * phi_a + b)
        phi_star = (34.0 - b) / a if a != 0 else float("nan")
        out.append(f"  fit: angle = {a:.4f}*phi + {b:.3f};  residuals = {np.round(resid, 2).tolist()};  max|resid| = {np.abs(resid).max():.2f} deg")
        out.append(f"  mean(angle) - mean(phi) = {ang_a.mean()-phi_a.mean():+.2f} deg")
        out.append(f"  calibration: phi input that would give a measured angle of 34 deg = {phi_star:.1f} deg")

    # Set 3: column height (ground_mu=2.0, phi=34).
    out.append("")
    out.append("Set 3 - column height (ground_mu=2.0, phi=34):")
    for r in rows:
        if r["prefix"] in ("g20_p34_h25", "g20_p34_h50", "g20_p34_h80"):
            out.append(f"  hfac={r['h']:.1f} (N={r['N']})  angle={s(r['angle'])} deg  height_ratio={s(r['height_ratio'],3)}"
                       f"  runout/D={s(r['runout'],3)}")

    text = "\n".join(out)
    print(text)
    path = os.path.join(log_dir, "phase3_6b_table.txt")
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print(f"\n[repose] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--repose", action="store_true", help="run the Step-6b Part-B repose study instead of the Scene A tests")
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    if args.repose:
        repose_study(log_dir)
        return

    rows: list[tuple[str, str, str, str]] = []

    # ---------------- per-diameter tests ----------------
    for D, prefix in PREFIXES.items():
        d = load(log_dir, prefix)
        if d is None:
            rows.append((f"D={D:.2f} run", "INCONCLUSIVE", "npz missing", f"{prefix}.npz"))
            continue
        voxel = float(d["voxel"])
        dt = float(d["dt"])
        g = d["gravity"].astype(np.float64)  # (3,)
        g_z = float(g[2])
        M = float(d["total_mass"])

        # --- test 1: rest -------------------------------------------------------------------------
        t = d["t"]
        mean_speed = d["mean_speed"]
        speed_25 = float(mean_speed[snap_index(t, 2.5)])
        rest_t = first_rest_time(t, mean_speed, REST_SPEED_TOL)
        disp, ta, tb = max_disp_between(d, 2.5, 3.0)
        d6 = load(log_dir, prefix, "_6s")
        if d6 is not None:
            disp6, _, _ = max_disp_between(d6, 5.5, 6.0)
            t6 = d6["t"]
            rest_t6 = first_rest_time(t6, d6["mean_speed"], REST_SPEED_TOL)
            verdict = "PASS" if (disp6 < REST_DISP_TOL and rest_t6 is not None) else "FAIL"
            key = (f"3s: disp(2.5-3.0)={disp:.3e} m, rest_t={fmt(rest_t)} s, speed@2.5={speed_25:.3e}; "
                   f"6s: disp(5.5-6.0)={disp6:.3e} m, rest_t={fmt(rest_t6)} s")
            rows.append((f"D={D:.2f} T1 rest", verdict, key, f"disp<{REST_DISP_TOL:g} m"))
        else:
            if rest_t is None:
                verdict = "INCONCLUSIVE"
                key = (f"not settled by {t[-1]:.2f}s; disp(2.5-3.0)={disp:.3e} m, "
                       f"speed@2.5={speed_25:.3e} m/s")
            else:
                verdict = "PASS" if disp < REST_DISP_TOL else "FAIL"
                key = (f"disp(2.5-3.0)={disp:.3e} m ({ta:.2f}->{tb:.2f}s), rest_t={rest_t:.3f} s, "
                       f"speed@2.5={speed_25:.3e} m/s")
            rows.append((f"D={D:.2f} T1 rest", verdict, key, f"disp<{REST_DISP_TOL:g} m"))

        # --- test 2: angle of repose (informative) ------------------------------------------------
        q_final = d["snap_q"][-1].astype(np.float64)
        ang = flank_angles(q_final, voxel)
        plane_means = [ang[k]["mean"] for k in ("y0", "x0") if ang[k]["mean"] is not None]
        if plane_means:
            mean_ang = float(np.mean(plane_means))
            diff = mean_ang - float(d["soil_friction_angle_deg"])
            verdict = "PASS" if abs(diff) <= REPOSE_TOL_DEG else "FAIL"
            key = (f"y0 L/R={fmt(ang['y0']['left'],3)}/{fmt(ang['y0']['right'],3)}, "
                   f"x0 L/R={fmt(ang['x0']['left'],3)}/{fmt(ang['x0']['right'],3)}, "
                   f"mean={mean_ang:.2f} deg, phi={float(d['soil_friction_angle_deg']):.0f} deg, diff={diff:+.2f}")
        else:
            verdict = "INCONCLUSIVE"
            key = f"no usable flank; {ang['y0']['note']} / {ang['x0']['note']}"
        rows.append((f"D={D:.2f} T2 repose", verdict, key, f"|mean-phi|<={REPOSE_TOL_DEG:g} deg"))

        # --- test 3: mass conservation and no fall-through -----------------------------------------
        mass = d["mass"]
        rel = float((mass.max() - mass.min()) / mass.mean())
        min_z = float(d["min_z"].min())
        verdict = "PASS" if (rel < MASS_REL_TOL and min_z > -voxel) else "FAIL"
        rows.append((f"D={D:.2f} T3 mass", verdict,
                     f"rel_drift={rel:.3e}, min_z={min_z:.4f} m, -voxel={-voxel:.4f} m",
                     f"<{MASS_REL_TOL:g} and min_z>-voxel"))

        # --- test 4: ground weight check + diagnostic ---------------------------------------------
        win = (t >= 2.5 - 1e-9) & (t <= 3.0 + 1e-9)
        fz = d["ground_force"][win, 2]
        fz_mean = float(fz.mean())
        fz_std = float(fz.std())
        expected = M * g_z  # signed: g_z is negative, so expected is negative for a downward push
        rel_err = abs(abs(fz_mean) - abs(expected)) / abs(expected)
        verdict = "PASS" if rel_err < WEIGHT_REL_TOL else "FAIL"
        # Diagnostic: Fz - M*(g + a_com_z) over the whole run (reveals sign/units/one-step-delay issues).
        a_com = np.gradient(d["com_vel"][:, 2], t)
        diag = d["ground_force"][:, 2] - M * (g_z + a_com)
        rows.append((f"D={D:.2f} T4 weight", verdict,
                     f"Fz_mean={fz_mean:.3f} N (std {fz_std:.3f}), expected={expected:.3f} N, "
                     f"rel_err={rel_err:.3e}; diag mean={diag.mean():.3f} std={diag.std():.3f} "
                     f"min={diag.min():.3f} max={diag.max():.3f} N",
                     f"|Fz| within {WEIGHT_REL_TOL:.0%} of M*g"))

        # --- test 6 (per run): no NaN -------------------------------------------------------------
        nan_ok = not (
            np.isnan(d["snap_q"]).any() or np.isnan(d["snap_v"]).any()
            or np.isnan(d["ground_force"]).any() or np.isnan(d["com"]).any()
        )
        log_ok = os.path.exists(os.path.join(log_dir, prefix + ".log")) and (
            "RUN_DONE" in open(os.path.join(log_dir, prefix + ".log")).read())
        verdict = "PASS" if (nan_ok and log_ok) else "FAIL"
        rows.append((f"D={D:.2f} T6 no-fail", verdict,
                     f"nan_ok={nan_ok}, RUN_DONE_in_log={log_ok}", "no NaN and RUN_DONE"))

        # --- test 7: performance -------------------------------------------------------------------
        st = d["step_time"]
        med = float(np.median(st[1:])) if st.size > 1 else float("nan")
        first = float(st[0]) if st.size else float("nan")
        ms_per_sim_s = 1e3 * med / dt
        dv, peak = vram_peak_minus_idle(log_dir, prefix)
        vram_ok = (dv is not None) and (dv < VRAM_GB_TOL * 1024.0)
        verdict = "PASS" if (med < STEP_MS_TOL and vram_ok) else "FAIL"
        rows.append((f"D={D:.2f} T7 perf", verdict,
                     f"first={1e3*first:.1f} ms, median={1e3*med:.2f} ms, "
                     f"ms/sim_s={ms_per_sim_s:.0f}, peak_vram_delta={fmt(dv,4)} MiB (peak {fmt(peak,5)})",
                     f"median<{STEP_MS_TOL:g} ms and delta<{VRAM_GB_TOL:g} GB"))

    # ---------------- test 5: reproducibility (D=0.15) ----------------
    a = load(log_dir, "sceneA_D015")
    b = load(log_dir, "sceneA_D015_repeat")
    if a is None or b is None:
        rows.append(("T5 reproduc.", "INCONCLUSIVE", "sceneA_D015 or _repeat missing", f"<{REPRO_TOL:g} m"))
    else:
        n = min(a["snap_q"].shape[0], b["snap_q"].shape[0])
        diffs = [
            float(np.abs(a["snap_q"][i].astype(np.float64) - b["snap_q"][i].astype(np.float64)).max())
            for i in range(n)
        ]
        mx = max(diffs) if diffs else float("nan")
        verdict = "PASS" if mx < REPRO_TOL else "FAIL"
        rows.append(("T5 reproduc.", verdict,
                     f"max snapshot |dq|={mx:.3e} m over {n} snapshots (per-snapshot max={fmt(max(diffs),3)})",
                     f"<{REPRO_TOL:g} m"))

    # ---------------- print ----------------
    w = (20, 13, 105, 34)
    print("=" * (sum(w) + 3))
    print(f"{'test':<{w[0]}} {'verdict':<{w[1]}} {'key numbers':<{w[2]}} {'threshold':<{w[3]}}")
    print("-" * (sum(w) + 3))
    for r in rows:
        print(f"{r[0]:<{w[0]}} {r[1]:<{w[1]}} {r[2]:<{w[2]}} {r[3]:<{w[3]}}")
    print("=" * (sum(w) + 3))


if __name__ == "__main__":
    main()
