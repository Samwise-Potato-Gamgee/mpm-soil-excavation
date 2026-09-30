#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_resolution_study.py -- CPU-only resolution-study analysis for Step 6c.

Reads logs/<prefix>.npz files produced by mpm_run.py and reports, for every run:
  - the flank angle using the EXACT mpm_accept.flank_angles algorithm (imported and called),
    for both slabs (y0, x0), left/right/mean;
  - a transparent re-implementation of the same fit that additionally returns the peak height
    z_peak, z_peak/voxel (pile height in voxels), the number of fit bins per side and the
    horizontal span of the fit window in voxels (max-min centre / voxel), per side;
  - a validity flag VALID if z_peak/voxel >= 6 and the fit window has >= 4 bins per side,
    otherwise LOW_RES (reported, never filtered);
  - rest time (the mpm_accept criterion), mass drift, last-step ground force vs total weight,
    median step time, n_escaped, n_below and N.

The re-implementation is cross-checked against mpm_accept.flank_angles for every run (and
explicitly for the reference run logs/sceneA_D015.npz); a mismatch > 1e-9 deg aborts.

No Warp, no Newton, no GPU: numpy + stdlib only.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from mpm_accept import REST_SPEED_TOL, first_rest_time, flank_angles


def load(log_dir: str, prefix: str):
    """Load logs/<prefix>.npz, or return None if missing."""
    path = os.path.join(log_dir, prefix + ".npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def flank_angles_detailed(q: np.ndarray, voxel: float) -> dict:
    """Transparent re-implementation of mpm_accept.flank_angles with fit-window metadata.

    Mirrors mpm_accept.flank_angles exactly (same slab, same one-voxel bins, same upper envelope,
    same 20-80% height window, same polyfit) so the returned angle is bit-identical.
    """
    out = {}
    for axis, slab_axis, bin_axis in (("y0", 1, 0), ("x0", 0, 1)):
        rec = {
            "left": None, "right": None, "mean": None, "note": "",
            "z_peak": None, "z_peak_over_voxel": None,
            "left_bins": None, "right_bins": None,
            "left_span_voxels": None, "right_span_voxels": None,
        }
        slab = q[np.abs(q[:, slab_axis]) < voxel]
        if slab.shape[0] < 4:
            rec["note"] = "too few particles in slab"
            out[axis] = rec
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
            rec["note"] = "too few bins"
            out[axis] = rec
            continue
        peak = int(np.argmax(zmax))
        z_peak = float(zmax[peak])
        lo, hi = 0.2 * z_peak, 0.8 * z_peak
        rec["z_peak"] = z_peak
        rec["z_peak_over_voxel"] = z_peak / voxel
        left_mask = np.arange(centers.size) < peak
        right_mask = np.arange(centers.size) > peak

        def fit(side_mask):
            sel = side_mask & (zmax >= lo) & (zmax <= hi)
            if np.count_nonzero(sel) < 2:
                return None
            a, _b = np.polyfit(centers[sel], zmax[sel], 1)
            return math.degrees(math.atan(abs(float(a))))

        left = fit(left_mask)
        right = fit(right_mask)
        vals = [v for v in (left, right) if v is not None]
        rec["left"] = left
        rec["right"] = right
        rec["mean"] = float(np.mean(vals)) if vals else None

        for side, mask in (("left", left_mask), ("right", right_mask)):
            sel = mask & (zmax >= lo) & (zmax <= hi)
            nb = int(np.count_nonzero(sel))
            rec[side + "_bins"] = nb
            if nb >= 1:
                wc = centers[sel]
                rec[side + "_span_voxels"] = float((wc.max() - wc.min()) / voxel)
        out[axis] = rec
    return out


def m2_slab(q: np.ndarray, slab_axis: int, bin_axis: int,
            bin_width: float = 0.015, slab_half: float = 0.015) -> dict:
    """Method M2 on one slab: FIXED bin width and FIXED slab half-width.

    Slab: |coord[slab_axis]| < slab_half. Bins of fixed bin_width along bin_axis over the full
    spanned range (contiguous, so empty bins are visible). Upper envelope z_max per bin, peak =
    max, fit window = 20-80% of z_peak on each side of the peak, linear fit, angle = atan(|slope|).
    Returns left/right/mean angles plus fit-bin counts and the number of empty bins inside each
    side's fit window (between the outermost selected bins).
    """
    rec = {"left": None, "right": None, "mean": None, "z_peak": None, "note": "",
           "left_bins": 0, "right_bins": 0, "left_empty": 0, "right_empty": 0}
    slab = q[np.abs(q[:, slab_axis]) < slab_half]
    if slab.shape[0] < 4:
        rec["note"] = "too few particles in slab"
        return rec
    coord = slab[:, bin_axis]
    z = slab[:, 2]
    lo_edge = math.floor(float(coord.min()) / bin_width) * bin_width
    hi_edge = math.ceil(float(coord.max()) / bin_width) * bin_width + bin_width
    edges = np.arange(lo_edge, hi_edge, bin_width)
    if edges.size < 2:
        rec["note"] = "too few bins"
        return rec
    centers = 0.5 * (edges[:-1] + edges[1:])
    zmax = np.full(centers.size, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (coord >= a) & (coord < b)
        if np.any(m):
            zmax[i] = float(z[m].max())
    if np.all(np.isnan(zmax)):
        rec["note"] = "no non-empty bins"
        return rec
    peak = int(np.nanargmax(zmax))
    z_peak = float(zmax[peak])
    rec["z_peak"] = z_peak
    lo, hi = 0.2 * z_peak, 0.8 * z_peak
    nonempty = ~np.isnan(zmax)

    def side(side_mask):
        sel = side_mask & nonempty & (zmax >= lo) & (zmax <= hi)
        nb = int(np.count_nonzero(sel))
        if nb < 2:
            return None, nb, 0
        a, _b = np.polyfit(centers[sel], zmax[sel], 1)
        ang = math.degrees(math.atan(abs(float(a))))
        idx = np.nonzero(sel)[0]
        span = np.arange(idx.min(), idx.max() + 1)
        empty = int(np.count_nonzero(np.isnan(zmax[span])))
        return ang, nb, empty

    left, lb, le = side(np.arange(centers.size) < peak)
    right, rb, re = side(np.arange(centers.size) > peak)
    rec["left"], rec["right"] = left, right
    rec["left_bins"], rec["right_bins"] = lb, rb
    rec["left_empty"], rec["right_empty"] = le, re
    vals = [v for v in (left, right) if v is not None]
    rec["mean"] = float(np.mean(vals)) if vals else None
    return rec


def radial_profile_angle(q: np.ndarray, center: tuple, bin_width: float = 0.015) -> dict:
    """Method M3: radial upper-envelope profile with fixed 0.015 m bins.

    r = sqrt((x-cx)^2 + (y-cy)^2) over ALL particles, fixed-width r bins from min to max r,
    upper envelope z_max per bin. Outward-falling flank: bins with r larger than the r of the
    peak and z in [0.2, 0.8] * z_peak, linear fit of z_max vs r, angle = atan(|slope|).
    NOTE: the block base is square, not axisymmetric, so M3 averages over azimuth.
    """
    rec = {"angle": None, "bins": 0, "empty": 0, "z_peak": None, "note": ""}
    dx = q[:, 0] - center[0]
    dy = q[:, 1] - center[1]
    r = np.sqrt(dx * dx + dy * dy)
    z = q[:, 2]
    lo_edge = math.floor(float(r.min()) / bin_width) * bin_width
    hi_edge = math.ceil(float(r.max()) / bin_width) * bin_width + bin_width
    edges = np.arange(lo_edge, hi_edge, bin_width)
    if edges.size < 2:
        rec["note"] = "too few bins"
        return rec
    centers = 0.5 * (edges[:-1] + edges[1:])
    zmax = np.full(centers.size, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (r >= a) & (r < b)
        if np.any(m):
            zmax[i] = float(z[m].max())
    if np.all(np.isnan(zmax)):
        rec["note"] = "no non-empty bins"
        return rec
    peak = int(np.nanargmax(zmax))
    z_peak = float(zmax[peak])
    rec["z_peak"] = z_peak
    lo, hi = 0.2 * z_peak, 0.8 * z_peak
    nonempty = ~np.isnan(zmax)
    outward = np.arange(centers.size) > peak
    sel = outward & nonempty & (zmax >= lo) & (zmax <= hi)
    nb = int(np.count_nonzero(sel))
    rec["bins"] = nb
    if nb < 2:
        rec["note"] = "too few fit bins"
        return rec
    a, _b = np.polyfit(centers[sel], zmax[sel], 1)
    rec["angle"] = math.degrees(math.atan(abs(float(a))))
    idx = np.nonzero(sel)[0]
    span = np.arange(idx.min(), idx.max() + 1)
    rec["empty"] = int(np.count_nonzero(np.isnan(zmax[span])))
    return rec


def cross_check(q: np.ndarray, voxel: float, label: str) -> None:
    """Abort if the detailed re-implementation disagrees with mpm_accept.flank_angles."""
    fa = flank_angles(q, voxel)
    fd = flank_angles_detailed(q, voxel)
    for axis in ("y0", "x0"):
        for key in ("left", "right", "mean"):
            a, b = fa[axis][key], fd[axis][key]
            if (a is None) != (b is None):
                raise RuntimeError(f"{label}: flank None mismatch in {axis}/{key}: {a!r} vs {b!r}")
            if a is not None and abs(a - b) > 1.0e-9:
                raise RuntimeError(
                    f"{label}: flank mismatch in {axis}/{key}: mpm_accept={a!r} detailed={b!r} "
                    f"(|diff|={abs(a-b):.3e} deg > 1e-9)"
                )


def vram_peak_minus_idle(log_dir: str, prefix: str):
    """Peak VRAM [MiB] above the idle value, from the driver's <prefix>_vram.csv and idle file."""
    vram_path = os.path.join(log_dir, prefix + "_vram.csv")
    run_idle_path = os.path.join(log_dir, prefix + "_idle.txt")
    idle_path = os.path.join(log_dir, "idle_vram.txt")
    if not os.path.exists(vram_path):
        return None
    if os.path.exists(run_idle_path):
        idle = float(open(run_idle_path).read().strip())
    elif os.path.exists(idle_path):
        idle = float(open(idle_path).read().strip())
    else:
        idle = 0.0
    vals = [float(x) for x in open(vram_path).read().split() if x.strip()]
    if not vals:
        return None
    return max(vals) - idle


def analyze(d, prefix: str, log_dir: str) -> dict:
    """Compute all Step-6c quantities for one loaded npz."""
    voxel = float(d["voxel"])
    qf = d["final_q"].astype(np.float64) if "final_q" in d else d["snap_q"][-1].astype(np.float64)

    # ---- provenance (Step 6d-test1) and NaN/inf check ----
    dil = float(d["dilatancy_used"]) if "dilatancy_used" in d else 0.0
    sb_raw = d["strain_basis_used"] if "strain_basis_used" in d else "P0"
    sbasis = str(sb_raw.item()) if hasattr(sb_raw, "item") else str(sb_raw)
    arrs = [qf, d["ground_force"].astype(np.float64)]
    if "final_v" in d:
        arrs.append(d["final_v"].astype(np.float64))
    nan_flag = bool(any(not np.all(np.isfinite(a)) for a in arrs))

    cross_check(qf, voxel, prefix)

    fa = flank_angles(qf, voxel)
    fd = flank_angles_detailed(qf, voxel)

    # ---- initial-state geometry and mass (from the first snapshot) ----
    q0 = d["snap_q"][0].astype(np.float64)
    init_width = float(q0[:, 0].max() - q0[:, 0].min())
    init_height = float(q0[:, 2].max() - q0[:, 2].min())
    total_mass_arr = float(d["mass"][0])
    center = (float(q0[:, 0].mean()), float(q0[:, 1].mean()))

    # ---- voxel-independent methods M2 (fixed-bin slabs) and M3 (radial profile) ----
    m2_y0 = m2_slab(qf, 1, 0)
    m2_x0 = m2_slab(qf, 0, 1)
    m2_means = [m for m in (m2_y0["mean"], m2_x0["mean"]) if m is not None]
    m2_overall = float(np.mean(m2_means)) if m2_means else None
    m3 = radial_profile_angle(qf, center)
    runout_r = float(np.sqrt(qf[:, 0] ** 2 + qf[:, 1] ** 2).max())

    # ---- rest time (same criterion as mpm_accept.py:57-65, REST_SPEED_TOL mpm_accept.py:23) ----
    t = d["t"]
    rest_t = first_rest_time(t, d["mean_speed"], REST_SPEED_TOL)

    # ---- mass drift (same form as mpm_accept T3) ----
    mass = d["mass"]
    mass_drift = float((mass.max() - mass.min()) / mass.mean())

    # ---- ground force at the last step vs total weight ----
    M = float(d["total_mass"])
    gz = float(d["gravity"].astype(np.float64)[2])
    fz_last = float(d["ground_force"][-1, 2])
    expected = M * gz
    fz_reldiff = abs(abs(fz_last) - abs(expected)) / abs(expected) if expected != 0.0 else float("nan")
    fz_over_W = fz_last / expected if expected != 0.0 else float("nan")

    # ---- performance ----
    st = d["step_time"]
    med_step_ms = float(np.median(st[1:])) * 1.0e3 if st.size > 1 else float("nan")
    if st.size >= 3:
        k = st.size // 3
        med_first_ms = float(np.median(st[:k])) * 1.0e3
        med_last_ms = float(np.median(st[-k:])) * 1.0e3
    else:
        med_first_ms = float("nan")
        med_last_ms = float("nan")

    # ---- escape diagnostics (recomputed from positions; stored keys are a convenience) ----
    gh = float(d["ground_half_xy"])
    n_escaped = int(np.count_nonzero(
        (np.abs(qf[:, 0]) > gh) | (np.abs(qf[:, 1]) > gh)
    ))
    n_below = int(np.count_nonzero(qf[:, 2] < -1.0e-3))

    # ---- aggregate flank metadata ----
    def g(axis, key):
        return fd[axis][key]

    zpeaks = [g("y0", "z_peak"), g("x0", "z_peak")]
    zpeaks = [z for z in zpeaks if z is not None]
    zpeak_y0 = g("y0", "z_peak")
    zpeak_x0 = g("x0", "z_peak")
    zpeak_min = min(zpeaks) if zpeaks else None
    zpeak_vox_min = (zpeak_min / voxel) if zpeak_min is not None else None
    zpeak_max = max(zpeaks) if zpeaks else None
    z_peak_m = zpeak_max
    z_peak_m_vox = (z_peak_m / voxel) if z_peak_m is not None else None

    bin_keys = [("y0", "left_bins"), ("y0", "right_bins"), ("x0", "left_bins"), ("x0", "right_bins")]
    bins_present = [g(a, k) for a, k in bin_keys]
    bins_min = min(b for b in bins_present if b is not None) if any(b is not None for b in bins_present) else 0

    valid = (zpeak_vox_min is not None and zpeak_vox_min >= 6.0 and bins_min >= 4)

    def mean_angle():
        vals = [fa[a]["mean"] for a in ("y0", "x0") if fa[a]["mean"] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "prefix": prefix,
        "voxel": voxel,
        "hfac": float(d["block_height_factor"]),
        "N": int(d["n_particles"]),
        "n_steps": int(d["n_steps"]),
        "dt": float(d["dt"]),
        "zpeak_y0": zpeak_y0,
        "zpeak_x0": zpeak_x0,
        "zpeak_vox_min": zpeak_vox_min,
        "ang_y0_L": fa["y0"]["left"], "ang_y0_R": fa["y0"]["right"], "ang_y0_mean": fa["y0"]["mean"],
        "ang_x0_L": fa["x0"]["left"], "ang_x0_R": fa["x0"]["right"], "ang_x0_mean": fa["x0"]["mean"],
        "ang_mean": mean_angle(),
        "bins_y0_L": g("y0", "left_bins"), "bins_y0_R": g("y0", "right_bins"),
        "bins_x0_L": g("x0", "left_bins"), "bins_x0_R": g("x0", "right_bins"),
        "span_y0_L": g("y0", "left_span_voxels"), "span_y0_R": g("y0", "right_span_voxels"),
        "span_x0_L": g("x0", "left_span_voxels"), "span_x0_R": g("x0", "right_span_voxels"),
        "valid": "VALID" if valid else "LOW_RES",
        "rest_t": rest_t,
        "mass_drift": mass_drift,
        "Fz_last": fz_last,
        "Fz_over_W": fz_over_W,
        "Fz_reldiff": fz_reldiff,
        "med_step_ms": med_step_ms,
        "vram_delta": vram_peak_minus_idle(log_dir, prefix),
        "n_escaped": n_escaped,
        "n_below": n_below,
        "init_width": init_width,
        "init_height": init_height,
        "total_mass_arr": total_mass_arr,
        "z_peak_m": z_peak_m,
        "z_peak_m_vox": z_peak_m_vox,
        "m2_y0_L": m2_y0["left"], "m2_y0_R": m2_y0["right"], "m2_y0_mean": m2_y0["mean"],
        "m2_x0_L": m2_x0["left"], "m2_x0_R": m2_x0["right"], "m2_x0_mean": m2_x0["mean"],
        "m2_mean": m2_overall,
        "m2_y0_bins_L": m2_y0["left_bins"], "m2_y0_bins_R": m2_y0["right_bins"],
        "m2_x0_bins_L": m2_x0["left_bins"], "m2_x0_bins_R": m2_x0["right_bins"],
        "m2_y0_empty_L": m2_y0["left_empty"], "m2_y0_empty_R": m2_y0["right_empty"],
        "m2_x0_empty_L": m2_x0["left_empty"], "m2_x0_empty_R": m2_x0["right_empty"],
        "m3_angle": m3["angle"], "m3_bins": m3["bins"], "m3_empty": m3["empty"],
        "runout_r": runout_r,
        "med_first_ms": med_first_ms,
        "med_last_ms": med_last_ms,
        "dil": dil,
        "sbasis": sbasis,
        "nan": nan_flag,
        "dM2": None,
    }


# (key, header, width, decimals) -- decimals None means string/int
COLUMNS = [
    ("prefix", "run", 16, None),
    ("voxel", "voxel", 9, 5),
    ("hfac", "hfac", 5, 2),
    ("N", "N", 7, None),
    ("n_steps", "steps", 6, None),
    ("dt", "dt", 10, 6),
    ("zpeak_y0", "zpk_y0", 8, 4),
    ("zpeak_x0", "zpk_x0", 8, 4),
    ("zpeak_vox_min", "zpk_vox", 8, 2),
    ("ang_y0_L", "ay0_L", 7, 2),
    ("ang_y0_R", "ay0_R", 7, 2),
    ("ang_y0_mean", "ay0_m", 7, 2),
    ("ang_x0_L", "ax0_L", 7, 2),
    ("ang_x0_R", "ax0_R", 7, 2),
    ("ang_x0_mean", "ax0_m", 7, 2),
    ("ang_mean", "ang_m", 7, 2),
    ("bins_y0_L", "bn_y0L", 7, None),
    ("bins_y0_R", "bn_y0R", 7, None),
    ("bins_x0_L", "bn_x0L", 7, None),
    ("bins_x0_R", "bn_x0R", 7, None),
    ("span_y0_L", "sp_y0L", 7, 2),
    ("span_y0_R", "sp_y0R", 7, 2),
    ("span_x0_L", "sp_x0L", 7, 2),
    ("span_x0_R", "sp_x0R", 7, 2),
    ("valid", "valid", 8, None),
    ("rest_t", "rest_t", 8, 3),
    ("mass_drift", "m_drift", 10, 2),
    ("Fz_last", "Fz_last", 10, 3),
    ("Fz_over_W", "Fz/W", 8, 6),
    ("Fz_reldiff", "Fz_rel", 10, 2),
    ("med_step_ms", "med_ms", 8, 2),
    ("vram_delta", "vram_MiB", 8, 0),
    ("n_escaped", "esc", 5, None),
    ("n_below", "bel", 5, None),
    ("init_width", "initW", 9, 4),
    ("init_height", "initH", 9, 4),
    ("total_mass_arr", "mass_kg", 10, 4),
    ("z_peak_m", "zpk_m", 8, 4),
    ("z_peak_m_vox", "zpk_m/v", 8, 2),
    ("m2_y0_L", "m2y0L", 7, 2),
    ("m2_y0_R", "m2y0R", 7, 2),
    ("m2_y0_mean", "m2y0m", 7, 2),
    ("m2_x0_L", "m2x0L", 7, 2),
    ("m2_x0_R", "m2x0R", 7, 2),
    ("m2_x0_mean", "m2x0m", 7, 2),
    ("m2_mean", "m2_m", 7, 2),
    ("m2_y0_bins_L", "m2y0bL", 7, None),
    ("m2_y0_bins_R", "m2y0bR", 7, None),
    ("m2_x0_bins_L", "m2x0bL", 7, None),
    ("m2_x0_bins_R", "m2x0bR", 7, None),
    ("m2_y0_empty_L", "m2y0eL", 7, None),
    ("m2_y0_empty_R", "m2y0eR", 7, None),
    ("m2_x0_empty_L", "m2x0eL", 7, None),
    ("m2_x0_empty_R", "m2x0eR", 7, None),
    ("m3_angle", "m3_ang", 7, 2),
    ("m3_bins", "m3_bin", 7, None),
    ("m3_empty", "m3_emp", 7, None),
    ("runout_r", "runout", 8, 4),
    ("med_first_ms", "med_1st", 8, 2),
    ("med_last_ms", "med_lst", 8, 2),
    ("dil", "dil", 7, 3),
    ("sbasis", "sbasis", 6, None),
    ("dM2", "dM2", 7, 2),
    ("nan", "nan", 6, None),
]


def missing_row(prefix: str) -> dict:
    """Placeholder row for a run whose .npz is absent (run failed/timeout); used with --allow-missing."""
    row = {k: None for k, _h, _w, _n in COLUMNS}
    row["prefix"] = prefix
    row["valid"] = "MISSING"
    row["missing"] = True
    return row


def fmt_cell(value, decimals):
    if value is None:
        return "n/a"
    if decimals is None:
        return str(value)
    return f"{value:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", action="append", default=[],
                    help="npz prefix under --log-dir (repeatable, in table order)")
    ap.add_argument("--log-dir", default=None, help="default PROJECT/logs")
    ap.add_argument("--out", default=None, help="output table path (default PROJECT/logs/phase3_6c_table.txt)")
    ap.add_argument("--reference", default="sceneA_D015",
                    help="reference prefix for the explicit re-implementation cross-check")
    ap.add_argument("--one-line", default=None, metavar="PREFIX",
                    help="print one machine-readable line for PREFIX and exit (used by the GPU driver)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit a MISSING placeholder row for a prefix whose .npz is absent instead of aborting")
    args = ap.parse_args()

    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    out_path = args.out or os.path.join(log_dir, "phase3_6c_table.txt")

    if args.one_line is not None:
        d = load(log_dir, args.one_line)
        if d is None:
            raise SystemExit(f"missing {args.one_line}.npz under {log_dir}")
        r = analyze(d, args.one_line, log_dir)
        print(f"ONE_LINE prefix={r['prefix']} dil={r['dil']:.6g} sbasis={r['sbasis']} "
              f"N={r['N']} steps={r['n_steps']} nan={r['nan']} "
              f"med_ms={r['med_step_ms']:.4f} first_ms={r['med_first_ms']:.4f} last_ms={r['med_last_ms']:.4f}")
        return

    if not args.prefix:
        ap.error("--prefix is required (repeatable) unless --one-line is used")

    # Explicit reference cross-check (required).
    ref = load(log_dir, args.reference)
    if ref is None:
        raise SystemExit(f"reference {args.reference}.npz not found under {log_dir}")
    ref_q = ref["final_q"].astype(np.float64) if "final_q" in ref else ref["snap_q"][-1].astype(np.float64)
    cross_check(ref_q, float(ref["voxel"]), args.reference)
    print(f"[cross-check] {args.reference}: detailed re-implementation == mpm_accept.flank_angles "
          f"(both slabs, left/right/mean) within 1e-9 deg")

    rows = []
    for prefix in args.prefix:
        d = load(log_dir, prefix)
        if d is None:
            if not args.allow_missing:
                raise SystemExit(f"missing {prefix}.npz under {log_dir}")
            rows.append(missing_row(prefix))
            continue
        rows.append(analyze(d, prefix, log_dir))

    # dM2: M2 overall mean minus the M2 overall mean of the baseline with the same height factor
    # (baselines c_v015_h10 / c_v015_h25; None for other height factors or if unavailable).
    baseline_m2 = {}
    for hfac, bname in ((1.0, "c_v015_h10"), (2.5, "c_v015_h25")):
        bd = load(log_dir, bname)
        baseline_m2[hfac] = analyze(bd, bname, log_dir)["m2_mean"] if bd is not None else None
    for r in rows:
        r["dM2"] = None
        base = baseline_m2.get(round(float(r["hfac"]), 6)) if r.get("hfac") is not None else None
        if base is not None and r.get("m2_mean") is not None:
            r["dM2"] = r["m2_mean"] - base

    header = "  ".join(f"{h:>{w}}" for _k, h, w, _n in COLUMNS)
    lines = [
        "# Step 6c resolution study -- generated by scripts/mpm_resolution_study.py (CPU only)",
        "# flank angle: mpm_accept.flank_angles (identical detailed re-implementation, cross-checked <=1e-9 deg)",
        "# fit window: upper-envelope bins with z in [0.2,0.8]*z_peak, one-voxel bins, polyfit degree 1",
        "# valid: VALID iff min(z_peak)/voxel >= 6 and min fit bins over all four sides >= 4, else LOW_RES",
        f"# rest_t criterion: mpm_accept.first_rest_time(t, mean_speed, REST_SPEED_TOL={REST_SPEED_TOL:g} m/s): "
        "first time from which mean speed stays below the threshold (mpm_accept.py:23,57-65)",
        "# zpk_vox = min over slabs of z_peak/voxel; sp_* = fit-window horizontal span in voxels; "
        "esc = |x| or |y| > ground_half_xy; bel = z < -1e-3 m",
        "# M2: fixed 0.015 m bins, fixed 0.015 m slab half-width; M3: radial r=sqrt(x^2+y^2) profile",
        "# with fixed 0.015 m bins (square base is not axisymmetric, so M3 averages over azimuth)",
        "# initW/initH = max-min of x and z of snap_q[0]; mass_kg = total particle mass from the mass array",
        "",
        header,
        "-" * len(header),
    ]
    for r in rows:
        cells = [fmt_cell(r.get(k), nd) for k, _h, _w, nd in COLUMNS]
        lines.append("  ".join(f"{c:>{w}}" for c, (_k, _h, w, _n) in zip(cells, COLUMNS)))

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as fh:
        fh.write(text + "\n")
    print(f"\n[resolution-study] wrote {out_path}")


if __name__ == "__main__":
    main()
