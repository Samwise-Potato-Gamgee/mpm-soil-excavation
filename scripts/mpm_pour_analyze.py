#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_pour_analyze.py -- CPU analysis + pictures for the Step 6d-test2 full pour.

Reads logs/<prefix>.npz produced by scripts/mpm_pour_run.py (only final_q, active_mask and the
every-10-step speed record are needed).  Uses ONLY the ACTIVE particles and the emitter axis
(x = y = 0) as the heap centre.  Computes heap height, M3 radial flank angle, M2 fixed-bin slab
angles, a 90%-30% radial fit, rest time, splash fraction and a mass check, then writes one table to
logs/phase3_6d2_pour_table.txt and four PNGs per run (side, top, 3D, strip of snapshots).

No Warp, no Newton, no GPU: numpy + stdlib + PIL (Pillow).  No plotting library is installed.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False

VOXEL = 0.015
DT = 0.0025
SPACING = 0.0075
GROUND_HALF = 0.6
REST_THRESH = 1.0e-3


# --------------------------------------------------------------------------------------------------
# Angle methods (copied/adapted from scripts/mpm_resolution_study.py)
# --------------------------------------------------------------------------------------------------
def radial_profile(q, center=(0.0, 0.0), bin_width=0.015, frac_lo=0.2, frac_hi=0.8):
    """M3 radial upper-envelope profile with a configurable height window.

    Returns the fit angle, fit bin count, empty bins in the fit window, fit-window horizontal span
    in voxels, the radial peak (z_peak, r_peak), the outermost fit radius r_end, the fit line and
    the full (centers, zmax) envelope.
    """
    r = np.sqrt((q[:, 0] - center[0]) ** 2 + (q[:, 1] - center[1]) ** 2)
    z = q[:, 2]
    edges = np.arange(math.floor(float(r.min()) / bin_width) * bin_width,
                      math.ceil(float(r.max()) / bin_width) * bin_width + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    zmax = np.full(centers.size, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (r >= a) & (r < b)
        if np.any(m):
            zmax[i] = float(z[m].max())
    rec = {"angle": None, "bins": 0, "empty": 0, "span_vox": None, "z_peak": None,
           "r_peak": None, "r_end": None, "slope": None, "intercept": None, "note": "",
           "centers": centers, "zmax": zmax, "sel": None}
    if np.all(np.isnan(zmax)):
        rec["note"] = "no non-empty bins"
        return rec
    peak = int(np.nanargmax(zmax))
    z_peak = float(zmax[peak])
    rec["z_peak"] = z_peak
    rec["r_peak"] = float(centers[peak])
    lo, hi = frac_lo * z_peak, frac_hi * z_peak
    nonempty = ~np.isnan(zmax)
    sel = (np.arange(centers.size) > peak) & nonempty & (zmax >= lo) & (zmax <= hi)
    rec["sel"] = sel
    nb = int(np.count_nonzero(sel))
    rec["bins"] = nb
    if nb < 2:
        rec["note"] = "too few fit bins"
        return rec
    a, b = np.polyfit(centers[sel], zmax[sel], 1)
    rec["angle"] = math.degrees(math.atan(abs(float(a))))
    rec["slope"] = float(a)
    rec["intercept"] = float(b)
    idx = np.nonzero(sel)[0]
    span = np.arange(idx.min(), idx.max() + 1)
    rec["empty"] = int(np.count_nonzero(np.isnan(zmax[span])))
    rec["span_vox"] = float((centers[sel].max() - centers[sel].min()) / bin_width)
    rec["r_end"] = float(centers[sel].max())
    return rec


def m2_slab(q, slab_axis, bin_axis, bin_width=0.015, slab_half=0.015):
    """M2 fixed-bin slab (copy of mpm_resolution_study.m2_slab)."""
    rec = {"left": None, "right": None, "mean": None, "note": "",
           "left_bins": 0, "right_bins": 0, "left_empty": 0, "right_empty": 0}
    slab = q[np.abs(q[:, slab_axis]) < slab_half]
    if slab.shape[0] < 4:
        rec["note"] = "too few particles in slab"
        return rec
    coord = slab[:, bin_axis]
    z = slab[:, 2]
    edges = np.arange(math.floor(float(coord.min()) / bin_width) * bin_width,
                      math.ceil(float(coord.max()) / bin_width) * bin_width + bin_width, bin_width)
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
        return ang, nb, int(np.count_nonzero(np.isnan(zmax[span])))

    left, lb, le = side(np.arange(centers.size) < peak)
    right, rb, re = side(np.arange(centers.size) > peak)
    rec["left"], rec["right"] = left, right
    rec["left_bins"], rec["right_bins"] = lb, rb
    rec["left_empty"], rec["right_empty"] = le, re
    vals = [v for v in (left, right) if v is not None]
    rec["mean"] = float(np.mean(vals)) if vals else None
    return rec


def rest_time_s(check_steps, mean_speed, thresh=REST_THRESH):
    """First time from which the active mean speed stays below thresh (None if never)."""
    ms = np.asarray(mean_speed, dtype=float)
    cs = np.asarray(check_steps, dtype=float)
    if ms.size == 0:
        return None
    ok = ms < thresh
    bad = np.nonzero(~ok)[0]
    if bad.size == 0:
        return float(cs[0]) * DT
    if bad[-1] + 1 >= ms.size:
        return None
    return float(cs[bad[-1] + 1]) * DT


def vram_delta(log_dir, prefix):
    try:
        idle = float(open(os.path.join(log_dir, prefix + "_idle.txt")).read().strip())
    except Exception:
        idle = 0.0
    try:
        vals = [float(x) for x in open(os.path.join(log_dir, prefix + "_vram.csv")).read().split() if x.strip()]
    except Exception:
        return None
    return (max(vals) - idle) if vals else None


def analyze_run(log_dir, prefix):
    d = np.load(os.path.join(log_dir, prefix + ".npz"), allow_pickle=False)
    final_q = d["final_q"].astype(np.float64)
    active = d["active_mask"].astype(bool)
    q = final_q[active]
    n_active = int(q.shape[0])
    r = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)

    m3 = radial_profile(q, (0.0, 0.0), VOXEL, 0.2, 0.8)
    m3b = radial_profile(q, (0.0, 0.0), VOXEL, 0.3, 0.9)
    m2y = m2_slab(q, 1, 0)
    m2x = m2_slab(q, 0, 1)
    m2_means = [m for m in (m2y["mean"], m2x["mean"]) if m is not None]
    m2_mean = float(np.mean(m2_means)) if m2_means else None

    z_peak = m3["z_peak"]
    r_peak = m3["r_peak"]
    runout = float(r.max()) if r.size else float("nan")
    r_end80 = m3["r_end"]
    if r_end80 is not None:
        splash_n = int(np.count_nonzero(r > r_end80 + 0.05))
    else:
        splash_n = 0
    splash_frac = (splash_n / n_active) if n_active else float("nan")

    rest_s = rest_time_s(d["check_steps"], d["check_mean_speed"])
    step = d["step_time_ms"].astype(float)
    med_step = float(np.median(step)) if step.size else float("nan")

    exp_n = int(d["n_layers"]) * int(d["n_per_layer"])
    mass_ok = (n_active == exp_n)

    return {
        "prefix": prefix, "sbasis": str(d["strain_basis"].item()),
        "steps": int(d["total_steps"]), "stop": str(d["stop_reason"].item()),
        "n_active": n_active, "exp_n": exp_n,
        "z_peak": z_peak, "z_peak_vox": (z_peak / VOXEL) if z_peak is not None else None,
        "r_peak": r_peak, "runout": runout,
        "m3": m3, "m3b": m3b, "m2y": m2y, "m2x": m2x, "m2_mean": m2_mean,
        "valid": (m3["bins"] >= 4 and z_peak is not None and z_peak / VOXEL >= 6),
        "rest_s": rest_s, "splash_n": splash_n, "splash_frac": splash_frac,
        "mass_ok": mass_ok, "med_step": med_step, "vram": vram_delta(log_dir, prefix),
        "nan": bool(d["nan"]), "n_below": int(d["n_below"]), "n_outside": int(d["n_outside"]),
        "q": q, "d": d,
    }


# --------------------------------------------------------------------------------------------------
# Pictures (PIL)
# --------------------------------------------------------------------------------------------------
def _colormap(t):
    """t in [0,1] -> RGB via a simple blue->cyan->green->yellow->red ramp."""
    t = float(min(max(t, 0.0), 1.0))
    anchors = [(0.0, (20, 25, 110)), (0.25, (0, 175, 255)), (0.5, (0, 200, 70)),
               (0.75, (255, 220, 0)), (1.0, (215, 30, 0))]
    for (t0, c0), (t1, c1) in zip(anchors[:-1], anchors[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(round(c0[k] + f * (c1[k] - c0[k]))) for k in range(3))
    return anchors[-1][1]


def _dashed(draw, p0, p1, fill, width=1, dash=8, gap=6):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        a = pos
        b = min(pos + dash, length)
        draw.line([(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)], fill=fill, width=width)
        pos += dash + gap


def draw_side(path, q, m3, z_peak, runout):
    W, H = 1600, 700
    R = max(runout * 1.15, 0.25)
    zmax = max((z_peak or 0.1) * 1.25, 0.15)
    scale = min(W / (2.0 * R), H / zmax)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def px(x, z):
        return (W / 2.0 + x * scale, H - z * scale)

    # ground line
    draw.line([px(-R, 0.0), px(R, 0.0)], fill=(0, 0, 0), width=2)
    # particles in the y=0 slab
    slab = q[np.abs(q[:, 1]) < SPACING]
    zz = slab[:, 2]
    z_ref = max(float(np.nanmax(zz)), 1e-9)
    for p in slab:
        x, y, z = p
        c = _colormap(z / z_ref)
        draw.rectangle([px(x, z), (px(x, z)[0] + 1, px(x, z)[1] + 1)], fill=c)
    # M3 fit window on the right flank
    if m3["slope"] is not None and m3["r_end"] is not None and m3["z_peak"] is not None:
        r_fit = m3["centers"][m3["sel"]] if m3["sel"] is not None else None
        if r_fit is not None and r_fit.size:
            p0 = px(float(r_fit.min()), float(m3["slope"] * r_fit.min() + m3["intercept"]))
            p1 = px(float(r_fit.max()), float(m3["slope"] * r_fit.max() + m3["intercept"]))
            draw.line([p0, p1], fill=(200, 0, 200), width=5)
    # dashed 34 deg reference from the heap peak
    if z_peak is not None and m3["r_peak"] is not None:
        x0 = m3["r_peak"]
        x1 = x0 + z_peak / math.tan(math.radians(34.0))
        _dashed(draw, px(x0, z_peak), px(x1, 0.0), (0, 140, 0), width=3)
    draw.text((10, 8), f"side view |y|<{SPACING} m  R={R:.3f} m  z_peak={z_peak if z_peak is None else round(z_peak,4)} m",
              fill=(0, 0, 0))
    draw.text((10, H - 20), "x (m)", fill=(0, 0, 0))
    img.save(path)


def draw_top(path, q, runout):
    W = H = 800
    lim = 0.65
    scale = W / (2.0 * lim)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def px(x, y):
        return (W / 2.0 + x * scale, H / 2.0 - y * scale)

    def bbox(x0, y0, x1, y1):
        a, b = px(x0, y0), px(x1, y1)
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]

    draw.rectangle(bbox(-GROUND_HALF, -GROUND_HALF, GROUND_HALF, GROUND_HALF), outline=(0, 0, 0), width=2)
    if runout and np.isfinite(runout):
        draw.ellipse(bbox(-runout, -runout, runout, runout), outline=(200, 0, 200), width=2)
    z_ref = max(float(q[:, 2].max()), 1e-9)
    for p in q:
        c = _colormap(p[2] / z_ref)
        x, y = px(p[0], p[1])
        draw.rectangle([(x, y), (x + 1, y + 1)], fill=c)
    draw.text((10, 8), f"top view  ground |x|,|y|<={GROUND_HALF} m  run-out={runout:.4f} m", fill=(0, 0, 0))
    img.save(path)


def draw_3d(path, q, rng):
    W, H = 900, 700
    az = math.radians(35.0)
    el = math.radians(25.0)
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)

    def proj(x, y, z):
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        return (yr, z * ce - xr * se, xr * ce + z * se)

    n = q.shape[0]
    if n > 15000:
        idx = rng.choice(n, size=15000, replace=False)
        sub = q[idx]
    else:
        sub = q
    corners = np.array([[-GROUND_HALF, -GROUND_HALF, 0.0], [GROUND_HALF, -GROUND_HALF, 0.0],
                        [GROUND_HALF, GROUND_HALF, 0.0], [-GROUND_HALF, GROUND_HALF, 0.0]])
    pc = np.array([proj(*c) for c in corners])
    ps = np.array([proj(*p) for p in sub]) if sub.size else np.zeros((0, 3))
    allp = np.vstack([pc, ps]) if ps.size else pc
    x0, x1 = allp[:, 0].min(), allp[:, 0].max()
    y0, y1 = allp[:, 1].min(), allp[:, 1].max()
    scale = min((W - 80) / max(x1 - x0, 1e-6), (H - 80) / max(y1 - y0, 1e-6))

    def px(sx, sy):
        return (40 + (sx - x0) * scale, H - 40 - (sy - y0) * scale)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(4):
        a, b = pc[i], pc[(i + 1) % 4]
        draw.line([px(a[0], a[1]), px(b[0], b[1])], fill=(0, 0, 0), width=2)
    z_ref = max(float(q[:, 2].max()), 1e-9)
    if ps.size:
        order = np.argsort(ps[:, 2])[::-1]  # painter: draw far/high first
        for k in order:
            p = ps[k]
            pp = px(p[0], p[1])
            draw.rectangle([pp, (pp[0] + 1, pp[1] + 1)], fill=_colormap(sub[k, 2] / z_ref))
    draw.text((10, 8), f"3D scatter (<=15000 pts)  elev=25 az=35  n_shown={ps.shape[0]}", fill=(0, 0, 0))
    img.save(path)


def draw_strip(path, snap_list, step_list, z_peak, runout):
    npanel = len(snap_list)
    pw, H = 1600 // npanel, 700
    R = max(runout * 1.15, 0.25)
    zmax = max((z_peak or 0.1) * 1.25, 0.15)
    scale = min(pw / (2.0 * R), H / zmax)
    img = Image.new("RGB", (pw * npanel, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for pi, (q, step) in enumerate(zip(snap_list, step_list)):
        ox = pi * pw
        draw.line([(ox + pw / 2.0 - R * scale, H), (ox + pw / 2.0 + R * scale, H)], fill=(0, 0, 0), width=2)
        if q.shape[0]:
            slab = q[np.abs(q[:, 1]) < SPACING]
            z_ref = max(float(slab[:, 2].max()) if slab.shape[0] else 1.0, 1e-9)
            for p in slab:
                sx = ox + pw / 2.0 + p[0] * scale
                sy = H - p[2] * scale
                draw.rectangle([(sx, sy), (sx + 1, sy + 1)], fill=_colormap(p[2] / z_ref))
        draw.text((ox + 6, 8), f"step {step}  t={step*DT:.3f}s", fill=(0, 0, 0))
        if pi:
            draw.line([(ox, 0), (ox, H)], fill=(180, 180, 180), width=1)
    img.save(path)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+", help="run prefixes under --log-dir")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    out_path = args.out or os.path.join(log_dir, "phase3_6d2_pour_table.txt")

    results = [analyze_run(log_dir, p) for p in args.prefixes]
    rng = np.random.default_rng(0)

    hdr = ["run", "sbasis", "steps", "stop", "n_act", "z_peak", "zpk/vox", "r_peak", "runout",
           "m3", "m3bins", "m3empty", "m3span", "valid", "m3_90_30", "m2_y0m", "m2_x0m", "m2_mean",
           "rest_s", "splash_n", "splash_frac", "mass_ok", "n_below", "nan", "med_step", "vram"]
    lines = [
        "# Step 6d-test2 full pour -- scripts/mpm_pour_analyze.py (CPU only, active particles, centre (0,0))",
        "# M3 = radial upper envelope, 0.015 m bins, 20-80% window, fit outward of r_peak; VALID iff >=4 bins and z_peak/voxel>=6",
        "# m3_90_30 = same radial fit with the 90%-30% window (sensitivity check); M2 = fixed 0.015 m bins, 0.015 m slab",
        "",
        "  ".join(f"{h:>9}" for h in hdr),
        "-" * (11 * len(hdr)),
    ]

    def s(x, nd=3):
        return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    for r in results:
        row = [r["prefix"], r["sbasis"], r["steps"], r["stop"], r["n_active"],
               s(r["z_peak"], 4), s(r["z_peak_vox"], 2), s(r["r_peak"], 4), s(r["runout"], 4),
               s(r["m3"]["angle"], 2), r["m3"]["bins"], r["m3"]["empty"], s(r["m3"]["span_vox"], 2),
               "VALID" if r["valid"] else "LOW_RES", s(r["m3b"]["angle"], 2),
               s(r["m2y"]["mean"], 2), s(r["m2x"]["mean"], 2), s(r["m2_mean"], 2),
               s(r["rest_s"], 3), r["splash_n"], s(r["splash_frac"], 4), r["mass_ok"],
               r["n_below"], r["nan"], s(r["med_step"], 3), s(r["vram"], 0)]
        lines.append("  ".join(f"{str(v):>9}" for v in row))

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as fh:
        fh.write(text + "\n")

    # coarse radial envelope (every 4th bin)
    for r in results:
        c = r["m3"]["centers"]
        zm = r["m3"]["zmax"]
        print(f"\n[envelope] {r['prefix']} ({r['sbasis']}): r_centre z_max (every 4th bin)")
        parts = []
        for i in range(0, c.size, 4):
            zv = "nan" if not np.isfinite(zm[i]) else f"{zm[i]:.4f}"
            parts.append(f"({c[i]:.3f},{zv})")
        print("  " + " ".join(parts))
    print(f"\n[envelope] table written to {out_path}")

    if not HAVE_PIL:
        print("[figures] PIL unavailable: no PNG files written")
        return

    for r in results:
        q = r["q"]
        p = r["prefix"]
        draw_side(os.path.join(log_dir, f"pour_{p}_side.png"), q, r["m3"], r["z_peak"], r["runout"])
        draw_top(os.path.join(log_dir, f"pour_{p}_top.png"), q, r["runout"])
        draw_3d(os.path.join(log_dir, f"pour_{p}_3d.png"), q, rng)
        d = r["d"]
        snap_list = [d["snap_q_600"], d["snap_q_1200"], d["snap_q_1800"], d["snap_q_2236"], d["snap_q_end"]]
        step_list = [600, 1200, 1800, 2236, r["steps"]]
        draw_strip(os.path.join(log_dir, f"pour_{p}_strip.png"),
                   [np.asarray(x, dtype=np.float64) for x in snap_list], step_list, r["z_peak"], r["runout"])
        print(f"[figures] wrote pour_{p}_side.png pour_{p}_top.png pour_{p}_3d.png pour_{p}_strip.png")


if __name__ == "__main__":
    main()
