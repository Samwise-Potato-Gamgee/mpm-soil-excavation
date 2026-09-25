#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_kill_analyze.py -- CPU-only (numpy) analysis of the .npz files written by mpm_kill_test.py.

Loads the result files from PROJECT/logs/ and prints one verdict table (test | verdict | key numbers |
tolerance).  No Warp, no GPU, no Newton import.

Usage:
    python scripts/mpm_kill_analyze.py --log-dir PROJECT/logs

Tolerances (must match mpm_kill_test.py):
    T1 dead tolerance          : max killed displacement < 1e-6 m
    T2 equivalence             : max|K-R| <= 5% of max|K-C| and of max|R-C|
    T3 revival motion          : revived max displacement > 1e-3 m
    T3 history match           : max |history - expected| < 1e-5
    T5 ignored-without-notify  : mean displacement of flag-cleared-but-not-notified particles > 1e-4 m
"""

from __future__ import annotations

import argparse
import os

import numpy as np

T1_DEAD_TOL = 1e-6
T2_EQUIV_REL_TOL = 0.05
T3_MOVE_TOL = 1e-3
T3_HIST_TOL = 1e-5
T5_IGNORE_TOL = 1e-4


def load(log_dir, name):
    path = os.path.join(log_dir, name + ".npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=False)


def vram_peak_minus_idle(log_dir, name, idle):
    """Peak VRAM (MiB) from <name>_vram.csv minus the pre-experiment idle value."""
    path = os.path.join(log_dir, name + "_vram.csv")
    if not os.path.exists(path):
        return None, None
    vals = [float(x) for x in open(path).read().split() if x.strip() != ""]
    if not vals:
        return None, None
    return max(vals) - idle, max(vals)


def fmt(x, nd=6):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    idle = 0.0
    idle_path = os.path.join(log_dir, "idle_vram.txt")
    if os.path.exists(idle_path):
        idle = float(open(idle_path).read().strip())

    rows = []

    # ---------------- T1 ----------------
    t1 = load(log_dir, "t1_inplace")
    t1p = load(log_dir, "t1_park")
    for tag, d in (("T1a inplace", t1), ("T1b park", t1p)):
        if d is None:
            rows.append((tag, "INCONCLUSIVE", "result file missing", f"<{T1_DEAD_TOL:g} m"))
            continue
        mx = float(d["max_disp"])
        verdict = "PASS" if mx < T1_DEAD_TOL else "FAIL"
        rows.append((tag, verdict, f"max_disp={mx:.3e} m, mean={float(d['mean_disp']):.3e} m", f"<{T1_DEAD_TOL:g} m"))

    # ---------------- T2 (dense primary, sparse to expose ghost-cell effects) ----------------
    def t2_rows(prefix, label):
        out = []
        K, R, C = load(log_dir, prefix + "_K"), load(log_dir, prefix + "_R"), load(log_dir, prefix + "_C")
        if K is None or R is None or C is None:
            return [(f"T2 {label}", "INCONCLUSIVE", f"one of {prefix}_K/R/C missing", "n/a")]
        qK, qR, qC = K["q"], R["q"], C["q"]
        vK, vR, vC = K["v"], R["v"], C["v"]
        diff_KR = float(np.abs(qK - qR).max()) if qK.shape == qR.shape else float("nan")
        diff_KC = float(np.abs(qK - qC).max()) if qK.shape == qC.shape else float("nan")
        diff_RC = float(np.abs(qR - qC).max()) if qR.shape == qC.shape else float("nan")
        ok = (diff_KR <= T2_EQUIV_REL_TOL * max(diff_KC, 1e-12)) and (diff_KR <= T2_EQUIV_REL_TOL * max(diff_RC, 1e-12))
        out.append((
            f"T2 {label} particles", "PASS" if ok else "FAIL",
            f"|K-R|={diff_KR:.3e}, |K-C|={diff_KC:.3e}, |R-C|={diff_RC:.3e} m; "
            f"z_K={qK[:,2].mean():.4f} z_R={qR[:,2].mean():.4f} z_C={qC[:,2].mean():.4f}; "
            f"spd_K={np.linalg.norm(vK,axis=1).mean():.3e} spd_R={np.linalg.norm(vR,axis=1).mean():.3e} "
            f"spd_C={np.linalg.norm(vC,axis=1).mean():.3e}",
            f"|K-R| <= {T2_EQUIV_REL_TOL:.0%} of |K-C|",
        ))
        if all("force_hist" in d and d["force_hist"].size for d in (K, R, C)):
            FK, FR, FC = K["force_hist"], R["force_hist"], C["force_hist"]
            f_KR = float(np.abs(FK - FR).max())
            f_KC = float(np.abs(FK - FC).max())
            f_RC = float(np.abs(FR - FC).max())
            fok = (f_KR <= T2_EQUIV_REL_TOL * max(f_KC, 1e-12)) and (f_KR <= T2_EQUIV_REL_TOL * max(f_RC, 1e-12))
            out.append((
                f"T2 {label} cube wrench", "PASS" if fok else "FAIL",
                f"max|Fw_K-Fw_R|={f_KR:.3e}, max|Fw_K-Fw_C|={f_KC:.3e}, max|Fw_R-Fw_C|={f_RC:.3e}",
                f"<= {T2_EQUIV_REL_TOL:.0%} of K-C",
            ))
        else:
            out.append((f"T2 {label} cube wrench", "INCONCLUSIVE", "no force_hist (non-cube run?)", "n/a"))
        return out

    rows.extend(t2_rows("t2", "dense"))
    if os.path.exists(os.path.join(log_dir, "t2_sparse_K.npz")):
        rows.extend(t2_rows("t2_sparse", "sparse"))

    # ---------------- T3 ----------------
    t3 = load(log_dir, "t3_revive")
    if t3 is None:
        rows.append(("T3 revival", "INCONCLUSIVE", "result file missing", "n/a"))
    else:
        he = float(t3["history_err"])
        mv = float(t3["revived_max_disp"])
        verdict = "PASS" if (he < T3_HIST_TOL and mv > T3_MOVE_TOL) else "FAIL"
        rows.append((
            "T3 revival", verdict,
            f"history_err={he:.3e}, revived_max_disp={mv:.3e} m (mean {float(t3['revived_mean_disp']):.3e})",
            f"hist<{T3_HIST_TOL:g} and move>{T3_MOVE_TOL:g} m",
        ))

    # ---------------- T4 ----------------
    for grid in ("sparse", "dense"):
        for mode in ("none", "inplace", "park"):
            name = f"t4_{grid}_{mode}"
            d = load(log_dir, name)
            if d is None:
                rows.append((f"T4 {grid}/{mode}", "INCONCLUSIVE", "result file missing", "no error"))
                continue
            err = str(d["error"]) if "error" in d else ""
            st = d["step_times"]
            dv, peak = vram_peak_minus_idle(log_dir, name, idle)
            key = (f"step_ms={1e3*float(st.mean()):.2f}, active_cells={int(d['active_cells'])}, "
                   f"peak_vram_delta={fmt(dv,4)} MiB (peak {fmt(peak,5)})")
            if err:
                rows.append((f"T4 {grid}/{mode}", "FAIL", f"{key}; error={err}", "no error"))
            else:
                rows.append((f"T4 {grid}/{mode}", "PASS", key, "no error"))

    # ---------------- T5 ----------------
    t5 = load(log_dir, "t5_cost")
    if t5 is None:
        rows.append(("T5 notify cost", "INCONCLUSIVE", "result file missing", "n/a"))
    else:
        nm = float(t5["notify_median_s"])
        sn = float(t5["step_normal_mean"])
        sy = float(t5["step_notify_mean"])
        disp = float(t5["no_notify_mean_disp"])
        verdict = "PASS" if (nm >= 0.0 and disp > T5_IGNORE_TOL) else "FAIL"
        rows.append((
            "T5 notify cost", verdict,
            f"notify_median={1e3*nm:.3f} ms, step_normal={1e3*sn:.2f} ms, step_with_notify={1e3*sy:.2f} ms, "
            f"no_notify_mean_disp={disp:.3e} m",
            f"no-notify disp >{T5_IGNORE_TOL:g} m",
        ))

    # ---------------- print table ----------------
    w = (26, 13, 95, 30)
    print("=" * (w[0] + w[1] + w[2] + w[3] + 4))
    print(f"{'test':<{w[0]}} {'verdict':<{w[1]}} {'key numbers':<{w[2]}} {'tolerance':<{w[3]}}")
    print("-" * (w[0] + w[1] + w[2] + w[3] + 4))
    for r in rows:
        print(f"{r[0]:<{w[0]}} {r[1]:<{w[1]}} {r[2]:<{w[2]}} {r[3]:<{w[3]}}")
    print("=" * (w[0] + w[1] + w[2] + w[3] + 4))


if __name__ == "__main__":
    main()
