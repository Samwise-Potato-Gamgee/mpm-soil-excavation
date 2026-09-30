#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
mpm_density_check.py -- CPU-only density analysis of saved MPM result files.

For each selected run, using the FINAL particle positions and that run's voxel size v (spacing
s = v/2, one particle at 1700 kg/m^3 fills s^3, a fully packed voxel holds 8 particles):

  1. particles per occupied cell of size v, 2v, 4v (mean/median/5th/95th percentile, fraction of
     occupied cells with <4 particles, and the same for "interior" cells whose 26 neighbours are
     all occupied);
  2. occupied-volume estimates (a) occupied v-cells * v^3, (b) occupied 2v-cells * (2v)^3,
     (c) the height-envelope volume (max z per v-column + 0.5*s, times v^2), (d) nominal
     N * s^3, with the ratios a/d, b/d, c/d;
  3. the same for the initial state when the file stores it (snap_q[0]);
  4. a synthetic calibration on a perfect lattice block and the same block jittered by +-20% of a
     spacing;
  5. a vertical density profile for the two pour runs.

Deterministic, numpy + stdlib + PIL only.  Writes logs/phase3_density_table.txt and
logs/phase3_density.npz, plus optional logs/density_slice_<run>.png.
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

LOG = os.path.join(os.path.dirname(__file__), "..", "logs")
EPS = 1.0e-4  # added to pos/v before floor to keep lattice points on cell boundaries stable

# name, kind, strain/extra label
RUNS = [
    ("pour_P0", "pour", "P0"),
    ("pour_P1d", "pour", "P1d"),
    ("c_v015_h10", "collapse", "P0 h1.0 baseline"),
    ("e_P1d_h10", "collapse", "P1d h1.0"),
    ("e_Q1_h10", "collapse", "Q1 h1.0"),
    ("e_dil03_h10", "collapse", "P0 h1.0 dil0.3"),
    ("e_dil10_h10", "collapse", "P0 h1.0 dil1.0"),
    ("c_v015_h25", "collapse", "P0 h2.5 baseline"),
    ("e_P1d_h25", "collapse", "P1d h2.5"),
    ("c_v030_h10", "collapse", "P0 h1.0 v0.03"),
    ("c_v019_h10", "collapse", "P0 h1.0 v0.01875"),
    ("c_v010_h10", "collapse", "P0 h1.0 v0.01"),
]


# --------------------------------------------------------------------------------------------------
def occupied(pos, cell):
    idx = np.floor(pos / cell + EPS).astype(np.int64)
    cells, counts = np.unique(idx, axis=0, return_counts=True)
    return cells, counts.astype(np.int64)


def interior_mask(cells):
    """True for cells whose 26 neighbours are all occupied."""
    cs = set(map(tuple, cells.tolist()))
    out = np.zeros(cells.shape[0], dtype=bool)
    for i, c in enumerate(cells.tolist()):
        c0, c1, c2 = c
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if (c0 + dx, c1 + dy, c2 + dz) not in cs:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        out[i] = ok
    return out


def cell_stats(pos, cell):
    cells, counts = occupied(pos, cell)
    n_occ = int(cells.shape[0])
    if n_occ == 0:
        return dict(n_occ=0, med=float("nan"), mean=float("nan"), p5=float("nan"), p95=float("nan"),
                    frac_lt4=float("nan"), n_int=0, int_med=float("nan"), int_mean=float("nan"),
                    int_frac_lt4=float("nan"))
    im = interior_mask(cells)
    n_int = int(im.sum())
    return dict(
        n_occ=n_occ,
        med=float(np.median(counts)), mean=float(counts.mean()),
        p5=float(np.percentile(counts, 5)), p95=float(np.percentile(counts, 95)),
        frac_lt4=float(np.mean(counts < 4)),
        n_int=n_int,
        int_med=float(np.median(counts[im])) if n_int else float("nan"),
        int_mean=float(counts[im].mean()) if n_int else float("nan"),
        int_frac_lt4=float(np.mean(counts[im] < 4)) if n_int else float("nan"),
    )


def envelope_volume(pos, cell, spacing):
    idx = np.floor(pos[:, :2] / cell + EPS).astype(np.int64)
    cols, inv = np.unique(idx, axis=0, return_inverse=True)
    zmax = np.full(cols.shape[0], -np.inf)
    np.maximum.at(zmax, inv, pos[:, 2])
    return float(np.sum(zmax + 0.5 * spacing) * cell * cell)


def volume_estimates(pos, v, s):
    n = pos.shape[0]
    nominal = n * s ** 3
    a = int(occupied(pos, v)[0].shape[0]) * v ** 3
    b = int(occupied(pos, 2.0 * v)[0].shape[0]) * (2.0 * v) ** 3
    c = envelope_volume(pos, v, s)
    return dict(n=n, nominal=nominal, va=a, vb=b, vc=c,
                ra=a / nominal if nominal else float("nan"),
                rb=b / nominal if nominal else float("nan"),
                rc=c / nominal if nominal else float("nan"))


def vertical_profile(pos, v, s):
    z = pos[:, 2]
    k = np.floor(z / v + EPS).astype(np.int64)
    ks = np.unique(k)
    zs, cnt, frac = [], [], []
    for kk in ks:
        m = k == kk
        cc = int(m.sum())
        ncol = int(np.unique(np.floor(pos[m, :2] / v + EPS).astype(np.int64), axis=0).shape[0])
        zs.append((float(kk) + 0.5) * v)
        cnt.append(cc)
        frac.append(cc / (8.0 * ncol) if ncol else float("nan"))
    return np.asarray(zs), np.asarray(cnt, dtype=np.int64), np.asarray(frac)


def make_lattice(nx, ny, nz, s):
    xs = np.arange(nx + 1) * s
    ys = np.arange(ny + 1) * s
    zs = 0.5 * s + np.arange(nz + 1) * s
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def min_nn_distance(pos, v):
    """Minimum nearest-neighbour distance (m) via a v-cell bucket search over 27 neighbours."""
    idx = np.floor(pos / v + EPS).astype(np.int64)
    buckets: dict = {}
    for i, c in enumerate(idx.tolist()):
        buckets.setdefault(tuple(c), []).append(i)
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    mind = float("inf")
    for c, ii in buckets.items():
        pts = pos[ii]
        neigh = [pos[buckets[(c[0] + o[0], c[1] + o[1], c[2] + o[2])]]
                 for o in offs if (c[0] + o[0], c[1] + o[1], c[2] + o[2]) in buckets]
        if not neigh:
            continue
        nb = np.concatenate(neigh, axis=0)
        d = np.sqrt(((pts[:, None, :] - nb[None, :, :]) ** 2).sum(-1))
        d[d < 1.0e-9] = np.inf
        m = float(d.min())
        if m < mind:
            mind = m
    return mind


# --------------------------------------------------------------------------------------------------
def load_run(name):
    d = np.load(os.path.join(LOG, name + ".npz"), allow_pickle=False)
    v = float(d["voxel"])
    s = float(d["spacing"])
    q = d["final_q"].astype(np.float64)
    n_all = q.shape[0]
    if "active_mask" in d.files:
        active = d["active_mask"].astype(bool)
        q = q[active]
        note = f"active_mask ({int(active.sum())}/{n_all})"
    else:
        note = f"all {n_all}"
    init = None
    if "snap_q" in d.files:
        init = d["snap_q"][0].astype(np.float64)
    return dict(name=name, v=v, s=s, q=q, init=init, note=note)


def fmt_row(cols, widths):
    return "  ".join(f"{str(c):>{w}}" for c, w in zip(cols, widths))


def main() -> None:
    os.makedirs(LOG, exist_ok=True)
    out_lines: list[str] = []
    results = {}

    def emit(s):
        out_lines.append(s)

    # ---------------- per-run statistics ----------------
    run_stats = {}
    for name, kind, label in RUNS:
        r = load_run(name)
        st = {}
        for c, tag in ((r["v"], "v"), (2 * r["v"], "2v"), (4 * r["v"], "4v")):
            st[tag] = cell_stats(r["q"], c)
        vol = volume_estimates(r["q"], r["v"], r["s"])
        init_st = init_vol = None
        init_nn = None
        if r["init"] is not None:
            init_st = {tag: cell_stats(r["init"], c)
                       for c, tag in ((r["v"], "v"), (2 * r["v"], "2v"), (4 * r["v"], "4v"))}
            init_vol = volume_estimates(r["init"], r["v"], r["s"])
            init_nn = min_nn_distance(r["init"], r["v"])
        nn = min_nn_distance(r["q"], r["v"])
        prof = None
        if kind == "pour":
            prof = vertical_profile(r["q"], r["v"], r["s"])
        run_stats[name] = dict(r=r, kind=kind, label=label, st=st, vol=vol,
                               init_st=init_st, init_vol=init_vol, prof=prof, nn=nn, init_nn=init_nn)
        results[name] = dict(v=r["v"], s=r["s"], n=r["q"].shape[0], note=r["note"],
                             vol=vol, st=st, init_st=init_st, init_vol=init_vol, prof=prof,
                             nn=nn, init_nn=init_nn)
        print(f"loaded {name}: v={r['v']} N={r['q'].shape[0]} ({r['note']})")

    # ---------------- Table 1: identity ----------------
    emit("")
    emit("TABLE 1  run identity (final state used; pour uses emitted particles via active_mask)")
    hdr = ["run", "kind", "label", "voxel", "spacing", "N_used", "N_kept", "note"]
    W = [16, 9, 20, 8, 9, 8, 8, 22]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, kind, label in RUNS:
        s = run_stats[name]
        emit(fmt_row([name, kind, label, f"{s['r']['v']:.5f}", f"{s['r']['s']:.5f}",
                      s["r"]["q"].shape[0], results[name]["n"] - s["r"]["q"].shape[0],
                      results[name]["note"]], W))

    # ---------------- Table 2: per-cell at v ----------------
    emit("")
    emit("TABLE 2  particles per occupied cell at cell size v (final); <4 = fraction of occupied cells with <4")
    hdr = ["run", "n_cell_v", "mean", "median", "p5", "p95", "frac<4", "n_interior", "int_mean", "int_median", "int_frac<4"]
    W = [16, 9, 7, 7, 7, 7, 8, 10, 9, 10, 10]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, _k, _l in RUNS:
        s = run_stats[name]["st"]["v"]
        emit(fmt_row([name, s["n_occ"], f"{s['mean']:.3f}", f"{s['med']:.1f}", f"{s['p5']:.1f}",
                      f"{s['p95']:.1f}", f"{s['frac_lt4']:.3f}", s["n_int"], f"{s['int_mean']:.3f}",
                      f"{s['int_med']:.1f}", f"{s['int_frac_lt4']:.3f}"], W))

    # ---------------- Table 3: per-cell at 2v, 4v ----------------
    emit("")
    emit("TABLE 3  particles per occupied cell at 2v and 4v (final)")
    hdr = ["run", "2v_n", "2v_med", "2v_intmed", "2v_frac<4", "2v_int_frac<4",
           "4v_n", "4v_med", "4v_intmed", "4v_frac<4", "4v_int_frac<4"]
    W = [16, 7, 7, 9, 9, 11, 7, 7, 9, 9, 11]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, _k, _l in RUNS:
        a = run_stats[name]["st"]["2v"]
        b = run_stats[name]["st"]["4v"]
        emit(fmt_row([name, a["n_occ"], f"{a['med']:.1f}", f"{a['int_med']:.1f}", f"{a['frac_lt4']:.3f}",
                      f"{a['int_frac_lt4']:.3f}", b["n_occ"], f"{b['med']:.1f}", f"{b['int_med']:.1f}",
                      f"{b['frac_lt4']:.3f}", f"{b['int_frac_lt4']:.3f}"], W))

    # ---------------- Table 4: volumes final ----------------
    emit("")
    emit("TABLE 4  occupied-volume estimates (final): a=sum v-cells*v^3, b=sum 2v-cells*(2v)^3,")
    emit("         c=height-envelope, d=nominal=N*s^3; ratios a/d, b/d, c/d")
    hdr = ["run", "a", "b", "c", "d", "a/d", "b/d", "c/d"]
    W = [16, 11, 11, 11, 11, 7, 7, 7]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, _k, _l in RUNS:
        v = run_stats[name]["vol"]
        emit(fmt_row([name, f"{v['va']:.6f}", f"{v['vb']:.6f}", f"{v['vc']:.6f}", f"{v['nominal']:.6f}",
                      f"{v['ra']:.3f}", f"{v['rb']:.3f}", f"{v['rc']:.3f}"], W))

    # ---------------- Table 5: initial state ----------------
    emit("")
    emit("TABLE 5  initial state (snap_q[0]) where available: v-cell median and interior median, a/d and c/d")
    hdr = ["run", "init_N", "init_med_v", "init_intmed_v", "init_a/d", "init_c/d"]
    W = [16, 8, 11, 14, 10, 10]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, _k, _l in RUNS:
        iv = run_stats[name]["init_vol"]
        ist = run_stats[name]["init_st"]
        if iv is None:
            emit(fmt_row([name, "n/a", "n/a", "n/a", "n/a", "n/a"], W))
        else:
            emit(fmt_row([name, iv["n"], f"{ist['v']['med']:.1f}", f"{ist['v']['int_med']:.1f}",
                          f"{iv['ra']:.3f}", f"{iv['rc']:.3f}"], W))

    # ---------------- Table 6: synthetic calibration ----------------
    emit("")
    emit("TABLE 6  synthetic calibration: perfect lattice block 41x41x21 at v=0.015 (N=35301) and same jittered by +-20% s")
    cal = {}
    for label, pos in (("perfect", make_lattice(40, 40, 20, 0.0075).astype(np.float64)),
                       ("jitter20", None)):
        if pos is None:
            base = make_lattice(40, 40, 20, 0.0075).astype(np.float64)
            jr = np.random.default_rng(0).uniform(-0.2 * 0.0075, 0.2 * 0.0075, size=base.shape)
            pos = base + jr
        st = {tag: cell_stats(pos, c) for c, tag in ((0.015, "v"), (0.03, "2v"), (0.06, "4v"))}
        vol = volume_estimates(pos, 0.015, 0.0075)
        cal[label] = dict(st=st, vol=vol)
        emit(f"  {label:9s}  v: med={st['v']['med']:.1f} int_med={st['v']['int_med']:.1f} "
             f"int_frac<4={st['v']['int_frac_lt4']:.3f} | 2v: med={st['2v']['med']:.1f} "
             f"int_med={st['2v']['int_med']:.1f} | 4v: med={st['4v']['med']:.1f} "
             f"int_med={st['4v']['int_med']:.1f} | a/d={vol['ra']:.3f} b/d={vol['rb']:.3f} c/d={vol['rc']:.3f}")

    # ---------------- Table 7: vertical profiles (pours) ----------------
    emit("")
    emit("TABLE 7  vertical density profiles (pour runs): z, count, n_columns, frac=count/(8*n_columns)")
    for name in ("pour_P0", "pour_P1d"):
        p = run_stats[name]["prof"]
        z, c, f = p
        emit(f"  {name}: {z.size} slabs")
        for i in range(z.size):
            emit(f"    z={z[i]:.4f}  count={int(c[i]):5d}  ncol={int(round(c[i]/(8*f[i]))) if f[i] else 0:4d}  frac={f[i]:.3f}")

    # ---------------- Table 8: nearest-neighbour diagnostic ----------------
    emit("")
    emit("TABLE 8  nearest-neighbour distance (m): min over particles; min/v vs 1.0 = nominal lattice")
    hdr = ["run", "min_nn_final", "min_nn/v", "min_nn_init", "init_nn/v"]
    W = [16, 13, 10, 12, 11]
    emit(fmt_row(hdr, W))
    emit("-" * (sum(W) + 2 * (len(W) - 1)))
    for name, _k, _l in RUNS:
        s = run_stats[name]
        ini = "n/a" if s["init_nn"] is None else f"{s['init_nn']:.6f}"
        inir = "n/a" if s["init_nn"] is None else f"{s['init_nn']/s['r']['s']:.4f}"
        emit(fmt_row([name, f"{s['nn']:.6f}", f"{s['nn']/s['r']['s']:.4f}", ini, inir], W))

    text = "\n".join(out_lines) + "\n"
    with open(os.path.join(LOG, "phase3_density_table.txt"), "w") as fh:
        fh.write(text)
    print(text)

    # ---------------- npz ----------------
    save = {}
    for name in results:
        r = results[name]
        save[f"{name}__v"] = r["v"]
        save[f"{name}__s"] = r["s"]
        save[f"{name}__n"] = r["n"]
        save[f"{name}__note"] = r["note"]
        for tag in ("v", "2v", "4v"):
            for k in ("n_occ", "med", "mean", "p5", "p95", "frac_lt4", "n_int", "int_med",
                      "int_mean", "int_frac_lt4"):
                save[f"{name}__{tag}_{k}"] = r["st"][tag][k]
        for k in ("nominal", "va", "vb", "vc", "ra", "rb", "rc"):
            save[f"{name}__{k}"] = r["vol"][k]
        if r["init_vol"] is not None:
            for k in ("nominal", "va", "vb", "vc", "ra", "rb", "rc"):
                save[f"{name}__init_{k}"] = r["init_vol"][k]
            for tag in ("v", "2v", "4v"):
                save[f"{name}__init_{tag}_med"] = r["init_st"][tag]["med"]
                save[f"{name}__init_{tag}_int_med"] = r["init_st"][tag]["int_med"]
        if r["prof"] is not None:
            save[f"{name}__prof_z"], save[f"{name}__prof_count"], save[f"{name}__prof_frac"] = r["prof"]
        save[f"{name}__min_nn"] = r["nn"]
        if r["init_nn"] is not None:
            save[f"{name}__init_min_nn"] = r["init_nn"]
    for label, c in cal.items():
        for tag in ("v", "2v", "4v"):
            for k in ("med", "int_med", "int_frac_lt4"):
                save[f"synth_{label}__{tag}_{k}"] = c["st"][tag][k]
        for k in ("ra", "rb", "rc"):
            save[f"synth_{label}__{k}"] = c["vol"][k]
    np.savez_compressed(os.path.join(LOG, "phase3_density.npz"), **save)

    # ---------------- pictures ----------------
    if HAVE_PIL:
        for name in ("pour_P0", "pour_P1d", "c_v015_h10"):
            r = run_stats[name]
            pos = r["r"]["q"]
            v = r["r"]["v"]
            idx = np.floor(pos / v + EPS).astype(np.int64)
            cells, counts = np.unique(idx, axis=0, return_counts=True)
            cmap = {tuple(c): int(n) for c, n in zip(cells.tolist(), counts)}
            per = np.array([cmap[tuple(c)] for c in idx.tolist()], dtype=float)
            yc = float(np.median(pos[:, 1])) if name.startswith("pour") else 0.0
            slab = np.abs(pos[:, 1] - yc) < r["r"]["s"]
            W, H = 1500, 700
            xr = max(0.45, float(np.abs(pos[:, 0]).max()) * 1.1)
            zr = max(0.20, float(pos[:, 2].max()) * 1.1)
            sc = min(W / (2 * xr), H / zr)
            img = Image.new("RGB", (W, H), (255, 255, 255))
            dr = ImageDraw.Draw(img)
            x0, z0 = W / 2.0, H
            for p, cnt in zip(pos[slab], per[slab]):
                if cnt <= 8:
                    col = (255, int(255 * cnt / 8.0), 0)
                else:
                    col = (0, 255, min(255, int(255 * (cnt - 8) / 8.0)))
                px, py = x0 + p[0] * sc, z0 - p[2] * sc
                dr.rectangle([(px, py), (px + 2, py + 2)], fill=col)
            dr.line([(x0 - xr * sc, z0), (x0 + xr * sc, z0)], fill=(0, 0, 0), width=2)
            dr.text((8, 8), f"{name}: |y-{yc:.3f}|<{r['r']['s']}  colour=particles per v-cell (green~8)", fill=(0, 0, 0))
            img.save(os.path.join(LOG, f"density_slice_{name}.png"))

    print(f"\nwrote {os.path.join(LOG, 'phase3_density_table.txt')} and phase3_density.npz")


if __name__ == "__main__":
    main()
