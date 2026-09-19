"""Where each animal's slab actually sits, drawn twice: on that animal's own
image and on the atlas it was registered to.

One row per animal, two panels:

  left   the animal's own 20 um image, sagittal, with the slab boxed. This is
         the cut itself -- the slab is defined on this grid, so the box is
         exact.
  right  the atlas variant that animal was registered to, sagittal, with the
         same slab boxed. The warp is not rigid, so a plane of the sample is
         not a plane of the atlas: this box is the anterior-posterior range the
         slab's cells land in (1st to 99th percentile of their atlas
         coordinate), which is an approximation, and the residual of the
         straight-line fit behind it is printed on the panel.

Both sections are taken at the medio-lateral plane holding the most of the
structure that positioned the slab, because that is the plane the placement was
decided on. Yellow is the territory the run counts on, blue is that positioning
structure -- it is not counted unless it happens to lie inside the territory.

Panels are flipped so dorsal is up wherever the isocortex comes out below the
middle of the tissue: s10 was acquired the other way round (orientation
-1, 3, -2) and would otherwise sit upside down beside the other five.

    conda activate antsreg
    python -m stats.plot_slab --run /data/hdd12tb-1/fengyi/COMBINe/stats/0918_07_xsec_iso_isofrac30
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables, plot_style as ps  # noqa: E402
from stats.laminar import (ISOCORTEX_ID, cover_ids,  # noqa: E402
                           _labels_in_sample_path)
from stats.ontology import Ontology  # noqa: E402

# The atlas tif's axes, which are not the sample grid's: (DV, AP, ML).
ATLAS_AP, ATLAS_ML = 1, 2
# Enough cells to fit a straight line through a warp; reading all twelve classes
# for a picture is not worth the minutes.
FIT_CLASSES = ("neuron_GFP", "neuron_RFP", "glia_GFP", "glia_RFP")


def _find(d, suffix):
    hits = sorted(f for f in os.listdir(d) if f.endswith(suffix))
    return os.path.join(d, hits[0]) if hits else None


def atlas_paths(sample_dir, atlas_dir):
    """-> (annotation, reference) for the variant this sample was registered to.

    The variant is written into the file name as orientation + slicing, exactly
    as the sample's own yaml states them, so the name is rebuilt rather than
    guessed: two variants of the same atlas do not hold the same structures (the
    hemispheres of DeMBA P5 are not mirror images)."""
    cfg = yaml.safe_load(open(sorted(glob.glob(os.path.join(sample_dir, "*.yaml")))[0]))
    src = (cfg.get("atlas") or {}).get("source", "demba_p5")
    v = (cfg.get("atlas_variants") or {}).get(src, {})
    o = "_".join(str(int(x)) for x in v["orientation"])
    sl = "_".join("full" if s is None else f"{int(s[0])}-{int(s[1])}" for s in v["slicing"])
    stem = f"DeMBA_P5_{{}}_{o}__{sl}__pad20.tif"
    ann = os.path.join(atlas_dir, stem.format("annotation"))
    ref = os.path.join(atlas_dir, stem.format("reference"))
    if not os.path.exists(ann):
        raise FileNotFoundError(f"没有这个图谱变体：{ann}")
    return ann, (ref if os.path.exists(ref) else None)


def sample_panel(sample_dir, root_ids, cov):
    import nibabel as nib

    lab = np.asarray(nib.load(_labels_in_sample_path(sample_dir)).dataobj).astype(np.int64)
    ip = _find(sample_dir, "fine_20um.nii.gz")
    img = np.asarray(nib.load(ip).dataobj).astype(np.float32) if ip else None
    cm = np.isin(lab, cov)
    ml = int(np.argmax(cm.sum(axis=(1, 2))))          # grid is (ML, AP, DV)
    rm = np.isin(lab, root_ids)
    return (None if img is None else img[ml].T, rm[ml].T, cm[ml].T)


def atlas_panel(ann_path, ref_path, root_ids, cov):
    import tifffile

    ann = tifffile.imread(ann_path)
    ref = tifffile.imread(ref_path).astype(np.float32) if ref_path else None
    cm = np.isin(ann, cov)
    ml = int(np.argmax(cm.sum(axis=(0, ATLAS_AP))))    # atlas is (DV, AP, ML)
    rm = np.isin(ann, root_ids)
    return (None if ref is None else ref[:, :, ml], rm[:, :, ml], cm[:, :, ml])


def slab_in_atlas(sample_dir, root_ids, lo, hi):
    """-> (yt_lo, yt_hi, residual_um) for the cells the slab holds."""
    frames = []
    for c in FIT_CLASSES:
        p = os.path.join(sample_dir, "cell_registration", c, "cell_registration.csv")
        if os.path.exists(p):
            frames.append(cell_tables.read_cell_registration(p))
    if not frames:
        return np.nan, np.nan, np.nan
    d = pd.concat(frames, ignore_index=True)
    d = d[d["region_id"].isin(root_ids)]
    for c in ("yr", "yt"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["yr", "yt"])
    if d.empty:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(d["yr"].to_numpy(), d["yt"].to_numpy(), 1)
    resid = float(np.std(d["yt"].to_numpy() - (a + b * d["yr"].to_numpy()))) * 20.0
    inside = d[(d["yr"] >= lo) & (d["yr"] <= hi)]
    if inside.empty:
        return np.nan, np.nan, resid
    return (float(inside["yt"].quantile(0.01)), float(inside["yt"].quantile(0.99)), resid)


def _dorsal_up(img, root, cov):
    """s10 was acquired with dorsal down (orientation -1, 3, -2), so its panels
    come out upside down next to the other five. Flip whichever panels put the
    isocortex below the middle of the tissue: the cortex is the dorsal shell."""
    tissue = root | cov
    if img is not None:
        tissue = tissue | (img > np.percentile(img[img > 0], 40) if (img > 0).any() else tissue)
    rows = np.arange(root.shape[0])
    if not root.any() or not tissue.any():
        return img, root, cov
    if (rows[:, None] * root).sum() / root.sum() > (rows[:, None] * tissue).sum() / tissue.sum():
        f = lambda v: None if v is None else v[::-1]
        return f(img), f(root), f(cov)
    return img, root, cov


def _crop(img, root, cov, lo, hi, pad=6):
    """Trim the empty frame around the brain so the panels are the same kind of
    picture; never trim into the slab."""
    tissue = root | cov
    if img is not None and (img > 0).any():
        tissue = tissue | (img > 0)
    if not tissue.any():
        return img, root, cov, lo, hi
    r = np.flatnonzero(tissue.any(axis=1))
    c = np.flatnonzero(tissue.any(axis=0))
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad + 1, tissue.shape[0])
    c0 = max(min(c.min(), int(np.floor(lo))) - pad, 0)
    c1 = min(max(c.max(), int(np.ceil(hi))) + pad + 1, tissue.shape[1])
    f = lambda v: None if v is None else v[r0:r1, c0:c1]
    return f(img), f(root), f(cov), lo - c0, hi - c0


def draw(ax, img, root, cov, lo, hi, colour, box_label):
    from matplotlib.patches import Rectangle

    img, root, cov = _dorsal_up(img, root, cov)
    img, root, cov, lo, hi = _crop(img, root, cov, lo, hi)
    if img is not None:
        v = img[img > 0]
        top = np.percentile(v, 99.5) if v.size else 1.0
        ax.imshow(np.clip(img / max(top, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1,
                  origin="upper", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~root, np.ones_like(root, dtype=float)), cmap="autumn",
              alpha=0.40, origin="upper", interpolation="nearest", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~cov, np.zeros_like(cov, dtype=float)), cmap="winter",
              alpha=0.55, origin="upper", interpolation="nearest", vmin=0, vmax=1)
    if np.isfinite(lo) and np.isfinite(hi):
        h = root.shape[0] - 1
        ax.add_patch(Rectangle((lo, 0), hi - lo + 1, h, facecolor=colour, alpha=0.13, zorder=4))
        ax.add_patch(Rectangle((lo, 0), hi - lo + 1, h, fill=False, edgecolor=colour,
                               linewidth=2.2, zorder=5))
    ax.set_title(box_label, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--atlas-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    atlas_dir = args.atlas_dir or os.path.join(here, "atlas", "DeMBA")

    cfg = yaml.safe_load(open(os.path.join(args.run, "config_used.yaml")))
    qc = pd.read_csv(os.path.join(args.run, "laminar_slab_qc.csv"))
    if qc["lo"].isna().all():
        raise SystemExit(f"{args.run} 没有切片板，没什么可画的")
    lam = cfg.get("laminar") or {}
    slab = lam.get("slab") or {}
    if slab.get("axis", "yr") != "yr":
        raise SystemExit("这张图画的是冠状板，需要 slab.axis: yr")

    ont = Ontology.from_json(_ontology_path(cfg, args.run))
    root_id = int(lam.get("root_id", ISOCORTEX_ID))
    root_ids = cover_ids(ont, [root_id])
    cov_names = [str(c) for c in (slab.get("cover") or [root_id])]
    cov = cover_ids(ont, cov_names)
    root_acr = ont.acronyms[int(ont.order_of_ids([root_id])[0])]

    ga, gb = cfg["groups"]["a"], cfg["groups"]["b"]
    order = list(ga["samples"]) + list(gb["samples"])
    gof = {s: "a" for s in ga["samples"]}
    gof.update({s: "b" for s in gb["samples"]})

    fig, axes = plt.subplots(len(order), 2, figsize=(11.5, 2.55 * len(order)), squeeze=False)
    for r, s in enumerate(order):
        row = qc[qc["sample"] == s].iloc[0]
        sdir = cfg["samples"][s]["dir"]
        colour = ps.GROUP_COLORS[gof[s]]
        lo, hi = float(row["lo"]), float(row["hi"])

        img, rm, cm = sample_panel(sdir, root_ids, cov)
        draw(axes[r][0], img, rm, cm, lo, hi, colour,
             f"{s} — own image · yr {lo:.0f}-{hi:.0f} · {row['thickness_um']:.0f} um · "
             f"{row['root_volume_in_slab_mm3']:.2f} mm3 {root_acr} · "
             f"{int(row['cells_in_slab']):,} cells")

        ann, ref = atlas_paths(sdir, atlas_dir)
        alo, ahi, resid = slab_in_atlas(sdir, root_ids, lo, hi)
        aimg, arm, acm = atlas_panel(ann, ref, root_ids, cov)
        draw(axes[r][1], aimg, arm, acm, alo, ahi, colour,
             f"{s} — atlas {os.path.basename(ann).split('annotation_')[1][:12]} · "
             f"yt {alo:.0f}-{ahi:.0f} (approx, fit residual {resid:.0f} um)")
        axes[r][0].set_ylabel(s, fontsize=10, color=colour, fontweight="bold")

    fig.suptitle(
        f"Sampled range, per animal — left: the animal's own image (exact) · "
        f"right: its atlas variant (approximate)\n"
        f"sagittal at the plane richest in {'+'.join(cov_names)} · "
        f"yellow = {root_acr} (counted) · blue = {'+'.join(cov_names)} "
        f"(positions the slab only) · "
        f"anterior left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.30 / len(order)))
    out = args.out or os.path.join(args.run, "figures", "00_slab_sagittal.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ps.savefig(fig, out)


def _ontology_path(cfg, run_dir):
    p = cfg["ontology_json"]
    if os.path.isabs(p) and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (os.path.join(here, "stats", "configs"), here, run_dir):
        c = os.path.normpath(os.path.join(base, p))
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"找不到 ontology_json: {p}")


if __name__ == "__main__":
    main()
