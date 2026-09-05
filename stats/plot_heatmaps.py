"""Paint group differences onto atlas sections -- coronal and sagittal.

Reads region_stats.csv, colours every region by a chosen statistic at a chosen
ontology level, and renders a row of coronal and a row of sagittal sections.

What the colours mean, and what they do not:

* The value is per REGION, not per voxel. A region is a flat patch of colour;
  the map shows where a difference sits anatomically, not any within-region
  structure. It is not a voxel-wise statistical map and must not be read as one.
* Every voxel reads out its ancestor at the chosen level (see
  region_maps.RegionVolume.paint), so a level-5 map paints CA1's voxels with
  HPF's value.
* Regions with no row for that (level, class, metric) are left grey: not
  tested, filtered out by coverage or minimum count, or -- under gatekeeping --
  never reached because an ancestor was not significant. Grey means "no
  result", never "no difference".
* With `--significant-only` (the default), only regions passing p_adj < alpha
  are coloured. Levels configured as uncorrected are EXPLORATORY: the script
  marks those panels in the title, because a coloured region there has not
  survived any correction.

Usage:
    conda activate antsreg
    python -m stats.plot_heatmaps --config stats/configs/group_analysis.yaml \\
        --class-name GFP_any --metric Density --level 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import region_maps  # noqa: E402
from stats.group_stats import load_config  # noqa: E402

VALUE_COLUMNS = {
    "log2fc": ("log2 fold change (B/A)", "RdBu_r", True),
    "hedges_g": ("Hedges' g (B - A)", "RdBu_r", True),
    "neglog10p": ("-log10 p (uncorrected)", "viridis", False),
    "mean_a": ("group A mean", "viridis", False),
    "mean_b": ("group B mean", "viridis", False),
}


def section_positions(mask, axis, n):
    """`n` evenly spaced slice indices spanning the part of the axis that
    actually contains brain, so panels are not wasted on empty ends."""
    present = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    if len(present) == 0:
        raise ValueError("annotation is empty")
    lo, hi = present[0], present[-1]
    # inset by half a step so the first and last panel are not the very edge
    edges = np.linspace(lo, hi, n + 2)[1:-1]
    return np.round(edges).astype(int)


def draw(stats_df, ontology, region_vol, args, out_path, ml_offset=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label, cmap, diverging = VALUE_COLUMNS[args.value]
    col = args.value
    df = stats_df.copy()
    if col == "neglog10p":
        col = "_neglog10p"
        with np.errstate(divide="ignore"):
            df[col] = -np.log10(df["p_value"].clip(lower=1e-300))

    vals, n_regions = region_maps.values_by_order(
        df, ontology, args.level, args.class_name, args.metric, col,
        significant_only=args.significant_only, alpha=args.alpha)
    if n_regions == 0:
        print(f"  [warn] no rows for level={args.level} class={args.class_name} "
              f"metric={args.metric}"
              + (" passing the significance filter" if args.significant_only else ""))

    volume = region_vol.paint(vals, level=args.level)
    brain = region_vol.paint(np.ones(ontology.n), level=None, background=np.nan)
    brain_mask = np.isfinite(brain)

    if diverging:
        vmin, vmax = region_maps.symmetric_limits(volume)
    else:
        finite = volume[np.isfinite(volume)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.nanmin(finite)),
                                                         float(np.nanmax(finite)))

    coronal = section_positions(brain_mask, region_maps.AXIS_AP, args.n_coronal)
    sagittal = section_positions(brain_mask, region_maps.AXIS_ML, args.n_sagittal)

    ncols = max(len(coronal), len(sagittal))
    fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols, 5.6), constrained_layout=True)
    axes = np.atleast_2d(axes)

    def panel(ax, img, mask, title):
        # the brain outline in grey first, the statistic on top: a region left
        # grey is "no result", and that has to stay visible rather than reading
        # as background
        ax.imshow(np.where(mask, 0.85, np.nan), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return im

    im = None
    for i in range(ncols):
        if i < len(coronal):
            k = coronal[i]
            im = panel(axes[0, i], volume[k].T, brain_mask[k].T, f"coronal AP {k}")
        else:
            axes[0, i].axis("off")
        if i < len(sagittal):
            k = sagittal[i]
            im = panel(axes[1, i], volume[:, :, k].T, brain_mask[:, :, k].T,
                       f"sagittal ML {k + ml_offset}")
        else:
            axes[1, i].axis("off")

    exploratory = ""
    if "exploratory" in stats_df.columns:
        sub = stats_df[(stats_df["level"] == args.level)
                       & (stats_df["class_name"] == args.class_name)
                       & (stats_df["metric"] == args.metric)]
        if not sub.empty and bool(sub["exploratory"].iloc[0]):
            exploratory = "   [EXPLORATORY level - uncorrected, read as leads only]"
    filt = (f"p_adj < {args.alpha}" if args.significant_only else "all tested regions")
    fig.suptitle(f"{args.class_name} / {args.metric} / level {args.level}   "
                 f"({label}; {filt}; {n_regions} regions){exploratory}", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.7, label=label)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--annotation", default=None,
                    help="atlas annotation to draw on (default: the master DeMBA P5 "
                         "annotation next to the config's ontology_json)")
    ap.add_argument("--class-name", default="all_cells")
    ap.add_argument("--metric", default="Density")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--value", default="log2fc", choices=sorted(VALUE_COLUMNS))
    ap.add_argument("--hemisphere", default="right", choices=["right", "left", "both"])
    ap.add_argument("--n-coronal", type=int, default=6)
    ap.add_argument("--n-sagittal", type=int, default=4)
    ap.add_argument("--significant-only", dest="significant_only", action="store_true",
                    default=True)
    ap.add_argument("--all-regions", dest="significant_only", action="store_false",
                    help="colour every tested region, not only the significant ones")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", action="store_true",
                    help="render every (level, class, metric) that has at least one "
                         "region to colour, instead of the single combination given "
                         "by --level/--class-name/--metric")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = (cfg.get("output") or {}).get("dir", "./stats_output")
    stats_path = os.path.join(out_dir, "region_stats.csv")
    if not os.path.exists(stats_path):
        raise SystemExit(f"{stats_path} not found -- run stats.group_stats first")
    stats_df = pd.read_csv(stats_path)

    annotation = args.annotation or os.path.join(
        os.path.dirname(cfg["ontology_json"]), "DeMBA_P5_annotation.tif")
    print(f"Loading {annotation} ...")
    annot = region_maps.load_annotation(annotation)
    annot, ml_offset = region_maps.hemisphere_slice(annot, args.hemisphere)

    ontology = region_maps.load_ontology(cfg["ontology_json"])
    region_vol = region_maps.RegionVolume(annot, ontology)

    heat_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    if not args.batch:
        out = args.out or os.path.join(
            heat_dir, f"{args.class_name}_{args.metric}_L{args.level}_{args.value}.png")
        draw(stats_df, ontology, region_vol, args, out, ml_offset=ml_offset)
        return

    # Batch: one figure per combination that actually has something to show.
    # Combinations are taken from the table, so gatekeeping has already pruned
    # them -- there is no point rendering a level a class never reached.
    df = stats_df
    if args.significant_only and "p_adj" in df.columns:
        df = df[df["p_adj"] < args.alpha]
    combos = (df[["level", "class_name", "metric"]].drop_duplicates()
              .sort_values(["level", "class_name", "metric"]))
    if combos.empty:
        print("Nothing to draw: no combination has a region passing the filter.")
        return
    print(f"Batch: {len(combos)} figures")
    for _, row in combos.iterrows():
        args.level = int(row["level"])
        args.class_name = str(row["class_name"])
        args.metric = str(row["metric"])
        out = os.path.join(
            heat_dir, f"{args.class_name}_{args.metric}_L{args.level}_{args.value}.png")
        draw(stats_df, ontology, region_vol, args, out, ml_offset=ml_offset)


if __name__ == "__main__":
    main()
