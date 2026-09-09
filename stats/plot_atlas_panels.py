"""Atlas sections for slides: per-sample density maps, and a significance map.

Two modes, drawn with the same section geometry so they line up on a slide:

  per-sample     one panel per animal, region density painted on a coronal
                 section. This is the figure that answers "does the effect look
                 like a group difference, or like one animal?" before any test
                 is discussed.
  significance   one panel, group difference, coloured only where the region
                 passed correction. Red = higher in group B, blue = higher in A.

Three things this draws that a heatmap normally gets wrong:

**Two greys, not one.** A region can be uncoloured because nothing was ever
computed for it (excluded, below the coverage floor, too few cells) or because
it was tested and did not pass. Those mean opposite things and both get their
own grey and their own legend entry. One grey lets the room read "no result" as
"no difference", which is the single most common misreading of these maps.

**A region is a flat patch of colour.** The value is per region, not per voxel,
so this is not a voxel-wise statistical map and must not be presented as one.
Every voxel reads out its ancestor at the chosen level, so a level-5 map paints
all of CA1, CA3 and DG with HPF's number.

**Hemisphere.** The samples are hemispheres and the atlas was cropped to match,
so only one hemisphere is drawn. s10 is a left hemisphere registered against a
mirrored atlas, so its numbers are already in right-hemisphere coordinates -- no
flip is applied or needed here.

Usage:
    conda activate antsreg
    python -m stats.plot_atlas_panels --config stats/configs/tsc_marker_ungated.yaml \
        --mode per-sample --class-name GFP_any --metric Density --level 5
    python -m stats.plot_atlas_panels --config ... --mode significance \
        --class-name GFP_any --metric Density --level 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import plot_style as ps  # noqa: E402
from stats import region_maps  # noqa: E402
from stats.group_stats import class_vocabulary, load_config  # noqa: E402


def coronal_positions(brain_mask, n):
    """`n` AP indices spread over the part of the atlas that holds brain.

    n=1 gives the middle of that span, which is what a single-section slide
    wants; the ends are never used because a section through the last few
    voxels of the brain is mostly background."""
    present = np.where(brain_mask.any(axis=(1, 2)))[0]
    if present.size == 0:
        raise ValueError("annotation is empty")
    lo, hi = int(present[0]), int(present[-1])
    if n == 1:
        return [int(round((lo + hi) / 2))]
    return [int(round(v)) for v in np.linspace(lo, hi, n + 2)[1:-1]]


def coronal(volume, k, box=None):
    """One coronal section, oriented the way an anatomist expects.

    The volume is (AP, DV, ML), so volume[k] is already (DV, ML): rows run
    dorsal->ventral down the image and columns run across the hemisphere. No
    transpose -- transposing here is what turns a coronal section on its side.

    `box` is the tight bounding box of brain across the sections being drawn.
    Cropping to it is what stops a row of panels from being mostly background:
    the atlas volume is sized for the whole brain, and any one section fills
    well under half of it."""
    img = volume[k]
    if box is not None:
        r0, r1, c0, c1 = box
        img = img[r0:r1, c0:c1]
    return img


def brain_box(brain_mask, aps, pad=4):
    """-> (row0, row1, col0, col1) covering brain in every section in `aps`.

    Shared across panels on purpose: cropping each panel to its own extent
    would silently rescale the sections relative to each other, so a bigger
    structure would look the same size as a smaller one."""
    sub = brain_mask[aps]
    rows = np.where(sub.any(axis=(0, 2)))[0]
    cols = np.where(sub.any(axis=(0, 1)))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return (max(int(rows[0]) - pad, 0), int(rows[-1]) + 1 + pad,
            max(int(cols[0]) - pad, 0), int(cols[-1]) + 1 + pad)


def draw_section(ax, img, layers, cmap, vmin, vmax, title=None):
    """Paint the underlay greys first, then the data on top.

    Regions with no result at all are drawn by nobody, so they stay blank: an
    untested region has nothing to report, and painting it grey only invites the
    reader to read "no result" as "no difference". What does get an underlay is
    a region that WAS tested and did not pass, because that is a result."""
    from matplotlib.colors import ListedColormap
    for mask, colour in layers:
        ax.imshow(np.where(mask, 1.0, np.nan), cmap=ListedColormap([colour]),
                  vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    return im


class Maps:
    """Atlas + ontology + the stats table, prepared once."""

    def __init__(self, config_path, annotation=None, hemisphere="right"):
        self.cfg = load_config(config_path)
        self.out_dir = (self.cfg.get("output") or {}).get("dir", "./stats_output")
        stats_path = os.path.join(self.out_dir, "region_stats.csv")
        if not os.path.exists(stats_path):
            raise SystemExit(f"{stats_path} not found -- run stats.group_stats first")
        self.stats = pd.read_csv(stats_path, low_memory=False)

        ga, gb = self.cfg["groups"]["a"], self.cfg["groups"]["b"]
        self.group_name = {"a": ga.get("name", "A"), "b": gb.get("name", "B")}
        self.group_samples = {"a": list(ga["samples"]), "b": list(gb["samples"])}
        self.samples = self.group_samples["a"] + self.group_samples["b"]
        self.alpha = float((self.cfg.get("stats") or {}).get("alpha", 0.05))
        (self.base_classes, self.combined_classes,
         self.total_class) = class_vocabulary(self.cfg)

        annotation = annotation or os.path.join(
            os.path.dirname(self.cfg["ontology_json"]), "DeMBA_P5_annotation.tif")
        print(f"Loading {annotation} ...")
        annot = region_maps.load_annotation(annotation)
        annot, self.ml_offset = region_maps.hemisphere_slice(annot, hemisphere)
        self.ontology = region_maps.load_ontology(self.cfg["ontology_json"])
        self.rv = region_maps.RegionVolume(annot, self.ontology)
        self.brain = self.rv.paint(np.ones(self.ontology.n), level=None,
                                   background=np.nan)
        self.brain_mask = np.isfinite(self.brain)

    def all_classes(self):
        """Every class this run produced, total first, then the mutually
        exclusive base classes, then the remaining combined ones.

        The order is the reading order for a stack of figures: the total says
        whether anything moved at all, the base classes say which population
        moved, and the combined classes are the pooled views that only mean
        something once you know the first two."""
        rest = [c for c in self.combined_classes if c != self.total_class]
        return ([self.total_class] if self.total_class else []) + self.base_classes + rest

    def family(self, level, class_name, metric):
        return self.stats[(self.stats["level"] == level)
                          & (self.stats["class_name"] == class_name)
                          & (self.stats["metric"] == metric)]

    def painted(self, sub, column, level):
        """-> volume of `column` for the rows in `sub`, NaN elsewhere."""
        vals = np.full(self.ontology.n, np.nan)
        if not sub.empty:
            orders = self.ontology.order_of_ids(sub["id"].to_numpy(dtype=np.int64))
            ok = orders >= 0
            vals[orders[ok]] = sub[column].to_numpy(dtype=float)[ok]
        return self.rv.paint(vals, level=level)


def figure_per_sample(m, args, out_path):
    import matplotlib.pyplot as plt

    fam = m.family(args.level, args.class_name, args.metric)
    if fam.empty:
        # a skip, not an error: sweeping every class over every level always
        # hits combinations that region_filter emptied out, and one of those
        # must not take the other fifty figures down with it
        print(f"  [skip] no rows for {args.class_name}/{args.metric}/L{args.level}")
        return False
    missing = [s for s in m.samples if s not in fam.columns]
    if missing:
        raise SystemExit(f"region_stats.csv has no per-sample column for {missing}")

    vols = {s: m.painted(fam, s, args.level) for s in m.samples}
    stack = np.concatenate([v[np.isfinite(v)] for v in vols.values()])
    if stack.size == 0:
        print(f"  [skip] every sample column is empty for {args.class_name}/"
              f"{args.metric}/L{args.level}")
        return False

    # One scale for all panels. Per-panel scales would make every animal look
    # the same and destroy the only thing this figure is for. The high end is a
    # percentile, not the max, so one tiny hot region cannot flatten the rest.
    if args.log:
        for s in vols:
            with np.errstate(divide="ignore", invalid="ignore"):
                vols[s] = np.log10(np.where(vols[s] > 0, vols[s], np.nan))
        stack = np.concatenate([v[np.isfinite(v)] for v in vols.values()])
    vmin = float(np.percentile(stack, args.clip_low))
    vmax = float(np.percentile(stack, args.clip_high))
    cmap = ps.sequential_cmap("blue")

    aps = coronal_positions(m.brain_mask, args.n_coronal)
    box = brain_box(m.brain_mask, aps)
    nrows, ncols = len(aps), len(m.samples)
    panel_w, panel_h = _panel_size(box)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols + 1.8, panel_h * nrows + 1.9),
                             squeeze=False, constrained_layout=True)
    for r, k in enumerate(aps):
        for c, s in enumerate(m.samples):
            key = "a" if s in m.group_samples["a"] else "b"
            im = draw_section(axes[r, c], coronal(vols[s], k, box), [],
                              cmap, vmin, vmax, title=s if r == 0 else None)
            if r == 0:
                axes[r, c].title.set_color(ps.GROUP_COLORS[key])
                axes[r, c].title.set_fontweight("bold")
        axes[r, 0].set_ylabel(f"AP {k}", fontsize=8, color=ps.INK_SOFT)

    unit = "log10 cells / mm$^3$" if args.log else _unit(args.metric)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label(f"{args.class_name} · {unit}", fontsize=9)
    cb.outline.set_visible(False)

    handles = [_group_patch(m, k) for k in ("a", "b")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle(f"{args.class_name} · {args.metric} · level {args.level} — per sample\n"
                 f"shared colour scale ({args.clip_low:g}–{args.clip_high:g} percentile); "
                 f"one flat colour per region, not per voxel", fontsize=11)
    ps.savefig(fig, out_path)
    return True


def figure_significance(m, args, out_path):
    """The group difference, coloured where the test found something.

    Two tiers, because at 3 vs 3 the corrected set is often empty and an empty
    map is not the same statement as "nothing happened here":

      * filled + black outline -- passed the family's correction (p_adj < alpha)
      * filled, no outline     -- raw p < alpha only, did NOT survive correction

    The outline is an additive mark rather than a second colour scale, so both
    tiers stay on one diverging ramp and the reader compares magnitudes across
    them directly. An unoutlined region is a LEAD, not a finding: it is one of
    however many regions the family tested, and the whole point of the
    correction is that some of them come out this way by chance.

    A family where neither tier has a single region is not drawn at all. A page
    of uniform grey tells the reader nothing they cannot get from the index."""
    import matplotlib.pyplot as plt

    fam = m.family(args.level, args.class_name, args.metric)
    if fam.empty:
        print(f"  [skip] no rows for {args.class_name}/{args.metric}/L{args.level}")
        return False
    passed = fam[fam["p_adj"] < args.alpha]
    raw = fam[fam["p_value"] < args.alpha]      # a superset of `passed`
    if raw.empty:
        print(f"  [skip] nothing at raw p < {args.alpha} in {args.class_name}/"
              f"{args.metric}/L{args.level}")
        return False

    tested = np.isfinite(m.painted(fam, args.value, args.level))
    vol = m.painted(raw, args.value, args.level)
    outline = np.isfinite(m.painted(passed, args.value, args.level))
    finite = vol[np.isfinite(vol)]
    lim = float(np.max(np.abs(finite))) if finite.size else 1.0
    cmap = ps.diverging_cmap()

    aps = coronal_positions(m.brain_mask, args.n_coronal)
    box = brain_box(m.brain_mask, aps)
    ncols = len(aps)
    panel_w, panel_h = _panel_size(box)
    fig, axes = plt.subplots(1, ncols,
                             figsize=(panel_w * ncols + 2.4, panel_h + 2.4),
                             squeeze=False, constrained_layout=True)
    for c, k in enumerate(aps):
        ax = axes[0, c]
        im = draw_section(ax, coronal(vol, k, box),
                          [(coronal(tested, k, box), ps.NOT_SIG)],
                          cmap, -lim, lim, title=f"coronal AP {k}")
        sect = coronal(outline, k, box)
        if sect.any():
            # contour, not a patch: the region boundary is whatever the atlas
            # says it is, and tracing it from the mask keeps the outline exactly
            # on the painted area at any section
            ax.contour(sect.astype(float), levels=[0.5], colors=[ps.INK],
                       linewidths=1.2)

    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(_signed_label(args.value, m), fontsize=9)
    cb.outline.set_visible(False)

    correction = str(fam["correction"].iloc[0])
    # The first two keys carry the SAME fill on purpose: fill colour comes from
    # the colour bar (direction and magnitude), and the only thing separating
    # the two tiers is the outline. Giving them different fills would invent a
    # second colour meaning that the map does not use.
    handles = [
        _outlined_patch(f"outlined: p_adj < {args.alpha}, survives {correction}"),
        _grey_patch(ps.SWATCH_NEUTRAL,
                    f"not outlined: raw p < {args.alpha} only — a lead"),
        _grey_patch(ps.NOT_SIG, f"grey: tested, raw p ≥ {args.alpha}"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5)
    note = ""
    if args.level >= 6:
        # Deep levels are a whole-brain search with 3 vs 3 animals. A coloured
        # region there is a lead for the family, not a claim about that region.
        note = "\nlevel ≥ 6: read as a family-level result, not a per-region claim"
    fig.suptitle(f"{args.class_name} · {args.metric} · level {args.level} — "
                 f"{len(passed)} of {len(fam)} regions pass {correction}, "
                 f"{len(raw)} reach raw p < {args.alpha}" + note, fontsize=11)
    ps.savefig(fig, out_path)
    return True


def _panel_size(box, target_h=2.6):
    """Panel inches from the cropped section's own aspect, so the brain fills
    the panel instead of floating in a box sized for the whole atlas."""
    if box is None:
        return 2.1, target_h
    r0, r1, c0, c1 = box
    return target_h * (c1 - c0) / max(r1 - r0, 1), target_h


def _unit(metric):
    return {"Density": "cells / mm$^3$", "Count": "cells", "Volume": "mm$^3$",
            "Percentage": "% of class", "RelativeVolume": "% of volume",
            "RegionProportion": "% of base cells"}.get(metric, metric)


def _signed_label(value, m):
    a, b = m.group_name["a"], m.group_name["b"]
    if value == "log2fc":
        return f"log2 fold change   ({b} / {a})"
    return f"Hedges' g   ({b} − {a})"


def _grey_patch(colour, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=colour, edgecolor="none", label=label)


def _outlined_patch(label):
    """The legend key for the corrected tier: the outline IS the encoding, so
    the swatch carries no fill colour of its own."""
    from matplotlib.patches import Patch
    return Patch(facecolor=ps.SWATCH_NEUTRAL, edgecolor=ps.INK, linewidth=1.2,
                 label=label)


def _group_patch(m, key):
    from matplotlib.patches import Patch
    return Patch(facecolor="none", edgecolor=ps.GROUP_COLORS[key], linewidth=1.6,
                 label=f"{m.group_name[key]}: {', '.join(m.group_samples[key])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="per-sample",
                    choices=["per-sample", "significance", "both"])
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--class-name", default=None,
                    help="default: the config's own 'all cells' class "
                         "(all_cells on a marker run, MADM_all on a MADM run)")
    ap.add_argument("--metric", default="Density")
    ap.add_argument("--all-classes", action="store_true",
                    help="draw every class this config declared, not just one. "
                         "The atlas is loaded once for the whole sweep, which is "
                         "why this is a flag rather than a shell loop")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--levels", default=None,
                    help="comma-separated levels to sweep; overrides --level")
    ap.add_argument("--value", default="log2fc", choices=["log2fc", "hedges_g"],
                    help="significance mode: which signed statistic to colour by")
    ap.add_argument("--hemisphere", default="right", choices=["right", "left", "both"])
    ap.add_argument("--n-coronal", type=int, default=1,
                    help="1 = the middle of the brain's AP extent (slide default)")
    ap.add_argument("--log", action="store_true",
                    help="per-sample mode: log10 the values before mapping colour")
    ap.add_argument("--clip-low", type=float, default=2.0)
    ap.add_argument("--clip-high", type=float, default=98.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ps.apply_rcparams()
    m = Maps(args.config, args.annotation, args.hemisphere)
    if args.alpha is None:
        args.alpha = m.alpha
    if args.all_classes:
        classes = m.all_classes()
    else:
        classes = [args.class_name or m.total_class or "all_cells"]
    levels = ([int(x) for x in args.levels.split(",")] if args.levels
              else [args.level])
    base = args.out_dir or os.path.join(m.out_dir, "figures")
    density_dir = ps.figure_dir(base, "density")
    sig_dir = ps.figure_dir(base, "significance")

    n = 0
    for cls in classes:
        for level in levels:
            args.class_name, args.level = cls, level
            # the section count is part of the name: a 4-section survey and a
            # 1-section slide figure of the same family are different figures,
            # and without this the second call silently overwrites the first
            tag = f"{cls}_{args.metric}_L{level}"
            if args.n_coronal > 1:
                tag += f"_x{args.n_coronal}"
            if args.mode in ("per-sample", "both"):
                n += bool(figure_per_sample(
                    m, args, os.path.join(density_dir, f"map_per_sample_{tag}.png")))
            if args.mode in ("significance", "both"):
                n += bool(figure_significance(
                    m, args,
                    os.path.join(sig_dir, f"map_significance_{tag}_{args.value}.png")))
    print(f"{n} figure(s) for {len(classes)} class(es) x {len(levels)} level(s)")


if __name__ == "__main__":
    main()
