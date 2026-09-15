"""Group bar charts with every individual sample drawn on top.

Three figures the group meeting asks for, plus a generic mode:

    volume          region volumes, mm3, whole brain + major subdivisions
    cortex-count    cell counts in Cerebral cortex, one panel per marker class
    cortex-density  cell density in Cerebral cortex, one panel per marker class

Two decisions in here are not cosmetic.

**Small multiples, not one grouped axis.** The items being compared span two
orders of magnitude (GFP has ~10x the cells of GFP_RFP_Sox9; root has ~2x the
volume of Cerebrum). On a shared axis the small ones become invisible slivers and
the figure only communicates the ranking, which nobody needs. Each panel gets its
own y-axis anchored at zero instead.

**Every sample is drawn.** n = 3 vs 3. A bar plus an SD whisker at that n is a
picture of two numbers pretending to be a distribution, and it hides the thing
that actually matters here -- whether one animal is carrying the difference. The
dots are the data; the bar is a reading aid. Marker shape is fixed per sample
across every figure this module writes, so the same animal is followable from the
volume figure to the density figure.

Error bars are SD, not SEM. At n=3 an SEM bar is a third the size and reads as
precision the design does not have.

Usage:
    conda activate antsreg
    python -m stats.plot_bars --config stats/configs/tsc_marker_ungated.yaml --preset all

    # anything else
    python -m stats.plot_bars --config ... --regions "Hippocampal formation" \
        --metric Density --classes GFP_any,RFP_any --by class
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import plot_style as ps  # noqa: E402
from stats.group_stats import class_vocabulary, load_config  # noqa: E402

VOLUME_REGIONS = ["root", "Cerebrum", "Cerebral cortex", "Cerebral nuclei",
                  "Brain stem", "Interbrain", "Midbrain", "Hindbrain",
                  "fiber tracts", "ventricular systems"]

METRIC_UNITS = {
    "Count": "cells",
    "Density": "cells / mm$^3$",
    "Volume": "mm$^3$",
    "RelativeVolume": "% of analysed volume",
    "Percentage": "% of that class, whole brain",
    "RegionProportion": "% of base-class cells in region",
}


class Run:
    """Everything one stats run puts on disk, plus the config's grouping."""

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.out_dir = (self.cfg.get("output") or {}).get("dir", "./stats_output")
        stats_path = os.path.join(self.out_dir, "region_stats.csv")
        if not os.path.exists(stats_path):
            raise SystemExit(f"{stats_path} not found -- run stats.group_stats first")
        self.stats = pd.read_csv(stats_path, low_memory=False)
        vol_path = os.path.join(self.out_dir, "region_volumes.csv")
        self.volumes = pd.read_csv(vol_path) if os.path.exists(vol_path) else None

        ga, gb = self.cfg["groups"]["a"], self.cfg["groups"]["b"]
        self.group_name = {"a": ga.get("name", "A"), "b": gb.get("name", "B")}
        self.group_samples = {"a": list(ga["samples"]), "b": list(gb["samples"])}
        self.samples = self.group_samples["a"] + self.group_samples["b"]
        self.marker = ps.sample_style(self.samples)
        self.alpha = float((self.cfg.get("stats") or {}).get("alpha", 0.05))
        self.test_levels = tuple(
            (self.cfg.get("region_filter") or {}).get("levels") or (2, 3, 4, 5, 6, 7, 8))

        (self.base_classes, self.combined_classes,
         self.total_class) = class_vocabulary(self.cfg)

    def panel_classes(self):
        """Total first, then the mutually exclusive base classes.

        Deliberately excludes the other combined classes: they overlap each
        other, so a row of panels containing GFP_any beside GFP and GFP_RFP is
        showing the same cells two and three times."""
        return ([self.total_class] if self.total_class else []) + self.base_classes

    def row(self, region, metric, class_name):
        """-> the one region_stats row for this (region, metric, class), or None.

        A region appears once per level, and a name is unique in the ontology, so
        there is at most one row; if the name matched several ids the caller is
        asking something ambiguous and gets told."""
        m = ((self.stats["name"] == region) & (self.stats["metric"] == metric)
             & (self.stats["class_name"] == class_name))
        sub = self.stats[m]
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError(f"{region!r}/{metric}/{class_name} matched {len(sub)} rows")
        return sub.iloc[0]

    def volume_row(self, region):
        """Fallback for regions region_stats never tested -- notably `root`,
        which sits at level 0 while the test levels start at 2."""
        if self.volumes is None:
            return None
        sub = self.volumes[self.volumes["name"] == region]
        return None if sub.empty else sub.iloc[0]


class Item:
    """One panel: a label, the per-sample values behind it, and its test."""

    def __init__(self, label, values, p_adj=np.nan, p_raw=np.nan, g=np.nan,
                 note=None):
        self.label = label
        self.values = values          # {sample: float or nan}
        self.p_adj = p_adj
        self.p_raw = p_raw
        self.g = g
        self.note = note

    def group_values(self, run, key):
        return np.array([self.values.get(s, np.nan) for s in run.group_samples[key]],
                        dtype=float)


def item_from_stats(run, region, metric, class_name, label=None):
    r = run.row(region, metric, class_name)
    if r is None:
        return None
    return Item(label or region,
                {s: float(r[s]) if pd.notna(r.get(s)) else np.nan for s in run.samples},
                p_adj=float(r["p_adj"]) if pd.notna(r["p_adj"]) else np.nan,
                p_raw=float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
                g=float(r["hedges_g"]) if pd.notna(r["hedges_g"]) else np.nan,
                note=f"L{int(r['level'])} · {r['correction']} · m={int(r['m_family'])}")


def item_from_volumes(run, region, label=None, test_levels=(2, 3, 4, 5, 6, 7, 8)):
    """Volume for a region region_stats.csv holds no row for.

    region_volumes.csv is built from the same excluded-region mask as the tests,
    so its numbers are on the same footing -- what is missing is a p-value, and
    the panel says which of the two reasons applies. `root` is simply shallower
    than the shallowest tested level; anything deeper than that was dropped by
    region_filter, which is a statement about that region's data quality and
    belongs on the figure."""
    r = run.volume_row(region)
    if r is None:
        return None
    level = int(r["level"])
    if level < min(test_levels):
        note = f"level {level} · below the tested levels · no p-value"
    else:
        note = f"L{level} · dropped by region_filter (coverage / min count)"
    return Item(label or region,
                {s: float(r[f"abs_mm3:{s}"]) for s in run.samples
                 if f"abs_mm3:{s}" in r.index},
                note=note)


def both_p_text(item, alpha):
    """Raw and adjusted p, always both, for the per-region figures.

    ps.p_label collapses to 'n.s.' once nothing survives the correction, which
    is the right default on a summary slide but hides exactly the number that
    is wanted when going region by region."""
    if not np.isfinite(item.p_raw):
        return "not tested"
    adj = f"{item.p_adj:.3g}" if np.isfinite(item.p_adj) else "n/a"
    return f"p = {item.p_raw:.3g}\np_adj = {adj}"


def draw_panel(ax, run, item, metric, alpha, p_text=None):
    """One item: two bars, six dots, one annotation."""
    import matplotlib.pyplot as plt  # noqa: F401

    xs = [0, 1]
    means, sds = [], []
    for x, key in zip(xs, ("a", "b")):
        vals = item.group_values(run, key)
        ok = vals[np.isfinite(vals)]
        mean = float(ok.mean()) if ok.size else np.nan
        sd = float(ok.std(ddof=1)) if ok.size > 1 else 0.0
        means.append(mean)
        sds.append(sd)
        ax.bar(x, mean, width=0.62, color=ps.GROUP_FILL[key],
               edgecolor=ps.GROUP_COLORS[key], linewidth=1.4, zorder=1)
        if ok.size > 1:
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=ps.GROUP_COLORS[key],
                        elinewidth=1.4, capsize=5, capthick=1.4, zorder=2)
        # jitter is deterministic: the same animal sits in the same place in
        # every panel, so a reader can track it across the figure
        for i, s in enumerate(run.group_samples[key]):
            v = item.values.get(s, np.nan)
            if not np.isfinite(v):
                continue
            offset = (i - (len(run.group_samples[key]) - 1) / 2) * 0.17
            ax.plot(x + offset, v, marker=run.marker[s], markersize=6.5,
                    markerfacecolor="white", markeredgecolor=ps.INK,
                    markeredgewidth=1.2, linestyle="none", zorder=3)

    finite = np.array([v for v in item.values.values() if np.isfinite(v)])
    top = max([m + s for m, s in zip(means, sds) if np.isfinite(m)] +
              ([float(finite.max())] if finite.size else [1.0]))
    ax.set_ylim(0, top * 1.28 if top > 0 else 1.0)
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks(xs)
    ax.set_xticklabels([run.group_name["a"], run.group_name["b"]], fontsize=9)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    txt = p_text(item, alpha) if p_text else ps.p_label(item.p_adj, item.p_raw, alpha)
    sig = np.isfinite(item.p_adj) and item.p_adj < alpha
    style = dict(transform=ax.transAxes, ha="center", fontsize=8.5,
                 color=ps.INK if sig else ps.INK_SOFT,
                 fontweight="bold" if sig else "normal")
    if p_text:
        # two lines do not fit in the headroom without landing on the top dot,
        # so they go between the title and the axes
        ax.set_title(item.label, fontsize=10, pad=34)
        ax.text(0.5, 1.02, txt, va="bottom", **style)
    else:
        ax.set_title(item.label, fontsize=10, pad=14)
        ax.text(0.5, 0.965, txt, va="top", **style)
    if item.note:
        ax.text(0.5, -0.30, item.note, transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color=ps.INK_SOFT)


def figure(run, items, metric, title, out_path, ncols=None, alpha=0.05, p_text=None):
    import matplotlib.pyplot as plt

    items = [i for i in items if i is not None]
    if not items:
        print(f"  [skip] {title}: nothing to draw")
        return
    # balance the grid rather than filling rows to the cap: 6 panels at a cap
    # of 5 would leave a row of one, which reads as a missing panel
    cap = ncols or 5
    nrows = int(np.ceil(len(items) / cap))
    ncols = int(np.ceil(len(items) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3.1 * nrows),
                             squeeze=False)
    for ax, item in zip(axes.ravel(), items):
        draw_panel(ax, run, item, metric, alpha, p_text=p_text)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel(METRIC_UNITS.get(metric, metric))

    handles = _legend_handles(run)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    ps.savefig(fig, out_path)


def _legend_handles(run):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = []
    for key in ("a", "b"):
        h.append(Patch(facecolor=ps.GROUP_FILL[key], edgecolor=ps.GROUP_COLORS[key],
                       label=f"{run.group_name[key]} (mean ± SD)"))
    for s in run.samples:
        h.append(Line2D([], [], marker=run.marker[s], linestyle="none",
                        markerfacecolor="white", markeredgecolor=ps.INK,
                        markersize=6.5, label=s))
    return h


def preset_volume(run, out_dir, alpha, include_untested=False):
    """Volume panels for the regions the pipeline actually tested.

    Regions with no row -- `root`, which is shallower than the shallowest tested
    level, and anything region_filter dropped -- are left off. They would be a
    bar with no p-value sitting next to bars that have one, which invites the
    reader to compare them as if they were the same kind of statement.
    `--include-untested` puts them back, labelled with why they have no test."""
    # runs made before volume was tested once per region carry it under a class
    legacy = run.total_class or (run.base_classes[0] if run.base_classes else "all_cells")
    items = []
    for name in VOLUME_REGIONS:
        it = (item_from_stats(run, name, "Volume", "region")
              or item_from_stats(run, name, "Volume", legacy))
        if it is None and include_untested:
            it = item_from_volumes(run, name, test_levels=run.test_levels)
        items.append(it)
    figure(run, items, "Volume",
           "Region volume — warped atlas region inside each sample's brain mask",
           os.path.join(out_dir, "bars_volume.png"), ncols=5, alpha=alpha)
    coverage_figure(run, out_dir)


def coverage_figure(run, out_dir):
    """Volume's companion, and not optional.

    These are hemispheres cut by hand, so a small 'volume' can mean a small brain
    or a cut that took less of it. Coverage -- the fraction of the warped region
    that lands inside the brain mask -- separates the two, and a volume bar chart
    shown without it invites the room to read dissection as biology."""
    import matplotlib.pyplot as plt
    if run.volumes is None:
        return
    r = run.volumes[run.volumes["name"] == "root"]
    if r.empty:
        return
    r = r.iloc[0]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, s in enumerate(run.samples):
        key = "a" if s in run.group_samples["a"] else "b"
        cov = float(r[f"coverage:{s}"]) * 100
        ax.bar(i, cov, width=0.66, color=ps.GROUP_FILL[key],
               edgecolor=ps.GROUP_COLORS[key], linewidth=1.4, zorder=1)
        ax.text(i, cov + 0.4, f"{cov:.1f}", ha="center", va="bottom", fontsize=8.5,
                color=ps.INK_SOFT)
    ax.set_xticks(range(len(run.samples)))
    ax.set_xticklabels(run.samples)
    ax.set_ylim(min(80, min(float(r[f"coverage:{s}"]) * 100
                            for s in run.samples) - 4), 102)
    ax.set_ylabel("% of whole-brain volume inside the brain mask")
    ax.set_title("Tissue coverage per sample\n"
                 "(the dissection confound behind any volume difference)",
                 fontsize=10)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # its own handles, not _legend_handles: there is no mean and no SD on this
    # figure, one bar per animal, and a legend claiming otherwise is wrong
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=ps.GROUP_FILL[k],
                             edgecolor=ps.GROUP_COLORS[k], label=run.group_name[k])
                       for k in ("a", "b")],
              loc="lower left", ncol=2)
    fig.tight_layout()
    ps.savefig(fig, os.path.join(out_dir, "bars_coverage.png"))


MAX_STACK_HUES = len(ps.CLASS_COLORS)


def composition_partition(run):
    """-> [(group name, [base classes])] for the composition stack.

    A stacked bar can carry at most `MAX_STACK_HUES` segments: past that the
    palette runs out, and inventing more hues is exactly how a stack stops being
    colour-blind readable. The 12-class run is a 2 x 6 product, so the answer is
    not a longer palette, it is to spend position on one factor and colour on
    the other.

    The split is not guessed from class names. It is taken from the config's own
    combined classes: any set of them that is pairwise disjoint and covers the
    base classes exactly is a partition the config author declared (glia_all +
    non_glia_all here). Fewest groups wins, and a run whose base classes already
    fit in one bar is left as one bar."""
    base = list(run.base_classes)
    if len(base) <= MAX_STACK_HUES:
        return [(None, base)]
    combined = (run.cfg.get("combined_categories") or {})
    sets = {}
    for name, terms in combined.items():
        members = frozenset(t["class"] for t in terms if t.get("sign", "+") == "+")
        if members and members <= set(base) and len(members) <= MAX_STACK_HUES:
            sets[name] = members
    import itertools
    names = sorted(sets)
    for k in range(2, min(len(names), 6) + 1):
        for combo in itertools.combinations(names, k):
            picked = [sets[n] for n in combo]
            if len(frozenset().union(*picked)) != sum(len(x) for x in picked):
                continue  # overlapping, so cells would be counted twice
            if frozenset().union(*picked) == set(base):
                order = {c: i for i, c in enumerate(base)}
                return [(n, sorted(sets[n], key=order.get)) for n in combo]
    # No declared partition small enough. Chunking by position would put
    # unrelated classes in one bar and label it as if it meant something, so
    # say so instead and draw nothing.
    print(f"  [skip] composition: {len(base)} base classes exceed the "
          f"{MAX_STACK_HUES}-hue stack and no combined class set partitions them")
    return []


def composition_figure(run, out_dir, region):
    """Stacked composition of the mutually exclusive base classes, per sample.

    Not a group comparison -- a data-quality slide. It answers the question that
    gets asked before any result is believed: are the six animals even made of
    the same cells, or is one of them mostly RFP because its GFP channel was
    dim? Only the mutually exclusive base classes are stacked; the combined
    classes overlap, so a stack including them would sum past 100%.

    RegionProportion is the metric on purpose. It is a share of the base classes
    within the region, so a sample that simply detected fewer cells everywhere
    does not move on this figure -- only a shift in the mixture does.

    With more base classes than the palette can carry, the classes are split
    into panels (see composition_partition) and each panel is renormalised to
    100% within its own group. The group's real share of the region is printed
    over each bar, so renormalising does not hide the split between groups."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = composition_partition(run)
    if not groups:
        return
    rows = {}
    for _, classes in groups:
        for c in classes:
            r = run.row(region, "RegionProportion", c)
            if r is None:
                print(f"  [skip] composition: no RegionProportion row for {c} "
                      f"in {region}")
                return
            rows[c] = r

    def value(c, sample):
        v = rows[c][sample]
        return float(v) if pd.notna(v) else 0.0

    # a visible gap between the two groups, so the reader is not comparing
    # across the boundary by accident
    xs, x = {}, 0.0
    for key in ("a", "b"):
        for s_ in run.group_samples[key]:
            xs[s_] = x
            x += 1.0
        x += 0.55

    ncols = len(groups)
    fig, axes = plt.subplots(1, ncols, squeeze=False,
                             figsize=(ncols * (1.05 * len(run.samples) + 0.9) + 2.6, 5.0))
    for ax, (gname, classes) in zip(axes[0], groups):
        for s_ in run.samples:
            total = sum(value(c, s_) for c in classes)
            if total <= 0:
                continue
            bottom = 0.0
            for i, c in enumerate(classes):
                v = value(c, s_) / total * 100
                ax.bar(xs[s_], v, bottom=bottom, width=0.74,
                       color=ps.CLASS_COLORS[i], edgecolor="white", linewidth=1.6,
                       zorder=2)
                # in-segment labels are required, not optional: several of the
                # hues fall below 3:1 against white, so colour alone is not
                # enough to identify a segment
                if v >= 5.0:
                    ax.text(xs[s_], bottom + v / 2, f"{v:.0f}", ha="center",
                            va="center", fontsize=8, color="white",
                            fontweight="bold")
                bottom += v
            if gname is not None:
                ax.text(xs[s_], 102, f"{total:.0f}%", ha="center", va="bottom",
                        fontsize=8, color=ps.INK_SOFT)
        ax.set_xticks([xs[s_] for s_ in run.samples])
        labels = ax.set_xticklabels(run.samples)
        for lbl, s_ in zip(labels, run.samples):
            lbl.set_color(ps.GROUP_COLORS["a" if s_ in run.group_samples["a"] else "b"])
            lbl.set_fontweight("bold")
        # headroom for the real-share annotation, but ticks stop at 100 so the
        # bar is still read against a 0-100 axis
        ax.set_ylim(0, 100 if gname is None else 113)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if gname is not None:
            ax.set_title(gname, fontsize=10, fontweight="bold")
        ax.legend(handles=[Patch(facecolor=ps.CLASS_COLORS[i], edgecolor="white",
                                 label=c) for i, c in enumerate(classes)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
                  fontsize=8)
    axes[0, 0].set_ylabel("% within the group" if len(groups) > 1
                          else "% of base-class cells in the region")
    sub = (f"{run.group_name['a']} (blue labels) vs "
           f"{run.group_name['b']} (orange labels)")
    if len(groups) > 1:
        sub += ("; each panel renormalised to 100% within its group, "
                "the group's real share of the region printed above each bar")
    fig.suptitle(f"{region} — cell-class composition per sample\n{sub}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    slug = region.lower().replace(" ", "_").replace(",", "")
    ps.savefig(fig, os.path.join(out_dir, f"bars_composition_{slug}.png"))


def preset_region(run, out_dir, region, metric, alpha, classes=None):
    classes = classes or run.panel_classes()
    items = [item_from_stats(run, region, metric, c, label=c) for c in classes]
    slug = region.lower().replace(" ", "_").replace(",", "")
    figure(run, items, metric,
           f"{region} — {metric} by cell class",
           os.path.join(out_dir, f"bars_{slug}_{metric.lower()}.png"),
           ncols=4, alpha=alpha)


BY_REGION_METRICS = ("Count", "Density", "RegionProportion", "Volume", "RelativeVolume")
# tested once per region, not per class (group_stats.REGION_CLASS)
REGION_LEVEL_METRICS = ("Volume", "RelativeVolume")


def _safe_dirname(text):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")


def preset_by_region(run, out_dir, alpha):
    """One folder per tested level, one sub-folder per region, and in it one
    figure per metric with a panel per tested class.

    Built for going through regions one at a time, so every panel prints the
    raw and the adjusted p even when nothing survives, and no g: the six dots
    are the effect. A class a region was not tested for (below min_total_count)
    has no row and therefore no panel. With no test_classes every class in the
    run is drawn: the total, the other combined classes, then the base classes,
    five panels to a row."""
    classes = list(run.cfg.get("test_classes") or [])
    if not classes:
        combined = [c for c in run.combined_classes if c != run.total_class]
        classes = ([run.total_class] if run.total_class else []) + combined + run.base_classes
    metrics = [m for m in BY_REGION_METRICS if m in set(run.stats["metric"])]
    top = min(int(lv) for lv in run.stats["level"].unique())
    # a region below the shallowest tested level is filed under its ancestors
    # from that level down (L06/Isocortex_Isocortex/MO_Somatomotor_areas), with
    # the ancestor folders spelled exactly like their own folders one level up
    acronym_of = dict(zip(run.stats["name"], run.stats["acronym"]))
    ancestor_cols = [f"L{k}_name" for k in range(top, int(run.stats["level"].max()))]
    cols = ["level", "name", "acronym"] + [c for c in ancestor_cols if c in run.stats]
    regions = (run.stats[run.stats["class_name"].isin(classes)][cols]
               .drop_duplicates().sort_values(["level", "name"]))
    for row in regions.itertuples(index=False):
        level, name, acronym = int(row.level), row.name, row.acronym
        parts = [f"L{level:02d}"]
        for k in range(top, level):
            anc = getattr(row, f"L{k}_name", None)
            if isinstance(anc, str) and anc:
                parts.append(_safe_dirname(f"{acronym_of[anc]}_{anc}"
                                           if anc in acronym_of else anc))
        region_dir = os.path.join(out_dir, *parts, _safe_dirname(f"{acronym}_{name}"))
        for i, metric in enumerate(metrics, 1):
            if metric in REGION_LEVEL_METRICS:
                items = [item_from_stats(run, name, metric, "region", label=metric)]
                ncols = 1
            else:
                items = [item_from_stats(run, name, metric, c, label=c) for c in classes]
                ncols = min(len(classes), 5)
            figure(run, items, metric, f"{name} ({acronym}) — {metric}",
                   os.path.join(region_dir, f"{i}_{metric}.png"),
                   ncols=ncols, alpha=alpha, p_text=both_p_text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--preset", default="all",
                    choices=["all", "volume", "cortex-count", "cortex-density",
                             "composition", "by-region", "none"])
    ap.add_argument("--region", default="Cerebral cortex",
                    help="region for the count/density presets")
    ap.add_argument("--regions", default=None,
                    help="generic mode: comma-separated region names, one panel each")
    ap.add_argument("--classes", default=None,
                    help="generic mode: comma-separated class names, one panel each")
    ap.add_argument("--metric", default="Density")
    ap.add_argument("--include-untested", action="store_true",
                    help="also draw regions with no test row (root, and anything "
                         "region_filter dropped); off by default")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ps.apply_rcparams()
    run = Run(args.config)
    if args.preset == "by-region":
        preset_by_region(run, os.path.join(args.out_dir or os.path.join(run.out_dir, "figures"),
                                           "bars_by_region"), run.alpha)
        return
    out_dir = ps.figure_dir(args.out_dir or os.path.join(run.out_dir, "figures"),
                            "bars")

    if args.regions or args.classes:
        regions = [r.strip() for r in (args.regions or args.region).split(",")]
        default_cls = run.total_class or "all_cells"
        classes = [c.strip() for c in (args.classes or default_cls).split(",")]
        if len(regions) > 1 and len(classes) > 1:
            raise SystemExit("give several regions or several classes, not both -- "
                             "a panel has to mean one thing")
        if len(regions) > 1:
            items = [item_from_stats(run, r, args.metric, classes[0]) for r in regions]
            title = f"{classes[0]} — {args.metric}"
            slug = f"{classes[0]}_{args.metric.lower()}_by_region"
        else:
            items = [item_from_stats(run, regions[0], args.metric, c, label=c)
                     for c in classes]
            title = f"{regions[0]} — {args.metric} by cell class"
            slug = (regions[0].lower().replace(" ", "_").replace(",", "")
                    + f"_{args.metric.lower()}")
        figure(run, items, args.metric, title,
               os.path.join(out_dir, f"bars_{slug}.png"), alpha=run.alpha)
        return

    if args.preset in ("all", "volume"):
        preset_volume(run, out_dir, run.alpha, include_untested=args.include_untested)
    if args.preset in ("all", "cortex-count"):
        preset_region(run, out_dir, args.region, "Count", run.alpha)
    if args.preset in ("all", "cortex-density"):
        preset_region(run, out_dir, args.region, "Density", run.alpha)
    if args.preset in ("all", "composition"):
        composition_figure(run, out_dir, args.region)


if __name__ == "__main__":
    main()
