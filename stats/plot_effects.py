"""Whole-family views: a volcano and a forest of effect sizes.

The bar charts show a handful of regions and the atlas maps show where they sit.
Neither shows the thing a group meeting will ask about a 3-vs-3 screen: how many
regions were tested, and how far the ones that came out sit from the rest. These
two do.

**Volcano** -- one dot per region in one correction family (class, metric, level).
The dashed line is not p = 0.05. It is the raw p that the family's own correction
actually demanded, so the reader sees the price of testing 50 regions instead of
being shown a threshold that was never used.

**Forest** -- Hedges' g with its 95% interval, one row per region. At n = 3 vs 3
those intervals are wide, often several units of g. That is not a defect of the
plot; it is the design, and a forest is the honest way to put it on a slide. A
bar chart of the same data hides it.

Both plots are per FAMILY. A family is exactly one (class, metric, level) triple,
because that is the set the p-values were corrected within -- mixing levels on one
volcano would put points on it whose p_adj came from different denominators.

Usage:
    conda activate antsreg
    python -m stats.plot_effects --config stats/configs/tsc_marker_ungated.yaml \
        --class-name GFP_any --metric Density --levels 2,3,5
    python -m stats.plot_effects --config ... --kind forest \
        --class-name GFP_any --metric Density --level 5 --top 20
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

UP = "#d63b3a"      # higher in group B
DOWN = "#2a78d6"    # higher in group A
MUTED = "#b9b8b2"


class Run:
    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.out_dir = (self.cfg.get("output") or {}).get("dir", "./stats_output")
        path = os.path.join(self.out_dir, "region_stats.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found -- run stats.group_stats first")
        self.stats = pd.read_csv(path, low_memory=False)
        ga, gb = self.cfg["groups"]["a"], self.cfg["groups"]["b"]
        self.group_name = {"a": ga.get("name", "A"), "b": gb.get("name", "B")}
        self.alpha = float((self.cfg.get("stats") or {}).get("alpha", 0.05))
        (self.base_classes, self.combined_classes,
         self.total_class) = class_vocabulary(self.cfg)

    def all_classes(self):
        """Total first, then the mutually exclusive base classes, then the rest
        of the combined ones -- the order a stack of these figures is read in."""
        rest = [c for c in self.combined_classes if c != self.total_class]
        return ([self.total_class] if self.total_class else []) + self.base_classes + rest

    def family(self, level, class_name, metric):
        return self.stats[(self.stats["level"] == level)
                          & (self.stats["class_name"] == class_name)
                          & (self.stats["metric"] == metric)].copy()


def effective_threshold(fam, alpha):
    """The raw p the family's correction actually required.

    Under BH this is the largest raw p among the rows that passed; under Holm the
    same reading holds. When nothing passed there is no such p, and the caller
    draws no line rather than drawing 0.05 -- which was never the threshold."""
    passed = fam[fam["p_adj"] < alpha]
    if passed.empty:
        return None
    return float(passed["p_value"].max())


def volcano(run, fam, level, class_name, metric, alpha, ax):
    fam = fam[np.isfinite(fam["log2fc"]) & np.isfinite(fam["p_value"])]
    if fam.empty:
        ax.axis("off")
        return
    y = -np.log10(fam["p_value"].clip(lower=1e-300))
    sig = fam["p_adj"] < alpha
    up = fam["log2fc"] > 0

    ax.scatter(fam.loc[~sig, "log2fc"], y[~sig], s=26, c=MUTED,
               edgecolors="white", linewidths=0.6, zorder=2)
    for mask, colour in ((sig & up, UP), (sig & ~up, DOWN)):
        if mask.any():
            ax.scatter(fam.loc[mask, "log2fc"], y[mask], s=64, c=colour,
                       edgecolors="white", linewidths=1.2, zorder=3)

    thr = effective_threshold(fam, alpha)
    if thr is not None:
        ax.axhline(-np.log10(thr), color=ps.INK_SOFT, linestyle="--", linewidth=1,
                   zorder=1)
        # left-aligned on purpose: the regions that clear the line are the ones
        # getting labelled, and they sit near it, so a right-aligned note lands
        # on top of the very names the figure exists to show
        ax.text(0.01, -np.log10(thr), f"p_adj = {alpha}  ⇒  raw p ≤ {thr:.2g}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=8, color=ps.INK_SOFT)
    ax.axvline(0, color=ps.GRID, linewidth=1, zorder=1)

    # Only significant regions get a label. Labelling every point is how a
    # volcano becomes unreadable, and the near-threshold names are exactly the
    # ones a reader should not be memorising off a 3-vs-3 screen.
    for _, r in fam[sig].iterrows():
        ax.annotate(str(r["acronym"]), (r["log2fc"], -np.log10(max(r["p_value"], 1e-300))),
                    textcoords="offset points", xytext=(7, 3), fontsize=9,
                    color=ps.INK, fontweight="bold")

    n_sig = int(sig.sum())
    ax.set_title(f"level {level} — {n_sig} of {len(fam)} pass "
                 f"({fam['correction'].iloc[0]})", fontsize=10)
    ax.set_xlabel(f"log2 fold change  ({run.group_name['b']} / {run.group_name['a']})")
    ax.grid(True, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    lim = float(np.nanmax(np.abs(fam["log2fc"]))) * 1.28 or 1.0
    ax.set_xlim(-lim, lim)


def figure_volcano(run, args, out_path):
    import matplotlib.pyplot as plt
    levels = [int(x) for x in args.levels.split(",")]
    fams = [(L, run.family(L, args.class_name, args.metric)) for L in levels]
    fams = [(L, f) for L, f in fams if not f.empty]
    if not fams:
        # a skip, not an error: sweeping every class hits combinations that
        # region_filter emptied out, and one of those must not abort the rest
        print(f"  [skip] no rows for {args.class_name}/{args.metric} at levels {levels}")
        return False
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4.0),
                             squeeze=False, constrained_layout=True)
    for ax, (L, f) in zip(axes[0], fams):
        volcano(run, f, L, args.class_name, args.metric, args.alpha, ax)
    axes[0, 0].set_ylabel("-log10 raw p")
    fig.legend(handles=_direction_handles(run), loc="outside lower center", ncol=3)
    fig.suptitle(f"{args.class_name} · {args.metric} — one dot per region, "
                 f"one panel per correction family", fontsize=12)
    ps.savefig(fig, out_path)
    return True


def figure_forest(run, args, out_path):
    import matplotlib.pyplot as plt
    fam = run.family(args.level, args.class_name, args.metric)
    if fam.empty:
        print(f"  [skip] no rows for {args.class_name}/{args.metric}/L{args.level}")
        return False
    fam = fam[np.isfinite(fam["hedges_g"])]
    if args.significant_only:
        fam = fam[fam["p_adj"] < args.alpha]
        if fam.empty:
            print(f"  [skip] nothing passed the correction in {args.class_name}/"
                  f"{args.metric}/L{args.level}; drop --significant-only to see "
                  f"the whole family")
            return False
    else:
        fam = fam.reindex(fam["hedges_g"].abs().sort_values(ascending=False).index)
        fam = fam.head(args.top)
    fam = fam.sort_values("hedges_g")

    n = len(fam)
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * n + 2.2), constrained_layout=True)
    ys = np.arange(n)
    lo = fam["g_ci_lo"].to_numpy(dtype=float)
    hi = fam["g_ci_hi"].to_numpy(dtype=float)
    g = fam["hedges_g"].to_numpy(dtype=float)
    sig = (fam["p_adj"] < args.alpha).to_numpy()
    colours = np.where(g > 0, UP, DOWN)

    for i in range(n):
        c = colours[i] if sig[i] else MUTED
        ax.plot([lo[i], hi[i]], [ys[i], ys[i]], color=c,
                linewidth=2.4 if sig[i] else 1.6, solid_capstyle="round", zorder=2)
        ax.plot(g[i], ys[i], marker="o", markersize=8 if sig[i] else 6,
                color=c, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.axvline(0, color=ps.INK_SOFT, linewidth=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{a}" for a in fam["acronym"]], fontsize=9)
    for tick, s in zip(ax.get_yticklabels(), sig):
        tick.set_fontweight("bold" if s else "normal")
        tick.set_color(ps.INK if s else ps.INK_SOFT)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel(f"Hedges' g with 95% CI   ({run.group_name['b']} − {run.group_name['a']})")
    ax.xaxis.grid(True, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    what = ("regions passing the correction" if args.significant_only
            else f"top {n} of {len(run.family(args.level, args.class_name, args.metric))} "
                 f"by |g|")
    fig.legend(handles=_direction_handles(run), loc="outside lower center", ncol=3)
    ax.set_title(f"{args.class_name} · {args.metric} · level {args.level}\n{what}   "
                 f"— intervals are wide because n = 3 vs 3", fontsize=11)
    ps.savefig(fig, out_path)
    return True


def figure_summary(run, args, out_path):
    """One tile per (class, level): how many regions passed, out of how many.

    Sweeping every class produces dozens of maps and volcanoes, and a stack that
    size is unusable without an index. This is the index: it says which of those
    figures has anything on it before anyone opens them, and it is also the only
    place the whole screen is visible at once.

    Read it as a map of where to look, NOT as a result. A tile showing 3/50 has
    not been corrected for the fact that this grid contains many families -- the
    correction happened inside each family, which is exactly one tile."""
    import matplotlib.pyplot as plt

    classes = run.all_classes()
    levels = sorted(int(x) for x in run.stats["level"].unique())
    counts = np.full((len(classes), len(levels)), np.nan)
    totals = np.zeros_like(counts)
    for i, c in enumerate(classes):
        for j, L in enumerate(levels):
            fam = run.family(L, c, args.metric)
            if fam.empty:
                continue
            totals[i, j] = len(fam)
            counts[i, j] = int((fam["p_adj"] < args.alpha).sum())

    # Discrete steps, and zero gets its own neutral. The gap between 0 and 1
    # region is categorical here (nothing to open vs something to open), and on
    # a continuous ramp anchored at zero it would be an imperceptible shade
    # change -- the reader would have to read every number to find the tiles
    # worth looking at, which defeats the point of an index.
    cmap = ps.sequential_cmap("blue")
    steps = [1, 2, 4, 8, 16]
    def tile_colour(k):
        if k == 0:
            return "#f0efec"
        idx = sum(k >= t for t in steps) - 1
        return cmap(0.30 + 0.68 * idx / max(len(steps) - 1, 1))

    fig, ax = plt.subplots(figsize=(1.05 * len(levels) + 3.6,
                                    0.46 * len(classes) + 2.4))
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    for i in range(len(classes)):
        for j in range(len(levels)):
            if np.isnan(counts[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9,
                        color=ps.INK_SOFT)
                continue
            k = int(counts[i, j])
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88,
                                       facecolor=tile_colour(k), edgecolor="none"))
            ax.text(j, i, f"{k}/{int(totals[i, j])}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if k else "normal",
                    color="white" if k >= 4 else ps.INK)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"L{L}" for L in levels])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    for lbl, c in zip(ax.get_yticklabels(), classes):
        if c == run.total_class:
            lbl.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{args.metric} — regions passing p_adj < {args.alpha}, "
                 f"out of the regions tested\n"
                 f"one tile = one correction family; “—” = nothing tested",
                 fontsize=11)
    fig.tight_layout()
    ps.savefig(fig, out_path)
    return True


def _direction_handles(run):
    from matplotlib.lines import Line2D
    mk = lambda c, lbl: Line2D([], [], marker="o", linestyle="none", color=c,  # noqa: E731
                               markersize=8, markeredgecolor="white", label=lbl)
    return [mk(UP, f"higher in {run.group_name['b']} (passes correction)"),
            mk(DOWN, f"higher in {run.group_name['a']} (passes correction)"),
            mk(MUTED, "does not pass")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", default="both",
                    choices=["volcano", "forest", "summary", "both"])
    ap.add_argument("--all-classes", action="store_true",
                    help="draw every class this config declared, not just one")
    ap.add_argument("--class-name", default=None,
                    help="default: the config's own 'all cells' class "
                         "(all_cells on a marker run, MADM_all on a MADM run)")
    ap.add_argument("--metric", default="Density")
    ap.add_argument("--levels", default="2,3,5", help="volcano: one panel per level")
    ap.add_argument("--level", type=int, default=5, help="forest: a single family")
    ap.add_argument("--top", type=int, default=20,
                    help="forest: how many regions to show, ranked by |g|")
    ap.add_argument("--significant-only", action="store_true",
                    help="forest: show only regions passing the correction")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ps.apply_rcparams()
    run = Run(args.config)
    if args.alpha is None:
        args.alpha = run.alpha
    base = args.out_dir or os.path.join(run.out_dir, "figures")

    if args.kind == "summary":
        figure_summary(run, args, os.path.join(
            ps.figure_dir(base, "index"), f"summary_{args.metric}.png"))
        return


    classes = (run.all_classes() if args.all_classes
               else [args.class_name or run.total_class or "all_cells"])
    n = 0
    for cls in classes:
        args.class_name = cls
        if args.kind in ("volcano", "both"):
            # the directory is created here rather than up front so a run that
            # never draws a forest does not leave an empty forest/ behind
            n += bool(figure_volcano(run, args, os.path.join(
                ps.figure_dir(base, "volcano"), f"volcano_{cls}_{args.metric}.png")))
        if args.kind in ("forest", "both"):
            suffix = "sig" if args.significant_only else f"top{args.top}"
            n += bool(figure_forest(run, args, os.path.join(
                ps.figure_dir(base, "forest"),
                f"forest_{cls}_{args.metric}_L{args.level}_{suffix}.png")))
    print(f"{n} figure(s) for {len(classes)} class(es)")


if __name__ == "__main__":
    main()
