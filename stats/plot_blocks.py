"""Figures for stats/block_stats.py -- the block analysis as bars, not a sheet.

Three figures, and each answers a different question. Reading them in this
order is the point; taken out of order the third one lies.

    primary_<metric>       one panel per class, two bars, six dots. HOW MANY
                           cells are in the blocks. This is the analysis, and
                           it is the only one of the three with a defensible
                           p-value at n = 3 vs 3.
    blocks_<class>         every block side by side, plus that block's log2fc
                           underneath. WHERE. Exploratory by construction: 28
                           tests at 3 vs 3, and the blocks are 600 um apart so
                           they are not independent of each other either.
    gradient_<class>       the same per-block effects against the anatomical
                           axis instead of against the block number. A
                           migration phenotype is a gradient; this is the plot
                           that would show one, and the flat cloud it usually
                           shows is a real answer.

Two things this module deliberately refuses to draw.

**No stars.** ps.p_label prints the adjusted p and the family it came from.
With six animals a star is an invitation to read a bar as a finding.

**No bar without its dots.** At n=3 a bar and an SD whisker are two numbers
wearing the costume of a distribution. The dots are the data; every sample
keeps the same marker across every figure here and in stats/plot_bars.py, so
one animal can be followed from one figure to the next.

Colour comes from stats/plot_style.py unchanged: groups are categorical (two
fixed hues by group letter), a signed log2fc is polarity (blue below zero, red
above, near-white at zero). Nothing in here invents a colour.

Usage:
    conda activate antsreg
    python -m stats.plot_blocks --config stats/configs/tsc_blocks.yaml
    python -m stats.plot_blocks --config ... --metrics Count,Share \
        --classes glia_all,non_glia_all
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import plot_bars, plot_style as ps  # noqa: E402
from stats.group_stats import load_config  # noqa: E402

# Share is stored as a fraction and drawn as a percentage: 0.036 on an axis is
# unreadable, 3.6% is not. SCALE is applied to the values AND to nothing else,
# so the tests behind the panel are untouched.
METRIC_UNITS = {
    "Count": "cells in block",
    "Density": "cells / mm$^3$",
    "CoveredDensity": "cells / mm$^3$ of tissue",
    "Share": "% of that sample's cells in blocks",
}
SCALE = {"Share": 100.0}

# Axis names as tools/pick_blocks.py derives them on the native DeMBA P5 grid.
AXIS_LABEL = {
    "AP": "anterior - posterior (mm)",
    "DV": "dorsal - ventral (mm)",
    "ML_from_midline": "distance from midline (mm)",
}
# The bar chart's x is a block id, not a coordinate, so it says which way the
# blocks were sorted and not what the numbers mean.
ORDER_LABEL = {"AP": "front to back", "DV": "dorsal to ventral",
               "ML_from_midline": "midline outwards"}


class BlockRun:
    """Everything one block_stats run leaves on disk, plus the config's groups.

    Reads the CSVs rather than taking the in-memory result, so the same code
    serves the end of a run and a redraw three days later."""

    def __init__(self, config_path):
        self.cfg = load_config(config_path)
        self.out_dir = (self.cfg.get("output") or {}).get("dir", "./stats_output")
        long_path = os.path.join(self.out_dir, "block_counts_long.csv")
        if not os.path.exists(long_path):
            raise SystemExit(f"{long_path} 不存在 -- 先跑 stats.block_stats")
        self.long = pd.read_csv(long_path)
        self.primary = self._read("primary.csv")
        self.per_block = self._read("per_block.csv")
        self.share_block = self._read("redistribution.csv")
        self.share_global = self._read("redistribution_global.csv")
        self.hetero = self._read("heterogeneity.csv")
        self.positions = self._read("block_positions.csv")

        ga, gb = self.cfg["groups"]["a"], self.cfg["groups"]["b"]
        self.group_name = {"a": ga.get("name", "A"), "b": gb.get("name", "B")}
        self.group_samples = {"a": list(ga["samples"]), "b": list(gb["samples"])}
        self.samples = self.group_samples["a"] + self.group_samples["b"]
        self.marker = ps.sample_style(self.samples)
        self.alpha = float((self.cfg.get("stats") or {}).get("alpha", 0.05))
        self.fig_dir = os.path.join(self.out_dir, "figures")

    def _read(self, name):
        path = os.path.join(self.out_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def classes(self):
        return list(dict.fromkeys(self.long["class_name"]))

    def block_order(self, axis="AP"):
        """Blocks sorted front to back. Anatomy is the only ordering a reader
        can reason about; block number is the order pick_blocks happened to
        draw them in, which is nothing."""
        if self.positions.empty:
            return sorted(self.long["block"].unique())
        return list(self.positions.sort_values(axis)["block"])

    def tests_for(self, metric):
        """-> the per-block test table for this metric. Share lives in its own
        file because it is a different family, tested against a different
        null."""
        src = self.share_block if metric == "Share" else self.per_block
        if src.empty:
            return pd.DataFrame()
        return src[src.get("metric", metric) == metric] if "metric" in src else src

    def values(self, cls, metric):
        """-> {block: {sample: value}}, already scaled for display."""
        sub = self.long[self.long.class_name == cls]
        wide = sub.pivot_table(index="block", columns="sample", values=metric)
        return wide * SCALE.get(metric, 1.0)


def primary_figure(run, metric, out_dir):
    """One panel per class: each sample's MEAN over all blocks, tested 3 v 3."""
    items = []
    for cls in run.classes():
        wide = run.values(cls, metric)
        if wide.empty or wide.isna().all().all():
            continue
        means = wide.mean(axis=0)
        row = run.primary[(run.primary.class_name == cls)
                          & (run.primary.metric == metric)] if not run.primary.empty \
            else pd.DataFrame()
        p_adj = float(row["p_adj"].iloc[0]) if len(row) else np.nan
        p_raw = float(row["p_value"].iloc[0]) if len(row) else np.nan
        g = float(row["hedges_g"].iloc[0]) if len(row) else np.nan
        note = f"g = {g:+.2f}" if np.isfinite(g) else None
        items.append(plot_bars.Item(
            cls, {s: float(means.get(s, np.nan)) for s in run.samples},
            p_adj=p_adj, p_raw=p_raw, g=g, note=note))
    if not items:
        return
    n_blocks = run.long["block"].nunique()
    # plot_bars labels the y-axis from its own METRIC_UNITS, which knows the
    # whole-region metrics and not these; setdefault teaches it Share and the
    # per-block wording without overriding anything it already defines.
    plot_bars.METRIC_UNITS.setdefault(metric, METRIC_UNITS.get(metric, metric))
    plot_bars.figure(
        run, items, metric,
        f"{metric} per block, averaged over {n_blocks} blocks — "
        f"{run.group_name['a']} vs {run.group_name['b']} (n = 3 vs 3)",
        os.path.join(out_dir, f"primary_{metric}.png"), alpha=run.alpha)


def _hetero_note(run, cls, metric):
    if run.hetero.empty:
        return ""
    row = run.hetero[(run.hetero.class_name == cls)
                     & (run.hetero.metric == metric)]
    if not len(row):
        return ""
    r = row.iloc[0]
    bits = []
    # the permutation p is the one to print: the chi2 and Spearman p beside it
    # assume known SEs and independent blocks, and neither holds here
    if np.isfinite(r.get("I2", np.nan)):
        q = r.get("p_Q_perm", np.nan)
        tail = f"perm p = {q:.2g}" if np.isfinite(q) else f"χ² p = {r['p_heterogeneity']:.2g}"
        bits.append(f"I² = {r['I2'] * 100:.0f}%  ({tail})")
    rho = r.get("rho_AP", np.nan)
    if np.isfinite(rho):
        q = r.get("p_AP_perm", np.nan)
        tail = f"perm p = {q:.2g}" if np.isfinite(q) else f"p = {r['p_AP']:.2g}"
        bits.append(f"ρ vs A-P = {rho:+.2f} ({tail})")
    return "   ·   ".join(bits)


def blocks_figure(run, cls, metric, out_dir, axis="AP"):
    """Every block side by side, with that block's log2fc underneath.

    Two stacked panels sharing one x, never two y-scales on one axis: the
    counts and the log ratio are different measurements and a twin axis would
    let the reader compare heights that are not comparable."""
    import matplotlib.pyplot as plt

    wide = run.values(cls, metric)
    order = [b for b in run.block_order(axis) if b in wide.index]
    if not order:
        return
    wide = wide.loc[order]
    tests = run.tests_for(metric)
    tests = tests[tests.class_name == cls] if not tests.empty else pd.DataFrame()
    q = (tests.set_index("block")["p_adj"] if len(tests) else pd.Series(dtype=float))
    fc = (tests.set_index("block")["log2fc"] if len(tests) else pd.Series(dtype=float))

    x = np.arange(len(order), dtype=float)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(max(7.0, 0.46 * len(order) + 2.2), 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12})

    for sign, key in ((-1, "a"), (1, "b")):
        vals = wide[run.group_samples[key]].to_numpy(float)
        mean = np.nanmean(vals, axis=1)
        sd = np.nanstd(vals, axis=1, ddof=1)
        ax.bar(x + sign * 0.20, mean, width=0.36, color=ps.GROUP_FILL[key],
               edgecolor=ps.GROUP_COLORS[key], linewidth=1.0, zorder=1)
        ax.errorbar(x + sign * 0.20, mean, yerr=sd, fmt="none",
                    ecolor=ps.GROUP_COLORS[key], elinewidth=1.0, capsize=2.5,
                    capthick=1.0, zorder=2)
        for i, s in enumerate(run.group_samples[key]):
            off = (i - (len(run.group_samples[key]) - 1) / 2) * 0.085
            ax.plot(x + sign * 0.20 + off, wide[s].to_numpy(float),
                    marker=run.marker[s], markersize=3.6, markerfacecolor="white",
                    markeredgecolor=ps.INK, markeredgewidth=0.7, linestyle="none",
                    zorder=3)

    ax.set_ylabel(METRIC_UNITS.get(metric, metric))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.8, len(order) - 0.2)

    # a passing block gets a mark and its q printed, not a star: at 3 v 3 with
    # this many blocks the mark means "look here next", not "this is true"
    hits = [i for i, b in enumerate(order)
            if np.isfinite(q.get(b, np.nan)) and q.get(b) < run.alpha]
    top = ax.get_ylim()[1]
    for i in hits:
        ax.plot(x[i], top * 0.965, marker="v", markersize=6,
                color=ps.INK, linestyle="none", zorder=4)
        ax.text(x[i], top * 0.995, f"q={q[order[i]]:.3g}", ha="center",
                va="bottom", fontsize=6.5, color=ps.INK)
    ax.set_ylim(0, top * 1.10)

    vals = np.array([fc.get(b, np.nan) for b in order], dtype=float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    cmap = ps.diverging_cmap()
    colors = [cmap(0.5 + 0.5 * (v / lim)) if np.isfinite(v) else ps.NO_RESULT
              for v in vals]
    ax2.bar(x, np.nan_to_num(vals), width=0.66, color=colors,
            edgecolor=ps.INK_SOFT, linewidth=0.4, zorder=2)
    ax2.axhline(0, color=ps.INK_SOFT, linewidth=0.8, zorder=1)
    pooled = (run.hetero[(run.hetero.class_name == cls)
                         & (run.hetero.metric == metric)]
              if not run.hetero.empty else pd.DataFrame())
    if len(pooled) and np.isfinite(pooled["pooled_log2fc"].iloc[0]):
        ax2.axhline(float(pooled["pooled_log2fc"].iloc[0]), color=ps.INK,
                    linewidth=1.0, linestyle="--", zorder=3,
                    label="pooled over blocks")
        ax2.legend(loc="upper right", fontsize=7.5)
    ax2.set_ylabel("log2 fold change")
    ax2.set_ylim(-lim * 1.25, lim * 1.25)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(b)) for b in order], fontsize=7)
    ax2.set_xlabel(f"block id, ordered {ORDER_LABEL[axis]}")

    handles = plot_bars._legend_handles(run)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               bbox_to_anchor=(0.5, -0.035))
    note = _hetero_note(run, cls, metric)
    fig.suptitle(f"{cls} — {metric} in each block\n{note}" if note
                 else f"{cls} — {metric} in each block", fontsize=11.5, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.155)
    ps.savefig(fig, os.path.join(out_dir, f"blocks_{cls}_{metric}.png"))


def gradient_figure(run, cls, metric, out_dir):
    """Per-block effect against position on each anatomical axis.

    The question this answers is the one 28 separate t-tests cannot: a
    migration phenotype moves cells ALONG an axis, so it shows up as a slope
    here even when no single block passes its own test. A flat cloud is
    likewise informative -- it says there is no tangential gradient to find,
    and the blocks are noisy copies of one number."""
    import matplotlib.pyplot as plt

    tests = run.tests_for(metric)
    tests = tests[tests.class_name == cls] if not tests.empty else pd.DataFrame()
    if tests.empty or run.positions.empty:
        return
    sub = tests.merge(run.positions, on="block", how="left")
    row = (run.hetero[(run.hetero.class_name == cls)
                      & (run.hetero.metric == metric)]
           if not run.hetero.empty else pd.DataFrame())

    axes_here = [a for a in ("AP", "DV", "ML_from_midline") if a in sub.columns]
    fig, axs = plt.subplots(1, len(axes_here), figsize=(3.5 * len(axes_here), 3.4),
                            squeeze=False, sharey=True)
    cmap = ps.diverging_cmap()
    lim = float(np.nanmax(np.abs(sub["log2fc"]))) or 1.0
    for ax, axis in zip(axs[0], axes_here):
        xv = sub[axis].to_numpy(float)
        yv = sub["log2fc"].to_numpy(float)
        ax.axhline(0, color=ps.INK_SOFT, linewidth=0.8, zorder=1)
        ax.scatter(xv, yv, s=46, zorder=3, linewidths=0.8, edgecolors=ps.INK,
                   c=[cmap(0.5 + 0.5 * (v / lim)) if np.isfinite(v) else ps.NO_RESULT
                      for v in yv])
        ok = np.isfinite(xv) & np.isfinite(yv)
        if ok.sum() >= 3:
            b, a0 = np.polyfit(xv[ok], yv[ok], 1)
            xs = np.linspace(xv[ok].min(), xv[ok].max(), 2)
            ax.plot(xs, a0 + b * xs, color=ps.INK, linewidth=1.2, zorder=2)
        ax.set_xlabel(AXIS_LABEL[axis])
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if len(row):
            rho = row.iloc[0].get(f"rho_{axis}", np.nan)
            perm = row.iloc[0].get(f"p_{axis}_perm", np.nan)
            p = perm if np.isfinite(perm) else row.iloc[0].get(f"p_{axis}", np.nan)
            kind = "perm p" if np.isfinite(perm) else "p"
            if np.isfinite(rho):
                ax.set_title(f"ρ = {rho:+.2f}, {kind} = {p:.2g}", fontsize=9.5,
                             color=ps.INK_SOFT)
    axs[0, 0].set_ylabel("log2 fold change, one point per block")
    fig.suptitle(f"{cls} — {metric}: is the difference a gradient?\n"
                 "perm p relabels the six animals all 10 ways, so it cannot go "
                 "below 0.1 — read the slope and the rank, not significance",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    ps.savefig(fig, os.path.join(out_dir, f"gradient_{cls}_{metric}.png"))


def render_all(config_path, metrics=None, classes=None):
    """Every figure for one run. Called at the end of stats.block_stats and by
    this module's own main(). Takes the config PATH, not the loaded config, so
    that the two entry points resolve output.dir the same way."""
    ps.apply_rcparams()
    run = BlockRun(config_path)
    have = [m for m in ("Count", "Density", "CoveredDensity", "Share")
            if m in run.long.columns]
    metrics = [m for m in (metrics or ["Count", "Share"]) if m in have]
    classes = classes or run.classes()
    bars = ps.figure_dir(run.fig_dir, "bars")
    forest = ps.figure_dir(run.fig_dir, "forest")
    for metric in metrics:
        primary_figure(run, metric, bars)
        for cls in classes:
            if run.values(cls, metric).isna().all().all():
                continue
            blocks_figure(run, cls, metric, bars)
            gradient_figure(run, cls, metric, forest)
    return run.fig_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", default="Count,Share",
                        help="逗号分隔，默认 Count,Share")
    parser.add_argument("--classes", default=None, help="逗号分隔，默认全部")
    args = parser.parse_args()
    out = render_all(args.config,
                     metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
                     classes=([c.strip() for c in args.classes.split(",")]
                              if args.classes else None))
    print(f"\n图写到 -> {out}")


if __name__ == "__main__":
    main()
