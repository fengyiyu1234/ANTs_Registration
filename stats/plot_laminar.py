"""Bar charts for a laminar.py run: one figure per (scheme, pooling, metric).

Layout: one ROW per readout, one COLUMN per depth bin. The bins of a
LaminarShare readout sum to 1 -- a rise in one bin is a fall in another -- so
they sit side by side in one row and are read together, never as separate
findings. Each panel is the plot_bars panel: group bars (mean +- SD), every
animal as a dot with a fixed marker, raw p and p_adj (family = the bins of that
row), and the permutation rank out of 10, which at 3 vs 3 is the more honest
number (p_perm cannot go below 0.1).

Plus a QC figure of the ORIGINAL layer labels per animal, because the binning
hides exactly the L1 problem (s10 high, s12t low) that carries the apparent
superficial -> deep shift.

    conda activate antsreg
    python -m stats.plot_laminar --run /data/hdd12tb-1/fengyi/COMBINe/stats/0914_03_laminar_iso
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import plot_style as ps  # noqa: E402
from stats.plot_bars import Item, _legend_handles, both_p_text, draw_panel  # noqa: E402

READOUT_ORDER = ["MADM_all", "glia_all", "non_glia_all", "Sox9_pos", "Sox9_neg",
                 "glia_MADM", "glia_MADM_Sox9", "non_glia_MADM", "non_glia_MADM_Sox9"]

UNITS = {
    "LaminarShare": "% of this class's\n{root} cells",
    "BinComposition": "% of all labelled\ncells in the bin",
    "Count": "cells",
}
SUBTITLE = {
    "LaminarShare": "bins in a row sum to 100% — read them together",
    "BinComposition": "class share within each bin",
    "Count": "raw counts, reference only (carry labelling efficiency)",
}
TOKEN_ORDER = ["1", "2/3", "4", "5", "6a", "6b", "unassigned"]


class LaminarRun:
    """The bits of plot_bars.Run that draw_panel and _legend_handles touch."""

    def __init__(self, run_dir):
        with open(os.path.join(run_dir, "config_used.yaml")) as f:
            self.cfg = yaml.safe_load(f)
        ga, gb = self.cfg["groups"]["a"], self.cfg["groups"]["b"]
        self.group_name = {"a": ga.get("name", "A"), "b": gb.get("name", "B")}
        self.group_samples = {"a": list(ga["samples"]), "b": list(gb["samples"])}
        self.samples = self.group_samples["a"] + self.group_samples["b"]
        self.marker = ps.sample_style(self.samples)
        self.alpha = float((self.cfg.get("stats") or {}).get("alpha", 0.05))
        self.tests = pd.read_csv(os.path.join(run_dir, "laminar_tests.csv"))
        self.token_qc = pd.read_csv(os.path.join(run_dir, "laminar_token_qc.csv"),
                                    dtype={"token": str})
        self.schemes = (self.cfg.get("laminar") or {}).get("schemes") or {}
        self.root = self._root_acronym(run_dir)
        slab = (self.cfg.get("laminar") or {}).get("slab")
        self.slab = self._slab_note(run_dir, slab) if slab else ""

    def _slab_note(self, run_dir, slab):
        """A run that counts only part of the root must say so on every figure:
        the shares are of the slab, not of the whole structure."""
        qc = pd.read_csv(os.path.join(run_dir, "laminar_slab_qc.csv"))
        lo, hi = qc["lo"].min(), qc["hi"].max()
        thick = qc["thickness_um"]
        mode = str(slab.get("mode", "quantile"))
        return (f"{slab.get('axis', 'yt')} {lo:.0f}-{hi:.0f} ({mode}), "
                f"{thick.min():.0f}-{thick.max():.0f} um thick, "
                f"{qc['frac_kept'].min() * 100:.0f}-{qc['frac_kept'].max() * 100:.0f}% of cells")

    def _root_acronym(self, run_dir):
        """The territory the whole run is inside, for the axis labels and titles.
        Taken from the layer map rather than hard-coded: a run restricted to one
        area must not be labelled 'Isocortex'."""
        root_id = int((self.cfg.get("laminar") or {}).get("root_id", 315))
        lmap = pd.read_csv(os.path.join(run_dir, "laminar_layer_map.csv"))
        hit = lmap.loc[lmap["id"] == root_id, "acronym"]
        return str(hit.iloc[0]) if len(hit) else f"id {root_id}"


def _undefined(r, samples):
    """area_mean leaves a row empty when no area clears `area_min_cells` in every
    animal -- e.g. glia_MADM_Sox9 inside SS, where s12t holds 123 cells in all of
    it. Draw the gap instead of crashing on the NaN."""
    return all(pd.isna(r[s]) for s in samples)


def _item(r, samples, scale):
    note = (f"perm rank {int(r['perm_rank'])}/{int(r['perm_of'])}"
            if pd.notna(r["perm_rank"]) else "")
    if not bool(r["loo_keeps_sign"]):
        note += (" · " if note else "") + "sign flips leaving one out"
    return Item(r["bin"], {s: float(r[s]) * scale for s in samples},
                p_adj=float(r["p_adj"]), p_raw=float(r["p_value"]),
                g=float(r["hedges_g"]), note=note)


def grid_figure(run, scheme, pooling, metric, out_path):
    import matplotlib.pyplot as plt

    sub = run.tests[(run.tests["scheme"] == scheme) & (run.tests["pooling"] == pooling)
                    & (run.tests["metric"] == metric)]
    if sub.empty:
        return
    bins = list(run.schemes.get(scheme) or dict.fromkeys(sub["bin"]))
    present = set(sub["readout"])
    readouts = ([r for r in READOUT_ORDER if r in present]
                + [r for r in dict.fromkeys(sub["readout"]) if r not in READOUT_ORDER])
    scale = 1.0 if metric == "Count" else 100.0

    nrows, ncols = len(readouts), len(bins)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols + 0.9, 3.45 * nrows),
                             squeeze=False)
    for i, ro in enumerate(readouts):
        for j, b in enumerate(bins):
            ax = axes[i, j]
            row = sub[(sub["readout"] == ro) & (sub["bin"] == b)]
            if row.empty or _undefined(row.iloc[0], run.samples):
                ax.axis("off")
                if not row.empty:
                    ax.text(0.5, 0.5, f"{b}\nundefined:\nno area reaches\narea_min_cells\nin every animal",
                            ha="center", va="center", fontsize=8, color=ps.INK_SOFT,
                            transform=ax.transAxes)
                continue
            draw_panel(ax, run, _item(row.iloc[0], run.samples, scale), metric,
                       run.alpha, p_text=both_p_text)
        root = f"{run.root} slab" if run.slab else run.root
        axes[i, 0].set_ylabel(f"{ro}\n{UNITS.get(metric, metric).format(root=root)}",
                              fontsize=9.5)
        axes[i, 0].yaxis.label.set_fontweight("bold")

    role = sub["role"].iloc[0]
    handles = _legend_handles(run)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               bbox_to_anchor=(0.5, 0.0))
    slab = f"\nslab: {run.slab}" if run.slab else ""
    fig.suptitle(f"{run.root} laminar — {metric}, {scheme}, {pooling}  [{role}]\n"
                 f"{SUBTITLE.get(metric, '')}; BH family = the bins of one row{slab}",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985), h_pad=2.2)
    ps.savefig(fig, out_path)


def token_qc_figure(run, out_path):
    """One panel per original layer label, one bar per animal."""
    import matplotlib.pyplot as plt

    qc = run.token_qc.set_index("token")
    tokens = [t for t in TOKEN_ORDER if t in qc.index]
    fig, axes = plt.subplots(1, len(tokens), figsize=(2.3 * len(tokens), 3.4), squeeze=False)
    for ax, t in zip(axes[0], tokens):
        vals = [float(qc.loc[t, s]) * 100 for s in run.samples]
        for k, (s, v) in enumerate(zip(run.samples, vals)):
            key = "a" if s in run.group_samples["a"] else "b"
            ax.bar(k, v, width=0.7, color=ps.GROUP_FILL[key],
                   edgecolor=ps.GROUP_COLORS[key], linewidth=1.2, zorder=1)
            ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                    color=ps.INK_SOFT)
        ax.set_xticks(range(len(run.samples)))
        labels = ax.set_xticklabels(run.samples, rotation=90, fontsize=8)
        for lbl, s in zip(labels, run.samples):
            lbl.set_color(ps.GROUP_COLORS["a" if s in run.group_samples["a"] else "b"])
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
        ax.set_title(f"layer {t}" if t != "unassigned" else t, fontsize=10)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel(f"% of the animal's\n{run.root} cells"
                          + (" in the slab" if run.slab else ""))
    fig.suptitle(f"{run.root} — cells on each ORIGINAL layer label, per animal "
                 f"({run.group_name['a']} blue, {run.group_name['b']} orange)\n"
                 "a thin-layer registration / detection problem shows up here; "
                 "once layers are binned it no longer can", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ps.savefig(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="laminar.py output directory")
    ap.add_argument("--out-dir", default=None, help="default: <run>/figures")
    args = ap.parse_args()

    ps.apply_rcparams()
    run = LaminarRun(args.run)
    out = args.out_dir or os.path.join(args.run, "figures")
    token_qc_figure(run, os.path.join(out, "00_layer_label_qc.png"))
    combos = (run.tests[["role", "scheme", "pooling", "metric"]].drop_duplicates())
    order = {"primary": 0, "secondary": 1, "reference": 2}
    combos = combos.assign(o=combos["role"].map(order)).sort_values(
        ["scheme", "o", "pooling", "metric"])
    for c in combos.itertuples(index=False):
        grid_figure(run, c.scheme, c.pooling, c.metric,
                    os.path.join(out, c.scheme, f"{c.role}_{c.metric}_{c.pooling}.png"))


if __name__ == "__main__":
    main()
