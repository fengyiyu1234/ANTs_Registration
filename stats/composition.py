"""Within-animal composition: each class as a fraction of all labelled cells.

WHY THIS EXISTS SEPARATELY
--------------------------
In this design the genotype is a property of the ANIMAL -- experimental is a
full TSC knockout, control is a heterozygote -- and GFP, RFP and GFP/RFP cells
in one animal all carry it. Colour says nothing about genotype, so there is no
internal control: injection dose, labelling efficiency, clearing and section
quality all land directly on any count. Measured, the GFP:RFP ratio spans 0.44
to 0.98 across six brains, which is the size of that technical term.

The one thing that does divide it out is a COMPOSITION: a class over all
labelled cells in the same animal. Labelling efficiency appears in numerator
and denominator and cancels. On this dataset that is not cosmetic -- absolute
Sox9+ density gives g = -0.50, the Sox9+ fraction gives g = -1.55.

Reads the block_counts_long.csv that block_stats.py writes, so it inherits
whatever blocks, QC and class map that run used.

    conda activate antsreg
    python -m stats.composition --run <block_stats output dir>
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def load_groups(run_dir):
    """Group membership is recorded per sample in the long table."""
    df = pd.read_csv(os.path.join(run_dir, "block_counts_long.csv"))
    return df


def compose(df, base_classes, numerators, samples_a, samples_b):
    tot = df.groupby(["sample", "class_name"])["Count"].sum().unstack()
    missing = [c for c in base_classes if c not in tot.columns]
    if missing:
        raise ValueError(f"这些基础类不在表里: {missing}")
    denom = tot[base_classes].sum(axis=1)
    rows = []
    for name, parts in numerators.items():
        frac = tot[parts].sum(axis=1) / denom
        a = frac[samples_a].to_numpy(float)
        b = frac[samples_b].to_numpy(float)
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        # Hedges' small-sample correction for n_a = n_b = 3.
        g = (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))
        p = float(sp_stats.ttest_ind(a, b, equal_var=False)[1])
        # Exact permutation over all C(6,3)/2 = 10 distinct splits. It cannot
        # go below 0.1, so it is a RANK, reported next to Welch rather than
        # instead of it: it makes no distributional assumption, which is worth
        # having when the whole sample is six numbers.
        v = np.concatenate([a, b])
        obs = abs(b.mean() - a.mean())
        ge = sum(abs(v[list(c)].mean() - v[[i for i in range(len(v)) if i not in c]].mean())
                 >= obs - 1e-12 for c in combinations(range(len(v)), len(a)))
        loo = {}
        for drop in list(samples_a) + list(samples_b):
            aa = [frac[s] for s in samples_a if s != drop]
            bb = [frac[s] for s in samples_b if s != drop]
            if len(aa) < 2 or len(bb) < 2:
                continue
            loo[drop] = float(np.log2(np.mean(bb) / np.mean(aa)))
        rows.append({
            "readout": name,
            **{f"frac_{s}": float(frac[s]) for s in list(samples_a) + list(samples_b)},
            "mean_a": a.mean(), "sd_a": a.std(ddof=1),
            "mean_b": b.mean(), "sd_b": b.std(ddof=1),
            "log2fc": float(np.log2(b.mean() / a.mean())),
            "hedges_g": float(g), "p_welch": p,
            "perm_rank": ge // 2, "perm_of": len(list(combinations(range(6), 3))) // 2,
            "p_perm": ge / len(list(combinations(range(6), 3))),
            "loo_min_log2fc": min(loo.values()), "loo_max_log2fc": max(loo.values()),
            "loo_keeps_sign": len({np.sign(v) for v in loo.values()}) == 1,
        })
    return pd.DataFrame(rows)


def readouts_from_config(cfg):
    """-> (base classes, {readout name: [base classes summing into it]}).

    Base classes come from the config's class_map and are mutually exclusive
    and exhaustive, so their sum is the denominator: every labelled cell.
    Readouts are each base class on its own plus every combined_categories
    entry, minus any entry that IS the whole denominator -- a fraction of
    everything by everything is 1.0 in every animal and only clutters the
    table.
    """
    base = list((cfg.get("class_map") or {}).keys())
    if not base:
        raise ValueError("config 里没有 class_map。")
    out = {f"{c}_fraction": [c] for c in base}
    for name, terms in (cfg.get("combined_categories") or {}).items():
        parts = []
        for t in terms:
            cls = t["class"] if isinstance(t, dict) else t
            sign = "+" if not isinstance(t, dict) else t.get("sign", "+")
            if sign != "+":
                parts = None
                break
            parts.append(cls)
        # A signed combination is not a subset of the labelled cells, so it is
        # not a composition and gets no row.
        if parts and set(parts) != set(base):
            out[f"{name}_fraction"] = parts
    return base, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="block_stats 的输出目录")
    ap.add_argument("--config", required=True,
                    help="那次 block_stats 用的 config，类和分组都从这里读")
    args = ap.parse_args()

    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stats.group_stats import load_config
    cfg = load_config(args.config)
    groups = cfg["groups"]
    a, b = groups["a"]["samples"], groups["b"]["samples"]
    name_a = groups["a"].get("name", "A")
    name_b = groups["b"].get("name", "B")
    base, numerators = readouts_from_config(cfg)

    df = load_groups(args.run)
    out = compose(df, base, numerators, a, b)
    path = os.path.join(args.run, "composition.csv")
    out.to_csv(path, index=False)

    show = out[["readout", "mean_a", "sd_a", "mean_b", "sd_b", "log2fc",
                "hedges_g", "p_welch", "perm_rank", "loo_keeps_sign"]]
    print(f"{name_a} {', '.join(a)}   {name_b} {', '.join(b)}")
    print(f"分母 = 同一只动物的全部标记细胞（{' + '.join(base)}），"
          f"所以标记效率被约掉\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nperm_rank 是 10 种分法里的排名，1 最强；置换 p 的下限是 0.10，"
          f"不是显著性判定")
    print(f"写出 -> {path}")


if __name__ == "__main__":
    main()
