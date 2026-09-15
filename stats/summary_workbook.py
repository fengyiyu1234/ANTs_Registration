"""One workbook per block_stats run, organised around three questions.

WHY THIS EXISTS
---------------
block_stats.py writes nine loosely related tables and composition.py writes a
tenth somewhere else, and nothing says which table answers which question. The
three questions actually being asked are:

    Q1  Each animal averaged over ALL boxes, then compared between groups.
        Counts, densities, and cell-type proportions. This is the analysis.
    Q2  The SAME box position compared between groups, box by box.
        This is the only thing that can say whether the cells sit elsewhere.
    Q3  Quality check: the same box position must not carry different amounts
        of TISSUE in the two groups. A box is fixed in atlas space so its
        volume is identical in every sample by construction -- what can differ
        is how much of it is tissue rather than a tear or a hole, and if that
        differs by group then every Q1 and Q2 number in that box is confounded.

Q3 is not decoration. Read it before Q1: a count difference in a box whose
tissue fraction also differs by group is a staining/clearing result, not a
biological one.

Nothing here treats boxes as replicates. Q1 averages boxes within an animal
first, so n stays 3 vs 3; Q2 is one 3 vs 3 test per box and is exploratory.

    conda activate antsreg
    python -m stats.summary_workbook \
        --run    /data/.../0911_03_stats_roi_mid50 \
        --config stats/configs/tsc_roi_mid50.yaml
"""
import argparse
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats.composition import readouts_from_config  # noqa: E402
from stats.group_stats import (adjust_pvalues, hedges_g, load_config,  # noqa: E402
                               two_sample_ttest)

LEVEL_METRICS = ("Count", "Density", "CoveredDensity")

# Output files are English-only. The comments and the terminal are not the
# deliverable; the workbook is, and it travels to people and tools that will not
# render CJK. Enforced rather than remembered -- a Chinese string added to a
# sheet three months from now should fail the run, not ship silently.
CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def _assert_ascii_text(frame, sheet):
    for col in frame.columns:
        if CJK.search(str(col)):
            raise ValueError(f"sheet {sheet}: column name is not English: {col}")
        if frame[col].dtype != object:
            continue
        for v in frame[col].dropna().unique():
            if isinstance(v, str) and CJK.search(v):
                raise ValueError(f"sheet {sheet}, column {col}: output files are "
                                 f"English-only, found: {v[:60]}")


# ── helpers ──────────────────────────────────────────────────────────────────
def _test_rows(mat_a, mat_b, labels, test, correction, alpha):
    """Welch (or t) + Hedges' g + BH, on rows of a (n_rows, n_samples) matrix."""
    mat_a, mat_b = np.asarray(mat_a, float), np.asarray(mat_b, float)
    p = two_sample_ttest(mat_a, mat_b, test=test)
    g, lo, hi = hedges_g(mat_a, mat_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma, mb = np.nanmean(mat_a, axis=1), np.nanmean(mat_b, axis=1)
        log2fc = np.log2(np.where(mb > 0, mb, np.nan) / np.where(ma > 0, ma, np.nan))
    out = pd.DataFrame(labels)
    out["mean_a"], out["sd_a"] = ma, np.nanstd(mat_a, axis=1, ddof=1)
    out["mean_b"], out["sd_b"] = mb, np.nanstd(mat_b, axis=1, ddof=1)
    out["diff_pct"] = (mb - ma) / np.where(ma != 0, ma, np.nan) * 100.0
    out["log2fc"], out["hedges_g"] = log2fc, g
    out["g_ci_lo"], out["g_ci_hi"] = lo, hi
    out["p_value"] = p
    out["p_adj"] = adjust_pvalues(p, correction)
    out["significant"] = out["p_adj"] < alpha
    out["correction"], out["m_family"], out["alpha"] = correction, len(p), alpha
    return out


def _permutation_rank(a, b):
    """Exact relabelling over every distinct split. -> (rank, of, p, floor).

    With three animals per group there are only C(6,3)/2 = 10 distinct splits,
    so the smallest p this design can return is 0.1. It is a RANK, not a
    significance test, and it is here because it assumes nothing about the
    distribution -- which is worth having when the whole sample is six numbers.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = np.concatenate([a, b])
    if np.isnan(v).any() or len(a) != len(b):
        return np.nan, np.nan, np.nan, np.nan
    obs = abs(b.mean() - a.mean())
    splits = list(combinations(range(len(v)), len(a)))
    ge = sum(abs(v[list(c)].mean() - v[[i for i in range(len(v)) if i not in c]].mean())
             >= obs - 1e-12 for c in splits)
    return ge // 2, len(splits) // 2, ge / len(splits), 2.0 / len(splits)


def _loo(series, samples_a, samples_b):
    """Drop each animal in turn. -> (worst surviving fraction, sign holds?).

    At 3 vs 3 one brain is a third of a group, so an effect that only exists
    with a particular brain included is a statement about that brain. Signed
    on purpose: a drop that flips the sign scores negative, not large."""
    full_a = np.nanmean([series[s] for s in samples_a])
    full_b = np.nanmean([series[s] for s in samples_b])
    if not (full_a > 0 and full_b > 0):
        return np.nan, np.nan
    base = np.log2(full_b / full_a)
    fracs = []
    for drop in list(samples_a) + list(samples_b):
        aa = [series[s] for s in samples_a if s != drop]
        bb = [series[s] for s in samples_b if s != drop]
        if len(aa) < 2 or len(bb) < 2:
            continue
        ma, mb = np.nanmean(aa), np.nanmean(bb)
        if not (ma > 0 and mb > 0):
            continue
        fracs.append(np.log2(mb / ma) / base if base != 0 else np.nan)
    if not fracs:
        return np.nan, np.nan
    return float(np.nanmin(fracs)), bool(np.all(np.sign(fracs) > 0))


# ── Q1: each animal averaged over all boxes ──────────────────────────────────
def per_sample_table(long, base, frac_readouts, samples):
    """-> tidy [readout, metric, <one column per sample>].

    Level metrics are the animal's MEAN over boxes. Proportions come in two
    flavours and both are reported, because they answer slightly different
    questions and can disagree when boxes hold very different cell numbers:

      Fraction_pooled    all boxes' counts summed, then divided. The estimator
                         with the smaller variance, and the one composition.py
                         uses. Big boxes weigh more.
      Fraction_boxmean   the fraction computed inside each box, then averaged.
                         Literally "average over the boxes". Every box weighs
                         the same, so a nearly empty box counts as much as a
                         full one -- noisier, and undefined where a box holds
                         no labelled cells at all.
    """
    rows = []
    for metric in LEVEL_METRICS:
        if metric not in long.columns:
            continue
        wide = long.pivot_table(index="class_name", columns="sample",
                                values=metric, aggfunc="mean")
        for cls, r in wide.iterrows():
            rows.append({"readout": cls, "metric": metric,
                         **{s: r.get(s, np.nan) for s in samples}})

    # Denominator = every labelled cell in the same animal, so labelling
    # efficiency, injection dose and clearing all cancel.
    tot = long.pivot_table(index="class_name", columns="sample",
                           values="Count", aggfunc="sum")
    denom = tot.loc[base].sum(axis=0)
    for name, parts in frac_readouts.items():
        frac = tot.loc[parts].sum(axis=0) / denom.replace(0, np.nan)
        rows.append({"readout": name, "metric": "Fraction_pooled",
                     **{s: float(frac.get(s, np.nan)) for s in samples}})

    per_box = long.pivot_table(index=["block", "class_name"], columns="sample",
                               values="Count", aggfunc="sum")
    box_denom = per_box.groupby(level="block").apply(
        lambda g: g.loc[[(b, c) for b, c in g.index if c in base]].sum())
    for name, parts in frac_readouts.items():
        num = per_box.groupby(level="block").apply(
            lambda g: g.loc[[(b, c) for b, c in g.index if c in parts]].sum())
        f = (num / box_denom.replace(0, np.nan)).mean(axis=0, skipna=True)
        rows.append({"readout": name, "metric": "Fraction_boxmean",
                     **{s: float(f.get(s, np.nan)) for s in samples}})
    return pd.DataFrame(rows)


def q1_group(per_sample, samples_a, samples_b, test, correction, alpha):
    out = []
    for metric, sub in per_sample.groupby("metric", sort=False):
        labels = [{"readout": r, "metric": metric} for r in sub["readout"]]
        res = _test_rows(sub[samples_a].to_numpy(float),
                         sub[samples_b].to_numpy(float),
                         labels, test, correction, alpha)
        perm, loo = [], []
        for _, r in sub.iterrows():
            perm.append(_permutation_rank([r[s] for s in samples_a],
                                          [r[s] for s in samples_b]))
            loo.append(_loo(r, samples_a, samples_b))
        res[["perm_rank", "perm_of", "p_perm", "p_perm_floor"]] = perm
        res[["loo_worst_frac", "loo_keeps_sign"]] = loo
        out.append(res)
    out = pd.concat(out, ignore_index=True)
    order = {m: i for i, m in enumerate(
        list(LEVEL_METRICS) + ["Fraction_pooled", "Fraction_boxmean"])}
    return (out.assign(_o=out["metric"].map(order))
            .sort_values(["_o", "p_value"]).drop(columns="_o")
            .reset_index(drop=True))


# ── Q2: the same box position, between groups ────────────────────────────────
def q2_per_box(long, base, frac_readouts, samples_a, samples_b, positions,
               test, correction, alpha):
    """One test per (box, readout, metric), BH within each (readout, metric).

    Exploratory by construction. It says WHERE, for an effect Q1 already
    supports; on its own, at 3 vs 3 and one test per box, it is a map of which
    box drew the lucky animal."""
    samples = list(samples_a) + list(samples_b)
    frames = []
    for metric in list(LEVEL_METRICS) + ["Share"]:
        if metric not in long.columns:
            continue
        for cls, sub in long.groupby("class_name", sort=False):
            wide = sub.pivot_table(index="block", columns="sample", values=metric)
            if not set(samples) <= set(wide.columns):
                continue
            wide = wide.dropna(subset=samples)
            if wide.empty:
                continue
            labels = [{"block": int(b), "readout": cls, "metric": metric}
                      for b in wide.index]
            frames.append(_test_rows(wide[samples_a].to_numpy(float),
                                     wide[samples_b].to_numpy(float),
                                     labels, test, correction, alpha))

    per_box = long.pivot_table(index=["block", "class_name"], columns="sample",
                               values="Count", aggfunc="sum")
    box_denom = per_box.groupby(level="block").apply(
        lambda g: g.loc[[(b, c) for b, c in g.index if c in base]].sum())
    for name, parts in frac_readouts.items():
        num = per_box.groupby(level="block").apply(
            lambda g: g.loc[[(b, c) for b, c in g.index if c in parts]].sum())
        wide = (num / box_denom.replace(0, np.nan)).dropna(subset=samples)
        if wide.empty:
            continue
        labels = [{"block": int(b), "readout": name, "metric": "Fraction"}
                  for b in wide.index]
        frames.append(_test_rows(wide[samples_a].to_numpy(float),
                                 wide[samples_b].to_numpy(float),
                                 labels, test, correction, alpha))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["exploratory"] = True
    if not positions.empty:
        out = out.merge(positions, on="block", how="left")
    return out.sort_values(["metric", "readout", "p_value"]).reset_index(drop=True)


def q2_values(long, base, frac_readouts, samples):
    """The numbers behind Q2, one row per (box, readout), samples as columns."""
    frames = []
    for metric in list(LEVEL_METRICS) + ["Share"]:
        if metric not in long.columns:
            continue
        w = long.pivot_table(index=["block", "class_name"], columns="sample",
                             values=metric).reset_index()
        w.insert(2, "metric", metric)
        frames.append(w.rename(columns={"class_name": "readout"}))
    per_box = long.pivot_table(index=["block", "class_name"], columns="sample",
                               values="Count", aggfunc="sum")
    box_denom = per_box.groupby(level="block").apply(
        lambda g: g.loc[[(b, c) for b, c in g.index if c in base]].sum())
    for name, parts in frac_readouts.items():
        num = per_box.groupby(level="block").apply(
            lambda g: g.loc[[(b, c) for b, c in g.index if c in parts]].sum())
        w = (num / box_denom.replace(0, np.nan)).reset_index()
        w.insert(1, "readout", name)
        w.insert(2, "metric", "Fraction")
        frames.append(w)
    out = pd.concat(frames, ignore_index=True)
    cols = ["block", "readout", "metric"] + [s for s in samples if s in out.columns]
    return out[cols].sort_values(["metric", "readout", "block"]).reset_index(drop=True)


# ── Q3: does the same box hold the same amount of tissue in both groups ──────
def q3_volume(darkness, block_mm3, samples_a, samples_b, test, correction, alpha,
              min_ratio=0.90, kept_blocks=None):
    """Group comparison of TISSUE volume inside each box.

    The box is defined in atlas space, so its voxel set -- and therefore its
    volume -- is byte-identical in every sample. The only thing that can differ
    is how much of that volume is tissue rather than a tear, a fold or a hole,
    which is what dark_frac measures on the warped intensity image. (The warped
    brain mask cannot do this job: it is an envelope and reads ~1.00 even on a
    box that is a fifth empty.)

    A box that fails here is disqualified as evidence, not merely noisy: a
    group difference in how much tissue is present produces a group difference
    in cell count with no biology in it at all. Read this sheet before Q1."""
    if darkness.empty:
        return pd.DataFrame()
    d = darkness.copy()
    d["block"] = d["block"].astype(int)
    if "tissue_frac" not in d:
        d["tissue_frac"] = 1.0 - d["dark_frac"]
    d["block_mm3"] = d["block"].map(block_mm3).astype(float)
    d["tissue_mm3"] = d["tissue_frac"] * d["block_mm3"]
    samples = list(samples_a) + list(samples_b)
    wide = d.pivot_table(index="block", columns="sample", values="tissue_mm3")
    if not set(samples) <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.dropna(subset=samples)
    if wide.empty:
        return pd.DataFrame()
    labels = [{"block": int(b), "metric": "tissue_mm3",
               "atlas_mm3": float(block_mm3.get(int(b), np.nan))}
              for b in wide.index]
    out = _test_rows(wide[samples_a].to_numpy(float),
                     wide[samples_b].to_numpy(float),
                     labels, test, correction, alpha)
    fr = d.pivot_table(index="block", columns="sample", values="tissue_frac")
    out["tissue_frac_min"] = fr.loc[wide.index, samples].min(axis=1).to_numpy()
    out["tissue_frac_max"] = fr.loc[wide.index, samples].max(axis=1).to_numpy()
    out["worst_sample"] = fr.loc[wide.index, samples].idxmin(axis=1).to_numpy()
    out["min_over_max"] = out["tissue_frac_min"] / out["tissue_frac_max"]
    out["ratio_b_over_a"] = out["mean_b"] / out["mean_a"].replace(0, np.nan)
    cc = d.pivot_table(index="block", columns="sample", values="dark_cc_frac")
    out["dark_cc_frac_max"] = cc.loc[wide.index, samples].max(axis=1).to_numpy()
    # Two ways to fail, catching different things. The group test catches a
    # systematic difference between the groups. The spread catches ONE torn
    # brain, which the test can miss precisely because that brain also widens
    # its own group's variance -- and one brain is a third of a group here.
    #
    # The spread threshold is qc.volume_min_ratio, the same number block_stats
    # used when it decided what to drop, so this sheet and the dropped list
    # agree instead of quietly disagreeing.
    out["fail_group_diff"] = out["p_adj"] < alpha
    out["fail_spread"] = out["min_over_max"] < min_ratio
    out["min_ratio_used"] = min_ratio
    out["usable"] = ~(out["fail_group_diff"] | out["fail_spread"])
    if kept_blocks is not None:
        out["in_stats"] = out["block"].isin(set(kept_blocks))
    return out.sort_values(["usable", "min_over_max", "p_value"]).reset_index(drop=True)


def q3_values(darkness, block_mm3, samples):
    if darkness.empty:
        return pd.DataFrame()
    d = darkness.copy()
    d["block"] = d["block"].astype(int)
    if "tissue_frac" not in d:
        d["tissue_frac"] = 1.0 - d["dark_frac"]
    d["block_mm3"] = d["block"].map(block_mm3).astype(float)
    d["tissue_mm3"] = d["tissue_frac"] * d["block_mm3"]
    frames = []
    for metric in ("tissue_mm3", "tissue_frac", "dark_frac", "dark_cc_frac"):
        w = d.pivot_table(index="block", columns="sample", values=metric).reset_index()
        w.insert(1, "metric", metric)
        frames.append(w)
    out = pd.concat(frames, ignore_index=True)
    cols = ["block", "metric"] + [s for s in samples if s in out.columns]
    return out[cols].sort_values(["metric", "block"]).reset_index(drop=True)


# ── assembly ─────────────────────────────────────────────────────────────────
def readme_frame(meta):
    return pd.DataFrame([
        ("what this is", "Three questions from one block_stats run, one sheet "
                         "group per question."),
        ("read in this order", "Q3 first, then Q1, then Q2. A box that fails Q3 "
                               "has missing tissue mixed into its Q1 and Q2 "
                               "numbers, and that is not biology."),
        ("", ""),
        ("Q1_Group", "Each animal averaged over ALL boxes first, then compared "
                     f"between groups. n is {meta['n_a']} vs {meta['n_b']}. "
                     "Boxes are never treated as replicates. This is the "
                     "analysis."),
        ("Q1_PerSample", "The numbers behind Q1: one row per (readout, metric), "
                         "one column per animal. Plot from this sheet."),
        ("Q2_PerBox", "The SAME box position compared between groups, one test "
                      "per (box, readout, metric), BH within each (readout, "
                      "metric). Exploratory: it says WHERE for an effect Q1 "
                      "already supports, and does not stand on its own."),
        ("Q2_PerBox_Values", "The numbers behind Q2."),
        ("Q2_Spatial", "Did the cells move. Share is a box's count over the "
                       "SAME animal's total across all boxes, so every "
                       "animal-level term divides out and only location is "
                       "left. I2 near 0 means the boxes are noisy copies of one "
                       "number and no per-box claim is supportable."),
        ("Q3_BoxVolume", "Does the same box position hold the same amount of "
                         "TISSUE in both groups. A box is defined in atlas "
                         "space, so its voxel set is identical in every animal "
                         "and its atlas volume cannot differ by construction; "
                         "what can differ is how much of it is tissue rather "
                         "than a tear or a hole. fail_group_diff is a "
                         "systematic difference between groups after BH. "
                         "fail_spread is min/max of the tissue fraction across "
                         "animals below qc.volume_min_ratio -- it catches ONE "
                         "torn brain, which the test can miss precisely because "
                         "that brain also widens its own group's variance. The "
                         "threshold is the same one block_stats used to decide "
                         "what to drop, and in_stats says whether the box "
                         "reached Q1 and Q2."),
        ("Q3_BoxVolume_PerSample", "Per box per animal: tissue volume, tissue "
                                   "fraction, dark fraction."),
        ("Blocks", "Box definitions: atlas volume, centre, dominant region."),
        ("", ""),
        ("Count", "Cells of that class inside the box."),
        ("Density", "Count / the box's atlas volume. For cubes the divisor is a "
                    "constant, so p and g cannot differ from Count; columns each "
                    "keep their own volume, and only there are the two genuinely "
                    "different statistics."),
        ("CoveredDensity", "Count / (volume x coverage). Coverage comes from the "
                           "warped brain mask, which is an envelope and reads "
                           "~1.00 almost everywhere, so it does NOT detect "
                           "cracks. What detects them is Q3's dark fraction."),
        ("Fraction_pooled", "Counts summed over all boxes, then divided. The "
                            "denominator is every labelled cell in the same "
                            "animal, so labelling efficiency cancels. Smaller "
                            "variance; big boxes weigh more."),
        ("Fraction_boxmean", "The fraction computed inside each box, then "
                             "averaged over boxes. Literally 'averaged over the "
                             "boxes'. Every box weighs the same, so it is "
                             "noisier, and it is undefined where a box holds no "
                             "labelled cells."),
        ("perm_rank", "Rank among all 10 distinct ways to split six animals, 1 "
                      "strongest. The permutation p cannot go below 0.10, so "
                      "this is a rank, not a significance call."),
        ("loo_worst_frac", "How much of the effect survives dropping any single "
                           "animal, signed. Near 0, or negative, means the "
                           "effect is one brain."),
        ("", ""),
        ("run", meta["run"]),
        ("config", meta["config"]),
        ("blocks", f"{meta['n_blocks']}, shape {meta['shape']}, median atlas "
                   f"volume {meta['median_mm3']:.4f} mm3, inside "
                   f"{meta['regions']}"),
        ("dropped", meta["dropped"]),
        ("groups", f"A {meta['name_a']} = {', '.join(meta['samples_a'])}   "
                   f"B {meta['name_b']} = {', '.join(meta['samples_b'])}"),
        ("sign of log2fc and g", "Positive means group B (experimental) is "
                                 "higher."),
        ("generated", meta["generated"]),
    ], columns=["item", "meaning"])


def build(run_dir, cfg, out_path):
    groups = cfg["groups"]
    samples_a = list(groups["a"]["samples"])
    samples_b = list(groups["b"]["samples"])
    samples = samples_a + samples_b
    st = cfg.get("stats") or {}
    test = st.get("test", "welch")
    correction = st.get("correction", "bh")
    alpha = float(st.get("alpha", 0.05))

    long = pd.read_csv(os.path.join(run_dir, "block_counts_long.csv"))
    long["block"] = long["block"].astype(int)
    base, frac_readouts = readouts_from_config(cfg)
    # A config edited after the run silently changes what the class names mean.
    # Fail loudly: every proportion in this workbook has those base classes as
    # its denominator, so a mismatch would not be a missing row, it would be a
    # wrong number.
    have = set(long["class_name"].unique())
    missing = [c for c in base if c not in have]
    if missing:
        raise SystemExit(
            f"The config's class_map does not match this run's tables.\n"
            f"  base classes the config asks for: {base}\n"
            f"  classes actually in the table:    {sorted(have)}\n"
            f"  missing: {missing}\n"
            f"This config was edited after that block_stats run. Re-run "
            f"block_stats with the current config first, then this script.")

    with open(os.path.join(run_dir, "blocks_used.json"), encoding="utf-8") as f:
        doc = json.load(f)
    block_mm3 = {int(b["block"]): float(b.get("volume_mm3", np.nan))
                 for b in doc["blocks"]}
    if not doc["blocks"] or np.isnan(list(block_mm3.values())[0]):
        one = float(doc.get("block_um", 0)) ** 3 / 1e9
        block_mm3 = {int(b["block"]): one for b in doc["blocks"]}

    pos_path = os.path.join(run_dir, "block_positions.csv")
    positions = pd.read_csv(pos_path) if os.path.exists(pos_path) else pd.DataFrame()
    if not positions.empty:
        positions["block"] = positions["block"].astype(int)

    qc_path = os.path.join(run_dir, "qc_blocks.csv")
    darkness = pd.read_csv(qc_path) if os.path.exists(qc_path) else pd.DataFrame()

    per_sample = per_sample_table(long, base, frac_readouts, samples)
    q1 = q1_group(per_sample, samples_a, samples_b, test, correction, alpha)
    q2 = q2_per_box(long, base, frac_readouts, samples_a, samples_b,
                    positions, test, correction, alpha)
    q2v = q2_values(long, base, frac_readouts, samples)
    q3 = q3_volume(darkness, block_mm3, samples_a, samples_b, test, correction,
                   alpha, float((cfg.get("qc") or {}).get("volume_min_ratio", 0.90)),
                   kept_blocks=sorted(long["block"].unique()))
    q3v = q3_values(darkness, block_mm3, samples)

    het_path = os.path.join(run_dir, "heterogeneity.csv")
    hetero = pd.DataFrame()
    if os.path.exists(het_path) and os.path.getsize(het_path) > 1:
        hetero = pd.read_csv(het_path)
    glob_path = os.path.join(run_dir, "redistribution_global.csv")
    if os.path.exists(glob_path):
        g = pd.read_csv(glob_path)
        hetero = (g if hetero.empty else hetero.merge(g, on="class_name", how="outer"))

    blocks = pd.DataFrame([
        {"block": int(b["block"]),
         "atlas_mm3": float(b.get("volume_mm3", np.nan)),
         "n_voxels": b.get("n_voxels"),
         "top_region": (b.get("composition") or [{}])[0].get("name", ""),
         "top_region_pct": (b.get("composition") or [{}])[0].get("pct", np.nan),
         "composition": "; ".join(f"{c['name']} {c['pct']}%"
                                  for c in (b.get("composition") or [])[:5])}
        for b in doc["blocks"]])
    if not positions.empty:
        blocks = blocks.merge(positions.drop(columns=["top_region"], errors="ignore"),
                              on="block", how="left")

    n_blocks = int(long["block"].nunique())
    meta = dict(
        run=run_dir, config=cfg.get("_path", ""), n_blocks=n_blocks,
        shape=doc.get("block_shape", "cube"),
        median_mm3=float(np.median([v for v in block_mm3.values()])),
        regions=", ".join(doc.get("regions", [])),
        dropped=(f"qc.action={(cfg.get('qc') or {}).get('action', 'report')}; "
                 f"boxes entering the statistics: "
                 f"{n_blocks} / {len(doc['blocks'])}"),
        n_a=len(samples_a), n_b=len(samples_b),
        name_a=groups["a"].get("name", "A"), name_b=groups["b"].get("name", "B"),
        samples_a=samples_a, samples_b=samples_b,
        generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))

    # One box has no "where": the same-position comparison IS Q1 and Share is
    # 1.0 in every sample by definition. Blank the per-box family rather than
    # report 44 tests that are Q1 wearing a different label.
    single = n_blocks < 2
    if single:
        q2, q2v = pd.DataFrame(), pd.DataFrame()
    for name, frame in (("Q1_Group", q1), ("Q1_PerSample", per_sample),
                        ("Q2_PerBox", q2), ("Q2_Spatial", hetero),
                        ("Q3_BoxVolume", q3), ("Blocks", blocks),
                        ("ReadMe", readme_frame(meta))):
        if not frame.empty:
            _assert_ascii_text(frame, name)
    with pd.ExcelWriter(out_path) as w:
        readme_frame(meta).to_excel(w, sheet_name="ReadMe", index=False)
        q1.to_excel(w, sheet_name="Q1_Group", index=False)
        per_sample.to_excel(w, sheet_name="Q1_PerSample", index=False)
        if single:
            # Say so in the sheet: an empty sheet reads like a bug.
            note = pd.DataFrame([{"note": "This run has a single box, so Q2 "
                                  "is undefined: the same-position comparison "
                                  "IS Q1 and Share is 1.0 in every animal. Use "
                                  "a multi-box layout for spatial questions."}])
            note.to_excel(w, sheet_name="Q2_PerBox", index=False)
        else:
            q2.to_excel(w, sheet_name="Q2_PerBox", index=False)
            q2v.to_excel(w, sheet_name="Q2_PerBox_Values", index=False)
            if not hetero.empty:
                hetero.to_excel(w, sheet_name="Q2_Spatial", index=False)
        if not q3.empty:
            q3.to_excel(w, sheet_name="Q3_BoxVolume", index=False)
            q3v.to_excel(w, sheet_name="Q3_BoxVolume_PerSample", index=False)
        blocks.to_excel(w, sheet_name="Blocks", index=False)
    return q1, q2, q3, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="block_stats output directory")
    ap.add_argument("--config", required=True,
                    help="the config THAT run used")
    ap.add_argument("--out-dir", default=None,
                    help="result directory, named <date>_<nth run that day>_"
                         "<feature>, e.g. 0912_01_summary_roi_mid50. "
                         "Defaults to the --run directory")
    ap.add_argument("--out", default=None,
                    help="explicit xlsx path; overrides --out-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_path"] = os.path.abspath(args.config)
    # A result directory is a unit: the workbook plus enough to say where it
    # came from. Without source.json a folder named by date and a guessed
    # feature is unreproducible three weeks later.
    out_dir = args.out_dir or args.run
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, "summary.xlsx")
    q1, q2, q3, meta = build(args.run, cfg, out)
    if args.out_dir:
        with open(os.path.join(out_dir, "source.json"), "w", encoding="utf-8") as f:
            json.dump({"block_stats_run": os.path.abspath(args.run),
                       "config": cfg["_path"],
                       "n_blocks": meta["n_blocks"],
                       "block_shape": meta["shape"],
                       "groups": {meta["name_a"]: meta["samples_a"],
                                  meta["name_b"]: meta["samples_b"]},
                       "generated": meta["generated"]}, f, indent=2,
                      ensure_ascii=False)

    print(f"{meta['name_a']} {', '.join(meta['samples_a'])}   "
          f"{meta['name_b']} {', '.join(meta['samples_b'])}   "
          f"{meta['n_blocks']} boxes")
    if q3.empty:
        print("\nQ3 volume check: skipped -- this run wrote no "
              "qc_blocks.csv. It was added to block_stats later; re-run "
              "block_stats to get it.")
    else:
        bad = q3[~q3["usable"]]
        print(f"\nQ3 box volume check: {len(q3) - len(bad)} / {len(q3)} "
              f"boxes pass")
        if len(bad):
            print(bad[["block", "mean_a", "mean_b", "ratio_b_over_a",
                       "p_value", "p_adj", "tissue_frac_min", "tissue_frac_max",
                       "min_over_max", "worst_sample", "fail_group_diff",
                       "fail_spread", "in_stats"]]
                  .head(15).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    show = q1[q1.metric.isin(["Count", "Fraction_pooled"])]
    print("\nQ1 group comparison (Count and Fraction_pooled)")
    print(show[["readout", "metric", "mean_a", "mean_b", "log2fc", "hedges_g",
                "p_value", "p_adj", "perm_rank", "loo_keeps_sign"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    if not q2.empty:
        sig = q2[q2["significant"]]
        print(f"\nQ2 per box: {len(q2)} tests, {len(sig)} significant after "
              f"BH, {int((q2.p_value < 0.05).sum())} at nominal p<0.05 "
              f"(chance expectation {0.05 * len(q2):.1f})")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
