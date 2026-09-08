"""Group comparison of per-region cell counts and densities.

Successor to ClearMap/stats_vis/stats_group_compare.py, rebuilt on this
project's own outputs -- no ClearMap import anywhere. What changed and why:

  ontology      ClearMap's `ano` -> stats.ontology.Ontology, built from
                atlas/DeMBA/CCF_v3_ontology.json (see that module).
  region column cell_registration.csv column 9 is a RAW CCF id here, not
                ClearMap's graph_order (see stats.cell_tables).
  volumes       volume/result.mhd -> <run>/*_labels_in_sample.nii.gz, with
                <run>/*_brain_mask.nii.gz giving per-region tissue coverage
                (see stats.region_volumes).
  classes       classify_by: marker collapses the YOLO neuron/glia call and
                keeps only the marker co-expression signature.
  statistics    Welch + BH as before, plus Hedges' g with a CI, because at
                n=3 vs 3 a p-value alone is not readable (see below).

Reading the statistics at n=3 vs 3
----------------------------------
* An exact permutation test cannot work here at all: 6 samples split 3/3 give
  C(6,3)/2 = 10 distinct labelings, so the smallest attainable two-sided
  permutation p is 1/10 = 0.1. That is why this runs Welch's t-test (which
  buys power by assuming normality) and reports effect sizes next to it.
* Welch at n=3 has ~2-4 degrees of freedom; reaching p<0.05 needs roughly
  |Hedges' g| ~ 1.8 for a RAW p<0.05, and the threshold climbs steeply once
  BH divides alpha by the family size: |g| ~ 4.2 at m=17, ~6.7 at m=98.
  Testing every region at every level then FDR-correcting will normally leave
  nothing. Use region_filter to state a small region set up front -- family
  size is the dominant term in what survives.
* Report mean_a/mean_b, log2fc and hedges_g with its CI. Rank by effect size,
  use p_adj as a filter, not as the result.

Usage:
    conda activate antsreg
    python -m stats.group_stats --config stats/configs/tsc_marker.yaml
"""
import argparse
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import special, stats as sp_stats

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.ontology import Ontology  # noqa: E402
from stats.region_volumes import build_per_sample_volumes, build_reference_volumes  # noqa: E402

METRICS = ("Count", "Percentage", "Density", "RegionProportion",
           "Volume", "RelativeVolume")
CORRECTIONS = ("bh", "holm", "bonferroni", "none")
LOST_LABEL = "Lost cells"


# ================= config =================

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("ontology_json", "samples", "groups"):
        if key not in cfg:
            raise ValueError(f"config is missing required key '{key}'")
    root = os.path.dirname(os.path.abspath(path))

    def _abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

    cfg["ontology_json"] = _abs(cfg["ontology_json"])
    for name, entry in cfg["samples"].items():
        if "dir" not in entry:
            raise ValueError(f"sample '{name}' has no 'dir'")
        entry["dir"] = _abs(entry["dir"])
        if not os.path.isdir(entry["dir"]):
            raise FileNotFoundError(f"sample '{name}': {entry['dir']} does not exist")
    for key in ("a", "b"):
        for s in cfg["groups"][key]["samples"]:
            if s not in cfg["samples"]:
                raise ValueError(f"group {key} lists sample '{s}', which has no samples: entry")

    ref = cfg.get("reference_annotation")
    if ref and ref.get("path"):
        ref["path"] = _abs(ref["path"])
    out = cfg.get("output")
    if out:
        for key in ("dir", "volume_cache"):
            if out.get(key):
                out[key] = _abs(out[key])
    return cfg


# ================= counting =================

def collect_counts(sample_dirs, classes, group_samples, ontology, classify_by,
                   exclude_mask=None):
    """-> (counts_df, direct_counts_df, totals_df).

    counts are hierarchical rollups; direct counts are not (they drive the
    'Lost cells' rows). totals_df carries n_valid and n_total per (sample,
    class) so the out-of-atlas fraction stays visible downstream instead of
    being silently dropped here."""
    orders = np.arange(ontology.n)
    count_frames, direct_frames, total_rows = [], [], []
    for group_key, samples in group_samples.items():
        for sample in samples:
            sdir = sample_dirs[sample]
            for cls in classes:
                dirs = cell_tables.resolve_class_dirs(sdir, cls, classify_by)
                if not dirs:
                    print(f"  [warn] {sample}: no folder matching class '{cls}', treating as zero")
                bins = np.zeros(ontology.n)
                direct = np.zeros(ontology.n)
                n_valid = n_total = n_unknown = n_excluded = 0
                for d in dirs:
                    b, dd, nv, nt, nu, nx = cell_tables.class_counts(
                        cell_tables.class_csv_path(sdir, d), ontology,
                        exclude_mask=exclude_mask)
                    bins += b
                    direct += dd
                    n_valid += nv
                    n_total += nt
                    n_unknown += nu
                    n_excluded += nx
                if n_unknown:
                    print(f"  [warn] {sample}/{cls}: {n_unknown} cells carry a structure id "
                          "absent from the ontology and are excluded")
                count_frames.append(pd.DataFrame({
                    "group": group_key, "sample": sample, "class_name": cls,
                    "order": orders, "count": bins}))
                direct_frames.append(pd.DataFrame({
                    "group": group_key, "sample": sample, "class_name": cls,
                    "order": orders, "direct_count": direct}))
                total_rows.append({
                    "group": group_key, "sample": sample, "class_name": cls,
                    "source_dirs": "+".join(sorted(dirs)),
                    "total_valid": n_valid, "total_cells": n_total,
                    "n_background": n_total - n_valid - n_unknown - n_excluded,
                    "n_unknown_id": n_unknown, "n_excluded_region": n_excluded})
    return (pd.concat(count_frames, ignore_index=True),
            pd.concat(direct_frames, ignore_index=True),
            pd.DataFrame(total_rows))


def combine_categories(counts_df, totals_df, direct_df, formulas, classes):
    """Signed (+/-) aggregates over existing classes, e.g. Sox9_any =
    GFP_Sox9 + RFP_Sox9 + GFP_RFP_Sox9.

    Exact at every level because hierarchical rollup is linear:
    rollup(A) + rollup(B) == rollup(A + B) for every region, so the arithmetic
    can be done on the already-rolled-up counts. The marker classes are
    mutually exclusive (one composite label per physical cell, upstream in
    brain_detector), so in practice every term is '+'; signed terms exist so
    the same engine stays correct for class sets that do nest.

    -> (combined_counts, combined_totals, combined_direct, resolved_formulas)"""
    empty = (counts_df.iloc[0:0], totals_df.iloc[0:0], direct_df.iloc[0:0], {})
    if not formulas:
        return empty

    def _weights(terms, name):
        w = {}
        for term in terms:
            sign = 1 if str(term.get("sign", "+")).strip() == "+" else -1
            c = term["class"]
            if c not in classes:
                key = cell_tables.normalize_class_key(c)
                match = next((cl for cl in classes
                              if cell_tables.normalize_class_key(cl) == key), None)
                if match is None:
                    print(f"  [warn] combined category '{name}': component '{c}' not among "
                          f"{classes}, skipped")
                    continue
                c = match
            w[c] = w.get(c, 0) + sign
        return w

    c_frames, t_frames, d_frames, resolved = [], [], [], {}
    for name, terms in formulas.items():
        w = _weights(terms, name)
        if not w:
            print(f"  [warn] combined category '{name}': no valid components, skipped")
            continue

        sub = counts_df[counts_df["class_name"].isin(w)].copy()
        sub["count"] *= sub["class_name"].map(w)
        agg = sub.groupby(["group", "sample", "order"], as_index=False)["count"].sum()
        agg["class_name"] = name
        c_frames.append(agg[["group", "sample", "class_name", "order", "count"]])

        subd = direct_df[direct_df["class_name"].isin(w)].copy()
        subd["direct_count"] *= subd["class_name"].map(w)
        aggd = subd.groupby(["group", "sample", "order"], as_index=False)["direct_count"].sum()
        aggd["class_name"] = name
        d_frames.append(aggd[["group", "sample", "class_name", "order", "direct_count"]])

        subt = totals_df[totals_df["class_name"].isin(w)].copy()
        tcols = ["total_valid", "total_cells", "n_background", "n_unknown_id",
                 "n_excluded_region"]
        for col in tcols:
            subt[col] = subt[col] * subt["class_name"].map(w)
        aggt = subt.groupby(["group", "sample"], as_index=False)[tcols].sum()
        aggt["class_name"] = name
        aggt["source_dirs"] = ""
        t_frames.append(aggt[totals_df.columns])

        resolved[name] = " ".join(
            (c if s > 0 else f"-{c}") if i == 0 else (f"+ {c}" if s > 0 else f"- {c}")
            for i, (c, s) in enumerate(w.items()))

    if not c_frames:
        return empty
    return (pd.concat(c_frames, ignore_index=True),
            pd.concat(t_frames, ignore_index=True),
            pd.concat(d_frames, ignore_index=True),
            resolved)


# ================= metrics =================

def region_totals(counts_df, base_classes):
    """Per (sample, order) sum over the mutually-exclusive base classes --
    the denominator of RegionProportion. Combined categories are excluded on
    purpose: they overlap each other, so summing them would count cells more
    than once and the proportions would not close to 100%."""
    sub = counts_df[counts_df["class_name"].isin(base_classes)]
    return sub.groupby(["sample", "order"], as_index=False)["count"].sum().rename(
        columns={"count": "region_total"})


def metric_matrix(counts_df, totals_df, volumes, region_tot, class_name, metric,
                  group_key, samples, ontology, density_denominator):
    """(n_orders, n_samples) matrix for one class/metric/group, rows ordered
    by `order`, columns by `samples`."""
    n = ontology.n
    sub = counts_df[(counts_df["class_name"] == class_name) & (counts_df["group"] == group_key)]
    mat = (sub.pivot(index="order", columns="sample", values="count")
           .reindex(index=np.arange(n), columns=samples).to_numpy(dtype=float))

    if metric == "Count":
        return mat

    if metric == "Percentage":
        tot = (totals_df[(totals_df["class_name"] == class_name)
                         & (totals_df["group"] == group_key)]
               .set_index("sample")["total_valid"].reindex(samples).to_numpy(dtype=float))
        out = np.full_like(mat, np.nan)
        ok = tot > 0
        out[:, ok] = mat[:, ok] / tot[ok] * 100.0
        return out

    if metric == "RegionProportion":
        den = (region_tot.pivot(index="order", columns="sample", values="region_total")
               .reindex(index=np.arange(n), columns=samples).to_numpy(dtype=float))
        out = np.full_like(mat, np.nan)
        ok = den > 0
        out[ok] = mat[ok] / den[ok] * 100.0
        return out

    if metric == "RelativeVolume":
        # Share of the volume of the regions under analysis (excluded subtrees
        # are already gone from the root total -- see region_volumes.rollup_volumes),
        # so it is comparable across brains of different size and across
        # samples whose field of view differs.
        col = "relative_covered_pct" if density_denominator == "covered" else "relative_pct"
        return (volumes.pivot(index="order", columns="sample", values=col)
                .reindex(index=np.arange(n), columns=samples).to_numpy(dtype=float))

    if metric in ("Density", "Volume"):
        col = "covered_volume_mm3" if density_denominator == "covered" else "volume_mm3"
        vol = (volumes.pivot(index="order", columns="sample", values=col)
               .reindex(index=np.arange(n), columns=samples).to_numpy(dtype=float))
        if metric == "Volume":
            return vol
        out = np.full_like(mat, np.nan)
        ok = vol > 0
        out[ok] = mat[ok] / vol[ok]
        return out

    raise ValueError(f"unknown metric: {metric}")


def coverage_matrix(volumes, samples, ontology):
    return (volumes.pivot(index="order", columns="sample", values="coverage")
            .reindex(index=np.arange(ontology.n), columns=samples).to_numpy(dtype=float))


# ================= statistics =================

def benjamini_hochberg(p):
    """BH step-up FDR. NaN p-values are carried through as 1.0 (a region with
    no variance to test is not evidence of anything)."""
    p = np.asarray(p, dtype=float).copy()
    p[np.isnan(p)] = 1.0
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def welch_ttest(a, b):
    """Backwards-compatible alias for two_sample_ttest(..., test="welch")."""
    return two_sample_ttest(a, b, test="welch")


def _inv_trigamma(x):
    """Smyth (2004) appendix: Newton iteration for trigamma^-1."""
    if not np.isfinite(x) or x <= 0:
        return np.inf
    y = 0.5 + 1.0 / x
    for _ in range(60):
        tri = special.polygamma(1, y)
        step = tri * (1 - tri / x) / special.polygamma(2, y)
        y += step
        if abs(step / y) < 1e-10:
            break
    return y


# A prior fitted from a handful of regions is noise, not information. limma is
# normally run on thousands of features; here a family is 8-40 regions. Below
# MIN_FEATURES the variance prior is not estimable and the test falls back to
# Welch; between that and MIN_FOR_TREND a single global prior is used, because
# a lowess trend through ~15 points would follow the points rather than a trend.
# Measured consequence of NOT having this guard: level 2 (a family of 8-9)
# produced prior_df between 0.6 and 49.1 across families -- i.e. anywhere from
# no shrinkage to almost total shrinkage, decided by noise.
MIN_FEATURES_FOR_MODERATION = 12
MIN_FEATURES_FOR_TREND = 25


def moderated_ttest(a, b, trend=True, lowess_frac=0.6):
    """Empirical-Bayes moderated t-test (limma; Smyth 2004), per family.

    Why this exists: at n=3 vs 3 every region's test has 4 residual degrees of
    freedom, and that -- not multiplicity -- is what stops deep levels from
    reaching a corrected threshold. Shrinking each region's variance toward a
    prior fitted across the regions in the same family buys back degrees of
    freedom (measured here: 4 -> 7-9), which is enough to move levels 7 and 8
    from zero survivors to a dozen.

    What it assumes, and why it is defensible here: the regions in one family
    share a variance distribution. Borrowing across them is legitimate because
    WITHIN a single ontology level the regions are disjoint -- verified by the
    identity sum(rollup at L) == root - sum(direct above L). It would NOT be
    legitimate across levels, where a cell is counted in every ancestor.

    trend=True fits the prior variance as a function of the mean (limma-trend)
    instead of using one global prior. That matters: cell density's residual
    variance rises with the mean (Spearman rho = +0.57 at level 8 on this
    data), and with a global prior the number of survivors swings wildly with
    an arbitrary log transform (11 vs 1 at level 7). With the trend the two
    scales agree to within a couple of regions, which is the sign the prior is
    no longer absorbing a mean-variance relationship it should be modelling.

    MEASURED CALIBRATION -- read this before using it. The marginal p-values
    are correct: under a simulated global null they reject at 5.2% (m=50),
    5.0% (m=200), 5.0% (m=5000) against a nominal 5%, so the estimator itself
    is right. But BH on top of them is NOT calibrated at the family sizes this
    project has. Under the global null, P(at least one BH rejection) should be
    <= 5%; measured over 1500 simulations at n=3 vs 3:

        m=15  global prior 9.9%   trend 10.6%
        m=30  global prior 7.7%   trend 14.9%
        m=50  global prior 6.5%   trend 11.3%
        m=200 global prior 5.3%   (converges as m grows)

    The cause is that every region shares the same estimated prior (s0^2, d0),
    which is a common-mode dependence BH does not absorb; the trend makes it
    worse because it estimates more shared structure. For contrast, Welch on
    the same data rejects at only 3.2-3.6% marginally -- it is CONSERVATIVE at
    n=3. So the two bracket the truth rather than one being right:

        welch + BH        conservative, misses real effects
        moderated_t + BH  liberal, FWER 1.3-3x nominal at these family sizes

    Regions found by both are the defensible ones. Do not report a
    moderated_t-only hit as error-controlled.

    Returns (p, info) where info records the fitted prior for the report.
    """
    n1, n2 = a.shape[1], b.shape[1]
    d = n1 + n2 - 2
    if d <= 0:
        return np.ones(a.shape[0]), {}
    s2 = ((n1 - 1) * np.nanvar(a, axis=1, ddof=1)
          + (n2 - 1) * np.nanvar(b, axis=1, ddof=1)) / d
    ok = np.isfinite(s2) & (s2 > 0)
    p = np.ones(a.shape[0], dtype=float)
    if ok.sum() < MIN_FEATURES_FOR_MODERATION:
        return two_sample_ttest(a, b, test="welch"), {
            "fallback": f"welch ({int(ok.sum())} regions < {MIN_FEATURES_FOR_MODERATION})"}

    z = np.log(s2[ok])
    e = z - special.digamma(d / 2) + np.log(d / 2)
    means = np.nanmean(np.concatenate([a, b], axis=1), axis=1)[ok]

    used_trend = bool(trend and ok.sum() >= MIN_FEATURES_FOR_TREND)
    if used_trend:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        loc = lowess(e, means, frac=lowess_frac, return_sorted=False)
        resid = e - loc
    else:
        loc = np.full_like(e, e.mean())
        resid = e - e.mean()

    G = len(e)
    rhs = float(np.mean(resid ** 2) * G / (G - 1) - special.polygamma(1, d / 2))
    if rhs <= 0:
        d0 = np.inf
        s02 = np.exp(loc + special.digamma(d / 2) - np.log(d / 2))
    else:
        d0 = 2 * _inv_trigamma(rhs)
        s02 = np.exp(loc + special.digamma(d0 / 2) - np.log(d0 / 2))

    s2_mod = s02 if np.isinf(d0) else (d0 * s02 + d * s2[ok]) / (d0 + d)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (np.nanmean(b, axis=1)[ok] - np.nanmean(a, axis=1)[ok]) / np.sqrt(
            s2_mod * (1.0 / n1 + 1.0 / n2))
    df_total = d + (d0 if np.isfinite(d0) else 1e6)
    p[ok] = 2 * sp_stats.t.sf(np.abs(t), df_total)
    p[~np.isfinite(p)] = 1.0
    return p, {"d0": float(d0), "df_total": float(df_total), "n_features": int(G),
               "trend": used_trend}


def holm(p):
    """Holm-Bonferroni step-down. Controls the family-wise error rate -- the
    probability of even one false positive -- and is uniformly more powerful
    than plain Bonferroni, so there is no reason to prefer Bonferroni.

    Use it where the family is small and the claim needs to be strong: at the
    coarse ontology levels there are 8-18 regions, and those are the
    pre-specified primary hypotheses."""
    p = np.asarray(p, dtype=float).copy()
    p[np.isnan(p)] = 1.0
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)
    out = np.empty(m, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def adjust_pvalues(p, method):
    """Dispatch for the per-level correction. 'none' returns the raw p-values
    unchanged -- legitimate only for levels reached through gatekeeping and
    reported as exploratory (see run_all)."""
    method = (method or "none").lower()
    if method == "bh":
        return benjamini_hochberg(p)
    if method == "holm":
        return holm(p)
    if method == "bonferroni":
        q = np.asarray(p, dtype=float).copy()
        q[np.isnan(q)] = 1.0
        return np.clip(q * len(q), 0.0, 1.0)
    if method == "none":
        q = np.asarray(p, dtype=float).copy()
        q[np.isnan(q)] = 1.0
        return q
    raise ValueError(f"unknown correction '{method}'; known: {CORRECTIONS}")


TESTS = ("welch", "student", "moderated_t")


def two_sample_ttest(a, b, test="welch"):
    """Two-sample t-test per row, with the degenerate case forced to p=1.

    test="welch" (default) does not assume the groups share a variance and
    uses the Welch-Satterthwaite degrees of freedom. At n=3 vs 3 that is 4.0
    when the variances match and falls toward 2.4 as they diverge, so it costs
    almost nothing when Student's assumption holds and protects the error rate
    when it does not -- and disease groups are routinely more variable than
    controls. test="student" pools the variances and always gets df=4; use it
    only if you can argue the variances are equal.

    When BOTH groups have zero variance, Welch's standard error is zero and
    scipy returns t=inf, p=0.0 -- so "1 cell in every control, 0 in every
    experimental" comes out as p=0 and sails through FDR. Two constant groups
    carry no information about variability, so their difference is
    unmeasurable, not infinitely significant. This is the same condition under
    which hedges_g is NaN, and it is common in small regions where counts are
    0 or 1.

    NaN p-values (scipy's own degenerate output) are likewise treated as
    non-significant."""
    if test not in TESTS:
        raise ValueError(f"unknown test '{test}'; known: {TESTS}")
    equal_var = test == "student"
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # scipy warns "catastrophic cancellation ... data are nearly identical"
        # for exactly the near-constant rows the degenerate guard below turns
        # into p=1, so the warning has nothing left to tell us
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p = sp_stats.ttest_ind(a, b, axis=1, equal_var=equal_var, nan_policy="omit")
    p = np.asarray(p, dtype=float)
    p[np.isnan(p)] = 1.0
    with np.errstate(invalid="ignore"):
        degenerate = ((np.nanvar(a, axis=1, ddof=1) == 0)
                      & (np.nanvar(b, axis=1, ddof=1) == 0))
    p[degenerate] = 1.0
    return p


def hedges_g(a, b):
    """Bias-corrected standardised mean difference (b - a) with an approximate
    95% CI. Small-sample correction matters here: at n=3/3 Cohen's d
    overstates the effect by ~15%.

    NaN where the pooled SD is zero (both groups constant) -- not an infinite
    effect, just an unmeasurable one."""
    na = np.sum(~np.isnan(a), axis=1)
    nb = np.sum(~np.isnan(b), axis=1)
    ma, mb = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        va, vb = np.nanvar(a, axis=1, ddof=1), np.nanvar(b, axis=1, ddof=1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = np.where(pooled > 0, (mb - ma) / pooled, np.nan)
        J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
        g = J * d
        var_g = (na + nb) / (na * nb) + g ** 2 / (2.0 * (na + nb - 2))
        se = np.sqrt(var_g)
    return g, g - 1.96 * se, g + 1.96 * se


def run_level_tests(mat_a, mat_b, metadata, keep_orders, level, class_name, metric,
                    samples_a, samples_b, count_gate=None, correction="bh",
                    test="welch", alpha=0.05, variance_trend=True):
    """Test the regions of one ontology level, then BH-correct within that
    level alone -- each level is its own hypothesis family, and their sizes
    differ by an order of magnitude (18 regions at level 2, 324 at level 7)."""
    gate = keep_orders if count_gate is None else (keep_orders & count_gate)
    idx = np.where((metadata["level"].to_numpy() == level) & gate)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    a, b = mat_a[idx], mat_b[idx]
    total = np.nansum(a, axis=1) + np.nansum(b, axis=1)
    testable = (np.sum(~np.isnan(a), axis=1) >= 2) & (np.sum(~np.isnan(b), axis=1) >= 2)
    keep = (total > 0) & testable
    idx, a, b = idx[keep], a[keep], b[keep]
    if len(idx) == 0:
        return pd.DataFrame()

    mean_a, mean_b = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = np.where(mean_a > 0, mean_b / mean_a, np.nan)
        log2fc = np.log2(fold, out=np.full_like(fold, np.nan), where=fold > 0)

    mod_info = {}
    if test == "moderated_t":
        p, mod_info = moderated_ttest(a, b, trend=variance_trend)
    else:
        p = two_sample_ttest(a, b, test=test)
    # The effect size stays on the raw per-region variance on purpose: it is a
    # description of this region, not a test statistic, and shrinking it would
    # make the reported effect depend on the other regions in the family.
    g, g_lo, g_hi = hedges_g(a, b)

    out = metadata.iloc[idx].reset_index(drop=True).copy()
    out["class_name"] = class_name
    out["metric"] = metric
    out["test"] = "welch" if mod_info.get("fallback") else test
    out["test_note"] = mod_info.get("fallback", "")
    out["prior_df"] = mod_info.get("d0", np.nan)
    out["test_df"] = mod_info.get("df_total", float(len(samples_a) + len(samples_b) - 2))
    out["n_a"] = np.sum(~np.isnan(a), axis=1)
    out["n_b"] = np.sum(~np.isnan(b), axis=1)
    out["mean_a"], out["mean_b"] = mean_a, mean_b
    out["sd_a"] = np.nanstd(a, axis=1, ddof=1)
    out["sd_b"] = np.nanstd(b, axis=1, ddof=1)
    out["fold_change"], out["log2fc"] = fold, log2fc
    out["hedges_g"], out["g_ci_lo"], out["g_ci_hi"] = g, g_lo, g_hi
    out["p_value"] = p
    out["correction"] = correction
    out["m_family"] = len(idx)
    out["p_adj"] = adjust_pvalues(p, correction)

    # Family-level enrichment, carried on every row of the family.
    #
    # This is the only interpretable thing an UNCORRECTED level has to offer.
    # Gatekeeping narrows where you look, but it controls no error rate: with
    # m regions tested at alpha, m*alpha of them reach p<alpha by chance even
    # if every one is null. So a per-region claim is not available there --
    # but a claim about the SET is: seeing n_raw_sig hits where only
    # expected_false were predicted is itself testable (binomial, one-sided),
    # and expected_false / n_raw_sig estimates what fraction of the list is
    # noise. Report those two numbers instead of pretending the p-values were
    # corrected.
    n_raw = int((p < alpha).sum())
    m = len(idx)
    out["alpha"] = alpha
    out["n_raw_sig_in_family"] = n_raw
    out["expected_false_in_family"] = m * alpha
    out["family_enrichment_p"] = (
        float(sp_stats.binomtest(n_raw, m, alpha, alternative="greater").pvalue)
        if m > 0 else np.nan)
    out["est_false_frac_in_family"] = (min(1.0, m * alpha / n_raw) if n_raw > 0 else np.nan)
    # An uncorrected level is descriptive: it is only defensible when reached
    # through gatekeeping, and its rows must never be called "significant".
    out["exploratory"] = (correction or "none").lower() == "none"
    for i, s in enumerate(samples_a):
        out[s] = a[:, i]
    for j, s in enumerate(samples_b):
        out[s] = b[:, j]
    return out


def run_group_comparison(counts_df, totals_df, volumes, region_tot, metadata, class_names,
                         samples_a, samples_b, ontology, keep_orders, levels,
                         metrics, density_denominator):
    rows = []
    for cls in class_names:
        for metric in metrics:
            ma = metric_matrix(counts_df, totals_df, volumes, region_tot, cls, metric,
                               "a", samples_a, ontology, density_denominator)
            mb = metric_matrix(counts_df, totals_df, volumes, region_tot, cls, metric,
                               "b", samples_b, ontology, density_denominator)
            for level in levels:
                res = run_level_tests(ma, mb, metadata, keep_orders, level, cls, metric,
                                      samples_a, samples_b)
                if not res.empty:
                    rows.append(res)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan)


# ================= region filtering =================

def build_exclude_mask(cfg, ontology):
    """Regions dropped from the analysis entirely, from region_filter.exclude_ids.

    Computed before anything else because both the cell counts and the region
    volumes have to have it applied to their DIRECT (un-rolled) values -- see
    the note in build_region_filter."""
    rf = cfg.get("region_filter") or {}
    mask = np.zeros(ontology.n, dtype=bool)
    exclude_ids = rf.get("exclude_ids")
    if exclude_ids:
        mask[ontology.descendants_of(exclude_ids)] = True
        names = ", ".join(str(ontology.names[o]) for o in ontology.order_of_ids(exclude_ids)
                          if o >= 0)
        print(f"Excluded from the whole analysis: {int(mask.sum())} regions under [{names}]")
    return mask


def build_region_filter(cfg, volumes, metadata, ontology, samples, exclude_mask):
    """-> (keep_orders boolean array, levels to test, per-region drop report).

    Coverage is applied by NaN-ing the offending (region, sample) cells before
    the tests, not by dropping the region outright -- so a region truncated in
    one sample is still tested on the rest, and n_a/n_b in the output record
    how many samples actually contributed."""
    rf = cfg.get("region_filter") or {}
    keep = np.ones(ontology.n, dtype=bool)

    include_ids = rf.get("include_ids")
    if include_ids:
        sel = np.zeros(ontology.n, dtype=bool)
        sel[ontology.descendants_of(include_ids)] = True
        keep &= sel
        print(f"Region whitelist: {len(include_ids)} root id(s) -> {int(sel.sum())} regions")

    # Exclusions are applied to the DIRECT counts before rollup (see
    # cell_tables.class_counts / region_volumes.rollup_volumes), so an excluded
    # subtree leaves the analysis entirely -- it is gone from every ancestor's
    # count and volume, and from the Percentage and RelativeVolume denominators.
    # That is what "exclude before computing" has to mean for structures lost
    # during preparation: leaving them in the denominators would make every
    # other region's share depend on how much of the cerebellum survived.
    keep &= ~exclude_mask

    levels = rf.get("levels")
    if levels is None:
        levels = list(range(int(metadata["level"].max()) + 1))
    else:
        levels = [int(v) for v in levels]

    min_count = float(rf.get("min_total_count", 0))
    min_cov = float(rf.get("min_coverage", 0.0))
    cov = coverage_matrix(volumes, samples, ontology)
    dropped = (cov < min_cov) if min_cov > 0 else np.zeros_like(cov, dtype=bool)
    with warnings.catch_warnings():
        # a region absent from every sample is an all-NaN row; NaN is the right
        # answer for it, so the "All-NaN slice" warning is just noise
        warnings.simplefilter("ignore", RuntimeWarning)
        min_cov_seen = np.nanmin(cov, axis=1) if cov.size else np.full(ontology.n, np.nan)
    report = pd.DataFrame({
        "order": np.arange(ontology.n),
        "n_samples_below_min_coverage": dropped.sum(axis=1),
        "min_coverage_seen": min_cov_seen,
    })
    return keep, levels, min_cov, min_count, report


def apply_coverage_mask(mat, cov, min_cov):
    if min_cov <= 0:
        return mat
    out = mat.copy()
    out[cov < min_cov] = np.nan
    return out


# ================= orchestration =================

def run_all(cfg):
    ontology = Ontology.from_json(cfg["ontology_json"])
    metadata = ontology.metadata_frame()

    groups = cfg["groups"]
    name_a = groups["a"].get("name", "A")
    name_b = groups["b"].get("name", "B")
    samples_a = list(groups["a"]["samples"])
    samples_b = list(groups["b"]["samples"])
    all_samples = samples_a + samples_b
    sample_dirs = {s: cfg["samples"][s]["dir"] for s in all_samples}

    n_perm = math.comb(len(all_samples), len(samples_a)) // 2
    print(f"Group A ({name_a}): {samples_a}")
    print(f"Group B ({name_b}): {samples_b}")
    print(f"  n={len(samples_a)} vs {len(samples_b)}: an exact permutation test could not go "
          f"below p={1.0 / n_perm:.3g} here; Welch's t-test is used, effect sizes reported.")

    classify_by = cfg.get("classify_by", "marker")
    classes = cell_tables.discover_classes(
        [sample_dirs[s] for s in all_samples], classify_by, cfg.get("classes"))
    print(f"classify_by={classify_by}; classes: {classes}")
    for problem in cell_tables.check_class_resolution(sample_dirs, classes, classify_by):
        print(f"  [WARN] {problem}")

    out_dir = (cfg.get("output") or {}).get("dir", "./stats_output")
    os.makedirs(out_dir, exist_ok=True)

    exclude_mask = build_exclude_mask(cfg, ontology)

    volumes = build_per_sample_volumes(
        sample_dirs, ontology,
        use_mask=bool(cfg.get("use_brain_mask", True)),
        cache_path=(cfg.get("output") or {}).get("volume_cache")
        or os.path.join(out_dir, "per_sample_region_direct_volumes.csv"),
        force=bool(cfg.get("force_recompute_volumes", False)),
        exclude_mask=exclude_mask)

    counts_df, direct_df, totals_df = collect_counts(
        sample_dirs, classes, {"a": samples_a, "b": samples_b}, ontology, classify_by,
        exclude_mask=exclude_mask)

    comb_counts, comb_totals, comb_direct, formulas = combine_categories(
        counts_df, totals_df, direct_df, cfg.get("combined_categories"), classes)
    if formulas:
        print(f"Combined categories: {formulas}")

    region_tot = region_totals(counts_df, classes)
    keep_orders, levels, min_cov, min_count, cov_report = build_region_filter(
        cfg, volumes, metadata, ontology, all_samples, exclude_mask)
    metrics = [m for m in (cfg.get("metrics") or METRICS)]
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric '{m}'; known: {METRICS}")
    density_denominator = cfg.get("density_denominator", "covered")

    cov_a = coverage_matrix(volumes, samples_a, ontology)
    cov_b = coverage_matrix(volumes, samples_b, ontology)

    stats_cfg = cfg.get("stats") or {}
    alpha = float(stats_cfg.get("alpha", 0.05))
    test = str(stats_cfg.get("test", "welch")).lower()
    if test not in TESTS:
        raise ValueError(f"unknown stats.test '{test}'; known: {TESTS}")
    variance_trend = bool(stats_cfg.get("variance_trend", True))
    if test == "moderated_t":
        print(f"  test=moderated_t: each family's per-region variance is shrunk toward a "
              f"prior fitted across that family"
              f"{' as a function of the mean (limma-trend)' if variance_trend else ''}. "
              f"Valid because regions within one level are disjoint. Families smaller "
              f"than {MIN_FEATURES_FOR_MODERATION} regions fall back to Welch, and the "
              f"mean-variance trend needs {MIN_FEATURES_FOR_TREND}; see the `test` and "
              f"`prior_df` columns for what each family actually used.")
        print(f"  [WARN] moderated_t is ANTICONSERVATIVE at these family sizes. Simulated "
              f"at n=3 vs 3 under the global null, BH's family-wise error rate came out "
              f"6.5-15% against a nominal 5% (worse with variance_trend on), because "
              f"every region shares one estimated prior. Welch is conservative on the "
              f"same data (3.2-3.6% marginal vs 5% nominal). Treat the two as brackets: "
              f"report regions found by BOTH as established, and moderated_t-only hits "
              f"as leads. See moderated_ttest's docstring for the numbers.")
    default_corr = str(stats_cfg.get("correction", "bh")).lower()
    corr_by_level = {int(k): str(v).lower()
                     for k, v in (stats_cfg.get("correction_by_level") or {}).items()}
    for c in [default_corr, *corr_by_level.values()]:
        if c not in CORRECTIONS:
            raise ValueError(f"unknown correction '{c}'; known: {CORRECTIONS}")
    gatekeeping = bool(stats_cfg.get("gatekeeping", False))
    levels = sorted(levels)

    if gatekeeping:
        print(f"Gatekeeping ON: a region is tested at a level only if its ancestor at the "
              f"previous tested level reached p_adj < {alpha}. Levels tested in order {levels}.")
    uncorrected = [lv for lv in levels if corr_by_level.get(lv, default_corr) == "none"]
    if uncorrected and not gatekeeping:
        print(f"  [WARN] levels {uncorrected} run uncorrected WITHOUT gatekeeping. With ~150 "
              f"regions, about {0.05 * 150:.0f} will reach p<0.05 by chance alone. Turn on "
              f"stats.gatekeeping, or treat those rows as descriptive only.")

    def _compare(cdf, tdf, names):
        rows = []
        for cls in names:
            # One gate per class, from its Count matrix, applied to every
            # metric: a region holding a handful of cells cannot support a
            # density or proportion comparison either, and leaving those in
            # inflates the family with rows that can only be noise.
            gate = None
            if min_count > 0:
                ca = metric_matrix(cdf, tdf, volumes, region_tot, cls, "Count", "a",
                                   samples_a, ontology, density_denominator)
                cb = metric_matrix(cdf, tdf, volumes, region_tot, cls, "Count", "b",
                                   samples_b, ontology, density_denominator)
                gate = (np.nansum(ca, axis=1) + np.nansum(cb, axis=1)) >= min_count
            for metric in metrics:
                ma = apply_coverage_mask(metric_matrix(
                    cdf, tdf, volumes, region_tot, cls, metric, "a", samples_a,
                    ontology, density_denominator), cov_a, min_cov)
                mb = apply_coverage_mask(metric_matrix(
                    cdf, tdf, volumes, region_tot, cls, metric, "b", samples_b,
                    ontology, density_denominator), cov_b, min_cov)
                # Gatekeeping runs down the tested levels within one
                # (class, metric) chain: each level narrows what the next one
                # may look at, which is what makes an uncorrected deep level
                # defensible -- it is a description of a branch already
                # established at a coarse level, not a fresh search.
                open_orders = keep_orders.copy()
                for level in levels:
                    corr = corr_by_level.get(level, default_corr)
                    res = run_level_tests(ma, mb, metadata, open_orders, level, cls, metric,
                                          samples_a, samples_b, count_gate=gate,
                                          correction=corr, test=test, alpha=alpha,
                                          variance_trend=variance_trend)
                    if not res.empty:
                        res["gated"] = gatekeeping
                        rows.append(res)
                    if gatekeeping:
                        sig = (res["order"][res["p_adj"] < alpha].tolist()
                               if not res.empty else [])
                        open_orders = keep_orders & ontology.subtree_mask(sig)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    result = _compare(counts_df, totals_df, classes)
    if not result.empty:
        result["formula"] = ""
    if formulas:
        combined = _compare(comb_counts, comb_totals, list(formulas))
        if not combined.empty:
            combined["formula"] = combined["class_name"].map(formulas)
            result = pd.concat([result, combined], ignore_index=True) if not result.empty else combined

    if not result.empty:
        result = result.merge(cov_report, on="order", how="left")
        ref_cfg = cfg.get("reference_annotation")
        if ref_cfg:
            ref = build_reference_volumes(
                ref_cfg["path"], ref_cfg.get("voxel_size_um", [20.0, 20.0, 20.0]), ontology,
                cache_path=os.path.join(out_dir, "reference_region_volumes.csv"))
            result = result.merge(ref[["order", "ref_volume_mm3"]], on="order", how="left")
        result = result.replace([np.inf, -np.inf], np.nan)

    return dict(result=result, name_a=name_a, name_b=name_b,
                samples_a=samples_a, samples_b=samples_b, classes=classes,
                counts_df=counts_df, direct_df=direct_df, totals_df=totals_df,
                comb_counts=comb_counts, comb_direct=comb_direct, comb_totals=comb_totals,
                formulas=formulas, volumes=volumes, region_tot=region_tot,
                metadata=metadata, ontology=ontology, out_dir=out_dir,
                levels=levels,
                min_coverage=min_cov, min_total_count=min_count,
                exclude_mask=exclude_mask, alpha=alpha, gatekeeping=gatekeeping, test=test,
                variance_trend=variance_trend,
                correction=default_corr, correction_by_level=corr_by_level,
                density_denominator=density_denominator,
                metrics=metrics, classify_by=classify_by)


# ================= outputs =================

def _readme_frame(r):
    rows = [
        ("level", "Ontology depth from root (0 = whole brain, higher = finer)"),
        ("order", "Dense region index from stats/ontology.py (stable for one ontology JSON; "
                  "join across tools on 'id', not on this)"),
        ("id / acronym / name", "CCF structure"),
        ("class_name", f"Cell class. classify_by={r['classify_by']}: "
                       "'marker' means the YOLO neuron/glia call is discarded and neuron_* and "
                       "glia_* of the same marker signature are summed"),
        ("formula", "Resolved +/- formula, for combined categories only (blank otherwise)"),
        ("metric", "Count = cells in region + descendants. "
                   "Percentage = % of this class's whole-brain cells. "
                   f"Density = cells per mm^3 of that sample's own "
                   f"{'tissue-covered ' if r['density_denominator'] == 'covered' else ''}"
                   "warped region volume. "
                   "RegionProportion = % of all cells of all base classes in that same region "
                   "(immune to per-sample detection-efficiency differences). "
                   "Volume = the region volume itself, mm^3"),
        ("n_a / n_b", f"Samples actually contributing (a={r['name_a']}, b={r['name_b']}); "
                      "below the group size where coverage masking dropped a sample"),
        ("mean_a / mean_b / sd_a / sd_b", "Group mean and SD of the metric"),
        ("fold_change / log2fc", "mean_b / mean_a; NaN where mean_a is 0"),
        ("hedges_g / g_ci_lo / g_ci_hi", "Bias-corrected standardised difference (b - a) and its "
                                         "approximate 95% CI. Rank findings by this, not by p"),
        ("p_value", "Welch's t-test, uncorrected"),
        ("correction", "Multiple-testing correction applied to this level: bh (FDR), holm "
                       "(family-wise error rate), bonferroni, or none"),
        ("m_family", "Number of regions in this row's correction family -- one "
                     "(class, metric, level) combination. The adjusted threshold scales with it"),
        ("p_adj", "Adjusted p-value under `correction`. Equals p_value when correction is none"),
        ("exploratory", "TRUE where correction is none. Those rows are DESCRIPTIVE. Gatekeeping "
                        "narrows where you looked but controls no error rate, so no single region "
                        "on such a level is established -- read the family columns below instead"),
        ("n_raw_sig_in_family / expected_false_in_family",
         "How many regions in this (class, metric, level) family reached p < alpha, against how "
         "many were expected to by chance (m x alpha). The gap is the signal"),
        ("family_enrichment_p",
         "One-sided binomial p for seeing that many sub-alpha regions in a family of m if every "
         "one were null. This is the claim an uncorrected level CAN support: not 'this region "
         "differs' but 'this set of regions is enriched for differences'"),
        ("est_false_frac_in_family",
         "expected_false / n_raw_sig -- the fraction of that family's sub-alpha list expected to "
         "be noise. Rank the list by effect size; the top of it is the least likely to be noise"),
        ("gated", "TRUE if gatekeeping was on: this region was only tested because its ancestor "
                  "at the previous tested level was significant. Gatekeeping is what makes an "
                  "uncorrected deep level defensible -- without it, an uncorrected level is a "
                  "free search"),
        ("n_samples_below_min_coverage", f"Samples where this region's tissue coverage fell below "
                                         f"{r['min_coverage']:.2f} and was excluded"),
        ("(excluded regions)", f"{int(r['exclude_mask'].sum())} regions were removed from the "
                               "analysis before any rollup, so they are absent from every "
                               "ancestor's Count/Volume and from the Percentage and "
                               "RelativeVolume denominators"),
        ("(rows not shown)", f"Regions holding fewer than {r['min_total_count']:.0f} cells of a "
                             "class across all samples are not tested for that class at all"),
        ("min_coverage_seen", "Lowest coverage this region reached in any sample"),
        ("<sample name>", "That sample's own value for this row's metric"),
        ("CAUTION", f"n={len(r['samples_a'])} vs {len(r['samples_b'])}. An exact permutation test "
                    f"bottoms out at p="
                    f"{2.0 / math.comb(len(r['samples_a']) + len(r['samples_b']), len(r['samples_a'])):.3g}, "
                    "so this uses Welch's t-test, which assumes normality. |g| of about 1.8 is "
                    "needed for a raw p<0.05, and about 4.2 to clear BH in a family of 17 "
                    "regions or 6.7 in a family of 98. Treat p_adj as a filter and the effect "
                    "size, with its CI, as the result"),
    ]
    return pd.DataFrame(rows, columns=["column", "description"])


def write_outputs(r):
    out_dir = r["out_dir"]
    df = r["result"]
    out_cfg_long = os.path.join(out_dir, "region_stats.csv")
    df.to_csv(out_cfg_long, index=False)
    print(f"Wrote long-format table: {out_cfg_long} ({len(df)} rows)")

    if df.empty:
        return
    xlsx = os.path.join(out_dir, "region_stats_by_level.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame({"Statistical methods": describe_methods(r).split("\n")}).to_excel(
            w, sheet_name="Methods", index=False)
        _readme_frame(r).to_excel(w, sheet_name="ReadMe", index=False)
        cov = r["volumes"].merge(r["metadata"][["order", "id", "acronym", "name", "level"]],
                                 on="order", how="left")
        (cov[cov["voxel_count"] > 0]
         .sort_values(["sample", "level", "order"])
         .to_excel(w, sheet_name="Region_Volumes", index=False))
        for level in sorted(df["level"].unique()):
            sheet = df[df["level"] == level].sort_values(
                ["class_name", "metric", "p_value"])
            if not sheet.empty:
                sheet.to_excel(w, sheet_name=f"L{int(level):02d}", index=False)
    print(f"Wrote workbook: {xlsx}")


_TEST_NAMES = {"welch": "Welch's unequal-variance t-test",
               "student": "Student's pooled-variance t-test",
               "moderated_t": "empirical-Bayes moderated t-test (limma; Smyth 2004)"}
_CORR_NAMES = {"bh": "Benjamini-Hochberg FDR",
               "holm": "Holm-Bonferroni (family-wise error rate)",
               "bonferroni": "Bonferroni (family-wise error rate)",
               "none": "no correction -- DESCRIPTIVE ONLY"}


def describe_methods(r):
    """A written account of what was actually run, generated from the settings
    in force rather than from the config file, so it cannot drift away from the
    numbers next to it.

    Printed at the end of every run and saved as methods.md. It exists because
    the procedure here is not a single test -- it is a per-level chain with
    different corrections and a gatekeeping rule -- and that is not something a
    reader can reconstruct from a p-value column."""
    na, nb = len(r["samples_a"]), len(r["samples_b"])
    n_perm = math.comb(na + nb, na)
    lines = []
    add = lines.append

    add("# Statistical methods")
    add("")
    add(f"Generated by stats/group_stats.py on the run in `{r['out_dir']}`.")
    add("")
    add("## Design")
    add(f"- **Groups**: {r['name_a']} (n={na}: {', '.join(r['samples_a'])}) vs "
        f"{r['name_b']} (n={nb}: {', '.join(r['samples_b'])}).")
    add(f"- **Cell classes**: `classify_by: {r['classify_by']}` -> "
        f"{', '.join(r['classes'])}."
        + (" YOLO's neuron/glia call is discarded; neuron_* and glia_* sharing a marker "
           "signature are summed." if r["classify_by"] == "marker" else ""))
    if r["formulas"]:
        add(f"- **Aggregate classes**: " + "; ".join(f"{k} = {v}" for k, v in r["formulas"].items())
            + ".")
    add(f"- **Metrics**: {', '.join(r['metrics'])}. Density uses each sample's own warped "
        f"region volume"
        + (", restricted to tissue inside its brain mask" if r["density_denominator"] == "covered"
           else "") + ".")
    add("")

    add("## Regions")
    if r["exclude_mask"].any():
        names = ", ".join(sorted({str(r["ontology"].names[o])
                                  for o in np.where(r["exclude_mask"])[0]
                                  if r["ontology"].parent_order[o] < 0
                                  or not r["exclude_mask"][r["ontology"].parent_order[o]]}))
        add(f"- **Excluded before any aggregation** ({int(r['exclude_mask'].sum())} regions): "
            f"{names}. The exclusion is applied to the per-region counts and volumes before "
            "the hierarchical rollup, so these regions are absent from every ancestor's total "
            "and from the Percentage and RelativeVolume denominators.")
    if r["min_coverage"] > 0:
        add(f"- **Tissue coverage**: a region was dropped for a given sample when less than "
            f"{r['min_coverage']:.0%} of its warped volume fell inside that sample's brain "
            f"mask. The region was still tested on the remaining samples; n_a/n_b record how "
            f"many contributed.")
    if r["min_total_count"] > 0:
        add(f"- **Minimum count**: a region was tested for a class only if it held at least "
            f"{r['min_total_count']:.0f} cells of that class summed across all samples.")
    add("")

    add("## Testing")
    add(f"- **Test**: {_TEST_NAMES[r['test']]}, two-sided, applied per region.")
    if r["test"] == "moderated_t":
        res = r.get("result")
        dfs = ""
        if res is not None and not res.empty and "test_df" in res.columns:
            d = res["test_df"].dropna()
            if not d.empty:
                dfs = (f" Residual degrees of freedom rose from {na + nb - 2} to "
                       f"{d.min():.1f}-{d.max():.1f} across families.")
        add(f"  Each family's per-region variance was shrunk toward a prior estimated "
            f"across the regions of that same family"
            + (", fitted as a function of the region mean (limma-trend)"
               if r.get("variance_trend") else " (a single global prior)")
            + f".{dfs}")
        add("  This borrows information across regions, which is what buys back the "
            "degrees of freedom a 3-vs-3 design cannot supply. It is legitimate within "
            "a level because regions at one ontology level are disjoint; it would not "
            "be across levels, where each cell is counted in every ancestor. The prior "
            "is estimated from tens of regions rather than the thousands this method "
            "is usually applied to, so `prior_df` in the output should be read as an "
            "estimate with its own uncertainty.")
    add(f"- **Effect size**: Hedges' g (bias-corrected standardised mean difference, "
        f"J = 1 - 3/(4N-9) = {1 - 3 / (4 * (na + nb) - 9):.2f} at N={na + nb}), with an "
        f"approximate 95% confidence interval.")
    add(f"- **Families**: each (cell class, metric, ontology level) combination is corrected "
        f"as its own family. Levels tested: {', '.join(str(v) for v in sorted(r['levels']))}.")
    corr_lines = []
    for lv in sorted(r["levels"]):
        c = r["correction_by_level"].get(lv, r["correction"])
        corr_lines.append(f"  - level {lv}: {_CORR_NAMES[c]}")
    add(f"- **Correction per level** (alpha = {r['alpha']}):")
    lines.extend(corr_lines)
    if r["gatekeeping"]:
        add(f"- **Gatekeeping**: levels were tested in ascending order and a region was tested "
            f"at a level only if its ancestor at the previous tested level reached "
            f"p_adj < {r['alpha']}. Uncorrected levels are therefore descriptions of branches "
            f"already established at a corrected level, not an unrestricted search.")
    else:
        add("- **Gatekeeping**: off. Every level was tested over all regions passing the "
            "filters above.")
    uncorrected = sorted(lv for lv in r["levels"]
                         if r["correction_by_level"].get(lv, r["correction"]) == "none")
    if uncorrected:
        add(f"- **Exploratory levels**: {uncorrected}. Rows from these levels carry "
            f"`exploratory = TRUE`. Gatekeeping restricts where these levels look but "
            f"controls no error rate, so NO INDIVIDUAL REGION on them is established. "
            f"What they support is a statement about the set: `n_raw_sig_in_family` "
            f"against `expected_false_in_family` (= m x alpha), tested by "
            f"`family_enrichment_p` (one-sided binomial), with "
            f"`est_false_frac_in_family` estimating what fraction of the list is noise. "
            f"Report the enrichment and the effect-size ranking, not per-region p-values.")
    add("")

    # The family-level numbers are the substance of any uncorrected level, so
    # spell them out rather than leaving the reader to recompute them.
    res = r.get("result")
    if uncorrected and res is not None and not res.empty:
        rows = res[res["level"].isin(uncorrected)]
        if not rows.empty:
            add("### Set-level enrichment on the uncorrected levels")
            add("")
            add("| level | class | metric | m | p<alpha | expected by chance | "
                "binomial p | est. false fraction |")
            add("|---|---|---|---|---|---|---|---|")
            key = ["level", "class_name", "metric", "m_family", "n_raw_sig_in_family",
                   "expected_false_in_family", "family_enrichment_p",
                   "est_false_frac_in_family"]
            seen = rows[key].drop_duplicates().sort_values(["level", "class_name", "metric"])
            for _, q in seen.iterrows():
                if q["n_raw_sig_in_family"] == 0:
                    continue
                add(f"| {int(q['level'])} | {q['class_name']} | {q['metric']} | "
                    f"{int(q['m_family'])} | {int(q['n_raw_sig_in_family'])} | "
                    f"{q['expected_false_in_family']:.2f} | "
                    f"{q['family_enrichment_p']:.2g} | "
                    f"{q['est_false_frac_in_family']:.0%} |")
            add("")

    add("## Limitations at this sample size")
    add(f"- With {na} vs {nb} samples there are C({na + nb},{na}) = {n_perm} labelings, so an "
        f"exact permutation test (and equivalently the Mann-Whitney U test) cannot return a "
        f"two-sided p below {2.0 / n_perm:.2g}. A parametric test is the only route to "
        f"p < 0.05, at the cost of assuming approximate normality.")
    add("- Reaching a raw p < 0.05 requires |g| of roughly 1.8; after correction the "
        "requirement scales with family size (about 4.2 in a family of 17, 6.7 in a family "
        "of 98).")
    add("- The confidence interval on g is wide at this n (g = 4 carries roughly [0.8, 7.2]), "
        "so effect sizes should be read as orders of magnitude.")
    add("- Region volumes come from the atlas annotation warped into each sample. The "
        "registration was guided by hand-drawn masks on a few coarse structures, which gives "
        "those structures manual-segmentation accuracy; deeper subregions are interpolated "
        "within that envelope.")
    return "\n".join(lines)


def write_methods(r):
    text = describe_methods(r)
    path = os.path.join(r["out_dir"], "methods.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
    print(f"Also saved to: {path}")
    return text


def write_volume_report(r):
    """Per-region volumetry, absolute and relative, one row per region.

    Separate from the cell-count outputs because it answers a different
    question (how big is the structure, not how many cells are in it) and
    because it is the one table that is read region-by-region rather than
    scanned for hits. Excluded regions are absent: they were dropped before
    the rollup, so they are not in any ancestor's volume either, and
    relative_pct is a share of what remains.

    coverage and volume_ratio_to_median are carried on every row on purpose --
    an absolute volume from a region that is only partly in the field of view
    is not a measurement of that structure, and those two columns are how you
    see it (e.g. s18 cerebellum: coverage 0.33, ratio to median 0.28)."""
    vol = r["volumes"].merge(
        r["metadata"][["order", "id", "acronym", "name", "level"]], on="order", how="left")
    vol = vol[vol["voxel_count"] > 0].copy()

    group_of = {s: r["name_a"] for s in r["samples_a"]}
    group_of.update({s: r["name_b"] for s in r["samples_b"]})
    vol["group"] = vol["sample"].map(group_of)

    samples = r["samples_a"] + r["samples_b"]
    wide = None
    for value, prefix in (("volume_mm3", "abs_mm3"), ("relative_pct", "rel_pct"),
                          ("coverage", "coverage")):
        piv = (vol.pivot(index="order", columns="sample", values=value)
               .reindex(columns=samples))
        piv.columns = [f"{prefix}:{c}" for c in piv.columns]
        wide = piv if wide is None else wide.join(piv)

    meta = r["metadata"].set_index("order")[["id", "acronym", "name", "level"]]
    wide = meta.join(wide, how="right")
    for value, prefix in (("volume_mm3", "abs_mm3"), ("relative_pct", "rel_pct")):
        for gname, gsamples in ((r["name_a"], r["samples_a"]), (r["name_b"], r["samples_b"])):
            cols = [f"{prefix}:{s}" for s in gsamples]
            wide[f"{prefix}_mean:{gname}"] = wide[cols].mean(axis=1)
            wide[f"{prefix}_sd:{gname}"] = wide[cols].std(axis=1, ddof=1)
    wide["min_coverage"] = wide[[f"coverage:{s}" for s in samples]].min(axis=1)
    wide = wide.reset_index().sort_values(["level", "order"])

    path = os.path.join(r["out_dir"], "region_volumes.csv")
    wide.to_csv(path, index=False)
    print(f"Wrote volume report: {path} ({len(wide)} regions)")

    xlsx = os.path.join(r["out_dir"], "region_volumes.xlsx")
    notes = pd.DataFrame([
        ("abs_mm3:<sample>", "Absolute volume of the region (plus all its descendants) in that "
                             "sample's own space, from the warped atlas annotation"),
        ("rel_pct:<sample>", "That volume as a percentage of the total volume of the regions "
                             "under analysis in the same sample (excluded regions are not in "
                             "the total). Cancels overall brain size and field-of-view differences"),
        ("coverage:<sample>", "Fraction of the warped region lying inside that sample's brain "
                              "mask. Below ~0.8 the absolute volume is not a measurement of the "
                              "structure, only of the part that was imaged"),
        ("min_coverage", "Lowest coverage across all samples for this region"),
        ("CAVEAT", "The warped annotation follows the registration, which was guided by "
                   "hand-drawn masks on a few coarse structures. Volumes at and near those "
                   "structures carry manual-segmentation accuracy; deeper subregions are "
                   "interpolated by SyN inside that envelope and do not."),
    ], columns=["column", "description"])
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        notes.to_excel(w, sheet_name="ReadMe", index=False)
        for level in sorted(wide["level"].unique()):
            sheet = wide[wide["level"] == level]
            if not sheet.empty:
                sheet.to_excel(w, sheet_name=f"L{int(level):02d}", index=False)
    print(f"Wrote volume workbook: {xlsx}")


def write_per_sample_tables(r):
    """One workbook per sample: every class x every region with
    count/percentage/density/coverage, zero rows kept -- a full registration
    inventory for QC, independent of any group comparison."""
    sample_dir = os.path.join(r["out_dir"], "per_sample")
    os.makedirs(sample_dir, exist_ok=True)

    counts = pd.concat([r["counts_df"], r["comb_counts"]], ignore_index=True)
    totals = pd.concat([r["totals_df"], r["comb_totals"]], ignore_index=True)
    meta = r["metadata"].set_index("order")
    orders = r["metadata"]["order"].to_numpy()
    vol_col = "covered_volume_mm3" if r["density_denominator"] == "covered" else "volume_mm3"

    for sample in r["samples_a"] + r["samples_b"]:
        sub = counts[counts["sample"] == sample]
        tot = totals[totals["sample"] == sample].set_index("class_name")["total_valid"]
        sv = r["volumes"][r["volumes"]["sample"] == sample].set_index("order")
        vol = sv[vol_col].reindex(orders)
        cov = sv["coverage"].reindex(orders)
        rtot = (r["region_tot"][r["region_tot"]["sample"] == sample]
                .set_index("order")["region_total"].reindex(orders).fillna(0.0))

        pieces = []
        for cls, g in sub.groupby("class_name"):
            count = g.set_index("order")["count"].reindex(orders).fillna(0.0)
            tv = float(tot.get(cls, 0) or 0)
            piece = meta.loc[orders, ["id", "acronym", "name", "level"]].copy()
            piece["order"] = orders
            piece["class_name"] = cls
            piece["formula"] = r["formulas"].get(cls, "")
            piece["count"] = count.to_numpy()
            piece["percentage"] = count.to_numpy() / tv * 100.0 if tv > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                piece["density"] = np.where(vol.to_numpy() > 0,
                                            count.to_numpy() / vol.to_numpy(), np.nan)
                piece["region_proportion"] = np.where(rtot.to_numpy() > 0,
                                                      count.to_numpy() / rtot.to_numpy() * 100.0,
                                                      np.nan)
            piece["volume_mm3"] = vol.to_numpy()
            piece["coverage"] = cov.to_numpy()
            pieces.append(piece.reset_index(drop=True))

        table = pd.concat(pieces, ignore_index=True)
        path = os.path.join(sample_dir, f"{sample}_region_summary.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for level in sorted(table["level"].unique()):
                sheet = table[table["level"] == level].sort_values(["class_name", "order"])
                if not sheet.empty:
                    sheet.to_excel(w, sheet_name=f"L{int(level):02d}", index=False)
        print(f"Wrote per-sample summary: {path}")


def _safe_sheet_name(name):
    return (re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31]) or "Sheet"


def build_tree_rows(rollup, direct, ontology, names, levels):
    """DFS over regions with nonzero rollup. A node where cells stopped
    without resolving to any subregion emits a synthetic 'Lost cells' child
    holding exactly direct[node], so sibling counts always sum to the parent
    -- that gap is why a class can look complete at level 2 and be missing
    cells at level 6."""
    rows = []

    def walk(order, path):
        count = rollup[order]
        if count <= 0:
            return
        path = path + [(levels[order], names[order], count)]
        kids = ontology.children[order]
        if not kids:
            rows.append(path)
            return
        for c in kids:
            walk(c, path)
        if direct[order] > 1e-6:
            rows.append(path + [(levels[order] + 1, LOST_LABEL, direct[order])])

    walk(ontology.root_order, [])
    return rows


def tree_rows_to_frame(rows):
    if not rows:
        return pd.DataFrame()
    max_level = max(lvl for row in rows for lvl, _, _ in row)
    records = []
    for row in rows:
        rec = {}
        for lvl, name, count in row:
            rec[f"L{lvl:02d}_Name"] = name
            rec[f"L{lvl:02d}_Count"] = count
        records.append(rec)
    cols = [c for lvl in range(max_level + 1)
            for c in (f"L{lvl:02d}_Name", f"L{lvl:02d}_Count")]
    return pd.DataFrame.from_records(records, columns=cols)


def write_per_sample_trees(r):
    ontology = r["ontology"]
    counts = pd.concat([r["counts_df"], r["comb_counts"]], ignore_index=True)
    direct = pd.concat([r["direct_df"], r["comb_direct"]], ignore_index=True)
    totals = pd.concat([r["totals_df"], r["comb_totals"]], ignore_index=True)
    orders = r["metadata"]["order"].to_numpy()
    names = ontology.names
    levels = ontology.levels
    has_children = np.array([len(c) > 0 for c in ontology.children])
    sample_dir = os.path.join(r["out_dir"], "per_sample")
    os.makedirs(sample_dir, exist_ok=True)

    readme = pd.DataFrame([
        ("LevelX_Name / LevelX_Count", "Region and its hierarchical rollup count along this "
                                       "root-to-region path"),
        (f"'{LOST_LABEL}' row", "Cells that registered to the region above but never resolved to "
                                "any finer subregion"),
        ("Summary.lost_at_level", "Cells whose assignment stopped exactly at this level"),
        ("Summary.pct_cumulative_lost", "Share of this class's assigned cells that will be missing "
                                        "from any comparison finer than this level"),
    ], columns=["column", "description"])

    for sample in r["samples_a"] + r["samples_b"]:
        sub = counts[counts["sample"] == sample]
        subd = direct[direct["sample"] == sample]
        tot = totals[totals["sample"] == sample].set_index("class_name")["total_valid"]

        summary_rows, frames = [], {}
        for cls, g in sub.groupby("class_name"):
            roll = g.set_index("order")["count"].reindex(orders).fillna(0.0).to_numpy()
            dg = subd[subd["class_name"] == cls]
            dd = (dg.set_index("order")["direct_count"].reindex(orders).fillna(0.0).to_numpy()
                  if not dg.empty else np.zeros(len(orders)))
            frames[cls] = tree_rows_to_frame(build_tree_rows(roll, dd, ontology, names, levels))

            tv = float(tot.get(cls, 0) or 0)
            lost_by_level = pd.Series(dd * has_children).groupby(levels).sum()
            cum = 0.0
            for lvl in range(int(levels.max()) + 1):
                lost = float(lost_by_level.get(lvl, 0.0))
                cum += lost
                summary_rows.append({
                    "class_name": cls, "level": lvl, "lost_at_level": lost,
                    "cumulative_lost": cum, "total_valid": tv,
                    "pct_cumulative_lost": (cum / tv * 100.0) if tv else np.nan})

        path = os.path.join(sample_dir, f"{sample}_region_tree.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            readme.to_excel(w, sheet_name="ReadMe", index=False)
            pd.DataFrame(summary_rows).to_excel(w, sheet_name="Summary", index=False)
            for cls, frame in frames.items():
                if not frame.empty:
                    frame.to_excel(w, sheet_name=_safe_sheet_name(cls), index=False)
        print(f"Wrote per-sample tree: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-per-sample", action="store_true",
                    help="skip the per-sample QC workbooks (they are the slow part)")
    args = ap.parse_args()

    r = run_all(load_config(args.config))
    write_outputs(r)
    write_volume_report(r)
    if not args.no_per_sample:
        write_per_sample_tables(r)
        write_per_sample_trees(r)
    write_methods(r)


if __name__ == "__main__":
    main()
