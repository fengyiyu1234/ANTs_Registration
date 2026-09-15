"""Group comparison on fixed atlas-space blocks instead of whole regions.

WHY
---
A per-region Density is cells over that region's volume IN SAMPLE SPACE, so it
carries whatever scale difference clearing left behind. On this dataset that is
measurable, not hypothetical: whole-brain volume runs 49.1 to 82.4 mm3 across
six same-age brains, 26 of 28 level-2/3 structures come out 13-23% smaller in
one group, and RelativeVolume is flat -- a global scale difference, not
regional atrophy. Decomposing the six significant Density results in
0908_marker_ungated gives log2fc(Density) = log2fc(Count) - log2fc(Volume) with
the two terms roughly equal, so about half of each effect is the denominator.

A block fixed in ATLAS space has the same volume in every sample, so dividing
by it changes no statistic. The metric becomes "cells in one fixed anatomical
territory", immune to global shrinkage by construction. Expect this analysis to
land nearer the Count answer than the Density answer; if a result survives here
it is much harder to explain away.

WHAT IS AND IS NOT FIXED BY THIS
--------------------------------
Fixed:      per-sample scale, and the region-boundary wobble that comes with it.
Not fixed:  tissue actually MISSING inside a block. The atlas denominator does
            not shrink when tissue is gone, so a hole reads as a real decrease.
            Coverage from the warped brain mask cannot catch this: the mask is
            an envelope and reads 1.00 on every block of every sample, the two
            visibly torn ones included. What does catch it is the warped
            INTENSITY image -- block_darkness() reports, per block per sample,
            the fraction of the cube below a quarter of that sample's tissue
            median and the size of the largest connected dark component. Set
            qc.action to drop_block to exclude a flagged block from EVERY
            sample; see the comment above DARK_REL.
Not fixed:  n. Blocks are averaged to one number per sample before testing, so
            the primary test still has 3 vs 3. Treating blocks as replicates
            would be pseudo-replication and is not done anywhere here.

THREE FAMILIES
--------------
primary     one test per (class, metric) on the per-sample mean over all
            blocks. BH within each metric. This is the analysis.
per_block   one test per (block, class) at n=3 vs 3, BH within each
            (class, metric). Exploratory: it says WHERE, only for results the
            primary already supports, and is flagged as such in the output.
where       the blocks are at different places on purpose, so "do the two
            groups put their cells in the same places" is a question the block
            mean cannot answer -- cells moving between blocks leave it
            unchanged. Three tools for it, in the order they should be read:
            Heterogeneity (I2: is there any location structure at all, or are
            the blocks noisy copies of one number), the axis gradients (rho
            against ML/DV/AP -- a migration phenotype is a gradient, and one
            test per axis is the only version with power at this n), and
            Redistribution (per-block Share, plus an exact permutation on the
            whole share vector whose smallest possible p is 0.1).

Blocks come from Registration_toolkit's tools/pick_blocks.py as a blocks.json
holding, per block, the voxel bounds in each sample's OWN prepared atlas grid.
That per-sample indirection is not cosmetic: s10 is a left hemisphere prepared
with a mirrored orientation and a different crop, so one set of voxel numbers
cannot address all six samples.

Usage:
    conda activate antsreg
    python -m stats.block_stats --config stats/configs/tsc_blocks.yaml

Figures are written to <output.dir>/figures/ by the same run; --no-figures
skips them, and stats/plot_blocks.py redraws them from the CSVs alone.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import stats as sp_stats

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.group_stats import (adjust_pvalues, hedges_g, load_config,  # noqa: E402
                               two_sample_ttest)

# Columns 6-8 of cell_registration.csv are the voxel index in the sample's own
# PREPARED atlas grid -- the same frame blocks.json records its bounds in.
ATLAS_COLS = {6: "xt", 7: "yt", 8: "zt"}


def load_blocks(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not doc.get("blocks"):
        raise ValueError(f"{path} 里没有块。先跑 tools/pick_blocks.py。")
    return doc


def block_label_volume(doc, sample, shape, blocks_dir):
    """-> (uint16 volume of block ids, [block numbers in order]).

    One lookup array per sample beats testing every cell against every block:
    assigning a cell becomes a single fancy-index.

    Cubes are rebuilt from the six lo/hi bounds recorded per sample, which
    describe them exactly. A COLUMN is not a box in the sample's grid, so
    pick_blocks writes its label volume out instead and this loads it. Same
    array either way, so nothing downstream has to know which shape it got.
    """
    labels = (doc.get("per_sample_labels") or {}).get(sample)
    if labels:
        path = labels if os.path.isabs(labels) else os.path.join(blocks_dir, labels)
        vol = np.asarray(nib.load(path).dataobj).astype(np.uint16)
        if tuple(vol.shape) != tuple(shape):
            raise ValueError(
                f"{os.path.basename(path)} 的形状 {vol.shape} 和 {sample} 的图谱空间 "
                f"{tuple(shape)} 对不上。blocks.json 和这次的配准跑不是同一批。")
        return vol, sorted(int(v) for v in np.unique(vol) if v)

    vol = np.zeros(shape, dtype=np.uint16)
    used = []
    for b in doc["blocks"]:
        bounds = (b.get("per_sample_bounds") or {}).get(sample)
        if bounds is None:
            continue
        lo, hi = bounds["lo_xyz"], bounds["hi_xyz"]
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = b["block"]
        used.append(b["block"])
    return vol, used


def read_atlas_points(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None, names=range(20), engine="python")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return np.empty((0, 3), dtype=float)
    if df.empty:
        return np.empty((0, 3), dtype=float)
    out = df[[6, 7, 8]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return out[np.isfinite(out).all(axis=1)]


def resolve_class_map(cfg, sample_dirs):
    """Config class_map -> {label: [folder, ...]} per sample, matched through
    cell_tables.normalize_class_key so the 'glia_3_GFP' naming variant resolves
    with the rest. A folder claimed by no class is reported: those cells vanish
    from every count and every denominator, and a typo looks exactly like it."""
    raw = cell_tables.normalize_class_map(cfg.get("class_map"))
    if not raw:
        raise ValueError("这个分析需要显式的 class_map。")
    want = {label: {cell_tables.normalize_class_key(f) for f in folders}
            for label, folders in raw.items()}
    per_sample, unclaimed = {}, {}
    for sample, sdir in sample_dirs.items():
        folders = cell_tables.list_sample_classes(sdir)
        mapping, claimed = {}, set()
        for label, keys in want.items():
            hit = [f for f in folders if cell_tables.normalize_class_key(f) in keys]
            mapping[label] = hit
            claimed.update(hit)
        per_sample[sample] = mapping
        missed = [f for f in folders if f not in claimed]
        if missed:
            unclaimed[sample] = missed
    return list(raw), per_sample, unclaimed


# Per-block image QC. The warped brain mask is an envelope: it reads coverage
# 1.00 on every block of every sample here, including the two in s18 that are a
# fifth empty, so it can never be the thing that catches a hole. The warped
# INTENSITY image can. A tear, a bubble, or a block sitting where the sample
# ends reads as a contiguous run of near-zero voxels, and on this dataset the
# statistic is bimodal: a block is either exactly 0.0% dark or several percent,
# with nothing in between, so the exact cut barely matters.
#
# The cut is RELATIVE to that sample's own median block intensity, because the
# warped images are normalised per sample (medians run 0.088 to 0.152 across
# these six). Do NOT use Otsu here: on light-sheet it latches onto the bright
# cells and calls the rest of the tissue background.
DARK_REL = 0.25   # a voxel is dark below this fraction of the tissue median
DARK_MAX = 0.10   # a block is flagged above this dark fraction


def block_darkness(arr, vol, used, dark_rel=DARK_REL):
    """-> [{sample, block, tissue_ref, block_median, dark_frac, dark_cc_frac}].

    Driven off the per-sample LABEL VOLUME, not off block bounds, so it works
    the same for a cube and for a column that is not a box in this grid.

    dark_cc_frac is the LARGEST CONNECTED dark component as a fraction of the
    block, which is what separates one hole from speckle scattered through it.
    On the blocks that flag here the two are nearly equal, i.e. the dark voxels
    are a single object -- that is the signature of missing tissue, and a block
    where dark_frac is high but dark_cc_frac is small is more likely dim tissue
    or noise than a hole."""
    if not used:
        return []
    inside = vol > 0
    ref = float(np.median(arr[inside]))
    cut = dark_rel * ref
    boxes = ndimage.find_objects(vol.astype(np.int32))
    rows = []
    for blk in used:
        slc = boxes[int(blk) - 1]
        if slc is None:
            continue
        here = vol[slc] == blk
        v = arr[slc]
        dark = (v < cut) & here
        n_here = int(here.sum())
        if n_here == 0:
            continue
        frac = float(dark.sum() / n_here)
        cc = 0.0
        if dark.any():
            lab, n = ndimage.label(dark)
            if n:
                cc = float(np.bincount(lab.ravel())[1:].max() / n_here)
        rows.append({"block": int(blk), "tissue_ref": ref,
                     "block_median": float(np.median(v[here])),
                     "dark_frac": frac, "dark_cc_frac": cc})
    return rows


def count_cells(cfg, doc, sample_dirs, dark_rel=DARK_REL):
    """-> (counts [sample, block, class_name, count], coverage, darkness, classes).

    The warped image is read once per sample and used for both jobs: > 0 gives
    the envelope that coverage divides by, and the intensities themselves give
    the per-block darkness that actually detects a hole."""
    classes, per_sample_map, unclaimed = resolve_class_map(cfg, sample_dirs)
    for sample, missed in unclaimed.items():
        print(f"  WARN {sample}: 这些文件夹没有被任何类认领，它们的细胞会从所有"
              f"计数里消失: {', '.join(missed)}")

    rows, cov_rows, dark_rows = [], [], []
    for sample, sdir in sample_dirs.items():
        mask_path = _find_in_atlas(sdir)
        arr = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)
        mask = arr > 0
        vol, used = block_label_volume(doc, sample, arr.shape,
                                       os.path.dirname(os.path.abspath(cfg["blocks_json"])))
        if not used:
            raise ValueError(f"{sample} 在 blocks.json 里没有任何可用的块。")
        for row in block_darkness(arr, vol, used, dark_rel):
            row["sample"] = sample
            dark_rows.append(row)

        sizes = np.bincount(vol.ravel(), minlength=len(doc["blocks"]) + 1)
        covered = np.bincount(vol[mask].ravel(), minlength=len(doc["blocks"]) + 1)
        for b in used:
            cov_rows.append({"sample": sample, "block": b,
                             "voxels": int(sizes[b]),
                             "coverage": float(covered[b] / max(sizes[b], 1))})

        for label in classes:
            per_block = np.zeros(len(doc["blocks"]) + 1, dtype=np.int64)
            for folder in per_sample_map[sample][label]:
                pts = read_atlas_points(cell_tables.class_csv_path(sdir, folder))
                if len(pts) == 0:
                    continue
                idx = np.round(pts).astype(np.int64)
                inside = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
                idx = idx[inside]
                if len(idx) == 0:
                    continue
                hit = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
                per_block += np.bincount(hit, minlength=len(doc["blocks"]) + 1)
            for b in used:
                rows.append({"sample": sample, "block": b, "class_name": label,
                             "count": int(per_block[b])})
        print(f"  {sample}: {len(used)} 个块, "
              f"{sum(r['count'] for r in rows if r['sample'] == sample)} 个细胞落在块内")
    return (pd.DataFrame(rows), pd.DataFrame(cov_rows),
            pd.DataFrame(dark_rows), classes)


def _find_in_atlas(run_dir):
    hits = sorted(f for f in os.listdir(run_dir) if f.endswith("_in_atlas.nii.gz"))
    if not hits:
        raise FileNotFoundError(
            f"{run_dir} 里没有 *_in_atlas.nii.gz，算不了块的组织覆盖率。")
    return os.path.join(run_dir, hits[0])


def add_combined(counts, cfg, base_classes):
    """Signed sums over the mutually exclusive base classes. These overlap each
    other on purpose, which is why they never act as a denominator."""
    formulas = cfg.get("combined_categories") or {}
    if not formulas:
        return counts, []
    wide = counts.pivot_table(index=["sample", "block"], columns="class_name",
                              values="count", fill_value=0)
    extra, names = [], []
    for name, terms in formulas.items():
        total = np.zeros(len(wide), dtype=float)
        for term in terms:
            cls = term["class"] if isinstance(term, dict) else term
            sign = 1.0 if (not isinstance(term, dict)
                           or term.get("sign", "+") == "+") else -1.0
            if cls not in wide.columns:
                raise ValueError(f"combined_categories 里的 {cls!r} 不是已有的类。")
            total += sign * wide[cls].to_numpy(float)
        frame = wide.index.to_frame(index=False)
        frame["class_name"] = name
        frame["count"] = total
        extra.append(frame)
        names.append(name)
    return pd.concat([counts] + extra, ignore_index=True), names


def build_metrics(counts, coverage, block_mm3):
    """Count, atlas Density, and a coverage-corrected Density.

    `block_mm3` maps block number to its atlas volume. For cubes every entry is
    the same number, so Density cannot change a p-value relative to Count and
    is carried only because a density is what gets reported. Columns each keep
    a different volume -- cortex is not equally thick everywhere -- so there
    Density and Count are genuinely different statistics, and Density is the
    one that compares a thick column against a thin one fairly.

    CoveredDensity divides instead by the tissue actually present, which is the
    only correction available for a block with a hole in it."""
    df = counts.merge(coverage, on=["sample", "block"], how="left")
    df["Count"] = df["count"].astype(float)
    df["block_mm3"] = df["block"].map(block_mm3).astype(float)
    df["Density"] = df["Count"] / df["block_mm3"]
    cov = df["coverage"].replace(0, np.nan)
    df["CoveredDensity"] = df["Count"] / (df["block_mm3"] * cov)
    return df


def test_frame(mat_a, mat_b, labels, correction, test, alpha):
    p = two_sample_ttest(mat_a, mat_b, test=test)
    g, lo, hi = hedges_g(mat_a, mat_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma, mb = np.nanmean(mat_a, axis=1), np.nanmean(mat_b, axis=1)
        log2fc = np.log2(np.where(mb > 0, mb, np.nan) / np.where(ma > 0, ma, np.nan))
    out = pd.DataFrame(labels)
    out["mean_a"], out["mean_b"] = ma, mb
    out["sd_a"] = np.nanstd(mat_a, axis=1, ddof=1)
    out["sd_b"] = np.nanstd(mat_b, axis=1, ddof=1)
    out["log2fc"], out["hedges_g"] = log2fc, g
    out["g_ci_lo"], out["g_ci_hi"] = lo, hi
    out["p_value"] = p
    out["p_adj"] = adjust_pvalues(p, correction)
    out["correction"], out["m_family"], out["alpha"] = correction, len(p), alpha
    return out


# The native DeMBA P5 grid's axes, as tools/pick_blocks.py derives them:
# (ML, DV, AP) = (570, 400, 563). ML is folded to distance from the midline --
# a sample prepared from the other hemisphere is mirrored onto these same
# blocks, so which SIDE a block is on is not comparable across samples, while
# how far it is from the midline is.
AXES = ("ML_from_midline", "DV", "AP")


def block_positions(doc):
    """-> DataFrame [block, ML_from_midline, DV, AP, top_region], mm."""
    shape = doc["atlas"]["shape_xyz"]
    mid_mm = shape[0] * float(doc["atlas"]["res_um"]) / 1000.0 / 2.0
    rows = []
    for b in doc["blocks"]:
        # An ROI has no centre and no axis position: it is not somewhere, it is
        # everywhere inside itself. Nothing that uses positions runs on a single
        # block, so an empty frame is the honest answer rather than a made-up
        # centroid that would quietly become an anatomical claim.
        if "centre_mm_xyz" not in b:
            continue
        c = b["centre_mm_xyz"]
        comp = b.get("composition") or []
        rows.append({"block": int(b["block"]),
                     "ML_from_midline": abs(float(c[0]) - mid_mm),
                     "DV": float(c[1]), "AP": float(c[2]),
                     "top_region": comp[0]["name"] if comp else ""})
    if not rows:
        return pd.DataFrame(columns=["block", *AXES, "top_region"])
    return pd.DataFrame(rows)


def add_share(df):
    """Share = this block's count over that sample's total across all blocks.

    Every per-sample LEVEL term -- injection strength, clearing, how much of the
    guide survived -- is in that denominator too, so it divides out, and what is
    left is only WHERE the cells are. That is the question a migration phenotype
    asks, and it is the one question the block mean cannot answer: cells that
    move from one block to another leave the mean over blocks unchanged.

    A combined class built with a minus sign can go negative in a block, and a
    share of a signed total is not a proportion; those classes get NaN instead
    of a number that would look like one."""
    out = df.copy()
    tot = out.groupby(["sample", "class_name"])["Count"].transform("sum")
    out["Share"] = np.where(tot > 0, out["Count"] / tot, np.nan)
    signed = out.groupby("class_name")["Count"].transform("min") < 0
    out.loc[signed, "Share"] = np.nan
    return out


def block_heterogeneity(per_block, positions, n_a, n_b):
    """Is there ANY location structure in the effect, or is one number enough?

    Per-block log2fc with a delta-method SE, then Cochran's Q against the null
    that all blocks share one true effect. I2 near 0 says the blocks are noisy
    copies of the same number, and in that case "the effect is in block 17" is a
    statement about which block drew the lucky sample, not about anatomy. Only
    when Q rejects is there something for a per-block map to show.

    Then Spearman's rho of the block effect against each anatomical axis.
    Migration is a GRADIENT -- cells leave one end and arrive at the other -- so
    it is one test per axis rather than one per block, which is the only version
    of this question with any power at n=3 vs 3. Blocks 600 um apart are not
    independent, so rho's own p is optimistic; read the sign and the size, and
    treat the p as a screen."""
    rows = []
    for (cls, metric), sub in per_block.groupby(["class_name", "metric"], sort=False):
        sub = sub.merge(positions, on="block", how="left").reset_index(drop=True)
        y = sub["log2fc"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = ((sub["sd_a"] ** 2 / (n_a * sub["mean_a"] ** 2)
                    + sub["sd_b"] ** 2 / (n_b * sub["mean_b"] ** 2))
                   / np.log(2.0) ** 2).to_numpy(float)
        se = np.sqrt(var)
        ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
        k = int(ok.sum())
        rec = {"class_name": cls, "metric": metric, "n_blocks": k,
               "pooled_log2fc": np.nan, "Q": np.nan, "df": np.nan,
               "p_heterogeneity": np.nan, "I2": np.nan}
        if k >= 3:
            w = 1.0 / se[ok] ** 2
            ybar = float((w * y[ok]).sum() / w.sum())
            q = float((w * (y[ok] - ybar) ** 2).sum())
            dfree = k - 1
            rec.update(pooled_log2fc=ybar, Q=q, df=dfree,
                       p_heterogeneity=float(sp_stats.chi2.sf(q, dfree)),
                       I2=float(max(0.0, (q - dfree) / q)) if q > 0 else 0.0)
        for axis in AXES:
            rho = p = np.nan
            if k >= 4:
                rho, p = sp_stats.spearmanr(sub.loc[ok, axis].to_numpy(float), y[ok])
            rec[f"rho_{axis}"], rec[f"p_{axis}"] = float(rho), float(p)
        rows.append(rec)
    return pd.DataFrame(rows)


def _block_effect(mat, idx_a, idx_b):
    """-> (per-block log2fc, per-block SE of it) for one assignment of columns.

    Delta method on log2 of a ratio of means. Both terms use n=3, so the SE is
    itself a noisy number -- which is exactly why nothing downstream trusts it
    against a chi2 table."""
    a, b = mat[:, idx_a], mat[:, idx_b]
    with np.errstate(divide="ignore", invalid="ignore"):
        ma, mb = a.mean(axis=1), b.mean(axis=1)
        va = a.var(axis=1, ddof=1) / (a.shape[1] * ma ** 2)
        vb = b.var(axis=1, ddof=1) / (b.shape[1] * mb ** 2)
        y = np.log2(np.where(mb > 0, mb, np.nan) / np.where(ma > 0, ma, np.nan))
        se = np.sqrt((va + vb)) / np.log(2.0)
    return y, se


def _q_and_rho(y, se, coords):
    """Cochran's Q over blocks, plus Spearman rho against each axis."""
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    out = {"k": int(ok.sum()), "Q": np.nan, "pooled": np.nan}
    if ok.sum() >= 3:
        w = 1.0 / se[ok] ** 2
        ybar = float((w * y[ok]).sum() / w.sum())
        out["pooled"] = ybar
        out["Q"] = float((w * (y[ok] - ybar) ** 2).sum())
    for axis, c in coords.items():
        out[axis] = (float(sp_stats.spearmanr(c[ok], y[ok]).statistic)
                     if ok.sum() >= 4 else np.nan)
    return out


def spatial_permutation(df, positions, classes, metrics, samples_a, samples_b):
    """Exact group-label permutation for both spatial statistics.

    Neither textbook p-value survives contact with this design. Q against a chi2
    table assumes the per-block SEs are KNOWN; here each is estimated on 2
    degrees of freedom, so a block that happened to draw a small SD gets an
    enormous weight and Q inflates -- on the sparse classes the I2 it reports is
    mostly that artefact. Spearman's own p has the opposite problem: neighbouring
    blocks are 600 um apart and correlated, so its degrees of freedom are
    fiction.

    Relabelling the GROUPS fixes both at once. Every block keeps its position and
    its noise; only the contrast is reassigned, which is the null the question
    actually asks -- "is this more spatial structure than shuffling the six
    animals would produce". The price is the same floor as everywhere else in a
    3 vs 3: ten distinct splits, so the smallest p is 0.1, and the honest output
    is a rank, not a significance call."""
    names = list(samples_a) + list(samples_b)
    n_a = len(samples_a)
    splits = list(combinations(range(len(names)), n_a))
    obs_split = tuple(range(n_a))
    pos = positions.set_index("block")
    rows = []
    for metric in metrics:
        for cls in classes:
            wide = (df[df.class_name == cls]
                    .pivot_table(index="block", columns="sample", values=metric))
            if wide.empty or not set(names) <= set(wide.columns):
                continue
            wide = wide.dropna(subset=names)
            if len(wide) < 4:
                continue
            mat = wide[names].to_numpy(float)
            coords = {a: pos.loc[wide.index, a].to_numpy(float) for a in AXES}
            draws = []
            for combo in splits:
                rest = [i for i in range(len(names)) if i not in combo]
                y, se = _block_effect(mat, list(combo), rest)
                draws.append(_q_and_rho(y, se, coords))
            obs = draws[splits.index(obs_split)]
            rec = {"class_name": cls, "metric": metric, "n_splits": len(splits) // 2}
            # A permuted split and its complement give the same Q and mirrored
            # rho, so each distinct split is counted twice: the /2 is what makes
            # the rank read out of ten and not out of twenty.
            qs = np.array([d["Q"] for d in draws], dtype=float)
            if np.isfinite(obs["Q"]) and np.isfinite(qs).any():
                ge = int((qs >= obs["Q"] - 1e-12).sum())
                rec["Q_rank"] = ge // 2
                rec["p_Q_perm"] = float(ge / np.isfinite(qs).sum())
            for axis in AXES:
                r = np.array([abs(d[axis]) for d in draws], dtype=float)
                o = abs(obs[axis])
                if np.isfinite(o) and np.isfinite(r).any():
                    ge = int((r >= o - 1e-12).sum())
                    rec[f"rho_rank_{axis}"] = ge // 2
                    rec[f"p_{axis}_perm"] = float(ge / np.isfinite(r).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def redistribution(df, classes, samples_a, samples_b, correction, test, alpha):
    """-> (per-block Share tests, one whole-map permutation row per class).

    per block   one test per block on Share, BH within class. Shares sum to one,
                so the blocks are negatively coupled by construction: this is a
                map of where a shift sits, not 28 independent findings.
    whole map   the total-variation distance between the two groups' mean share
                vectors, which reads directly as "the fraction of cells that
                would have to change block for the two groups to match", against
                an EXACT permutation null over every way to split six samples
                three and three. There are only C(6,3)/2 = 10 distinct splits,
                so the smallest p this design can ever return is 0.1. It is
                therefore reported as a rank: rank 1 of 10 is the strongest
                statement six brains can make, and it is not a p < 0.05."""
    names = list(samples_a) + list(samples_b)
    n_a = len(samples_a)
    splits = [c for c in combinations(range(len(names)), n_a)]
    per_block, glob = [], []
    for cls in classes:
        wide = (df[df.class_name == cls]
                .pivot_table(index="block", columns="sample", values="Share"))
        if wide.empty or not set(names) <= set(wide.columns):
            continue
        wide = wide.dropna(subset=names)
        if wide.empty:
            continue
        labels = [{"block": int(b), "class_name": cls, "metric": "Share"}
                  for b in wide.index]
        per_block.append(test_frame(wide[samples_a].to_numpy(float),
                                    wide[samples_b].to_numpy(float),
                                    labels, correction, test, alpha))
        mat = wide[names].to_numpy(float)
        dists = []
        for combo in splits:
            rest = [i for i in range(len(names)) if i not in combo]
            dists.append(0.5 * float(np.abs(mat[:, list(combo)].mean(axis=1)
                                            - mat[:, rest].mean(axis=1)).sum()))
        obs = dists[splits.index(tuple(range(n_a)))]
        dists = np.array(dists)
        glob.append({"class_name": cls, "tv_distance": obs,
                     "rank_of": len(splits) // 2,
                     "rank": int((dists >= obs - 1e-12).sum() // 2),
                     "p_permutation": float((dists >= obs - 1e-12).sum() / len(splits)),
                     "p_floor": 2.0 / len(splits),
                     "null_median": float(np.median(dists)),
                     "null_max": float(dists.max())})
    per_block = (pd.concat(per_block, ignore_index=True) if per_block
                 else pd.DataFrame())
    if not per_block.empty:
        per_block["exploratory"] = True
    return per_block, pd.DataFrame(glob)


def leave_one_out(df, metric, classes, samples_a, samples_b, test):
    """Drop each sample in turn and re-test. At n=3 vs 3 this is not optional.

    One sample carries a third of a group, so a trend can be entirely one
    brain. On this dataset it is: MADM_all's +0.24 log2fc falls to +0.06
    without s18, the sample that fragmented, and Sox9_pos's -0.23 flips sign to
    +0.10 without s11, the sample with 13.6% of its guide painted as damage.
    A result whose direction depends on which brain is included is a statement
    about that brain, and the only way to see it is to look."""
    rows = []
    per_sample = (df[df.metric_col == metric] if "metric_col" in df else df)
    for cls in classes:
        means = (df[df.class_name == cls].groupby("sample")[metric].mean())
        if not set(samples_a + samples_b) <= set(means.index):
            continue
        full_a = means[samples_a].to_numpy(float)[None, :]
        full_b = means[samples_b].to_numpy(float)[None, :]
        base = float(np.log2(full_b.mean() / full_a.mean()))
        for drop in [None] + samples_a + samples_b:
            a = [s for s in samples_a if s != drop]
            b = [s for s in samples_b if s != drop]
            if len(a) < 2 or len(b) < 2:
                continue
            ma, mb = means[a].to_numpy(float), means[b].to_numpy(float)
            p = float(two_sample_ttest(ma[None, :], mb[None, :], test=test)[0])
            g = float(hedges_g(ma[None, :], mb[None, :])[0][0])
            rows.append({"class_name": cls, "metric": metric,
                         "dropped": drop or "(none)",
                         "mean_a": ma.mean(), "mean_b": mb.mean(),
                         "log2fc": float(np.log2(mb.mean() / ma.mean())),
                         "log2fc_full": base, "hedges_g": g, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # How much of the full-sample effect survives each single drop, as a
    # SIGNED fraction: 1.0 is unchanged, 0 is gone, negative is a sign flip.
    # Signed on purpose -- an unsigned ratio would score a drop that doubles a
    # negative effect the same as one that halves it.
    out["frac_of_full"] = out["log2fc"] / out["log2fc_full"].replace(0, np.nan)
    out.loc[out.dropped == "(none)", "frac_of_full"] = 1.0
    worst = (out[out.dropped != "(none)"]
             .groupby("class_name")["frac_of_full"].min().rename("worst_drop_frac"))
    return out.merge(worst, on="class_name", how="left")


def run(cfg):
    doc = load_blocks(cfg["blocks_json"])
    blocks = doc["blocks"]
    shape_mode = doc.get("block_shape", "cube")
    if shape_mode in ("column", "roi"):
        block_mm3 = {int(b["block"]): float(b["volume_mm3"]) for b in blocks}
    else:
        one = float(doc["block_um"]) ** 3 / 1e9
        block_mm3 = {int(b["block"]): one for b in blocks}
    median_mm3 = float(np.median(list(block_mm3.values())))
    groups = cfg["groups"]
    group_of, samples = {}, []
    for key in ("a", "b"):
        for s in groups[key]["samples"]:
            group_of[s] = groups[key].get("name", key.upper())
            samples.append(s)
    sample_dirs = {s: cfg["samples"][s]["dir"] for s in samples}
    samples_a = groups["a"]["samples"]
    samples_b = groups["b"]["samples"]

    if shape_mode == "roi":
        trim = ", ".join(f"{k} 两端各删 {float(v):.0%}"
                         for k, v in (doc.get("roi_trim") or {}).items())
        print(f"单个 ROI, {median_mm3:.1f} mm3（图谱空间，两半球）, "
              f"{', '.join(doc['regions'])} 的中段：{trim}")
    elif shape_mode == "column":
        print(f"{len(blocks)} 根柱子, {doc.get('column_um')} um 截面, 中位体积 "
              f"{median_mm3:.4f} mm3, 目标 {', '.join(doc['regions'])}")
    else:
        print(f"{len(blocks)} 个块, 每块 {doc['block_um']} um 立方 = "
              f"{median_mm3:.4f} mm3, 目标 {', '.join(doc['regions'])}")
    qc_cfg = cfg.get("qc") or {}
    dark_rel = float(qc_cfg.get("dark_rel", DARK_REL))
    dark_max = float(qc_cfg.get("dark_max", DARK_MAX))
    action = str(qc_cfg.get("action", "report"))
    min_ratio = float(qc_cfg.get("volume_min_ratio", 0.90))
    if action not in ("report", "drop_block"):
        raise ValueError("qc.action 只能是 report 或 drop_block。逐样本剔除会让"
                         "各样本的块平均算在不同的块集合上，那个比较不成立。")

    counts, coverage, darkness, base_classes = count_cells(
        cfg, doc, sample_dirs, dark_rel)
    darkness["dark_rel"] = dark_rel
    darkness["dark_max"] = dark_max
    # Tissue actually present in that block, as a fraction of its voxels. The
    # voxel SET is identical in every sample by construction -- the block is
    # defined in atlas space -- so this is the only thing that can differ, and
    # it is what "the same position gave a similar volume" has to mean.
    darkness["tissue_frac"] = 1.0 - darkness["dark_frac"]
    med = darkness.groupby("block")["tissue_frac"].transform("median")
    darkness["tissue_ratio"] = darkness["tissue_frac"] / med.replace(0, np.nan)
    worst = darkness.groupby("block")["tissue_ratio"].transform("min")
    darkness["block_worst_ratio"] = worst
    # Two ways to fail. Absolute: this sample's block is largely empty.
    # Relative: the samples disagree about how much tissue is there, which is
    # the failure that matters for a comparison and which an absolute cut
    # misses whenever every sample is mediocre in the same place.
    darkness["flagged"] = ((darkness["dark_frac"] >= dark_max)
                           | (worst < min_ratio))
    # A block flagged in ONE sample is dropped from EVERY sample or from none.
    # Dropping it per sample would leave each sample's block mean computed over
    # a different set of territories, and those means are not comparable.
    bad = sorted(int(b) for b in darkness.loc[darkness.flagged, "block"].unique())
    dropped = []
    if bad:
        shown = darkness[darkness.flagged].sort_values("tissue_ratio")
        for _, row in shown.head(15).iterrows():
            print(f"  QC {row['sample']} block {int(row['block'])}: "
                  f"组织占 {row['tissue_frac'] * 100:.1f}%，是该柱子六样本中位的 "
                  f"{row['tissue_ratio'] * 100:.0f}%，最大连通暗区 "
                  f"{row['dark_cc_frac'] * 100:.1f}%")
        if action == "drop_block":
            dropped = bad
            counts = counts[~counts.block.isin(dropped)].reset_index(drop=True)
            coverage = coverage[~coverage.block.isin(dropped)].reset_index(drop=True)
            print(f"  QC: 从所有样本里剔除块 {dropped}，剩 "
                  f"{len(blocks) - len(dropped)} 个")
        else:
            print(f"  QC: 只报告不剔除（qc.action=report）。可疑块 {bad}")

    counts, combined = add_combined(counts, cfg, base_classes)
    metrics = [m for m in cfg.get("metrics", ["Count", "Density", "CoveredDensity"])]
    df = add_share(build_metrics(counts, coverage, block_mm3))
    positions = block_positions(doc)
    all_classes = base_classes + combined

    # ── primary: per-sample mean over blocks ─────────────────────────────────
    primary = []
    for metric in metrics:
        per_sample = (df.groupby(["sample", "class_name"])[metric]
                      .mean().unstack("sample"))
        rows, mat_a, mat_b = [], [], []
        for cls in all_classes:
            if cls not in per_sample.index:
                continue
            rows.append({"class_name": cls, "metric": metric,
                         "n_blocks": int(df[df.class_name == cls]
                                         .groupby("sample")["block"].nunique().max())})
            mat_a.append(per_sample.loc[cls, samples_a].to_numpy(float))
            mat_b.append(per_sample.loc[cls, samples_b].to_numpy(float))
        if not rows:
            continue
        primary.append(test_frame(np.array(mat_a), np.array(mat_b), rows,
                                  cfg["stats"].get("correction", "bh"),
                                  cfg["stats"].get("test", "welch"),
                                  float(cfg["stats"].get("alpha", 0.05))))
    primary = pd.concat(primary, ignore_index=True)

    # ── secondary: one test per block, BH within each (class, metric) ────────
    per_block = []
    for metric in metrics:
        for cls in all_classes:
            sub = df[df.class_name == cls]
            wide = sub.pivot_table(index="block", columns="sample", values=metric)
            wide = wide.dropna(subset=samples_a + samples_b)
            if wide.empty:
                continue
            labels = [{"block": int(b), "class_name": cls, "metric": metric}
                      for b in wide.index]
            per_block.append(test_frame(
                wide[samples_a].to_numpy(float), wide[samples_b].to_numpy(float),
                labels, cfg["stats"].get("correction", "bh"),
                cfg["stats"].get("test", "welch"),
                float(cfg["stats"].get("alpha", 0.05))))
    per_block = pd.concat(per_block, ignore_index=True)
    per_block["exploratory"] = True

    loo = pd.concat(
        [leave_one_out(df, m, all_classes, samples_a, samples_b,
                       cfg["stats"].get("test", "welch")) for m in metrics],
        ignore_index=True)

    # ── where, not how many: heterogeneity, gradients, redistribution ────────
    # A single ROI has no "where": one block cannot be heterogeneous, cannot
    # have a gradient, and its Share is 1.0 in every sample by definition.
    single = len(df["block"].unique()) < 2
    hetero = (pd.DataFrame() if single else
              block_heterogeneity(per_block, positions,
                                  len(samples_a), len(samples_b)))
    share_block, share_global = ((pd.DataFrame(), pd.DataFrame()) if single else
        redistribution(
            df, all_classes, samples_a, samples_b,
            cfg["stats"].get("correction", "bh"), cfg["stats"].get("test", "welch"),
            float(cfg["stats"].get("alpha", 0.05))))
    if not share_block.empty and not hetero.empty:
        hetero = pd.concat(
            [hetero, block_heterogeneity(share_block, positions,
                                         len(samples_a), len(samples_b))],
            ignore_index=True)
    perm = (pd.DataFrame() if single else
            spatial_permutation(df, positions, all_classes,
                                list(metrics) + ["Share"], samples_a, samples_b))
    if not perm.empty:
        hetero = hetero.merge(perm, on=["class_name", "metric"], how="left")

    return dict(doc=doc, df=df, counts=counts, coverage=coverage, loo=loo,
                darkness=darkness, dropped_blocks=dropped, qc_action=action,
                primary=primary, per_block=per_block, classes=all_classes,
                metrics=metrics, group_of=group_of, samples=samples,
                block_mm3=median_mm3, shape_mode=shape_mode,
                cfg=cfg, positions=positions,
                hetero=hetero, share_block=share_block,
                share_global=share_global)


def write_outputs(r):
    out_dir = r["cfg"]["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    xlsx = os.path.join(out_dir, "block_stats.xlsx")

    readme = pd.DataFrame([
        ("what", "Group comparison on fixed atlas-space blocks, not whole regions."),
        ("why", "Per-region Density divides by a SAMPLE-SPACE volume, so it "
                "carries clearing shrinkage. Whole-brain volume here spans "
                "49.1-82.4 mm3 and 26/28 level-2/3 structures are 13-23% "
                "smaller in one group while RelativeVolume is flat: a global "
                "scale difference. A block fixed in atlas space has the same "
                "volume in every sample, so that term cancels."),
        ("blocks", (f"{len(r['doc']['blocks'])} columns of "
                    f"{r['doc'].get('column_um')} um cross-section following the "
                    f"cortical normal, median {r['block_mm3']:.4f} mm3, cut at "
                    f"both ends by the region mask so they run pia to white "
                    f"matter, inside {', '.join(r['doc']['regions'])}, seed "
                    f"{r['doc'].get('seed')}."
                    if r["shape_mode"] == "column" else
                    f"{len(r['doc']['blocks'])} cubes of {r['doc'].get('block_um')} um "
                    f"= {r['block_mm3']:.4f} mm3 each, inside "
                    f"{', '.join(r['doc']['regions'])}, min_purity "
                    f"{r['doc'].get('min_purity')}, seed {r['doc'].get('seed')}.")),
        ("layer caveat", ("Columns follow the cortical normal and are cut by "
                          "the region mask, so they run the full depth: "
                          "measured composition L1 9.3%, L2/3 17.9%, L4 5.3%, "
                          "L5 26.9%, L6a 37.1%, against 13.0 / 20.8 / 5.1 / "
                          "29.2 / 29.4 in the whole sheet. Upper layers are "
                          "present, L1 still under-sampled."
                          if r["shape_mode"] == "column" else
                          "A cube required to sit entirely inside Isocortex "
                          "cannot reach the upper layers: they are a thin outer "
                          "shell. These blocks sample DEEP cortex, mostly "
                          "layers 5 and 6a. An effect confined to layers 1-4 "
                          "will not appear here.")),
        ("Count", "Cells of that class inside the block."),
        ("Density", "Count / block atlas volume. The divisor is a constant, so "
                    "this cannot differ from Count in p or g -- it is the "
                    "reportable form, not extra information."),
        ("CoveredDensity", "Count / (block volume x coverage). The only "
                           "correction available for a block with missing "
                           "tissue. Note the warped brain mask is an envelope "
                           "and reads ~1.00 almost everywhere, so it does NOT "
                           "detect cracks; read Per_Block_Counts for outliers."),
        ("Block_QC", "Per block per sample: dark_frac is the fraction of the "
                     "cube below dark_rel x that sample's median block "
                     "intensity in the warped image, dark_cc_frac the largest "
                     "CONNECTED dark component. A hole makes the two nearly "
                     "equal; speckle does not. This is the check that works -- "
                     "the brain-mask coverage next door reads 1.00 even on a "
                     "block that is a fifth empty. A flagged block is dropped "
                     "from every sample or from none (qc.action), never from "
                     "one: per-sample dropping would average each sample over "
                     "a different set of territories."),
        ("Primary", "One test per (class, metric) on each sample's MEAN over "
                    "blocks. n is 3 vs 3. This is the analysis."),
        ("Per_Block", "One test per (block, class), n=3 vs 3, BH within each "
                      "(class, metric). Exploratory: says WHERE for an effect "
                      "the primary already supports. Not an independent result."),
        ("Leave_One_Out", "Every class re-tested with each sample dropped. At "
                          "3 vs 3 one brain is a third of a group; "
                          "worst_drop_frac is the fraction of the full "
                          "log2fc that survives the worst single drop. "
                          "Near 0, or negative, means the effect is one brain."),
        ("not fixed", "Blocks do not increase n, and a block is averaged into "
                      "its sample before testing -- blocks are never treated "
                      "as replicates."),
        ("Share", "That block's count over the SAME SAMPLE's total across all "
                  "blocks. Every per-sample level term divides out, so a "
                  "difference in Share means the cells sit in different places "
                  "-- the migration question. Cells moving between blocks leave "
                  "the block mean unchanged, so Primary cannot see it."),
        ("Redistribution", "Per-block Share tests, BH within class. Shares sum "
                           "to 1, so blocks are negatively coupled by "
                           "construction: a map of where a shift sits, not "
                           "independent findings."),
        ("Redistribution_Global", "Total-variation distance between the two "
                                  "groups' mean share vectors -- the fraction "
                                  "of cells that would have to change block for "
                                  "the groups to match -- against an exact "
                                  "permutation null over all 10 distinct 3v3 "
                                  "splits. The smallest p this design can "
                                  "return is 0.1, so read the RANK, not a "
                                  "significance call."),
        ("permutation columns", "p_Q_perm and p_<axis>_perm come from "
                                "relabelling the six animals every possible "
                                "way (10 distinct splits) and recomputing the "
                                "statistic. They replace the chi2 and Spearman "
                                "p on their left, both of which are wrong here: "
                                "chi2 assumes the per-block SEs are known when "
                                "they carry 2 df, and Spearman assumes blocks "
                                "600 um apart are independent. The permutation "
                                "cannot go below 0.1, so read the rank."),
        ("Heterogeneity", "Cochran's Q and I2 over the per-block effects, plus "
                          "Spearman rho against each anatomical axis. I2 near 0 "
                          "means the blocks are noisy copies of one number and "
                          "no per-block claim is supportable. A gradient (rho) "
                          "is one test per axis instead of one per block, which "
                          "is the only spatial question with power at this n. "
                          "Blocks are spatially correlated, so rho's p is "
                          "optimistic."),
        ("layers again", "These blocks sit in DEEP cortex only, so a radial "
                         "(pia-to-white-matter) migration phenotype is outside "
                         "what they can measure. Only tangential gradients are "
                         "testable here."),
    ], columns=["item", "meaning"])

    block_table = r["df"].pivot_table(index=["class_name", "block"],
                                      columns="sample", values="Count",
                                      fill_value=0).reset_index()
    cov_table = r["coverage"].pivot_table(index="block", columns="sample",
                                          values="coverage").reset_index()
    qc_table = (r["darkness"].pivot_table(index="block", columns="sample",
                                          values="dark_frac").reset_index()
                if not r["darkness"].empty else pd.DataFrame())

    with pd.ExcelWriter(xlsx) as writer:
        readme.to_excel(writer, sheet_name="ReadMe", index=False)
        r["primary"].to_excel(writer, sheet_name="Primary", index=False)
        r["per_block"].to_excel(writer, sheet_name="Per_Block", index=False)
        r["loo"].to_excel(writer, sheet_name="Leave_One_Out", index=False)
        if not r["share_block"].empty:
            r["share_block"].to_excel(writer, sheet_name="Redistribution",
                                      index=False)
        if not r["share_global"].empty:
            r["share_global"].to_excel(writer, sheet_name="Redistribution_Global",
                                       index=False)
        r["hetero"].to_excel(writer, sheet_name="Heterogeneity", index=False)
        r["positions"].to_excel(writer, sheet_name="Block_Positions", index=False)
        block_table.to_excel(writer, sheet_name="Per_Block_Counts", index=False)
        cov_table.to_excel(writer, sheet_name="Coverage", index=False)
        if not r["darkness"].empty:
            r["darkness"].to_excel(writer, sheet_name="Block_QC", index=False)
            qc_table.to_excel(writer, sheet_name="Block_QC_Dark", index=False)

    r["primary"].to_csv(os.path.join(out_dir, "primary.csv"), index=False)
    r["per_block"].to_csv(os.path.join(out_dir, "per_block.csv"), index=False)
    r["df"].to_csv(os.path.join(out_dir, "block_counts_long.csv"), index=False)
    r["hetero"].to_csv(os.path.join(out_dir, "heterogeneity.csv"), index=False)
    r["positions"].to_csv(os.path.join(out_dir, "block_positions.csv"), index=False)
    if not r["darkness"].empty:
        r["darkness"].to_csv(os.path.join(out_dir, "qc_blocks.csv"), index=False)
    if not r["share_block"].empty:
        r["share_block"].to_csv(os.path.join(out_dir, "redistribution.csv"),
                                index=False)
    if not r["share_global"].empty:
        r["share_global"].to_csv(
            os.path.join(out_dir, "redistribution_global.csv"), index=False)
    with open(os.path.join(out_dir, "blocks_used.json"), "w", encoding="utf-8") as f:
        json.dump(r["doc"], f, indent=2)
    return xlsx


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-figures", action="store_true",
                        help="只写表，不画图")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # load_config resolves ontology/sample/output paths against the config's
    # directory; blocks_json is this tool's own key, so it needs the same.
    root = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(cfg["blocks_json"]):
        cfg["blocks_json"] = os.path.normpath(os.path.join(root, cfg["blocks_json"]))
    r = run(cfg)

    alpha = float(cfg["stats"].get("alpha", 0.05))

    qc = r["darkness"]
    if not qc.empty:
        rel = float(qc["dark_rel"].iloc[0])
        min_ratio = float((cfg.get("qc") or {}).get("volume_min_ratio", 0.90))
        top = qc.sort_values("tissue_ratio").head(10)
        print("\n=== 块的成像质量：块里有多少体积是黑的 ===")
        print(f"  暗 = 低于该样本块内组织中位强度的 {rel:.0%}。"
              f"dark_cc = 最大连通暗区，和 dark_frac 接近说明是一个洞而不是散点")
        print(f"  tissue_ratio = 该样本这根柱子的组织量 / 六个样本在同一根柱子上的中位。"
              f"低于 {min_ratio:.0%} 判为各样本取到的体积不一致")
        show = top[["sample", "block", "tissue_frac", "tissue_ratio",
                    "dark_cc_frac", "flagged"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        if r["dropped_blocks"]:
            print(f"  已从所有样本剔除: {r['dropped_blocks']}")
        elif qc["flagged"].any():
            print(f"  qc.action=report，没有剔除任何块")

    print("\n=== Primary: 每样本先对块取平均，再做组间比较 (n=3 vs 3) ===")
    show = r["primary"][r["primary"].metric == "Count"].copy()
    show = show.sort_values("p_value")
    print(show[["class_name", "mean_a", "mean_b", "log2fc", "hedges_g",
                "p_value", "p_adj"]].to_string(index=False,
                                               float_format=lambda v: f"{v:9.4f}"))
    hits = r["primary"][r["primary"].p_adj < alpha]
    print(f"\nPrimary 显著项: {len(hits)}")
    if len(hits):
        print(hits[["class_name", "metric", "log2fc", "hedges_g",
                    "p_value", "p_adj"]].to_string(index=False))

    pb = r["per_block"]
    pbh = pb[pb.p_adj < alpha]
    print(f"\nPer_block 显著项 (exploratory): {len(pbh)} / {len(pb)}")
    if len(pbh):
        print(pbh.sort_values("p_adj").head(15)[
            ["block", "class_name", "metric", "log2fc", "hedges_g", "p_adj"]
        ].to_string(index=False))

    loo = r["loo"]
    if not loo.empty:
        sub = loo[(loo.metric == "Count") & (loo.dropped == "(none)")]
        print("\n=== 留一法：整体效应有多少靠单个样本撑着 ===")
        print(f"{'class':20s} {'log2fc(全部)':>13s} {'最差单样本剩余比例':>20s}")
        for _, row in sub.sort_values("worst_drop_frac").iterrows():
            print(f"{row.class_name:20s} {row.log2fc_full:13.3f} "
                  f"{row.worst_drop_frac:20.2f}")
        driven = sub[sub.worst_drop_frac < 0.4]
        if len(driven):
            worst = loo[(loo.metric == "Count") & (loo.dropped != "(none)")]
            for cls in driven.class_name:
                # The worst drop is the one that most WEAKENS the effect, which
                # for a negative effect is the largest log2fc, not the smallest
                # -- so select on the signed fraction, never on log2fc itself.
                w = worst[worst.class_name == cls].nsmallest(1, "frac_of_full").iloc[0]
                print(f"  {cls}: 去掉 {w.dropped} 之后 log2fc 从 "
                      f"{w.log2fc_full:+.3f} 变成 {w.log2fc:+.3f}")

    hetero = r["hetero"]
    if not hetero.empty:
        sub = hetero[hetero.metric == "Count"]
        print("\n=== 块间异质性：效应到底有没有位置结构 ===")
        print(f"  I2 接近 0 = {len(r['doc']['blocks'])} 个块只是同一个数的噪声副本，"
              "\"效应在第几号块\" 就没有意义")
        cols = ["class_name", "pooled_log2fc", "Q", "I2", "p_heterogeneity"]
        cols += [c for c in ("p_Q_perm",) if c in sub.columns]
        print("  p_heterogeneity 是 chi2 的，n=3 时权重不稳会高估；以 p_Q_perm 为准")
        print(sub[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:9.3f}"))
        print("\n=== 沿解剖轴的梯度（这是组间差异的梯度，不是细胞数本身的梯度）===")
        gap = r["doc"].get("column_um") or r["doc"].get("block_um")
        print(f"  块之间只隔 {gap} um，不独立，Spearman 自己的 p 偏乐观；"
              "p_*_perm 是打乱分组标签算出来的，下限 0.1")
        cols = ["class_name"]
        for a in AXES:
            cols += [c for c in (f"rho_{a}", f"p_{a}", f"p_{a}_perm")
                     if c in sub.columns]
        print(sub[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:7.3f}"))

    sg = r["share_global"]
    if not sg.empty:
        print("\n=== 重分布：两组把细胞放在同一批位置上吗 ===")
        print(f"  tv_distance = 要让两组的分布对上，需要换块的细胞比例。"
              f"3v3 只有 {int(sg['rank_of'].iloc[0])} 种不同的分法，"
              f"p 最小只能到 {sg['p_floor'].iloc[0]:.2f}，所以看排名不看显著性")
        print(sg[["class_name", "tv_distance", "null_median", "rank",
                  "rank_of", "p_permutation"]].to_string(
                      index=False, float_format=lambda v: f"{v:8.3f}"))
    sb = r["share_block"]
    if not sb.empty:
        sbh = sb[sb.p_adj < alpha]
        print(f"\nShare 的逐块显著项 (exploratory): {len(sbh)} / {len(sb)}")
        if len(sbh):
            print(sbh.sort_values("p_adj").head(15)[
                ["block", "class_name", "log2fc", "hedges_g", "p_adj"]
            ].to_string(index=False))

    xlsx = write_outputs(r)
    if not args.no_figures:
        try:
            from stats import plot_blocks
            plot_blocks.render_all(args.config)
        except ImportError as exc:
            print(f"跳过画图: {exc}")
    print(f"\n写出 -> {xlsx}")
    print(f"生成于 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
