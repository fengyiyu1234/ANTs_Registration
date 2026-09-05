"""Per-sample region volumes and tissue coverage -- the denominator side of
every density comparison.

Replaces the ClearMap-era `volume/result.mhd` (transformix-warped atlas) with
the two volumes this pipeline already writes into every run directory:

  <name>_labels_in_sample.nii.gz  atlas annotation warped into sample space,
                                  nearest-neighbour, so each region's voxel
                                  count IS that sample's own volume for it --
                                  brain-size differences between groups are
                                  absorbed here rather than becoming a fake
                                  density effect.
  <name>_brain_mask.nii.gz        where tissue actually is in this sample.

Why both: the samples are half-brains and the cut does not land on the
anatomical midline in every one of them (PROGRESS_LOG 2026-08-29). A region
straddling a short cut keeps its full warped volume but is missing the tissue,
so counting cells over the full warped volume systematically under-reports
its density. `coverage` = warped region voxels that are inside the brain mask,
and the default density denominator is the covered volume, so a truncated
region is measured over the part that was actually imaged. Regions whose
coverage falls below the config threshold are dropped for that sample instead
of being reported as a low density.

Note labels_in_sample is already zeroed outside registration.crop_for_registration,
so a region cropped away shrinks here too -- that is why `volume_ratio_to_median`
is also reported: a region much smaller in one sample than in the others is the
signature of a crop or a truncation, whatever the brain mask says.
"""
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd


def find_run_volumes(run_dir):
    """-> (labels_path, mask_path). mask_path is None if the run has none."""
    labels = sorted(glob.glob(os.path.join(run_dir, "*_labels_in_sample.nii.gz")))
    if not labels:
        raise FileNotFoundError(f"no *_labels_in_sample.nii.gz in {run_dir}")
    if len(labels) > 1:
        raise ValueError(f"multiple *_labels_in_sample.nii.gz in {run_dir}: {labels}")
    masks = sorted(glob.glob(os.path.join(run_dir, "*_brain_mask.nii.gz")))
    return labels[0], (masks[0] if masks else None)


def _voxel_offset(mask_img, ref_img):
    """Integer index offset that puts `mask_img`'s grid onto `ref_img`'s.

    The two are not always index-aligned: crop_for_registration crops in raw
    voxels, whose physical position need not be a whole number of 20 um
    registration voxels (s11: crop x=180 raw -> origin 468 um -> 23.4 voxels).
    Both grids are axis-aligned with identical spacing (every image this
    pipeline builds has identity direction), so the offset is a pure
    translation; it is rounded, and anything beyond a tenth of a voxel of
    rounding is reported since it slightly blurs the coverage estimate."""
    m_aff, r_aff = mask_img.affine, ref_img.affine
    if not np.allclose(m_aff[:3, :3], r_aff[:3, :3], atol=1e-4):
        raise ValueError("brain mask and label volume have different orientation/spacing; "
                         "cannot align by translation alone")
    shift = np.linalg.solve(r_aff[:3, :3], m_aff[:3, 3] - r_aff[:3, 3])
    rounded = np.round(shift).astype(int)
    return rounded, float(np.max(np.abs(shift - rounded)))


def _mask_on_reference_grid(mask_path, ref_img):
    """Boolean brain mask resampled (by integer translation) onto the label
    volume's grid. Voxels of the label grid not covered by the mask volume at
    all are False -- they are outside what registration ever looked at."""
    mask_img = nib.load(mask_path)
    offset, residual = _voxel_offset(mask_img, ref_img)
    mask = np.asarray(mask_img.dataobj) > 0
    out = np.zeros(ref_img.shape, dtype=bool)

    src_slices, dst_slices = [], []
    for axis in range(3):
        o = int(offset[axis])
        src_lo = max(0, -o)
        src_hi = min(mask.shape[axis], ref_img.shape[axis] - o)
        if src_hi <= src_lo:
            return out, residual  # no overlap at all
        src_slices.append(slice(src_lo, src_hi))
        dst_slices.append(slice(src_lo + o, src_hi + o))
    out[tuple(dst_slices)] = mask[tuple(src_slices)]
    return out, residual


def sample_region_volumes(run_dir, ontology, use_mask=True, verbose=True):
    """One row per ontology order for one sample, holding DIRECT (un-rolled)
    voxel counts.

    Columns: order, direct_voxel_count, direct_covered_voxel_count,
    voxel_mm3. The rollup is deliberately NOT done here: region exclusions
    have to be applied to the direct counts before rolling up (see
    rollup_volumes), so caching rolled-up numbers would bake one particular
    exclusion set into the cache."""
    labels_path, mask_path = find_run_volumes(run_dir)
    img = nib.load(labels_path)
    arr = np.asarray(img.dataobj)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    voxel_mm3 = float(np.prod(zooms)) * 1e-9  # um^3 -> mm^3

    ids, counts = np.unique(arr, return_counts=True)
    keep = ids > 0
    ids, counts = ids[keep].astype(np.int64), counts[keep].astype(float)
    orders = ontology.order_of_ids(ids)
    unknown = ids[orders < 0]
    if verbose and len(unknown):
        print(f"  [warn] {os.path.basename(run_dir)}: {len(unknown)} label id(s) not in the "
              f"ontology, e.g. {unknown[:5].tolist()} -- their volume is excluded")

    direct = np.zeros(ontology.n, dtype=float)
    np.add.at(direct, orders[orders >= 0], counts[orders >= 0])

    if use_mask and mask_path:
        mask, residual = _mask_on_reference_grid(mask_path, img)
        if verbose and residual > 0.1:
            print(f"  [note] {os.path.basename(run_dir)}: brain mask grid is offset from the "
                  f"label grid by {residual:.2f} voxel after rounding")
        m_ids, m_counts = np.unique(np.where(mask, arr, 0), return_counts=True)
        m_keep = m_ids > 0
        m_orders = ontology.order_of_ids(m_ids[m_keep].astype(np.int64))
        m_direct = np.zeros(ontology.n, dtype=float)
        np.add.at(m_direct, m_orders[m_orders >= 0], m_counts[m_keep][m_orders >= 0].astype(float))
    else:
        if verbose and use_mask:
            print(f"  [warn] {os.path.basename(run_dir)}: no *_brain_mask.nii.gz; "
                  "coverage set to 1.0 and truncated regions will NOT be detected")
        m_direct = direct.copy()

    return pd.DataFrame({
        "order": np.arange(ontology.n),
        "direct_voxel_count": direct,
        "direct_covered_voxel_count": m_direct,
        "voxel_mm3": voxel_mm3,
    })


def rollup_volumes(direct_df, ontology, exclude_mask=None):
    """Direct voxel counts -> the per-sample analysis table.

    Applies the region exclusions to the direct counts and only then rolls up,
    so an excluded subtree disappears from every ancestor's volume as well --
    including the root, which is what makes `relative_pct` a share of the
    regions actually under analysis rather than of the whole brain.

    Columns added: voxel_count, covered_voxel_count, volume_mm3,
    covered_volume_mm3, coverage, relative_pct, relative_covered_pct,
    volume_ratio_to_median.
    """
    frames = []
    for sample, g in direct_df.groupby("sample", sort=False):
        g = g.sort_values("order")
        direct = g["direct_voxel_count"].to_numpy(dtype=float)
        covered_direct = g["direct_covered_voxel_count"].to_numpy(dtype=float)
        voxel_mm3 = float(g["voxel_mm3"].iloc[0])
        if exclude_mask is not None and exclude_mask.any():
            direct = direct.copy(); direct[exclude_mask] = 0.0
            covered_direct = covered_direct.copy(); covered_direct[exclude_mask] = 0.0
        rolled = ontology.rollup(direct)
        covered = ontology.rollup(covered_direct)
        with np.errstate(divide="ignore", invalid="ignore"):
            coverage = np.where(rolled > 0, covered / rolled, np.nan)
        total = rolled[ontology.root_order]
        total_cov = covered[ontology.root_order]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = rolled / total * 100.0 if total > 0 else np.full_like(rolled, np.nan)
            rel_cov = covered / total_cov * 100.0 if total_cov > 0 else np.full_like(covered, np.nan)
        frames.append(pd.DataFrame({
            "sample": sample,
            "order": np.arange(ontology.n),
            "voxel_count": rolled,
            "covered_voxel_count": covered,
            "volume_mm3": rolled * voxel_mm3,
            "covered_volume_mm3": covered * voxel_mm3,
            "coverage": coverage,
            "relative_pct": rel,
            "relative_covered_pct": rel_cov,
        }))
    out = pd.concat(frames, ignore_index=True)
    median = out.groupby("order")["volume_mm3"].transform("median")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["volume_ratio_to_median"] = np.where(median > 0, out["volume_mm3"] / median, np.nan)
    return out


_DIRECT_COLUMNS = {"sample", "order", "direct_voxel_count",
                   "direct_covered_voxel_count", "voxel_mm3"}


def build_per_sample_volumes(sample_dirs, ontology, use_mask=True, cache_path=None,
                             force=False, exclude_mask=None):
    """sample_dirs: {sample name -> run directory}. Reads (or caches) the
    per-sample DIRECT voxel counts, then rolls them up with `exclude_mask`
    applied.

    The cache holds direct counts only, so changing the exclusion list does
    not invalidate it -- and an older cache written in the previous rolled-up
    format is detected by its columns and recomputed rather than silently
    misread."""
    direct_df = None
    if cache_path and os.path.exists(cache_path) and not force:
        cached = pd.read_csv(cache_path)
        if not _DIRECT_COLUMNS.issubset(cached.columns):
            print(f"  [note] cache {cache_path} predates the direct-count format -- recomputing")
        elif set(cached["sample"]) != set(sample_dirs):
            print(f"  [note] cache {cache_path} covers {sorted(set(cached['sample']))}, "
                  f"need {sorted(sample_dirs)} -- recomputing")
        else:
            print(f"Loaded per-sample region volumes from cache: {cache_path}")
            direct_df = cached

    if direct_df is None:
        frames = []
        for sample, run_dir in sample_dirs.items():
            print(f"Computing region volumes for '{sample}' from {run_dir} ...")
            one = sample_region_volumes(run_dir, ontology, use_mask=use_mask)
            one.insert(0, "sample", sample)
            frames.append(one)
        direct_df = pd.concat(frames, ignore_index=True)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            direct_df.to_csv(cache_path, index=False)
            print(f"Wrote per-sample region volumes: {cache_path}")

    # preserve the caller's sample order for stable output columns
    direct_df["sample"] = pd.Categorical(direct_df["sample"], categories=list(sample_dirs),
                                         ordered=True)
    direct_df = direct_df.sort_values(["sample", "order"])
    direct_df["sample"] = direct_df["sample"].astype(str)
    return rollup_volumes(direct_df, ontology, exclude_mask=exclude_mask)


def build_reference_volumes(annotation_path, voxel_size_um, ontology, cache_path=None, force=False):
    """Sample-independent region volumes from the master atlas annotation --
    a reference column only. Density is never computed from this: a fixed
    per-region constant would make Density a pure rescaling of Count and would
    hide exactly the brain-size difference between the groups."""
    if cache_path and os.path.exists(cache_path) and not force:
        print(f"Loaded reference region volumes from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Computing reference region volumes from {annotation_path} (one-time) ...")
    if str(annotation_path).endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(annotation_path)
    else:
        arr = np.asarray(nib.load(annotation_path).dataobj)

    ids, counts = np.unique(arr, return_counts=True)
    keep = ids > 0
    orders = ontology.order_of_ids(ids[keep].astype(np.int64))
    direct = np.zeros(ontology.n, dtype=float)
    np.add.at(direct, orders[orders >= 0], counts[keep][orders >= 0].astype(float))
    rolled = ontology.rollup(direct)

    voxel_mm3 = float(np.prod(np.asarray(voxel_size_um, dtype=float))) * 1e-9
    df = pd.DataFrame({
        "order": np.arange(ontology.n),
        "ref_voxel_count": rolled,
        "ref_volume_mm3": rolled * voxel_mm3,
    })
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
