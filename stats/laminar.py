"""Isocortex laminar distribution: every area's layers merged into depth bins,
then compared between groups.

WHY LAYERS AND NOT AREAS
------------------------
Cells reach the cortex by radial migration, and what migration decides is the
DEPTH a cell stops at, not the area it sits in. The per-region runs had
already shown that the Isocortex Sox9+ shift is one isocortex-wide shift seen
through every subregion (0913_01): dividing each area by its own animal's
isocortex removed most of the per-area effect. So the question is asked on
layers pooled across all isocortical areas.

WHY THESE BINS (measured 2026-09-14 on DeMBA P5 + the six TSC brains)
----------------------------------------------------------------------
* 6b is merged into L6. It is ~57 um thick in the atlas and 83% of its voxels
  sit within one voxel of its own boundary -- thinner than this dataset's
  registration residual. A layer that is all boundary cannot be counted.
* L1 is never a bin on its own. The L1/L2-3 boundary is the least reliable
  line in these registrations, in both directions: s10's cortex is partly
  compressed and pushes L2/3 cells into the L1 label (25.6% of its isocortical
  cells, rising to 52% along raw z), while in s12t the L1 label lies on the
  bright pial rim where almost nothing is detected (4.6%). The other four
  brains sit at 11.5-15.7%. That spread is tissue and detection, not biology.
* L4 is never a bin on its own either. Only 25 of 43 isocortical areas carry
  an L4 label; the agranular ones (MO, ACA, PL, ILA, ORB, AI, RSP, FRP, ECT,
  PERI) do not, so a pooled "L4" is sensory cortex only and the agranular
  areas' would-be L4 cells are already inside their L2/3 and L5.
* L5 vs L6 is taken as reliable: both are thick (~270 / ~500 um) and the
  corpus callosum below them registers well.

Hence the default: upper (1 + 2/3 + 4), L5, L6 (6a + 6b), plus superficial vs
deep as the coarsest summary. Every scheme must place every layer label the
atlas uses inside the root, or the run stops -- a layer silently left out would
shrink every denominator without anyone noticing.

METRICS
-------
  LaminarShare    cells of a class in a bin / cells of that class in all bins
                  of the same animal. PRIMARY. Labelling efficiency, dose and
                  brain size are in numerator and denominator and cancel; what
                  is left is only depth. Shares sum to 1 across a scheme's bins,
                  so a rise in one bin is a fall in the others: read the bins of
                  one readout together, never as independent findings.
  BinComposition  cells of a class in a bin / all labelled cells in that bin,
                  e.g. the Sox9+ fraction of L5. Complementary classes
                  (Sox9_pos / Sox9_neg) are the same test.
  Count           raw cells per bin, for reference only; it carries the
                  per-animal labelling efficiency (L1 cells range 7k-49k).

Two poolings of the proportions:
  pooled          counts summed over areas, then divided. Big areas (SS, MO)
                  weigh most. The primary estimator.
  area_mean       the proportion inside each area, then averaged over areas.
                  Every area weighs the same. Areas below `area_min_cells` in
                  ANY animal are dropped so all animals average the same set,
                  and a bin an area has no atlas label for is undefined there,
                  not zero.

  Density        cells of a class in a bin per mm3 that the bin occupies inside
                 the slab, measured on THIS animal's warped labels. Offered only
                 for a `volume` slab, where the denominator is a real measured
                 volume rather than the whole structure -- the ~20% brain-size
                 difference is inside the denominator, so it cannot reappear as
                 an effect the way it did in the whole-structure Density runs.
                 It is the primary readout when a volume slab is configured:
                 LaminarShare cannot see a change that moves every bin the same
                 way, and Count carries the animal's size.

    conda activate antsreg
    python -m stats.laminar --config stats/configs/tsc_laminar.yaml
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.composition import readouts_from_config  # noqa: E402
from stats.group_stats import (load_config, samples_beside_means,  # noqa: E402
                               save_config_copy)
from stats.ontology import Ontology  # noqa: E402
from stats.summary_workbook import (_assert_ascii_text, _loo,  # noqa: E402
                                    _permutation_rank, _test_rows)

ISOCORTEX_ID = 315
ATLAS_UM = 20.0
# The warped-coordinate columns of a cell_registration.csv, in the atlas grid
# every sample's yaml calls (x, y, z) = (ML, AP, DV).
ATLAS_AXES = ("xt", "yt", "zt")
# The sample's own 20 um grid -- the same grid labels_in_sample.nii.gz sits on,
# so a slab cut here has a volume that can simply be counted, with no transform
# and no fit in between. Axis order matches the nifti's (i, j, k).
SAMPLE_AXES = ("xr", "yr", "zr")
SLAB_AXES = ATLAS_AXES + SAMPLE_AXES

# "Primary motor area, Layer 1", "..., layer 2/3", and the ACAv labels that lost
# the word: "Anterior cingulate area, ventral part, 6a".
LAYER_NAME = re.compile(r"(?:,\s*layer\s*|\s+layer\s+|,\s*)(1|2/3|2|3|4|5|6a|6b)\s*$",
                        re.IGNORECASE)
TOKEN_ALIAS = {"2": "2/3", "3": "2/3"}

DEFAULT_SCHEMES = {
    "three_bin": {"upper": ["1", "2/3", "4"], "L5": ["5"], "L6": ["6a", "6b"]},
    "two_bin": {"superficial": ["1", "2/3", "4"], "deep": ["5", "6a", "6b"]},
}
UNASSIGNED = "unassigned"


def layer_token(name):
    m = LAYER_NAME.search(str(name))
    if not m:
        return None
    t = m.group(1).lower()
    return TOKEN_ALIAS.get(t, t)


def layer_map(ontology, root_id, schemes):
    """-> one row per structure under `root_id`: its layer token, the area it
    belongs to (the layer label's parent), and one `bin_<scheme>` column.

    Structures without a layer token (the root itself, area parents, a layer
    label the regex does not know) get no bin; cells on them are reported as
    `unassigned`, never dropped silently."""
    root = int(ontology.order_of_ids([root_id])[0])
    if root < 0:
        raise ValueError(f"root id {root_id} is not in the ontology")
    rows = []
    for o in ontology.descendant_orders([root]):
        tok = layer_token(ontology.names[o]) if o != root else None
        p = ontology.parent_order[o]
        rows.append({"order": o, "id": int(ontology.ids[o]), "name": ontology.names[o],
                     "acronym": ontology.acronyms[o],
                     "area": ontology.acronyms[p] if tok else "",
                     "token": tok, "leaf": not ontology.children[o]})
    df = pd.DataFrame(rows)

    # A layer label with children would be counted once itself and again through
    # them. CCF has none; fail loudly if an ontology ever does.
    bad = df[df["token"].notna() & ~df["leaf"]]
    if len(bad):
        raise ValueError(f"layer labels with children: {bad['name'].tolist()}")

    tokens = set(df["token"].dropna())
    if not tokens:
        raise ValueError(f"no layer labels under root id {root_id}")
    for scheme, bins in schemes.items():
        assigned = {}
        for b, toks in bins.items():
            for t in toks:
                t = TOKEN_ALIAS.get(str(t), str(t))
                if t in assigned:
                    raise ValueError(f"scheme '{scheme}': layer {t} is in both "
                                     f"'{assigned[t]}' and '{b}'")
                assigned[t] = b
        missing = tokens - set(assigned)
        if missing:
            raise ValueError(f"scheme '{scheme}' puts layer(s) {sorted(missing)} in no bin; "
                             "every layer the atlas uses must be placed")
        unknown = set(assigned) - tokens
        if unknown:
            raise ValueError(f"scheme '{scheme}' names layer(s) {sorted(unknown)}, "
                             f"which the atlas does not use under root {root_id}")
        df[f"bin_{scheme}"] = df["token"].map(assigned)
    return df


def read_sample(cfg, ontology, class_map, sample):
    """-> {class_name: DataFrame[order, <axis columns>]} for the cells of one
    animal that landed on a real structure. Read once per animal, because a
    slab has to be measured on the whole animal before any class is counted."""
    sdir = cfg["samples"][sample]["dir"]
    out = {}
    for cls in class_map:
        dirs = cell_tables.resolve_class_dirs(sdir, cls, class_map=class_map)
        if not dirs:
            print(f"  [warn] {sample}: class '{cls}' 没有对应的文件夹，按 0 计")
        parts = []
        for d in dirs:
            path = cell_tables.class_csv_path(sdir, d)
            if not os.path.exists(path):
                continue
            df = cell_tables.read_cell_registration(path)
            df = df[cell_tables.valid_region_mask(df)]
            keep = pd.DataFrame({"order": ontology.order_of_ids(
                df["region_id"].to_numpy(dtype=np.int64))})
            for ax in SLAB_AXES:
                keep[ax] = pd.to_numeric(df[ax], errors="coerce").to_numpy()
            parts.append(keep[keep["order"] >= 0])
        out[cls] = (pd.concat(parts, ignore_index=True) if parts
                    else pd.DataFrame(columns=["order", *SLAB_AXES]))
    return out


def _labels_in_sample_path(sample_dir):
    hits = sorted(f for f in os.listdir(sample_dir) if f.endswith("labels_in_sample.nii.gz"))
    if not hits:
        raise FileNotFoundError(f"{sample_dir} has no *_labels_in_sample.nii.gz; a volume "
                                "slab needs the atlas labels warped into the sample")
    return os.path.join(sample_dir, hits[0])


def cover_ids(ontology, names):
    """-> the structure ids of every `names` entry and its descendants.

    `names` are acronyms or ids: the structures the slab is asked to contain.
    They only position the slab; they are not the territory it is analysed on."""
    out = []
    for n in names:
        try:
            order = int(ontology.order_of_ids([int(n)])[0])
        except (TypeError, ValueError):
            hits = [k for k, a in enumerate(ontology.acronyms) if a == str(n)]
            if not hits:
                raise ValueError(f"slab cover '{n}' is not an acronym in the ontology")
            order = hits[0]
        if order < 0:
            raise ValueError(f"slab cover '{n}' is not in the ontology")
        out.extend(int(ontology.ids[d]) for d in ontology.descendant_orders([order]))
    return np.array(sorted(set(out)), dtype=np.int64)


def label_volume_profile(cfg, ontology, lmap, sample, axis, cover):
    """-> (root_profile, reference, cover_profile, um_per_voxel), all counted per
    plane of the sample's own 20 um grid.

    Read off <sample>_labels_in_sample.nii.gz, the atlas annotation warped into
    THIS animal, so every volume is the animal's own tissue, not the atlas's.

      root_profile[k, p]  voxels of lmap row k on plane p
      reference[p]        voxels of ANY structure on plane p -- the whole
                          cross-section, which is what the slab's volume
                          fraction is a fraction of
      cover_profile[p]    voxels of the cover structures on plane p
    """
    import nibabel as nib

    if axis not in SAMPLE_AXES:
        raise ValueError(f"a volume slab must be cut on a sample axis "
                         f"{SAMPLE_AXES}, not '{axis}' -- the volume is counted "
                         "on the sample's own grid")
    img = nib.load(_labels_in_sample_path(cfg["samples"][sample]["dir"]))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if not np.allclose(zooms, zooms[0]) or not np.isclose(zooms[0], ATLAS_UM):
        raise ValueError(f"{sample}: labels_in_sample is {zooms} um per voxel, "
                         f"expected isotropic {ATLAS_UM}")
    arr = np.asarray(img.dataobj).astype(np.int64)
    ax = SAMPLE_AXES.index(axis)
    other = tuple(k for k in range(3) if k != ax)

    reference = (arr > 0).sum(axis=other).astype(np.int64)
    cover_profile = np.isin(arr, cover).sum(axis=other).astype(np.int64)

    ids = lmap["id"].to_numpy(dtype=np.int64)
    srt = np.argsort(ids)
    ids_sorted = ids[srt]
    keep = np.isin(arr, ids_sorted)
    row = srt[np.searchsorted(ids_sorted, arr[keep])]
    plane = np.nonzero(keep)[ax]
    prof = np.zeros((len(lmap), arr.shape[ax]), dtype=np.int64)
    np.add.at(prof, (row, plane), 1)
    return prof, reference, cover_profile, float(zooms[0])


def place_volume_slab(reference, cover_profile, fraction=None, target=None):
    """-> (lo, hi, coverage) planes, inclusive, placed to contain as much of the
    cover structures as it can.

    Size is set either by `fraction` -- that share of `reference`'s total, so
    every animal gives up the same share of itself -- or by `target`, an
    absolute number of `reference` units, so every animal gives up the same
    amount. Raw cell counts are only comparable between animals under `target`,
    and then only if `reference` is the volume the cells are counted in: a slab
    holding the same amount of cross-section still holds different amounts of
    isocortex, because isocortex is 22.8-28.8% of these brains.

    Ties are broken towards the cover structures' own centre, because once a
    slab is thick enough to hold all of them every further shift scores the same
    and the search would otherwise drift to the front of the brain.

    The criterion reads only the warped atlas labels, never the cells, so it
    cannot tune the slab towards a result."""
    total = float(reference.sum())
    if total == 0:
        raise ValueError("the sample has no labelled voxels")
    if cover_profile.sum() == 0:
        raise ValueError("the cover structures have no voxels in this sample")
    if (fraction is None) == (target is None):
        raise ValueError("a volume slab needs exactly one of `fraction` or `volume_mm3`")
    want = fraction * total if fraction is not None else float(target)
    if want > total:
        raise ValueError(f"the slab asks for {want:.3g} but the whole reference is "
                         f"only {total:.3g}")
    cum = np.concatenate([[0.0], np.cumsum(reference.astype(float))])
    planes = np.arange(len(reference))
    centre = float((cover_profile * planes).sum() / cover_profile.sum())

    best = None
    for i in range(len(reference)):
        j = int(np.searchsorted(cum, cum[i] + want))
        if j >= len(reference):
            break
        got = float(cover_profile[i:j + 1].sum()) / float(cover_profile.sum())
        key = (round(got, 6), -abs((i + j) / 2.0 - centre))
        if best is None or key > best[0]:
            best = (key, i, j, got)
    if best is None:
        raise ValueError("no window of the requested size fits inside the volume")
    return best[1], best[2], best[3]


def place_length_slab(root_profile, cover_profile, fraction):
    """-> (lo, hi, coverage) planes, inclusive, whose THICKNESS is `fraction` of
    the root's own extent along the axis.

    Sizing on length rather than volume because a volume can be eaten by things
    that are not biology -- a compressed hemisphere, a torn piece, a slice the
    mask lost -- while the front-to-back extent of the isocortex survives all of
    those: it is set by the two ends of the structure, and a hole in the middle
    does not move them. The cost is that the animals then contribute different
    amounts of tissue, so it trades a volume artefact for a volume difference.
    Run it beside the volume sizing rather than instead of it."""
    present = np.flatnonzero(root_profile > 0)
    if present.size == 0:
        raise ValueError("the root has no voxels in this sample")
    extent = int(present[-1] - present[0] + 1)
    width = int(round(fraction * extent))
    if width < 1:
        raise ValueError(f"fraction {fraction} of {extent} planes rounds to nothing")
    if cover_profile.sum() == 0:
        raise ValueError("the cover structures have no voxels in this sample")
    planes = np.arange(len(root_profile))
    centre = float((cover_profile * planes).sum() / cover_profile.sum())
    cum = np.concatenate([[0.0], np.cumsum(cover_profile.astype(float))])
    total = float(cover_profile.sum())

    best = None
    for i in range(0, len(root_profile) - width + 1):
        j = i + width - 1
        got = float(cum[j + 1] - cum[i]) / total
        key = (round(got, 6), -abs((i + j) / 2.0 - centre))
        if best is None or key > best[0]:
            best = (key, i, j, got)
    return best[1], best[2], best[3]


def slab_bounds(per_class, in_root, slab):
    """-> (lo, hi) on the slab axis for one animal, plus what they were read off.

    `mode: quantile` puts the slab on the quantiles of THIS animal's own cells
    inside the root, so an animal whose brain sits a few hundred um off along
    the axis still gets the same piece of it. That is also the cost: the cut is
    made on the data, so a pure shift along the axis is absorbed into the slab
    instead of showing up in the result. `mode: absolute` takes lo/hi as atlas
    voxel indices, identical for every animal."""
    ax = slab.get("axis", "yt")
    if ax not in SLAB_AXES:
        raise ValueError(f"slab axis '{ax}' must be one of {SLAB_AXES}")
    mode = str(slab.get("mode", "quantile")).lower()
    if mode not in ("quantile", "absolute"):
        raise ValueError(f"slab mode '{mode}' must be 'quantile', 'absolute' or 'volume'")
    lo, hi = float(slab["lo"]), float(slab["hi"])
    if hi <= lo:
        raise ValueError(f"slab needs lo < hi, got {lo} .. {hi}")
    if mode == "absolute":
        return ax, lo, hi, np.nan
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"quantile slab needs 0 <= lo < hi <= 1, got {lo} .. {hi}")
    v = np.concatenate([d.loc[in_root[d["order"].to_numpy()], ax].to_numpy()
                        for d in per_class.values() if len(d)])
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"no cell inside the root carries a finite '{ax}'")
    return ax, float(np.quantile(v, lo)), float(np.quantile(v, hi)), float(v.size)


def count_cells(cfg, ontology, lmap, class_map, slab=None):
    """-> (long [group, sample, class_name, order, count], slab QC, slab volumes).

    `long` covers every structure under the root that holds at least one cell,
    counting only cells inside the slab when one is configured. A `volume` slab
    also returns the volume each structure occupies inside it, measured on the
    animal's own warped labels, which is what a Density readout divides by."""
    in_root = np.zeros(ontology.n, dtype=bool)
    in_root[lmap["order"].to_numpy()] = True
    is_volume = bool(slab) and str(slab.get("mode", "")).lower() == "volume"
    cids = cover_ids(ontology, (slab or {}).get("cover") or []) if is_volume else None
    if is_volume and not len(cids):
        raise ValueError("a volume slab needs `cover`: the structures that decide "
                         "where along the axis it sits")
    frames, qc, vols = [], [], []
    for key in ("a", "b"):
        for s in cfg["groups"][key]["samples"]:
            per_class = read_sample(cfg, ontology, class_map, s)
            ax, lo, hi, worst = "", np.nan, np.nan, np.nan
            vol_mm3 = vol_frac = root_in_slab = sized_mm3 = root_extent_um = np.nan
            sized_on = ""
            if is_volume:
                ax = slab.get("axis", "yr")
                prof, ref, cov, um = label_volume_profile(cfg, ontology, lmap, s, ax, cids)
                mm3 = (um ** 3) / 1e9
                # `reference: root` sizes the slab on the territory the cells are
                # counted in, which is what a raw count comparison needs; the
                # default sizes it on the whole cross-section.
                sized_on = str(slab.get("reference", "sample")).lower()
                if sized_on == "root":
                    ref_prof = prof.sum(axis=0)
                elif sized_on == "sample":
                    ref_prof = ref
                else:
                    raise ValueError(f"slab reference '{sized_on}' must be 'sample' or 'root'")
                if str(slab.get("size_by", "volume")).lower() == "length":
                    if sized_on != "root":
                        raise ValueError("size_by: length measures the root's own extent, "
                                         "so it needs reference: root")
                    lo, hi, worst = place_length_slab(ref_prof, cov, float(slab["fraction"]))
                else:
                    target = slab.get("volume_mm3")
                    lo, hi, worst = place_volume_slab(
                        ref_prof, cov,
                        fraction=(None if target is not None else float(slab["fraction"])),
                        target=(None if target is None else float(target) / mm3))
                per_order = prof[:, lo:hi + 1].sum(axis=1) * mm3
                vols.append(pd.DataFrame({"sample": s, "order": lmap["order"].to_numpy(),
                                          "volume_mm3": per_order,
                                          "root_volume_mm3": prof.sum() * mm3}))
                vol_mm3 = float(ref[lo:hi + 1].sum()) * mm3
                vol_frac = vol_mm3 / (float(ref.sum()) * mm3)
                root_in_slab = float(per_order.sum())
                sized_mm3 = float(ref_prof[lo:hi + 1].sum()) * mm3
                present = np.flatnonzero(prof.sum(axis=0) > 0)
                root_extent_um = float(present[-1] - present[0] + 1) * ATLAS_UM
            elif slab:
                ax, lo, hi, _ = slab_bounds(per_class, in_root, slab)
            n_root = n_kept = 0
            for cls, d in per_class.items():
                orders = d["order"].to_numpy()
                keep = in_root[orders]
                n_root += int(keep.sum())
                if slab:
                    v = d[ax].to_numpy()
                    keep = keep & np.isfinite(v) & (v >= lo) & (v <= hi)
                n_kept += int(keep.sum())
                bins = np.bincount(orders[keep], minlength=ontology.n).astype(float)
                nz = np.flatnonzero(bins)
                frames.append(pd.DataFrame({"group": key, "sample": s, "class_name": cls,
                                            "order": nz, "count": bins[nz]}))
            qc.append({"sample": s, "group": key,
                       "axis": ax, "lo": lo, "hi": hi,
                       "thickness_um": (hi - lo + 1) * ATLAS_UM if is_volume
                       else ((hi - lo) * ATLAS_UM if slab else np.nan),
                       "slab_volume_mm3": vol_mm3,
                       "sample_volume_mm3": (vol_mm3 / vol_frac) if vol_frac == vol_frac
                       else np.nan,
                       "volume_frac_of_sample": vol_frac,
                       "cover_coverage": worst,
                       "sized_on": sized_on,
                       "sized_by": str(slab.get("size_by", "volume")).lower(),
                       "sized_volume_mm3": sized_mm3,
                       "root_extent_um": root_extent_um,
                       "thickness_frac_of_root_extent":
                           ((hi - lo + 1) * ATLAS_UM / root_extent_um)
                           if root_extent_um == root_extent_um else np.nan,
                       "root_volume_in_slab_mm3": root_in_slab,
                       "cells_in_root": n_root, "cells_in_slab": n_kept,
                       "frac_kept": (n_kept / n_root) if n_root else np.nan})
    vol = pd.concat(vols, ignore_index=True) if vols else None
    return pd.concat(frames, ignore_index=True), pd.DataFrame(qc), vol


def share_readouts_from_config(cfg):
    """Every base class plus every all-'+' combined category, INCLUDING the one
    that is all cells: its laminar share is the most basic readout there is."""
    base = list(cfg["class_map"].keys())
    out = {c: [c] for c in base}
    for name, terms in (cfg.get("combined_categories") or {}).items():
        parts = [t["class"] if isinstance(t, dict) else t for t in terms]
        if all((t.get("sign", "+") if isinstance(t, dict) else "+") == "+" for t in terms):
            out[name] = parts
    return base, out


def sample_values(long, lmap, scheme, bin_names, share_readouts, comp_readouts, base,
                  samples, area_min_cells, vol=None):
    """-> tidy [scheme, pooling, metric, readout, bin, <one column per sample>].

    With `vol` (a volume slab's per-structure mm3) every share readout also gets
    a Density: cells of that class in the bin per mm3 of that bin INSIDE THE
    SLAB, in the animal's own tissue. That is the readout a fixed-volume slab
    exists for -- LaminarShare cannot see a change that moves every layer the
    same way, and a raw Count carries the animal's brain size."""
    col = f"bin_{scheme}"
    d = long.merge(lmap[["order", "area", col]], on="order", how="inner")
    d = d[d[col].notna()].rename(columns={col: "bin"})
    areas = sorted(lmap.loc[lmap[col].notna(), "area"].unique())
    full = pd.MultiIndex.from_product([samples, areas], names=["sample", "area"])
    present = (lmap[lmap[col].notna()].groupby(["area", col]).size().unstack(col)
               .reindex(index=areas, columns=bin_names).notna())

    def agg(parts):
        return (d[d["class_name"].isin(parts)]
                .groupby(["sample", "area", "bin"])["count"].sum()
                .unstack("bin").reindex(index=full, columns=bin_names).fillna(0.0))

    rows = []

    def emit(metric, pooling, readout, wide):
        wide = wide.reindex(index=samples, columns=bin_names)
        for b in bin_names:
            rows.append({"scheme": scheme, "pooling": pooling, "metric": metric,
                         "readout": readout, "bin": b,
                         **{s: float(wide.loc[s, b]) for s in samples}})

    def area_mean(num, den, keep_area):
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = num / den.replace(0, np.nan)
        mask = present.reindex(frac.index.get_level_values("area")).to_numpy()
        frac = frac.where(mask)
        frac = frac[frac.index.get_level_values("area").isin(keep_area)]
        return frac.groupby(level="sample").mean()

    vol_wide = None
    if vol is not None:
        v = vol.merge(lmap[["order", "area", col]], on="order", how="inner")
        v = v[v[col].notna()].rename(columns={col: "bin"})
        vol_wide = (v.groupby(["sample", "bin"])["volume_mm3"].sum()
                    .unstack("bin").reindex(index=samples, columns=bin_names).fillna(0.0))

    for r, parts in share_readouts.items():
        c = agg(parts)
        pooled = c.groupby(level="sample").sum()
        emit("Count", "pooled", r, pooled)
        if vol_wide is not None:
            emit("Density", "pooled", r, pooled.div(vol_wide.replace(0, np.nan)))
        emit("LaminarShare", "pooled", r,
             pooled.div(pooled.sum(axis=1).replace(0, np.nan), axis=0))
        area_tot = c.sum(axis=1).unstack("area")
        keep = area_tot.columns[(area_tot >= area_min_cells).all(axis=0)]
        emit("LaminarShare", "area_mean", r,
             area_mean(c, pd.concat([c.sum(axis=1)] * len(bin_names), axis=1,
                                    keys=bin_names), keep))

    total = agg(base)
    for r, parts in comp_readouts.items():
        c = agg(parts)
        pooled_t = total.groupby(level="sample").sum()
        emit("BinComposition", "pooled", r,
             c.groupby(level="sample").sum() / pooled_t.replace(0, np.nan))
        tot_area = total.sum(axis=1).unstack("area")
        keep = tot_area.columns[(tot_area >= area_min_cells).all(axis=0)]
        emit("BinComposition", "area_mean", r, area_mean(c, total, keep))
    return pd.DataFrame(rows)


def run_tests(values, samples_a, samples_b, test, correction, alpha, primary_scheme,
              primary_metric="LaminarShare"):
    """One correction family per (scheme, pooling, metric, readout): the bins of
    one readout. Families are small on purpose -- that is the point of pooling."""
    keys = ["scheme", "pooling", "metric", "readout"]
    samples = list(samples_a) + list(samples_b)
    out = []
    for _, sub in values.groupby(keys, sort=False):
        sub = sub.reset_index(drop=True)
        res = _test_rows(sub[samples_a].to_numpy(float), sub[samples_b].to_numpy(float),
                         sub[keys + ["bin"]].to_dict("records"), test, correction, alpha)
        extra = []
        for _, r in sub.iterrows():
            rank, of, p_perm, floor = _permutation_rank(r[samples_a].to_numpy(float),
                                                        r[samples_b].to_numpy(float))
            # A row whose group means are equal has log2fc 0, so every
            # leave-one-out fraction is 0/0; that NaN is the answer, not news.
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                worst, keeps = _loo(r, samples_a, samples_b)
            row = {"perm_rank": rank, "perm_of": of, "p_perm": p_perm, "p_perm_floor": floor,
                   "loo_worst_frac": worst, "loo_keeps_sign": keeps}
            for s in samples:
                ma = np.nanmean([r[x] for x in samples_a if x != s])
                mb = np.nanmean([r[x] for x in samples_b if x != s])
                row[f"log2fc_without_{s}"] = (float(np.log2(mb / ma))
                                              if ma > 0 and mb > 0 else np.nan)
            extra.append(row)
        out.append(pd.concat([res, pd.DataFrame(extra), sub[samples]], axis=1))
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "role", np.select(
        [(df["scheme"] == primary_scheme) & (df["pooling"] == "pooled")
         & (df["metric"] == primary_metric),
         df["metric"] == "Count"] if primary_metric != "Count" else
        [(df["scheme"] == primary_scheme) & (df["pooling"] == "pooled")
         & (df["metric"] == "Count"),
         df["metric"] == "Count"],
        ["primary", "reference"], "secondary"))
    return samples_beside_means(df, samples)


def token_qc(long, lmap, samples):
    """Share of each animal's labelled cells on every ORIGINAL layer label, plus
    the unassigned ones. This is where a registration problem in a thin layer
    shows up (s10's L1), before any bin hides it."""
    d = long.merge(lmap[["order", "token"]], on="order", how="left")
    d["token"] = d["token"].fillna(UNASSIGNED)
    t = d.groupby(["token", "sample"])["count"].sum().unstack("sample").reindex(columns=samples)
    t = t.fillna(0.0)
    return t / t.sum(axis=0)


def run(cfg, out_dir=None):
    lam = cfg.get("laminar") or {}
    root_id = int(lam.get("root_id", ISOCORTEX_ID))
    schemes = lam.get("schemes") or DEFAULT_SCHEMES
    primary = lam.get("primary_scheme", next(iter(schemes)))
    if primary not in schemes:
        raise ValueError(f"primary_scheme '{primary}' is not one of {list(schemes)}")
    area_min = int(lam.get("area_min_cells", 50))
    st = cfg.get("stats") or {}
    test = str(st.get("test", "welch")).lower()
    correction = str(st.get("correction", "bh")).lower()
    alpha = float(st.get("alpha", 0.05))

    if not cfg.get("class_map"):
        raise ValueError("laminar 需要 class_map：比例的分母是它的基础类之和")
    class_map = cell_tables.normalize_class_map(cfg["class_map"])
    samples_a = list(cfg["groups"]["a"]["samples"])
    samples_b = list(cfg["groups"]["b"]["samples"])
    samples = samples_a + samples_b

    ontology = Ontology.from_json(cfg["ontology_json"])
    slab = lam.get("slab")
    lmap = layer_map(ontology, root_id, schemes)
    long, slab_qc, slab_vol = count_cells(cfg, ontology, lmap, class_map, slab)

    base, share_ro = share_readouts_from_config(cfg)
    _, comp = readouts_from_config(cfg)
    comp_ro = {re.sub(r"_fraction$", "", k): v for k, v in comp.items()}

    values = pd.concat([
        sample_values(long, lmap, scheme, list(bins), share_ro, comp_ro, base, samples,
                      area_min, slab_vol)
        for scheme, bins in schemes.items()], ignore_index=True)
    primary_metric = str(st.get("primary_metric")
                         or ("Density" if slab_vol is not None else "LaminarShare"))
    if primary_metric not in ("LaminarShare", "BinComposition", "Density", "Count"):
        raise ValueError(f"stats.primary_metric '{primary_metric}' is not a metric this "
                         "run produces")
    tests = run_tests(values, samples_a, samples_b, test, correction, alpha, primary,
                      primary_metric)
    qc = token_qc(long, lmap, samples)

    out_dir = out_dir or (cfg.get("output") or {}).get("dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        lab_counts = (long.groupby(["order", "sample"])["count"].sum()
                      .unstack("sample").reindex(columns=samples).fillna(0).reset_index())
        layer_table = lmap.merge(lab_counts, on="order", how="left").fillna(
            {s: 0 for s in samples})
        tables = {
            "laminar_tests.csv": tests,
            "laminar_per_sample.csv": values,
            "laminar_counts_long.csv": long.merge(
                lmap[["order", "name", "area", "token"]], on="order", how="left"),
            "laminar_layer_map.csv": layer_table,
            "laminar_token_qc.csv": qc.reset_index(),
            "laminar_slab_qc.csv": slab_qc,
        }
        if slab_vol is not None:
            tables["laminar_slab_volume.csv"] = slab_vol.merge(
                lmap[["order", "name", "area", "token"]], on="order", how="left")
        for name, frame in tables.items():
            _assert_ascii_text(frame, name)
            frame.to_csv(os.path.join(out_dir, name), index=False)
        save_config_copy(cfg, out_dir)
    return {"layer_map": lmap, "long": long, "values": values, "tests": tests, "token_qc": qc,
            "slab_qc": slab_qc, "slab": slab, "primary_metric": primary_metric,
            "samples_a": samples_a, "samples_b": samples_b, "primary": primary,
            "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="覆盖 config 里的 output.dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = run(cfg, args.out_dir)
    a, b = r["samples_a"], r["samples_b"]
    groups = cfg["groups"]
    print(f"{groups['a'].get('name', 'A')} {', '.join(a)}   "
          f"{groups['b'].get('name', 'B')} {', '.join(b)}\n")

    if r["slab"]:
        print("切片板（只有板内的细胞进入统计）。lo/hi 是图谱体素，thickness 是 um：")
        print(r["slab_qc"].to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
        print()

    print("原始层标签上的细胞占比（所有标记细胞）。薄层的配准问题在这里看，分箱之后就看不见了：")
    print(r["token_qc"].to_string(float_format=lambda v: f"{v:6.3f}"))

    t = r["tests"]
    show = t[t["role"] == "primary"][["readout", "bin", "mean_a", "mean_b", *a, *b,
                                      "log2fc", "hedges_g", "p_value", "p_adj",
                                      "perm_rank", "loo_keeps_sign"]]
    if r["primary_metric"] == "Count":
        print(f"\n主分析：{r['primary']}，pooled Count"
              f"（板内每个箱的细胞数；板按每只自己的 root 体积等比例切，所以这是"
              f"“同一份额的皮层里有多少细胞”）")
    elif r["primary_metric"] == "Density":
        print(f"\n主分析：{r['primary']}，pooled Density"
              f"（板内每个箱的细胞数 / 该箱在这只动物板内的实测体积，单位 cells/mm3）")
    else:
        print(f"\n主分析：{r['primary']}，pooled LaminarShare"
              f"（每类细胞在各箱的占比，同一类的几个箱加起来是 1，要一起读）")
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print("\nperm_rank 是 10 种分法里的排名，1 最强；置换 p 的下限是 0.10，不是显著性判定")
    if r["out_dir"]:
        print(f"写出 -> {r['out_dir']}")


if __name__ == "__main__":
    main()
