"""Read the pipeline's per-class cell_registration.csv files and bin cells
into ontology regions.

Two differences from the ClearMap-era reader this replaces:

* Column 9 of an ANTs-pipeline cell_registration.csv holds the RAW CCF
  structure id (see registration_ants.cell_points -- "mapped_id  raw atlas
  annotation label id (not a ClearMap graph_order)"). The old reader treated
  that column as ClearMap's `graph_order`, which for these files would map
  cells to the wrong regions without erroring. Ids go through
  Ontology.order_of_ids here instead.

* Cell classes can be collapsed by marker signature (`classify_by="marker"`),
  dropping the YOLO neuron/glia call. See marker_signature().
"""
import os
import re

import numpy as np
import pandas as pd

# Written by cell_points.assign_cell_regions, header-less:
COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "xt", "yt", "zt",
           "region_id", "region_name", "slice_name", "tile_name", "score"]

# A cell that landed outside the atlas after warping. Both spellings occur:
# "background" from the annotation's own id 0, "no label" from an id with no
# ontology entry.
BACKGROUND_NAMES = {"background", "no label", "nolabel", "none", "nan", ""}

# YOLO's morphology call, which marker-mode analysis discards.
SOMA_TYPES = ("neuron", "glia")


def read_cell_registration(csv_path):
    """-> DataFrame with int64 `region_id` and cleaned `region_name`.

    Over-allocates column names because rows can be ragged (a region name
    containing a comma is quoted, but older files predate the provenance
    columns entirely)."""
    df = pd.read_csv(csv_path, header=None, names=range(20), engine="python")
    for i, name in enumerate(COLUMNS):
        if i in df.columns:
            df = df.rename(columns={i: name})
    region_id = pd.to_numeric(df.get("region_id"), errors="coerce")
    df["region_id"] = region_id.fillna(-1).astype(np.int64)
    df["region_name"] = df.get("region_name").map(_clean_name)
    return df


def _clean_name(raw):
    """Strip numpy's byte-repr wrapper, e.g. b'Cerebral cortex' -> Cerebral cortex."""
    s = str(raw).strip()
    s = re.sub(r"^b['\"]", "", s)
    s = re.sub(r"['\"]$", "", s)
    return s.strip()


def valid_region_ids(df):
    """Structure ids of cells that actually landed inside the atlas.

    Filters on BOTH the name and the id: a cell outside the atlas is written
    with region_id 0 and name "background", and id 0 is not a real CCF
    structure, so either test alone would do -- but a file hand-edited by
    scripts/relabel_cells.py can carry an id whose name lookup failed, and
    those must not be counted as if they were assigned."""
    name = df["region_name"].astype(str).str.strip().str.lower()
    ok = (~name.isin(BACKGROUND_NAMES)) & (df["region_id"] > 0)
    return df.loc[ok, "region_id"].to_numpy(dtype=np.int64)


def class_counts(csv_path, ontology, exclude_mask=None):
    """-> (rollup, direct, n_valid, n_total, n_unknown_id, n_excluded).

    rollup[order] = cells in that region plus every descendant.
    direct[order] = cells whose assignment stopped exactly at that region.
    n_total - n_valid - n_excluded is the sample's out-of-atlas ("background")
    count. n_unknown_id counts cells whose id is not in the ontology at all --
    0 for a healthy run; anything else means the annotation and the ontology
    JSON disagree and the numbers below it are incomplete.

    exclude_mask: boolean array over orders marking regions to drop from the
    analysis entirely (structures damaged during preparation, say). It is
    applied to `direct` BEFORE the rollup, which is the whole point: zeroing a
    subtree's direct counts makes every ancestor's rollup -- up to and
    including the root -- exclude it too. So n_valid, and therefore the
    Percentage denominator, becomes "cells in the regions under analysis"
    rather than "cells anywhere in the brain". Excluding after the rollup
    would leave every ancestor still carrying the excluded cells."""
    zero = np.zeros(ontology.n, dtype=float)
    if not os.path.exists(csv_path):
        return zero, zero.copy(), 0, 0, 0, 0
    try:
        df = read_cell_registration(csv_path)
    except pd.errors.EmptyDataError:
        return zero, zero.copy(), 0, 0, 0, 0

    ids = valid_region_ids(df)
    if len(ids) == 0:
        return zero, zero.copy(), 0, len(df), 0, 0

    direct, unknown = ontology.bincount_ids(ids)
    n_excluded = 0
    if exclude_mask is not None and exclude_mask.any():
        n_excluded = int(direct[exclude_mask].sum())
        direct = direct.copy()
        direct[exclude_mask] = 0.0
    n_valid = len(ids) - unknown - n_excluded
    return ontology.rollup(direct), direct, n_valid, len(df), unknown, n_excluded


# ================= class-name handling =================

def normalize_class_key(name):
    """Token-set key for fuzzy class matching: lowercase, split on
    non-alphanumerics, drop pure-numeric tokens. That last part is what makes
    'glia_3_GFP' (s11/s12q/s12t) and 'glia_GFP' (s8/s10/s18) resolve to the
    same class.

    Where that '3' comes from: those samples' GFP tile folder was named
    'GFP_3', and brain_detector's stitcher._merge_class builds a class label
    by splitting on '_' and sorting the union of the marker tokens
    (markers = sorted(markers_a | markers_b)), so the '3' became a marker of
    its own and sorted ahead of GFP. It is a phantom -- it appears on exactly
    the GFP-positive classes and nowhere else (RFP and RFP_Sox9 carry no '3'),
    duplicates GFP's count in the detector's own single-marker table, and adds
    no cells. The two naming conventions are therefore strictly 1:1 once
    numeric tokens are dropped, and the counts underneath are unaffected.

    See check_class_resolution() for the guard that catches the case this rule
    would get wrong -- two genuinely different on-disk classes collapsing into
    one label."""
    tokens = re.split(r"[^a-zA-Z0-9]+", name)
    return frozenset(t.lower() for t in tokens if t and not t.isdigit())


def marker_signature(class_name):
    """Drop the YOLO soma-type prefix and any pure-numeric token, keeping the
    marker tokens in the detector's own order:

        neuron_3_GFP_RFP_Sox9 -> GFP_RFP_Sox9
        glia_RFP              -> RFP

    brain_detector writes one row per physical cell with one composite label
    ({soma_type}_{soma channels}_{TF}, markers joined in sorted order), so the
    marker signatures are mutually exclusive and collapsing the soma type is a
    plain sum -- no cell is counted twice. Verified empirically: across all
    12 class files of one sample, fewer than 0.3% of cells in any pair of
    files have a counterpart within 5 um in another file."""
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", class_name) if t]
    kept = [t for t in tokens if not t.isdigit() and t.lower() not in SOMA_TYPES]
    return "_".join(kept) if kept else class_name


def soma_type(class_name):
    """'neuron' / 'glia' / None -- kept only for the QC report, which reports
    what fraction of cells YOLO called glia per sample."""
    for t in re.split(r"[^a-zA-Z0-9]+", class_name):
        if t.lower() in SOMA_TYPES:
            return t.lower()
    return None


def list_sample_classes(sample_dir):
    reg_dir = os.path.join(sample_dir, "cell_registration")
    if not os.path.isdir(reg_dir):
        return []
    return sorted(d for d in os.listdir(reg_dir)
                  if os.path.isdir(os.path.join(reg_dir, d)))


def class_csv_path(sample_dir, class_dir):
    return os.path.join(sample_dir, "cell_registration", class_dir, "cell_registration.csv")


def resolve_class_dirs(sample_dir, class_label, classify_by="marker"):
    """Every on-disk class folder in this sample that belongs to `class_label`.

    In marker mode a label like 'GFP_RFP' resolves to BOTH neuron_*_GFP_RFP
    and glia_*_GFP_RFP, and their counts get summed -- that summation is the
    whole point of marker mode, so it happens here rather than being left to
    the caller. In full mode at most one folder matches."""
    actual = list_sample_classes(sample_dir)
    if classify_by == "marker":
        key = normalize_class_key(marker_signature(class_label))
        return [d for d in actual if normalize_class_key(marker_signature(d)) == key]
    if class_label in actual:
        return [class_label]
    key = normalize_class_key(class_label)
    return [d for d in actual if normalize_class_key(d) == key]


def check_class_resolution(sample_dirs, classes, classify_by="marker"):
    """Guard the fuzzy class matching. Returns a list of human-readable
    problems, empty when everything resolves cleanly.

    Two things can go wrong, and both would silently corrupt the counts:

    1. Two distinct on-disk folders in ONE sample collapsing into the same
       class label. Dropping numeric tokens is safe only as long as the token
       really is noise; if a sample ever held both 'neuron_GFP' and
       'neuron_3_GFP' as separate populations, summing them here would double
       count. (Marker mode legitimately merges one neuron_* with one glia_*
       folder, so the check is per (class, soma type), not per class.)
    2. The same class resolving to a different number of folders in different
       samples -- one sample contributing neuron+glia and another only neuron
       makes the group comparison an apples-to-oranges one.
    """
    problems = []
    per_sample = {}
    for sample, sdir in sample_dirs.items():
        for cls in classes:
            dirs = resolve_class_dirs(sdir, cls, classify_by)
            per_sample[(sample, cls)] = dirs
            buckets = {}
            for d in dirs:
                buckets.setdefault(soma_type(d), []).append(d)
            for st, group in buckets.items():
                if len(group) > 1:
                    problems.append(
                        f"{sample}: class '{cls}' (soma type {st}) matches {len(group)} "
                        f"folders {sorted(group)} -- these would be summed as if they were "
                        "one population; check whether the numeric-token rule is collapsing "
                        "two genuinely different classes")
    for cls in classes:
        counts = {s: len(per_sample[(s, cls)]) for s in sample_dirs}
        if len(set(counts.values())) > 1:
            problems.append(
                f"class '{cls}' resolves to a different number of folders per sample: "
                f"{counts} -- the group comparison would not be like for like")
    return problems


def discover_classes(sample_dirs, classify_by="marker", explicit=None):
    """Canonical class labels across samples. Folder names are grouped by
    normalize_class_key so naming variants collapse into one label; the
    shortest name in each group becomes the label."""
    if explicit:
        return list(explicit)
    names = sorted({n for d in sample_dirs for n in list_sample_classes(d)})
    if classify_by == "marker":
        names = sorted({marker_signature(n) for n in names})
    groups = {}
    for name in names:
        groups.setdefault(normalize_class_key(name), []).append(name)
    return sorted(min(g, key=lambda n: (len(n), n)) for g in groups.values())
