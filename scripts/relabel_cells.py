"""Non-interactive helper: rewrite the mapped_id / region columns of every
ClearMap-format `cell_registration.csv` under a sample's cell_registration/
folder, using a hand-corrected label volume from
scripts/edit_sample_labels.py instead of the original (uncorrected)
labels_in_sample.nii.gz.

This is the last step of the post-registration correction workflow: run
registration once -> scripts/edit_sample_labels.py to hand-fix wrong
region boundaries in sample space -> this script, to push that correction
into the per-cell region assignments ClearMap already computed.

Note the scope of what this fixes: cell_registration.csv's "resample space"
columns (3-5) are voxel indices on the exact same grid as
labels_in_sample.nii.gz, with no rescaling -- so relabeling a cell is a
direct index lookup into the corrected volume, no coordinate transform
involved. This corrects *region assignment* only (which structure a cell
counts under). It does NOT touch the "atlas space" columns (6-8) or
recompute anything about the underlying SyN deformation field -- those still
reflect the original (uncorrected) registration, since hand-editing
labels_in_sample doesn't retroactively change the transform itself.

Usage:
    conda activate antsreg
    python scripts/relabel_cells.py \\
        <labels_in_sample_corrected.nii.gz> <sample_dir> \\
        --ontology-json <ontology.json>
    # or: --atlas-source brainglobe --atlas-res-um 25

<sample_dir>: the directory containing cell_registration/<class_name>/cell_registration.csv
for every detected-cell class (same layout single_sample.py reads).

By default each CSV is backed up to cell_registration.csv.bak (skipped if a
.bak already exists, so re-runs never clobber the true original) before being
overwritten in place -- pass --no-backup to skip this.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils  # noqa: E402


def _load_structures(ontology_json, atlas_source, atlas_res_um):
    if ontology_json:
        return atlas_utils.load_ccf_ontology_json(ontology_json)
    if atlas_source == "brainglobe":
        _, _, structures = atlas_utils.get_allen_atlas(atlas_res_um)
        return structures
    raise ValueError("Pass either --ontology-json or --atlas-source brainglobe --atlas-res-um <res>")


def relabel_cell_csv(csv_path, corrected_labels, id_to_name, backup=True):
    """Rewrite the mapped_id (col 9) / region (col 10) columns of one
    cell_registration.csv in place, using corrected_labels (a 3D array,
    axis order (z, y, x) as read via sitk.GetArrayFromImage) looked up at
    each cell's "resample space" voxel index (cols 3-5, stored as x, y, z).

    Returns the number of cells whose mapped_id actually changed. Rows with
    an out-of-bounds or missing resample-space coordinate are left untouched
    (not dropped, not crashed on).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 11:
        raise ValueError(f"{csv_path}: expected at least 11 columns (0-10), got {df.shape[1]}")
    if len(df) == 0:
        return 0

    coords = df.iloc[:, 3:6].to_numpy(dtype=float)  # (x, y, z) per row
    valid = ~np.isnan(coords).any(axis=1)
    ix = np.full(len(df), -1, dtype=np.int64)
    iy = np.full(len(df), -1, dtype=np.int64)
    iz = np.full(len(df), -1, dtype=np.int64)
    ix[valid] = np.round(coords[valid, 0]).astype(np.int64)
    iy[valid] = np.round(coords[valid, 1]).astype(np.int64)
    iz[valid] = np.round(coords[valid, 2]).astype(np.int64)

    zdim, ydim, xdim = corrected_labels.shape
    in_bounds = valid & (ix >= 0) & (ix < xdim) & (iy >= 0) & (iy < ydim) & (iz >= 0) & (iz < zdim)
    n_out_of_bounds = int((~in_bounds).sum())
    if n_out_of_bounds:
        print(f"  WARNING: {csv_path}: {n_out_of_bounds} row(s) with missing/out-of-bounds "
              f"resample-space coords left unchanged")

    new_ids = corrected_labels[iz[in_bounds], iy[in_bounds], ix[in_bounds]]
    old_ids = pd.to_numeric(df.iloc[:, 9], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    n_changed = int(np.sum(old_ids[in_bounds] != new_ids))

    if backup:
        bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not bak_path.exists():
            shutil.copy2(csv_path, bak_path)

    df.loc[in_bounds, 9] = new_ids
    df.loc[in_bounds, 10] = [id_to_name.get(int(i), f"Region {int(i)}") for i in new_ids]
    df.to_csv(csv_path, header=False, index=False)
    return n_changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corrected_labels_path")
    parser.add_argument("sample_dir")
    parser.add_argument("--ontology-json", default=None)
    parser.add_argument("--atlas-source", choices=["brainglobe"], default=None)
    parser.add_argument("--atlas-res-um", type=float, default=25)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    structures = _load_structures(args.ontology_json, args.atlas_source, args.atlas_res_um)
    id_to_name = {sid: info["name"] for sid, info in structures.items()}

    corrected_labels = sitk.GetArrayFromImage(sitk.ReadImage(args.corrected_labels_path)).astype(np.int64)

    cell_reg_dir = Path(args.sample_dir) / "cell_registration"
    if not cell_reg_dir.exists():
        print(f"No cell_registration/ folder found under {args.sample_dir}")
        sys.exit(1)

    total_changed = 0
    n_files = 0
    for class_dir in sorted(cell_reg_dir.iterdir()):
        csv_path = class_dir / "cell_registration.csv"
        if not class_dir.is_dir() or not csv_path.exists():
            continue
        n_changed = relabel_cell_csv(csv_path, corrected_labels, id_to_name, backup=not args.no_backup)
        print(f"{class_dir.name}: {n_changed} cell(s) relabeled -> {csv_path}")
        total_changed += n_changed
        n_files += 1

    print(f"\nDone. {n_files} file(s) processed, {total_changed} cell(s) relabeled in total.")


if __name__ == "__main__":
    main()
