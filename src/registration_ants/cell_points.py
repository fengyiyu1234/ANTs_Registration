"""Read ClearMap-format cell-centroid CSVs (cell_centroids/<prefix><class>.csv,
in the ORIGINAL full-resolution pixel coordinates cells were detected at) and
assign each cell a brain region by transforming its coordinates through the
registration this pipeline already computed -- the ANTs-side equivalent of
cellMap.py's process_cell_class().

Produces files in the exact same cell_registration/<class>/cell_registration.csv
layout ClearMap's cellMap.py writes (header-less, columns below), so existing
tooling keeps working unchanged: scripts/relabel_cells.py
(reads/rewrites columns 3-10), and the napari viewers in
../ClearMap/stats_vis/ (single_sample.py, stats_img_vis_ui.py) which already
detect and read either pipeline's output.

Column layout (0-indexed, no header):
  0-2   x,y,z   raw pixel-space centroid, exactly as given in the source CSV
  3-5   xr,yr,zr  voxel index in the UNCROPPED fine/isotropic sample image --
                  same grid as the pipeline's <name>_fine_<target>um.nii.gz,
                  regardless of whether registration.crop_for_registration
                  cropped a sub-region of it for the actual registration
  6-8   xt,yt,zt  voxel index in the atlas annotation's own native grid
  9     mapped_id   raw atlas annotation label id (not a ClearMap graph_order)
  10    name        region name, via the same atlas_structures dict used
                    elsewhere in this pipeline (get_allen_atlas /
                    load_ccf_ontology_json)
  11-13 slice_name, tile_name, score  passthrough provenance from the source
                    cell_centroids CSV (NaN-filled if that CSV predates them)

No .npy structured-array output (unlike cellMap.py) -- nothing on the ANTs
side reads it; the CSV alone is the established interchange format here.
"""
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import transforms

logger = logging.getLogger(__name__)


def read_centroid_csv(csv_path):
    """Read one cell_centroids/<prefix><class>.csv file, tolerating both the
    run_inference.py header format (cx,cy,z,score,slice_name,tile_name) and
    older header-less x,y,z-only files. Always returns a DataFrame with cx/cy/z
    plus score/slice_name/tile_name (NaN-filled if absent from the source).
    """
    expected = ['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']
    df = pd.read_csv(csv_path)
    cols = [str(c).strip().lower() for c in df.columns]
    if cols[:3] not in (['cx', 'cy', 'z'], ['x', 'y', 'z']):
        # First row wasn't actually a header -- re-read raw and name columns
        # positionally.
        df = pd.read_csv(csv_path, header=None)
        df.columns = expected[:df.shape[1]]
    else:
        df.columns = cols
        if df.columns[0] == 'x':
            df = df.rename(columns={'x': 'cx', 'y': 'cy'})
    for col in ('score', 'slice_name', 'tile_name'):
        if col not in df.columns:
            df[col] = np.nan
    return df


def _physical_to_index(img, physical_xyz):
    """Vectorized point -> voxel-index conversion: index = (point - origin) /
    spacing. Valid because every ANTs image built in this codebase uses
    identity direction (never set otherwise -- see io_utils.crop_to_bounds's
    same assumption). Avoids a slow per-point Python loop over
    ants.transform_physical_point_to_index for potentially tens of thousands
    of cells."""
    return (physical_xyz - np.array(img.origin)) / np.array(img.spacing)


def assign_cell_regions(cell_centroids_dir, output_dir, voxel_size_um, sample_fine_img, reg,
                         atlas_structures=None, prefix='ob_'):
    """Process every <prefix><class>.csv under cell_centroids_dir: convert
    original-pixel-space centroids to physical microns, transform through
    `reg` into atlas space, look up the atlas region at each cell, and write
    output_dir/cell_registration/<class>/cell_registration.csv.

    voxel_size_um: (sx, sy, sz) in microns of the RAW images cells were
    detected on -- typically finer than sample.voxel_size_um (the
    registration-input resolution). This is the direct ants-native
    replacement for cellMap.py's `ratio = VOXEL_SIZE_STITCHED/VOXEL_SIZE_ORIGINAL`
    + two-stage pixel rescaling: since every ANTs image here shares one
    physical-space frame regardless of resampling/cropping, a cell's physical
    position is just its raw pixel index times this voxel size -- no
    intermediate "stitched" rescale needed.

    sample_fine_img: the UNCROPPED fine/isotropic ants image from pipeline
    step 1 (before any registration.crop_for_registration crop) -- fixes the
    "resample space" columns to the grid of <name>_fine_<target>um.nii.gz on
    disk, regardless of what subregion was actually fed to registration.

    reg: dict returned by register.register_to_atlas/register_to_allen (needs
    'fwdtransforms' and 'atlas_annotation').

    atlas_structures: {id: {'name': ..., ...}} dict (e.g. from
    atlas_utils.get_allen_atlas or load_ccf_ontology_json), or None to skip
    name lookup (ids still get written, names fall back to "Region <id>").

    Returns the list of class names processed.
    """
    voxel_size_um = np.array(voxel_size_um, dtype=float)
    atlas_annotation = reg['atlas_annotation']
    atlas_arr = atlas_annotation.numpy()
    atlas_shape = np.array(atlas_arr.shape)
    id_to_name = {sid: info['name'] for sid, info in (atlas_structures or {}).items()}

    csv_paths = sorted(glob.glob(os.path.join(cell_centroids_dir, f'{prefix}*.csv')))
    processed = []
    for csv_path in csv_paths:
        class_name = os.path.basename(csv_path)[len(prefix):-len('.csv')]

        try:
            df_src = read_centroid_csv(csv_path)
        except Exception as e:
            logger.warning("Skipping class %s: failed to read CSV (%s).", class_name, e)
            continue
        if len(df_src) == 0:
            logger.warning("Skipping class %s: 0 cells found in csv.", class_name)
            continue

        raw_xyz = np.ascontiguousarray(df_src[['cx', 'cy', 'z']].values, dtype=float)
        physical_xyz = raw_xyz * voxel_size_um

        pts_sample = pd.DataFrame({'x': physical_xyz[:, 0], 'y': physical_xyz[:, 1], 'z': physical_xyz[:, 2]})
        pts_atlas = transforms.transform_cell_points(pts_sample, reg, direction='sample_to_atlas')
        atlas_phys = pts_atlas[['x', 'y', 'z']].values

        resample_idx = _physical_to_index(sample_fine_img, physical_xyz)
        atlas_idx = _physical_to_index(atlas_annotation, atlas_phys)
        atlas_idx_round = np.round(atlas_idx).astype(int)

        valid = np.all((atlas_idx_round >= 0) & (atlas_idx_round < atlas_shape), axis=1)
        raw_ids = np.zeros(len(atlas_idx_round), dtype=int)
        names = np.full(len(atlas_idx_round), 'no label', dtype=object)
        if np.any(valid):
            v = atlas_idx_round[valid]
            ids_valid = atlas_arr[v[:, 0], v[:, 1], v[:, 2]].astype(int)
            raw_ids[valid] = ids_valid
            names[valid] = [
                'background' if i == 0 else id_to_name.get(int(i), f'Region {int(i)}')
                for i in ids_valid
            ]

        slice_names = df_src['slice_name'].fillna('').astype(str).values
        tile_names = df_src['tile_name'].fillna('').astype(str).values
        scores = pd.to_numeric(df_src['score'], errors='coerce').fillna(0.0).values.astype(float)

        out_df = pd.DataFrame({
            0: raw_xyz[:, 0], 1: raw_xyz[:, 1], 2: raw_xyz[:, 2],
            3: resample_idx[:, 0], 4: resample_idx[:, 1], 5: resample_idx[:, 2],
            6: atlas_idx[:, 0], 7: atlas_idx[:, 1], 8: atlas_idx[:, 2],
            9: raw_ids, 10: names,
            11: slice_names, 12: tile_names, 13: scores,
        })

        class_dir = Path(output_dir) / 'cell_registration' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(class_dir / 'cell_registration.csv', header=False, index=False)
        logger.info("%s: %d cell(s) -> %s", class_name, len(out_df), class_dir / 'cell_registration.csv')
        processed.append(class_name)

    return processed
