"""Smoke tests for the cellMap.py-derived features ported in this session:
atlas_utils.reorient_volume/prepare_custom_atlas (atlas orientation+cropping),
io_utils.crop_to_bounds (origin-preserving sample crop before registration),
and cell_points.assign_cell_regions (cell centroid -> atlas region assignment).

Synthetic data only, same manual assert-based style as test_pipeline_smoke.py
(no pytest). Run manually: `python tests/test_new_features_smoke.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils, cell_points, io_utils  # noqa: E402


def test_reorient_volume():
    print("1. atlas_utils.reorient_volume...")
    # Distinct values per voxel so permutation/flips are checkable exactly.
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # (x,y,z)

    identity = atlas_utils.reorient_volume(arr, None)
    assert identity is arr or np.array_equal(identity, arr)

    # orientation=(1,3,2): keep axis0, swap axis1<->axis2 (destination axis1
    # comes from source axis3=z, destination axis2 comes from source axis2=y).
    swapped = atlas_utils.reorient_volume(arr, (1, 3, 2))
    assert swapped.shape == (2, 4, 3)
    assert np.array_equal(swapped, arr.transpose(0, 2, 1))

    # orientation=(-1,2,3): flip axis0 only.
    flipped = atlas_utils.reorient_volume(arr, (-1, 2, 3))
    assert flipped.shape == arr.shape
    assert np.array_equal(flipped, arr[::-1, :, :])

    try:
        atlas_utils.reorient_volume(arr, (1, 1, 3))
        assert False, "duplicate-axis orientation must raise"
    except ValueError:
        pass
    print("   OK (identity / permute+flip / duplicate-axis validation all correct)")


def test_prepare_custom_atlas(tmp_dir):
    print("2. atlas_utils.prepare_custom_atlas (reorient + crop + caching)...")
    atlas_dir = tmp_dir / "raw_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # Build a small raw "atlas" with distinct values, stored the same way
    # every other raw TIFF in this codebase is (z,y,x) on disk.
    arr_xyz = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    template_path = atlas_dir / "template.tif"
    annotation_path = atlas_dir / "annotation.tif"
    tifffile.imwrite(str(template_path), np.transpose(arr_xyz, (2, 1, 0)))
    tifffile.imwrite(str(annotation_path), np.transpose(arr_xyz, (2, 1, 0)))

    orientation = (1, 3, 2)
    slicing = [[1, 3], None, None]
    template_img, annotation_img = atlas_utils.prepare_custom_atlas(
        str(template_path), str(annotation_path), resolution_um=25,
        orientation=orientation, slicing=slicing,
    )

    expected = atlas_utils.reorient_volume(arr_xyz, orientation)[1:3, :, :]
    assert template_img.numpy().shape == expected.shape
    assert np.array_equal(template_img.numpy(), expected)
    assert all(abs(s - 25.0) < 1e-6 for s in template_img.spacing)

    postfix = atlas_utils._atlas_prep_postfix(orientation, slicing)
    cached_path = atlas_dir / f"template_{postfix}.tif"
    assert cached_path.exists()
    mtime_before = cached_path.stat().st_mtime_ns

    # Re-run with overwrite=False: cached file must be reused, not rewritten.
    atlas_utils.prepare_custom_atlas(
        str(template_path), str(annotation_path), resolution_um=25,
        orientation=orientation, slicing=slicing,
    )
    assert cached_path.stat().st_mtime_ns == mtime_before, "cached atlas file was needlessly recomputed"

    # No orientation/slicing -> falls through to plain load_custom_atlas.
    plain_template, _ = atlas_utils.prepare_custom_atlas(str(template_path), str(annotation_path), resolution_um=25)
    assert np.array_equal(plain_template.numpy(), arr_xyz)
    print("   OK (reorient+crop matches manual computation, caching skips recompute, plain passthrough works)")


def test_crop_to_bounds():
    print("3. io_utils.crop_to_bounds (origin-preserving crop)...")
    import ants

    full = np.zeros((20, 20, 20), dtype=np.float32)
    full[10, 12, 8] = 1.0
    img_full = ants.from_numpy(full, spacing=(2.0, 2.0, 2.0))

    cropped = io_utils.crop_to_bounds(img_full, x=(5, 15), y=None, z=(0, 10))
    assert cropped.shape == (10, 20, 10)
    assert np.array_equal(np.array(cropped.origin), np.array([10.0, 0.0, 0.0]))

    # The physical location of the hot voxel (from the FULL image) must map
    # back to the correct index in the cropped image -- this is the property
    # the whole crop-before-registration design depends on (a transform
    # computed with the cropped image stays valid for points in the original
    # image's physical space, with no manual crop-offset correction needed).
    phys = ants.transform_index_to_physical_point(img_full, (10, 12, 8))
    idx_in_crop = ants.transform_physical_point_to_index(cropped, phys)
    idx_in_crop = tuple(int(round(v)) for v in idx_in_crop)
    assert idx_in_crop == (5, 12, 8), f"expected (5,12,8), got {idx_in_crop}"
    assert cropped.numpy()[idx_in_crop] == 1.0
    print("   OK (shape/origin correct, physical-space correspondence preserved)")


def test_read_centroid_csv(tmp_dir):
    print("4. cell_points.read_centroid_csv (tolerant reader)...")
    legacy_path = tmp_dir / "ob_legacy.csv"
    pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_csv(legacy_path, header=False, index=False)
    df = cell_points.read_centroid_csv(legacy_path)
    assert list(df["cx"]) == [1.0, 4.0]
    assert df["score"].isna().all()

    modern_path = tmp_dir / "ob_modern.csv"
    pd.DataFrame({
        "cx": [1.0], "cy": [2.0], "z": [3.0],
        "score": [0.9], "slice_name": ["s1"], "tile_name": ["t1"],
    }).to_csv(modern_path, index=False)
    df2 = cell_points.read_centroid_csv(modern_path)
    assert df2["slice_name"].iloc[0] == "s1" and df2["score"].iloc[0] == 0.9
    print("   OK (legacy x,y,z-only and modern header formats both read correctly)")


def test_assign_cell_regions(tmp_dir):
    print("5. cell_points.assign_cell_regions (region assignment end-to-end)...")
    import ants

    atlas_arr = np.zeros((10, 10, 10), dtype=np.float32)
    atlas_arr[2:4, 2:4, 2:4] = 5   # "RegionFive"
    atlas_arr[6:8, 6:8, 6:8] = 7   # present in atlas but missing from structures dict (tests fallback)
    atlas_annotation = ants.from_numpy(atlas_arr, spacing=(25.0, 25.0, 25.0))

    sample_fine_img = ants.from_numpy(np.zeros((20, 20, 20), dtype=np.float32), spacing=(50.0, 50.0, 50.0))
    reg = {"fwdtransforms": [], "atlas_annotation": atlas_annotation}  # [] == identity (verified: ants passes points through unchanged)
    atlas_structures = {5: {"name": "RegionFive"}}

    voxel_size_um = (5.0, 5.0, 5.0)
    centroids_dir = tmp_dir / "cell_centroids"
    centroids_dir.mkdir(parents=True, exist_ok=True)
    # raw pixel coords chosen so raw_xyz * voxel_size_um lands exactly at the
    # atlas-index centers below (physical = raw_xyz * voxel_size_um, atlas
    # spacing=25 -> atlas_idx = physical/25).
    rows = pd.DataFrame({
        "cx": [15.0, 35.0, 0.0, 2000.0],
        "cy": [15.0, 35.0, 0.0, 2000.0],
        "z":  [15.0, 35.0, 0.0, 2000.0],
        "score": [0.5, 0.6, 0.7, 0.1],
        "slice_name": ["s0", "s1", "s2", "s3"],
        "tile_name": ["t0", "t1", "t2", "t3"],
    })
    rows.to_csv(centroids_dir / "ob_testclass.csv", index=False)

    output_dir = tmp_dir / "out"
    classes = cell_points.assign_cell_regions(
        str(centroids_dir), str(output_dir), voxel_size_um, sample_fine_img, reg,
        atlas_structures=atlas_structures, prefix="ob_",
    )
    assert classes == ["testclass"]

    out_csv = output_dir / "cell_registration" / "testclass" / "cell_registration.csv"
    assert out_csv.exists()
    result = pd.read_csv(out_csv, header=None)
    assert result.shape == (4, 14)

    # Row 0: physical (75,75,75) -> atlas index (3,3,3) -> id 5 -> "RegionFive".
    assert np.allclose(result.iloc[0, 6:9].values.astype(float), [3, 3, 3])
    assert result.iloc[0, 9] == 5 and result.iloc[0, 10] == "RegionFive"
    assert np.allclose(result.iloc[0, 3:6].values.astype(float), [1.5, 1.5, 1.5])  # 75/50

    # Row 1: atlas index (7,7,7) -> id 7, not in atlas_structures -> fallback name.
    assert np.allclose(result.iloc[1, 6:9].values.astype(float), [7, 7, 7])
    assert result.iloc[1, 9] == 7 and result.iloc[1, 10] == "Region 7"

    # Row 2: index (0,0,0) -> id 0 -> "background" (inside bounds).
    assert result.iloc[2, 9] == 0 and result.iloc[2, 10] == "background"

    # Row 3: way outside the atlas volume -> "no label", not conflated with background.
    assert result.iloc[3, 10] == "no label"

    assert list(result.iloc[:, 11]) == ["s0", "s1", "s2", "s3"]
    assert np.allclose(result.iloc[:, 13].values.astype(float), [0.5, 0.6, 0.7, 0.1])
    print("   OK (resample/atlas indices, region names, background vs out-of-bounds, provenance columns all correct)")


def main():
    tmp_dir = Path("/tmp/claude-1004/-home-fyu7-My-project-Registration-ants/4b6da97c-5cac-42de-b398-9c89d0ce513e/scratchpad/new_features_smoketest")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_reorient_volume()
    test_prepare_custom_atlas(tmp_dir)
    test_crop_to_bounds()
    test_read_centroid_csv(tmp_dir)
    test_assign_cell_regions(tmp_dir)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
