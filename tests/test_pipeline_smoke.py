"""End-to-end smoke test: synthetic anisotropic TIFF -> resample -> preprocess
-> register to Allen CCF -> bidirectional transform application.

Not a scientific accuracy check (the synthetic blob and the mismatched atlas
resolution used here make the actual registration meaningless) — this only
verifies every function in the pipeline runs, shapes/spacing come out right,
and the transform direction plumbing (fwd vs inv) is wired correctly. Run
manually: `python tests/test_pipeline_smoke.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import io_utils, preprocess, register, transforms  # noqa: E402


def make_synthetic_stack(path, shape_zyx=(50, 80, 80)):
    """A layered ellipsoid blob + noise, standing in for a tiny 'brain'."""
    z, y, x = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zz - cz) / (z * 0.35)) ** 2 + ((yy - cy) / (y * 0.35)) ** 2 + ((xx - cx) / (x * 0.35)) ** 2)
    blob = (r < 1).astype(np.float32) * 1000
    blob += (r < 0.5).astype(np.float32) * 500
    rng = np.random.default_rng(0)
    blob += rng.normal(0, 20, size=blob.shape)
    arr = np.clip(blob, 0, None).astype(np.uint16)
    tifffile.imwrite(path, arr)


def main():
    tmp_dir = Path("/tmp/claude-1004/-home-fyu7-My-project-Registration-ants/3434ed6b-afa1-4c05-956b-9aeb172d1731/scratchpad/antsreg_smoketest")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_tiff = tmp_dir / "synthetic_sample.tif"

    # Synthetic voxel sizes chosen only to keep this test fast (small array)
    # while still exercising both resample_to_isotropic branches: xy (20um)
    # is finer than target (25um) -> anti-alias + downsample; z (90um) is
    # coarser than target -> plain upsample interpolation. These are NOT the
    # user's real acquisition values.
    voxel_size_xyz = (20.0, 20.0, 90.0)
    target_um = 25.0

    print("1. Generating synthetic anisotropic TIFF stack...")
    make_synthetic_stack(raw_tiff)

    print("2. io_utils: load + resample to isotropic...")
    img = io_utils.load_tiff_stack_as_ants(raw_tiff, voxel_size_xyz)
    assert img.spacing == voxel_size_xyz, f"spacing mismatch: {img.spacing}"

    iso_out = tmp_dir / "sample_iso.nii.gz"
    img_iso = io_utils.convert_to_isotropic_nifti(raw_tiff, voxel_size_xyz, target_um, iso_out)
    assert iso_out.exists(), "NIfTI not written"
    for s in img_iso.spacing:
        assert abs(s - target_um) < 1e-3, f"isotropic spacing wrong: {img_iso.spacing}"
    print(f"   isotropic shape={img_iso.shape}, spacing={img_iso.spacing}")

    print("3. preprocess: N4 + normalize...")
    img_prep = preprocess.preprocess_for_registration(img_iso)
    arr_prep = img_prep.numpy()
    assert np.isfinite(arr_prep).all(), "non-finite values after preprocessing"
    assert 0.0 <= arr_prep.min() and arr_prep.max() <= 1.0 + 1e-5, "normalization out of [0,1]"
    assert img_prep.spacing == img_iso.spacing

    print("4. register: sample -> Allen CCF 100um (cached, fast)...")
    reg = register.register_to_allen(img_prep, atlas_res_um=100, type_of_transform="SyNRA")
    assert "fwdtransforms" in reg and "invtransforms" in reg
    assert reg["fwdtransforms"] != reg["invtransforms"], "fwd/inv transform lists should differ"

    print("5. transforms: bidirectional image warps...")
    sample_in_atlas = transforms.warp_sample_to_atlas(img_prep, reg)
    assert sample_in_atlas.shape == reg["atlas_template"].shape

    labels_in_sample = transforms.warp_labels_to_sample(img_prep, reg)
    assert labels_in_sample.shape == img_prep.shape
    label_vals = set(np.unique(labels_in_sample.numpy()))
    atlas_vals = set(np.unique(reg["atlas_annotation"].numpy()))
    assert label_vals.issubset(atlas_vals), "genericLabel interpolation introduced new label ids"

    print("6. transforms: point transform (atlas centroid -> sample -> atlas)...")
    atlas_center_phys = np.array(reg["atlas_template"].shape) / 2 * np.array(reg["atlas_template"].spacing)
    pts_atlas = pd.DataFrame({"x": [atlas_center_phys[0]], "y": [atlas_center_phys[1]], "z": [atlas_center_phys[2]]})
    pts_sample = transforms.transform_cell_points(pts_atlas, reg, direction="atlas_to_sample")
    pts_back = transforms.transform_cell_points(pts_sample, reg, direction="sample_to_atlas")
    assert np.isfinite(pts_sample[["x", "y", "z"]].values).all()
    assert np.isfinite(pts_back[["x", "y", "z"]].values).all()
    delta = np.abs(pts_back[["x", "y", "z"]].values - pts_atlas[["x", "y", "z"]].values)
    print(f"   round-trip point delta (physical units, meaningless here since inputs are synthetic): {delta}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
