"""Recovery test for the 2D sagittal-section pipeline on a synthetic atlas.

A phantom brain is built whose sagittal planes differ with distance from the
midline (a lateral "ventricle" and a rod whose depth changes with ML). One
oblique plane is cut out at a known position/tilt, mirrored, rotated,
rescaled, contrast-changed and noised into a "section" TIFF at 10 um pixels.
The pipeline then has to recover the plane, the in-plane pose, and -- the
end-to-end check of all the transform plumbing -- the 3D atlas position of
individual cells placed on the section.

Run manually: `python tests/test_section2d_smoke.py` (about a minute; set
TMPDIR to choose where the files go).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ants  # noqa: E402
from registration_ants import section2d  # noqa: E402

RES = 40.0
TRUTH = section2d.PlaneParams(ml_um=1000.0, yaw_deg=6.0, roll_deg=-4.0)
SIZE_RATIO, ROTATION_DEG, PIXEL_UM = 1.05, 110.0, 10.0


def make_phantom():
    shape = (140, 180, 110)                               # ML, AP, DV
    x, y, z = np.meshgrid(*(np.arange(s, dtype=float) for s in shape), indexing="ij")
    dx = x - 70
    r = np.sqrt((dx / 55) ** 2 + ((y - 90) / 80) ** 2 + ((z - 55) / 45) ** 2)
    ann = np.zeros(shape, np.uint32)
    ann[r < 1] = 10
    ann[(r < 1) & (r > 0.8) & (z < 55)] = 20              # dorsal shell
    ann[((dx / 20) ** 2 + ((y - 150) / 20) ** 2 + ((z - 45) / 20) ** 2) < 1] = 30
    ann[(((np.abs(dx) - 25) / 8) ** 2 + ((y - 80) / 25) ** 2 + ((z - 50) / 10) ** 2) < 1] = 40
    ann[(r < 0.9) & (np.abs(z - (40 + 0.6 * np.abs(dx))) < 4) & (y > 60) & (y < 110)] = 50
    lut = {0: 0.0, 10: 0.4, 20: 0.7, 30: 0.9, 40: 0.1, 50: 1.0}
    tmpl = np.vectorize(lut.get)(ann).astype(np.float32)
    rng = np.random.default_rng(0)
    tmpl += (ann > 0) * 0.1 * ndimage.gaussian_filter(rng.normal(size=shape), 2).astype(np.float32)
    structures = {k: {"name": f"region {k}", "acronym": f"R{k}", "structure_id_path": [k]} for k in lut if k}
    return tmpl, ann, structures


def make_section(atlas, out_dir):
    """Returns (tiff path, section->plane 2x2 A, section centre c) with
    plane_uv = A (p_section - c)."""
    tmpl, _, origin = atlas.sample_plane(TRUTH, PIXEL_UM)
    plane = ants.from_numpy(tmpl, spacing=(PIXEL_UM,) * 2, origin=origin)
    side = int(2.6 * max(atlas.half_extent_um) / PIXEL_UM)
    canvas = ants.from_numpy(np.zeros((side, side), np.float32), spacing=(PIXEL_UM,) * 2)
    c = np.array([side, side]) * PIXEL_UM / 2
    M = SIZE_RATIO * section2d._rot(ROTATION_DEG) @ np.diag([1.0, -1.0])   # plane -> section
    A = np.linalg.inv(M)
    tx_path = str(out_dir / "truth.mat")
    section2d._write_linear(A, -A @ c, tx_path)
    sec = ants.apply_transforms(canvas, plane, [tx_path], interpolator="linear").numpy()
    rng = np.random.default_rng(1)
    raw = np.clip(sec, 0, 1) ** 0.6 * 3000 + 200 + rng.normal(0, 60, sec.shape)
    path = out_dir / "section.tif"
    tifffile.imwrite(path, np.clip(raw, 0, 65535).astype(np.uint16).T)
    return path, A, c, sec


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="section2d_smoke_"))
    tmpl, ann, structures = make_phantom()
    atlas = section2d.SagittalAtlas(tmpl, ann, RES)
    print(f"phantom atlas, midline {atlas.midline_vox}, half-width {atlas.ml_max_um:.0f} um -> {out_dir}")

    tif, A, c, sec = make_section(atlas, out_dir)
    rng = np.random.default_rng(2)
    cols, rows = np.nonzero(sec > 0.05)
    pick = rng.choice(len(cols), 400, replace=False)
    cells = pd.DataFrame({"x": cols[pick], "y": rows[pick]})
    cells.to_csv(out_dir / "cells.csv", index=False)

    sec_cfg = {"name": "phantom", "image": str(tif), "pixel_size_um": PIXEL_UM, "cells_csv": str(out_dir / "cells.csv")}
    search = {**section2d.SEARCH_DEFAULTS, "search_res_um": 80, "fine_top_k": 2, "n_workers": 4}
    reg = {**section2d.REGISTRATION_DEFAULTS, "register_res_um": RES, "reg_iterations": [40, 20, 10]}
    summary = section2d.process_section(sec_cfg, atlas, search, reg, out_dir, structures, overwrite=True)

    print("\nrecovered:", {k: summary[k] for k in ("ml_um", "yaw_deg", "roll_deg", "size_ratio",
                                                  "rotation_deg", "mirrored", "far_gap")})
    assert abs(summary["ml_um"] - TRUTH.ml_um) <= 80, summary["ml_um"]
    assert abs(summary["yaw_deg"] - TRUTH.yaw_deg) <= 3, summary["yaw_deg"]
    assert abs(summary["roll_deg"] - TRUTH.roll_deg) <= 3, summary["roll_deg"]
    assert summary["mirrored"], "mirror not recovered"
    assert abs(summary["size_ratio"] - SIZE_RATIO) < 0.05, summary["size_ratio"]
    assert abs((summary["rotation_deg"] - ROTATION_DEG + 180) % 360 - 180) < 5, summary["rotation_deg"]

    got = pd.read_csv(out_dir / "phantom" / "cells_registered.csv")
    p = np.column_stack([cells["x"], cells["y"]]) * PIXEL_UM
    uv = (p - c) @ A.T
    truth_xyz = atlas.plane_to_atlas_um(uv[:, 0], uv[:, 1], TRUTH)
    err = np.linalg.norm(got[["atlas_x_um", "atlas_y_um", "atlas_z_um"]].to_numpy() - truth_xyz, axis=1)
    agree = (got["region_id"].to_numpy() == atlas.lookup(truth_xyz)).mean()
    print(f"cell 3D error: median {np.median(err):.0f} um, 90th pct {np.percentile(err, 90):.0f} um; "
          f"region agreement {100 * agree:.1f}%")
    assert np.median(err) < 60, np.median(err)
    assert agree > 0.85, agree
    for key in ("affine", "syn"):
        print(f"{key}: unlabelled {summary[f'{key}_unlabelled_pct']:.1f}%, outside {summary[f'{key}_outside_pct']:.1f}%")
    assert summary["syn_unlabelled_pct"] < 10
    print("\nSECTION2D SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
