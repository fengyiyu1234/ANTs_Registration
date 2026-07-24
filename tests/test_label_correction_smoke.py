"""Smoke test for the post-registration correction workflow:
mask_utils.interpolate_sparse_label_correction (scripts/edit_sample_labels.py's
export logic) and relabel_cell_csv (scripts/relabel_cells.py).

Synthetic data only, same manual assert-based style as test_pipeline_smoke.py
(no pytest). Run manually: `python tests/test_label_correction_smoke.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from registration_ants import mask_utils  # noqa: E402
from relabel_cells import relabel_cell_csv  # noqa: E402


def test_interpolate_sparse_label_correction():
    print("1. mask_utils.interpolate_sparse_label_correction...")
    shape = (5, 10, 10)  # z, y, x
    original = np.full(shape, 1, dtype=np.uint32)

    edited_z1 = original[1].copy()
    edited_z1[2:6, 2:6] = 2   # region A -> label 2
    edited_z1[7:9, 7:9] = 3   # region B -> label 3
    mask_z1 = edited_z1 != original[1]

    edited_z3 = original[3].copy()
    edited_z3[3:7, 3:7] = 2   # region A persists (shifted) -> label 2
    # region B: no edit at z=3 -> should taper to nothing by z=3
    mask_z3 = edited_z3 != original[3]

    keyframe_edits = {1: (mask_z1, edited_z1), 3: (mask_z3, edited_z3)}
    corrected = mask_utils.interpolate_sparse_label_correction(keyframe_edits, original)

    assert corrected.shape == shape
    assert np.array_equal(corrected[0], original[0]), "plane before first keyframe must stay untouched"
    assert np.array_equal(corrected[4], original[4]), "plane after last keyframe must stay untouched"
    assert np.array_equal(corrected[1], edited_z1), "keyframe plane must match the hand edit exactly"
    assert np.array_equal(corrected[3], edited_z3), "keyframe plane must match the hand edit exactly"

    area_label3_z1 = int(np.sum(corrected[1] == 3))
    area_label3_z2 = int(np.sum(corrected[2] == 3))
    area_label3_z3 = int(np.sum(corrected[3] == 3))
    assert area_label3_z1 > 0
    assert area_label3_z3 == 0, "label 3 has no edit at the far keyframe, so it must taper to nothing there"
    assert area_label3_z2 < area_label3_z1, "label 3's region should shrink en route to the far keyframe"
    assert set(np.unique(corrected[2])) <= {1, 2, 3}, "interpolated plane must not introduce stray label ids"
    print(f"   OK (label-3 area z1={area_label3_z1} -> z2={area_label3_z2} -> z3={area_label3_z3})")


def test_relabel_cell_csv(tmp_dir):
    print("2. relabel_cells.relabel_cell_csv...")
    corrected_labels = np.full((3, 4, 4), 10, dtype=np.int64)  # z, y, x
    corrected_labels[1, 1:3, 1:3] = 20  # a "special" sub-region

    csv_path = tmp_dir / "cell_registration.csv"
    rows = [
        # raw_x,raw_y,raw_z, resample_x,resample_y,resample_z, atlas_x,atlas_y,atlas_z, mapped_id, region
        [0, 0, 0,  1, 1, 1,  0, 0, 0,  999, "WrongRegion"],   # inside the 20-region -> should change
        [0, 0, 0,  0, 0, 0,  0, 0, 0,  10,  "Base"],          # already correct -> should NOT count as changed
        [0, 0, 0,  100, 100, 100,  0, 0, 0,  555, "OutOfBounds"],  # out of bounds -> untouched
        [0, 0, 0,  np.nan, np.nan, np.nan,  0, 0, 0,  777, "Missing"],  # missing coord -> untouched
    ]
    pd.DataFrame(rows).to_csv(csv_path, header=False, index=False)

    id_to_name = {10: "Base", 20: "Special"}
    n_changed = relabel_cell_csv(csv_path, corrected_labels, id_to_name, backup=True)
    assert n_changed == 1, f"expected exactly 1 changed row, got {n_changed}"

    bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    assert bak_path.exists(), "backup file must be created"
    bak_df = pd.read_csv(bak_path, header=None)
    assert bak_df.iloc[0, 9] == 999, "backup must preserve the pre-correction values"

    result = pd.read_csv(csv_path, header=None)
    assert result.iloc[0, 9] == 20 and result.iloc[0, 10] == "Special"
    assert result.iloc[1, 9] == 10 and result.iloc[1, 10] == "Base"
    assert result.iloc[2, 9] == 555, "out-of-bounds row must be left untouched"
    assert result.iloc[3, 9] == 777, "row with missing coords must be left untouched"

    # Re-run: backup must not be clobbered by a second pass.
    relabel_cell_csv(csv_path, corrected_labels, id_to_name, backup=True)
    bak_df_after = pd.read_csv(bak_path, header=None)
    assert bak_df_after.iloc[0, 9] == 999, "existing backup must not be overwritten on re-run"
    print(f"   OK ({n_changed} cell relabeled, backup preserved across re-run)")


def main():
    tmp_dir = Path("/tmp/claude-1004/-home-fyu7-My-project-Registration-ants/6835270b-acd3-4105-910e-d1d6b63854f2/scratchpad/label_correction_smoketest")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_interpolate_sparse_label_correction()
    test_relabel_cell_csv(tmp_dir)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
