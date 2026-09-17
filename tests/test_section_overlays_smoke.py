"""Checks for the two things in section_overlays.py that would fail silently:
labels placed on the picture by physical coordinate, and CCF's huge ids.

Run manually: `python tests/test_section_overlays_smoke.py` (a second, no files
written).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import section_overlays as ov  # noqa: E402

# A real CCF id past float32's exact range and far past any sane array length --
# a dense id -> colour table for this one id alone would be 1.8 GB.
BIG_ID = 614454277


def test_labels_on_grid_uses_physical_coordinates():
    # Label image: 4 x 4 voxels of 20 um, left half 7, right half BIG_ID,
    # origin offset by half a voxel (what downsample_section leaves behind).
    lab = np.where(np.arange(4)[None, :] < 2, 7, BIG_ID).repeat(4, 0).astype(np.uint32)
    origin, spacing = np.array([10.0, 10.0]), np.array([20.0, 20.0])
    # Display grid: 5 um pixels over the same field.
    xs = ys = np.arange(16) * 5.0 + 2.5
    out = ov._labels_on_grid(lab, origin, spacing, xs, ys)
    assert out.shape == (16, 16)
    # Nearest neighbour, so the 7 / BIG_ID border falls halfway between the
    # last 7 voxel's centre (x = 30 um) and the first BIG_ID one's (50 um):
    # at 40 um, between display columns 7 (37.5 um) and 8 (42.5 um).
    assert (out[:, :8] == 7).all(), "left half misplaced"
    assert (out[:, 8:] == BIG_ID).all(), "right half misplaced"

    # Outside the label image is background, not the border column smeared out.
    out = ov._labels_on_grid(lab, origin, spacing, np.array([-100.0, 30.0, 500.0]), np.array([30.0]))
    assert out.tolist() == [[0, 7, 0]], out


def test_encode_labels_and_colours():
    structures = {7: {"color_hex_triplet": "FF8000"}, BIG_ID: {"rgb_triplet": [1, 2, 3]}}
    lab = np.array([[0, 7], [BIG_ID, 7]], np.uint32)
    code, palette = ov.encode_labels(lab, structures)
    assert palette.shape == (3, 3), palette.shape           # background + 2 regions
    assert code.max() == 2 and palette.dtype == np.uint8
    assert palette[0].tolist() == [0, 0, 0], "code 0 must be background"
    assert palette[code[0, 1]].tolist() == [255, 128, 0], "hex triplet not used"
    assert palette[code[1, 0]].tolist() == [1, 2, 3], "rgb triplet not used"

    # Code 0 stays background even when no pixel is background.
    code, palette = ov.encode_labels(np.full((2, 2), 7, np.uint32), structures)
    assert palette[0].tolist() == [0, 0, 0] and (code == 1).all()

    # An id the ontology does not list still gets a stable, visible colour.
    _, palette = ov.encode_labels(np.array([[0, 999]], np.uint32), structures)
    assert palette[1].any() and (palette[1] == ov.region_rgb(999, structures)).all()


def test_render_overlays_leaves_background_alone():
    gray = np.full((8, 8), 100, np.uint8)
    lab = np.zeros((8, 8), np.uint32)
    lab[2:6, 2:6] = BIG_ID
    code, palette = ov.encode_labels(lab, {BIG_ID: {"rgb_triplet": [255, 0, 0]}})
    outline, filled = ov.render_overlays(gray, code, palette, alpha=0.5, line_width=1)
    assert outline.shape == filled.shape == (8, 8, 3) and outline.dtype == np.uint8
    for img in (outline, filled):
        assert (img[0, :] == 100).all(), "background must stay the image"
    # Outlines are drawn inside the region: the region's interior is untouched.
    assert (outline[3:5, 3:5] == 100).all()
    assert (outline[2, 2:6] == [255, 0, 0]).all()
    # The fill is the image blended with the colour, and it covers the inside.
    assert filled[4, 4].tolist() == [177, 50, 50], filled[4, 4]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all overlay checks passed")
