"""Smoke test for brain_mask.generate_brain_mask/suggest_crop (ClearMap's
Otsu-threshold auto brain-mask generator, ported for use as
ants.registration's moving_mask -- see config.yaml's mask.auto_brain_mask).

Synthetic data only, same manual assert-based style as test_pipeline_smoke.py
(no pytest). Run manually: `python tests/test_brain_mask_smoke.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import brain_mask  # noqa: E402


def make_synthetic_brain(shape=(60, 60, 40)):
    """A solid ellipsoid ("brain") off-center in a larger volume, standing in
    for a resampled sample surrounded by background."""
    x, y, z = shape
    xx, yy, zz = np.meshgrid(np.arange(x), np.arange(y), np.arange(z), indexing="ij")
    cx, cy, cz = x * 0.55, y * 0.45, z * 0.5
    r = np.sqrt(((xx - cx) / (x * 0.3)) ** 2 + ((yy - cy) / (y * 0.3)) ** 2 + ((zz - cz) / (z * 0.3)) ** 2)
    brain = (r < 1).astype(np.float32) * 800
    rng = np.random.default_rng(0)
    brain += rng.normal(0, 15, size=brain.shape)  # background noise floor
    true_mask = r < 1
    return np.clip(brain, 0, None).astype(np.float32), true_mask


def test_generate_brain_mask():
    print("1. brain_mask.generate_brain_mask...")
    arr, true_mask = make_synthetic_brain()
    mask, bbox = brain_mask.generate_brain_mask(arr, sigma=1.0, closing_radius=2, dilate_radius=1, slice_axis=2)

    assert mask.shape == arr.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}

    intersection = np.logical_and(mask, true_mask).sum()
    union = np.logical_or(mask, true_mask).sum()
    dice = 2 * intersection / (mask.sum() + true_mask.sum())
    assert dice > 0.9, f"mask should closely match the synthetic blob, dice={dice:.3f}"

    true_bbox = brain_mask._bounding_box(true_mask.astype(np.uint8))
    for (lo, hi), (true_lo, true_hi) in zip(bbox, true_bbox):
        assert abs(lo - true_lo) <= 3 and abs(hi - true_hi) <= 3, f"bbox {bbox} should track the true blob {true_bbox}"

    print(f"   OK (dice={dice:.3f} vs synthetic ground truth, bbox={bbox})")
    return bbox, arr.shape


def test_suggest_crop(bbox, shape):
    print("2. brain_mask.suggest_crop...")
    suggestion = brain_mask.suggest_crop(bbox, shape, padding=5)
    assert len(suggestion) == 3
    for (lo, hi), (true_lo, true_hi), size in zip(suggestion, bbox, shape):
        assert lo == max(0, true_lo - 5)
        assert hi == min(size, true_hi + 5)
        assert 0 <= lo < hi <= size

    # Edge case: a bbox touching the volume boundary must clip, not go negative/overflow.
    edge_bbox = ((0, 10), (5, 20), (30, 40))
    edge_suggestion = brain_mask.suggest_crop(edge_bbox, (40, 40, 40), padding=5)
    assert edge_suggestion[0] == [0, 15], "padding past axis start must clip to 0"
    assert edge_suggestion[2] == [25, 40], "padding past axis end must clip to shape"
    print(f"   OK (padded+clipped correctly: {suggestion}, boundary clipping verified)")


def main():
    bbox, shape = test_generate_brain_mask()
    test_suggest_crop(bbox, shape)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
