"""Paint a per-region statistic onto the atlas volume.

Shared by the static figure script (stats/plot_heatmaps.py) and the
interactive viewer (../Registration_toolkit/tools/stats_view.py), so the two
cannot disagree about what a voxel's colour means.

The one non-obvious operation is the level collapse. `region_stats.csv` holds
one row per (region, level), while the annotation labels each voxel with a
single structure that sits at whatever depth the atlas happened to label it.
To draw a level-5 map, every voxel has to read out its level-5 ANCESTOR: a
voxel labelled CA1 (level 8) is painted with HPF's value. Without that step a
level-5 map would only colour the handful of voxels whose own label happens to
be at level 5, which is not what "the level-5 result" means.

RegionVolume does the expensive part once (a voxel -> unique-label index map)
so that changing level, class, metric or statistic is a fancy-index away --
that is what makes the napari viewer interactive on a 64M-voxel hemisphere.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stats.ontology import Ontology  # noqa: E402

# DeMBA P5 / CCFv3 20 um master annotation, shape (AP, DV, ML) = (563, 400, 570).
# Confirmed empirically: axis 2 is the only one whose halves mirror each other
# (93% of voxels match), and 570 = 1140 CCFv3 10 um ML voxels at 20 um.
AXIS_AP, AXIS_DV, AXIS_ML = 0, 1, 2
AXIS_NAMES = {AXIS_AP: "AP", AXIS_DV: "DV", AXIS_ML: "ML"}
MIDLINE_ML = 285


def load_annotation(path):
    if str(path).endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(path)
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj)


def hemisphere_slice(annot, side="right"):
    """Crop to one hemisphere along ML. 'right' is the half the atlas was
    already cropped to for registration (ML >= midline); 'both' is a no-op.

    Five of the six samples are right hemispheres and one (s10) is a left
    hemisphere registered against a mirrored atlas, so every result is
    expressed in right-hemisphere coordinates regardless."""
    if side == "both":
        return annot, 0
    if annot.shape[AXIS_ML] <= MIDLINE_ML:
        return annot, 0  # already a hemisphere
    if side == "right":
        return annot[:, :, MIDLINE_ML:], MIDLINE_ML
    if side == "left":
        return annot[:, :, :MIDLINE_ML], 0
    raise ValueError(f"side must be right/left/both, got {side!r}")


class RegionVolume:
    """An annotation volume prepared for repeated repainting.

    `index` maps each voxel to a position in `ids` (int16: the atlas carries
    fewer than 700 distinct labels, so a dense id-indexed LUT -- which would
    need 2.4 GB for CCF's 6.1e8 max id -- is never built)."""

    def __init__(self, annot, ontology):
        self.ontology = ontology
        self.ids = np.unique(annot)
        if len(self.ids) > np.iinfo(np.int16).max:
            raise ValueError(f"{len(self.ids)} distinct labels exceeds the int16 index")
        self.index = np.searchsorted(self.ids, annot).astype(np.int16)
        self.orders = ontology.order_of_ids(self.ids)
        self.shape = annot.shape
        self._ancestor_cache = {}

    def _target_orders(self, level):
        """Which ontology node each unique label reads its value from."""
        if level is None:
            return self.orders
        if level not in self._ancestor_cache:
            self._ancestor_cache[level] = self.ontology.ancestor_at_level(level)
        anc = self._ancestor_cache[level]
        safe = np.clip(self.orders, 0, None)
        return np.where(self.orders >= 0, anc[safe], -1)

    def paint(self, value_by_order, level=None, background=np.nan):
        """-> float volume of the same shape, each voxel carrying its region's
        value at `level`. Voxels whose region has no value (not tested, or
        shallower than `level`) get `background`."""
        target = self._target_orders(level)
        vals = np.full(len(self.ids), background, dtype=float)
        ok = target >= 0
        vals[ok] = value_by_order[target[ok]]
        vals[self.ids == 0] = background  # label 0 is outside the brain
        return vals[self.index]

    def region_id_volume(self, level=None):
        """-> the structure id each voxel reads out at `level` (0 where none).
        Used for hover readout in the viewer."""
        target = self._target_orders(level)
        out = np.zeros(len(self.ids), dtype=np.int64)
        ok = target >= 0
        out[ok] = self.ontology.ids[target[ok]]
        out[self.ids == 0] = 0
        return out[self.index]


def values_by_order(stats_df, ontology, level, class_name, metric, value_col,
                    significant_only=False, alpha=0.05):
    """Pull one column of region_stats.csv into an order-indexed array.

    Rows are selected by (level, class, metric) -- the same triple that
    defines one correction family -- so the map shows exactly the numbers one
    family produced, never a mixture of levels."""
    sub = stats_df[(stats_df["level"] == level)
                   & (stats_df["class_name"] == class_name)
                   & (stats_df["metric"] == metric)]
    if significant_only and "p_adj" in sub.columns:
        sub = sub[sub["p_adj"] < alpha]
    out = np.full(ontology.n, np.nan)
    if sub.empty:
        return out, 0
    orders = ontology.order_of_ids(sub["id"].to_numpy(dtype=np.int64))
    ok = orders >= 0
    out[orders[ok]] = sub[value_col].to_numpy(dtype=float)[ok]
    return out, int(ok.sum())


def symmetric_limits(volume, percentile=99.0):
    """Colour limits centred on zero, robust to a few extreme regions. Used
    for signed statistics (log2fc, Hedges' g) so that the zero point of the
    diverging colormap is the no-difference point."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return -1.0, 1.0
    m = float(np.percentile(np.abs(finite), percentile))
    if m <= 0:
        m = float(np.max(np.abs(finite))) or 1.0
    return -m, m


def load_ontology(path):
    return Ontology.from_json(path)
