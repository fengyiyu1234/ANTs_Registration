"""Registration masks: automated atlas-side region exclusion (see
atlas_utils.build_region_exclusion_mask), and a sparse-plane-to-dense
interpolation workflow used by ../GT_tool_for_registration/paint_mask.py's `guide` kind for
turning a hand-painted outline into a full 3D mask.

Both functions here are axis-order-agnostic: "axis 0" just means whichever
axis the caller indexes keyframe planes by. The painting tool operates on
arrays read via SimpleITK (natural (z,y,x), axis 0 = z = the actual imaging
planes a person would scroll through) -- that's the convention these
functions are meant to be used with, but nothing here hardcodes it.
"""
import numpy as np
from scipy import ndimage


def _signed_distance(binary_plane):
    inside = ndimage.distance_transform_edt(binary_plane)
    outside = ndimage.distance_transform_edt(~binary_plane)
    return outside - inside


def interpolate_sparse_mask(keyframe_planes, full_shape):
    """Interpolate a sparse set of hand-painted 2D binary planes into a
    dense 3D binary mask, using signed-distance-field interpolation between
    consecutive keyframes -- smoother and more shape-consistent than
    linearly blending the raw 0/1 arrays, especially when the crack's
    cross-section changes shape a lot between painted planes.

    keyframe_planes: {index: 2D bool array} -- e.g. the planes a user
    painted a rough outline on. Any index NOT in this dict is treated as
    "not painted", not "no damage": interpolation only happens *between*
    the indices actually present, and planes outside
    [min(keyframe), max(keyframe)] are left empty (a crack has a bounded
    start/end -- it doesn't extend to the edges of the volume).
    full_shape: shape of the full 3D volume this mask will cover.
    """
    dense = np.zeros(full_shape, dtype=bool)
    indices = sorted(keyframe_planes)
    if not indices:
        return dense

    for idx in indices:
        dense[idx] = keyframe_planes[idx]

    for i0, i1 in zip(indices[:-1], indices[1:]):
        if i1 - i0 <= 1:
            continue
        sdf0 = _signed_distance(keyframe_planes[i0])
        sdf1 = _signed_distance(keyframe_planes[i1])
        for idx in range(i0 + 1, i1):
            t = (idx - i0) / (i1 - i0)
            dense[idx] = (1 - t) * sdf0 + t * sdf1 <= 0
    return dense


def interpolate_sparse_label_correction(keyframe_edits, original_labels):
    """Sparse per-plane hand corrections to a multi-label volume (e.g. Allen
    CCF region ids warped into sample space) -> a dense corrected volume,
    interpolating the correction itself between keyframes rather than
    overwriting whole planes -- see ../GT_tool_for_registration/edit_sample_labels.py, the
    interactive tool that produces `keyframe_edits`.

    keyframe_edits: {z: (edit_mask_2d, edited_labels_2d)} -- edit_mask_2d is
    a bool array marking only the pixels actually repainted on that plane
    (every other pixel on that plane, and every plane not in this dict, is
    left exactly as `original_labels`: this is a delta edit on top of the
    existing registration output, not a from-scratch resegmentation).
    edited_labels_2d gives the corrected label id at the painted pixels
    (values elsewhere are ignored).

    original_labels: full 3D label array (e.g. from labels_in_sample.nii.gz).

    Between two keyframe z's, each distinct label id present in either
    keyframe's edit gets its own signed-distance field (see
    interpolate_sparse_mask / _signed_distance), linearly blended by depth
    fraction; at each intermediate voxel, the label with the most-negative
    interpolated SDF wins (if none is negative -- i.e. no edited region's
    interpolated shape reaches that voxel -- the original label is kept
    untouched). This lets a correction that only appears at one keyframe
    naturally taper back to the original label towards the other keyframe,
    instead of extending indefinitely. Planes outside
    [min(keyframe_edits), max(keyframe_edits)] are left untouched, same as
    interpolate_sparse_mask.
    """
    corrected = original_labels.copy()
    indices = sorted(keyframe_edits)
    if not indices:
        return corrected

    for z in indices:
        edit_mask, edited_labels = keyframe_edits[z]
        corrected[z][edit_mask] = edited_labels[edit_mask]

    for z0, z1 in zip(indices[:-1], indices[1:]):
        if z1 - z0 <= 1:
            continue
        mask0, labels0 = keyframe_edits[z0]
        mask1, labels1 = keyframe_edits[z1]
        label_ids = set(np.unique(labels0[mask0])) | set(np.unique(labels1[mask1]))

        sdfs0 = {label: _signed_distance((labels0 == label) & mask0) for label in label_ids}
        sdfs1 = {label: _signed_distance((labels1 == label) & mask1) for label in label_ids}

        for z in range(z0 + 1, z1):
            t = (z - z0) / (z1 - z0)
            best_sdf = np.full(original_labels.shape[1:], np.inf)
            best_label = np.zeros(original_labels.shape[1:], dtype=original_labels.dtype)
            for label in label_ids:
                sdf = (1 - t) * sdfs0[label] + t * sdfs1[label]
                take = sdf < best_sdf
                best_sdf = np.where(take, sdf, best_sdf)
                best_label = np.where(take, label, best_label)
            inside = best_sdf <= 0
            corrected[z] = np.where(inside, best_label, original_labels[z])

    return corrected
