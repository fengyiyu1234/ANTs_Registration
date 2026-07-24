"""Otsu-threshold based tissue/brain mask generation, for use as
ants.registration's `moving_mask` (nonzero = used in the metric). Useful when
the sample and atlas differ a lot in shape/background extent (e.g. a small P5
mouse sample vs. an adult-proportioned atlas), since it keeps the intensity
metric from being diluted by background voxels.

Ported from ClearMap.Alignment.BrainMask (same algorithm, same defaults) --
that module doubled as elastix's `-fMask` generator there. Operates on plain
numpy arrays (not ANTs images), matching atlas_utils.build_region_exclusion_mask's
convention -- callers wrap the result into an ants image themselves (see
pipeline.py), since the correct spacing/origin/direction to use depends on
which pipeline-stage image the mask is generated from.
"""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import ball, convex_hull_image


def generate_brain_mask(arr, sigma=1.0, closing_radius=3, dilate_radius=2, slice_axis=2):
    """Generate a binary tissue mask from a resampled brain image.

    Signal is often concentrated in cortex, so a plain Otsu threshold marks
    weakly-labeled interior structures as background, leaving a hollow
    cortical "shell" instead of a solid brain mask. To fix that, each 2D
    slice along `slice_axis` (default: 2 = z, matching this codebase's (x,y,z)
    array convention and the coronal-section stacking axis) is solidified
    with its convex hull -- i.e. the outermost outline of that section is
    connected and everything inside it is filled -- rather than relying on 3D
    hole-filling, which requires the shell to be fully closed in 3D and fails
    wherever it isn't.

    Arguments
    ---------
    arr : (X,Y,Z) array
        The image to segment (e.g. sample_fine_prep.numpy()).
    sigma : float
        Gaussian pre-smoothing sigma (voxels) applied before Otsu
        thresholding, to reduce noise-driven false positives.
    closing_radius : int
        Radius (voxels) of the structuring element used for binary closing,
        applied before the per-slice convex hull (cleans up small noise/gaps
        so they don't distort a slice's hull).
    dilate_radius : int
        Radius (voxels) of a final binary dilation, giving the mask a small
        safety margin so it doesn't clip real tissue near the boundary.
    slice_axis : int
        Axis to take 2D slices along for the convex-hull fill (0=x, 1=y,
        2=z). Should match the axis along which sections were acquired.

    Returns
    -------
    mask : array
        Binary mask, same shape as arr, dtype uint8 (0/1).
    bbox : tuple
        ((x0, x1), (y0, y1), (z0, z1)) bounding box of the mask's nonzero
        extent, in voxels of arr.
    """
    arr = np.asarray(arr, dtype=np.float32)

    smoothed = ndimage.gaussian_filter(arr, sigma=sigma)
    mask = smoothed > threshold_otsu(smoothed)

    if closing_radius > 0:
        mask = ndimage.binary_closing(mask, structure=ball(closing_radius))

    mask = np.moveaxis(mask, slice_axis, 0)
    for i, section in enumerate(mask):
        if section.any():
            mask[i] = convex_hull_image(section)
    mask = np.moveaxis(mask, 0, slice_axis)

    # Clean up any noise blobs the per-slice hull turned solid but that are
    # still disconnected from the main (largest) brain volume in 3D.
    labeled, n_components = ndimage.label(mask)
    if n_components > 1:
        sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    if dilate_radius > 0:
        mask = ndimage.binary_dilation(mask, structure=ball(dilate_radius))

    mask = mask.astype(np.uint8)
    return mask, _bounding_box(mask)


def _bounding_box(mask):
    """((x0,x1), (y0,y1), (z0,z1)) of the mask's nonzero extent."""
    nonzero = np.argwhere(mask)
    if nonzero.size == 0:
        return tuple((0, s) for s in mask.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    return tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def suggest_crop(bbox, shape, padding=10):
    """Pad `bbox` by `padding` voxels per side, clipped to `shape` -- a
    ready-to-paste suggestion for registration.crop_for_registration in
    config.yaml."""
    return [[max(0, lo - padding), min(size, hi + padding)]
            for (lo, hi), size in zip(bbox, shape)]
