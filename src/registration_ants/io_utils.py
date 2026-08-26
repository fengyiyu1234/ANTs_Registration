"""Raw TIFF stacks -> isotropic ANTs images / NIfTI, handling anisotropic voxels.

ANTs/ITK registration operates in physical space (spacing/origin/direction in
the image header), so anisotropic voxels are not inherently a problem for the
algorithm. Resampling to an isotropic grid here is done to match the Allen CCF
atlas grid and keep data volume manageable, not because ANTs requires it.
"""
from pathlib import Path

import ants
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


_LABEL_DTYPE_NOTE = """Why atlas annotations must not be cast to float32.

An annotation volume looks like an image but is not one: each voxel holds an
ontology structure ID, a label rather than a measurement. float32 keeps only 24
bits of integer precision, so every ID above 2**24 = 16,777,216 is snapped to
the nearest representable value -- and CCFv3 has 127 such IDs, going up to
614,454,277. In the DeMBA P5 annotation that collapses 24 distinct float32
values, e.g. 484682516 (corpus callosum, body), 484682520 (optic radiation),
484682524 (auditory radiation) and 484682528 (commissural branch of stria
terminalis) all land on 484682528.0. Once that has happened the four are
indistinguishable, np.isin(annotation, [484682516]) matches nothing, and a
guide mask built from the ontology silently loses the whole corpus callosum
body (measured: 198,512 voxels, which is what made the callosum look like
three disconnected pieces in a horizontal view).

This is not a hypothetical: the copy of this atlas shipped with ClearMap had
already been damaged this way on disk, and DeMBA's own data descriptor warns
that elastix does the same thing (their workaround is to renumber IDs to small
sequential integers before warping and back afterwards).

Keeping the integer dtype on load is necessary but not sufficient: ANTs/ITK
itself is float32 throughout, so an annotation handed to ants.resample_image
or ants.apply_transforms comes back rounded regardless of the dtype it went in
as. Two kinds of consumer therefore exist here:

  - id lookups on the numpy array (pipeline._build_guide_regions_from_labels,
    build_region_exclusion_mask). These never touch ANTs -- it is the
    resulting BINARY mask, not the annotation, that reaches registration --
    so loading without the cast is all they need.
  - warping the annotation itself (transforms.warp_labels_to_sample). This
    must renumber the ids to 0..N-1 before the warp and map back after, the
    same workaround DeMBA used for elastix. Any new code that pushes the
    annotation through ANTs needs to do likewise."""


def load_tiff_stack_as_ants(raw_path, voxel_size_xyz, preserve_labels=False):
    """Read a TIFF stack and wrap it as an ANTs image with correct spacing.

    raw_path: path to a multi-page TIFF / BigTIFF volume.
    voxel_size_xyz: (sx, sy, sz) in microns, matching the physical x/y/z axes.
    preserve_labels: keep the file's own integer dtype instead of casting to
        float32 -- see _LABEL_DTYPE_NOTE. Pass True for an atlas ANNOTATION,
        leave False for anything whose voxels are intensities.

    tifffile returns array axis order (z, y, x); this is transposed to
    (x, y, z) so it lines up with the (sx, sy, sz) spacing order ANTs expects.
    """
    arr_zyx = tifffile.imread(str(raw_path))
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    if not preserve_labels:
        arr_xyz = arr_xyz.astype(np.float32)
    sx, sy, sz = voxel_size_xyz
    return ants.from_numpy(np.ascontiguousarray(arr_xyz), spacing=(sx, sy, sz))


def load_nifti_stack_as_ants(raw_path, voxel_size_xyz, preserve_labels=False):
    """Read a NIfTI volume and wrap it as an ANTs image with correct spacing,
    discarding the file's own header spacing/direction/origin.

    Unlike this codebase's own outputs, a NIfTI file from an external source
    (e.g. DevCCF's downloads) typically has spacing in mm and/or a non-identity
    direction matrix -- but every image built in this codebase assumes identity
    direction/origin and a spacing value that's directly microns (see
    crop_to_bounds's docstring). Trusting the file's own header here would
    silently misalign or rescale the registration, so it's never read.

    raw_path: path to a .nii/.nii.gz volume.
    voxel_size_xyz: (sx, sy, sz) in microns, matching the physical x/y/z axes.
    preserve_labels: read the file as unsigned integers instead of float32 --
        see _LABEL_DTYPE_NOTE. Pass True for an atlas ANNOTATION, leave False
        for anything whose voxels are intensities. The cast has to be avoided
        inside ants.image_read, not after it: image_read defaults to
        pixeltype='float', so by the time .numpy() hands the array back the
        large IDs have already been rounded.

    ants.image_read(...).numpy() returns the array in the same on-disk (i,j,k)
    order as e.g. nibabel's dataobj -- no reorientation happens on read, so
    (unlike the TIFF path) no transpose is needed here to reach (x,y,z) order.
    """
    pixeltype = "unsigned int" if preserve_labels else "float"
    arr_xyz = ants.image_read(str(raw_path), pixeltype=pixeltype).numpy()
    sx, sy, sz = voxel_size_xyz
    return ants.from_numpy(arr_xyz, spacing=(sx, sy, sz))


def resample_to_isotropic(img, target_um):
    """Resample an anisotropic ANTs image onto an isotropic grid at target_um.

    Axes finer than target_um are being downsampled: Gaussian-smoothed first
    (anti-alias) to avoid aliasing artifacts. Axes coarser than target_um are
    only being interpolated onto a finer grid — no smoothing needed, and this
    does not fabricate detail the original axis didn't have.
    """
    spacing = np.array(img.spacing, dtype=float)
    sigma = np.array([0.5 * (target_um / s) if s < target_um else 0.0 for s in spacing])
    if sigma.any():
        arr = gaussian_filter(img.numpy(), sigma=sigma)
        img = ants.from_numpy(arr, spacing=tuple(spacing), origin=img.origin, direction=img.direction)
    return ants.resample_image(img, (target_um,) * 3, use_voxels=False, interp_type=4)


def convert_to_isotropic_nifti(raw_path, voxel_size_xyz, target_um, out_path):
    """Raw TIFF stack -> isotropic NIfTI at target_um, written to out_path."""
    img = load_tiff_stack_as_ants(raw_path, voxel_size_xyz)
    img_iso = resample_to_isotropic(img, target_um)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(img_iso, str(out_path))
    return img_iso


def crop_bounds_to_grid(crop_cfg, raw_voxel_size_um, grid_spacing_um, grid_shape):
    """registration.crop_for_registration bounds (raw-tiff voxel-index space,
    x/y/z keys, each [start,end] or None/missing) -> [(lo,hi), ...] voxel-index
    bounds (half-open, clipped to grid_shape) on a DIFFERENT grid that shares
    the same physical origin (0) -- e.g. the full uncropped sample_fine grid,
    which crop_for_registration's own indices are never expressed in.

    Both grids start at physical origin 0 (raw_img and sample_fine both do --
    see load_tiff_stack_as_ants / resample_to_isotropic, neither ever passes a
    nonzero origin), so converting is a plain unit change: voxel_index *
    this_grid's_spacing = physical position = other_grid's_voxel_index *
    other_grid's_spacing.
    """
    bounds = []
    for axis, key in enumerate(("x", "y", "z")):
        spec = crop_cfg.get(key)
        lo_raw = 0 if not spec or spec[0] is None else spec[0]
        hi_raw = None if not spec or spec[1] is None else spec[1]
        lo = int(round(lo_raw * raw_voxel_size_um[axis] / grid_spacing_um))
        hi = grid_shape[axis] if hi_raw is None else int(round(hi_raw * raw_voxel_size_um[axis] / grid_spacing_um))
        bounds.append((max(0, lo), min(grid_shape[axis], hi)))
    return bounds


def crop_to_bounds(img, x=None, y=None, z=None):
    """Crop an ANTs image to explicit voxel index ranges per axis (any axis
    left as None keeps its full extent), shifting the cropped image's origin
    by the crop's start offset so it occupies the exact same physical-space
    sub-region as before cropping.

    This matters because every registration/point-transform in this codebase
    (ants.registration, ants.apply_transforms_to_points) works in absolute
    physical space, not array indices -- a transform computed with this
    cropped image as the moving/fixed image stays valid for points anywhere
    in the ORIGINAL (uncropped) image's physical space, with no separate
    crop-offset bookkeeping needed downstream (unlike ClearMap's manual
    CROP_OFFSET correction, which was only needed because its arrays carry no
    physical-space memory of their own). Verified empirically that
    ants.transform_physical_point_to_index against the cropped image
    correctly accounts for the shifted origin.

    x/y/z: (start, stop) voxel index pairs in img's own grid, or None to keep
    the full range on that axis -- matches the shape of the
    registration.crop_for_registration config block.
    """
    shape = img.shape
    bounds = []
    starts = []
    for axis, spec in enumerate((x, y, z)):
        if spec is None:
            bounds.append(slice(None))
            starts.append(0)
        else:
            start = 0 if spec[0] is None else int(spec[0])
            stop = shape[axis] if spec[1] is None else int(spec[1])
            bounds.append(slice(start, stop))
            starts.append(start)

    cropped_arr = np.ascontiguousarray(img.numpy()[tuple(bounds)])
    # Elementwise origin shift assumes identity direction, true for every
    # image built in this codebase (ants.from_numpy is never called with a
    # non-default direction anywhere -- see load_tiff_stack_as_ants,
    # atlas_utils.get_allen_atlas/load_custom_atlas).
    new_origin = tuple(np.array(img.origin) + np.array(starts) * np.array(img.spacing))
    return ants.from_numpy(cropped_arr, spacing=img.spacing, origin=new_origin, direction=img.direction)
