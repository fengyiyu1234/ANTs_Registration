"""Apply computed transforms: images (bidirectional) and cell-coordinate points.

All functions take `reg`, the dict returned by register.register_to_allen(...)
(carrying fwdtransforms/invtransforms plus the atlas template/annotation), so
callers don't need to separately track which transform list maps which way.
"""
from pathlib import Path

import ants


def warp_sample_to_atlas(sample_img, reg, interpolator="linear"):
    """Warp a sample-space image (e.g. a signal channel) into Allen atlas space."""
    return ants.apply_transforms(
        fixed=reg["atlas_template"], moving=sample_img,
        transformlist=reg["fwdtransforms"], interpolator=interpolator,
    )


def warp_labels_to_sample(sample_reference_img, reg):
    """Warp the Allen CCF annotation (region labels) into sample space.

    Uses genericLabel interpolation so discrete label ids are preserved —
    linear/B-spline interpolation would blend adjacent label ids into
    meaningless intermediate values.
    """
    return ants.apply_transforms(
        fixed=sample_reference_img, moving=reg["atlas_annotation"],
        transformlist=reg["invtransforms"], interpolator="genericLabel",
    )


def transform_cell_points(points_df, reg, direction="atlas_to_sample"):
    """Transform cell centroid coordinates through the registration's transforms.

    points_df: DataFrame with columns x, y, z in physical units (same units
        as the image spacing used during registration, typically microns).
    direction: 'atlas_to_sample' or 'sample_to_atlas'.

    ants.apply_transforms_to_points warps points in the OPPOSITE direction of
    ants.apply_transforms on images (see its own docstring: "point mapping
    goes the opposite direction of image mapping"). Since reg['fwdtransforms']
    warps an *image* sample->atlas and reg['invtransforms'] warps atlas
    ->sample, for *points* it's the reverse: fwdtransforms carries points
    atlas->sample, invtransforms carries points sample->atlas. Verified
    empirically against a real registration run: using fwdtransforms for
    sample_to_atlas put cells ~4.5mm away from where the same coordinate
    lands when warped through the image path (warp_sample_to_atlas), which
    is why essentially every cell was landing on atlas background before
    this was fixed.

    Far cheaper than warping a full-resolution label volume — use this for
    region-based cell counting instead of upsampling labels_in_sample back to
    native resolution.
    """
    transformlist = reg["fwdtransforms"] if direction == "atlas_to_sample" else reg["invtransforms"]
    return ants.apply_transforms_to_points(dim=3, points=points_df, transformlist=transformlist)


def load_saved_transforms(transforms_prefix):
    """Reconstruct {'fwdtransforms': [...], 'invtransforms': [...]} from the
    transform files a previous register_to_atlas(outprefix=transforms_prefix)
    call already wrote to disk -- for a separate process that never held the
    live `reg` dict (e.g. registration_eval.py), instead of re-running
    registration just to get the transform lists back.

    Same file-naming assumption already relied on in
    scripts/project_outline.py: a single-stage SyNRA/SyNOnly
    registration writes exactly {prefix}0GenericAffine.mat +
    {prefix}1Warp.nii.gz + {prefix}1InverseWarp.nii.gz (ANTs' standard
    naming), and fwdtransforms/invtransforms are built from those in the same
    order ants.registration() itself returns for that setup. NOT verified for
    a guide_regions-based two-stage registration (register_to_atlas's
    guide_regions branch) -- check the actual files written for that
    outprefix before trusting this helper for such a sample.
    """
    affine = f"{transforms_prefix}0GenericAffine.mat"
    warp = f"{transforms_prefix}1Warp.nii.gz"
    inverse_warp = f"{transforms_prefix}1InverseWarp.nii.gz"
    for f in (affine, warp, inverse_warp):
        if not Path(f).exists():
            raise FileNotFoundError(f"Expected transform file not found: {f}")
    return {
        "fwdtransforms": [warp, affine],
        "invtransforms": [affine, inverse_warp],
    }
