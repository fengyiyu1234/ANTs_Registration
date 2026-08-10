"""Apply computed transforms: images (bidirectional) and cell-coordinate points.

All functions take `reg`, the dict returned by register.register_to_allen(...)
(carrying fwdtransforms/invtransforms plus the atlas template/annotation), so
callers don't need to separately track which transform list maps which way.
"""
from pathlib import Path

import ants


def _mat_entries_to_invert(transformlist):
    """[True for each .mat (linear) entry, False for warp/displacement
    fields] -- built explicitly instead of trusting ants.apply_transforms's
    own whichtoinvert=None auto-inference. Per that function's own
    docstring, the auto default only special-cases the exact 2-element
    [matrix, warp] pattern a compound SyN-style registration's invtransforms
    happens to produce (e.g. [affine.mat, InverseWarp.nii.gz] ->
    (True, False)); anything else -- including a bare single .mat, which is
    exactly what invtransforms is for a plain
    type_of_transform="Affine"/"Rigid"/"Translation"/... registration with
    no deformable stage -- silently falls through to all-False ("don't
    invert anything"), applying the forward matrix un-inverted.

    Verified against a real run: warp_labels_to_sample against an
    Affine-only registration's invtransforms came back 100% empty with the
    default (every voxel mapped outside the atlas_annotation's bounds);
    passing this explicit list instead recovers the expected ~30% nonzero.

    Only ever needed for invtransforms -- every .mat in fwdtransforms is
    already in the right (forward) direction and must stay un-inverted,
    which is what ants.apply_transforms/apply_transforms_to_points already
    default to regardless of list shape, so fwdtransforms callers below
    don't need this.
    """
    return [str(t).endswith(".mat") for t in transformlist]


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
        whichtoinvert=_mat_entries_to_invert(reg["invtransforms"]),
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
    # See _mat_entries_to_invert's docstring -- same fragile ants
    # whichtoinvert=None auto-inference applies here too (it's the same
    # underlying C++ logic apply_transforms_to_points shares with
    # apply_transforms). Only invtransforms (sample_to_atlas) needs this;
    # fwdtransforms is already correct un-inverted regardless.
    whichtoinvert = _mat_entries_to_invert(transformlist) if direction == "sample_to_atlas" else None
    return ants.apply_transforms_to_points(dim=3, points=points_df, transformlist=transformlist,
                                            whichtoinvert=whichtoinvert)


def load_saved_transforms(transforms_prefix):
    """Reconstruct {'fwdtransforms': [...], 'invtransforms': [...]} from the
    transform files a previous register_to_atlas(outprefix=transforms_prefix)
    call already wrote to disk -- for a separate process that never held the
    live `reg` dict (e.g. ../GT_tool_for_registration/registration_eval.py),
    instead of re-running
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
    # Two possible layouts, distinguished by whether registration ran with a
    # landmark-derived initial_transform (registration.initial_transform in
    # config). With one, ANTs shifts every stage index up by one and writes
    # the initial field as stage 0; register_to_atlas additionally stages the
    # matching inverse field as {prefix}0InverseWarp.nii.gz precisely so this
    # function can rebuild the same chain it returned live.
    init_fwd = f"{transforms_prefix}0Warp.nii.gz"
    init_inv = f"{transforms_prefix}0InverseWarp.nii.gz"
    if Path(init_fwd).exists():
        affine = f"{transforms_prefix}1GenericAffine.mat"
        warp = f"{transforms_prefix}2Warp.nii.gz"
        inverse_warp = f"{transforms_prefix}2InverseWarp.nii.gz"
        required = (init_fwd, init_inv, affine, warp, inverse_warp)
        fwd = [warp, affine, init_fwd]
        inv = [init_inv, affine, inverse_warp]
    else:
        affine = f"{transforms_prefix}0GenericAffine.mat"
        warp = f"{transforms_prefix}1Warp.nii.gz"
        inverse_warp = f"{transforms_prefix}1InverseWarp.nii.gz"
        required = (affine, warp, inverse_warp)
        fwd = [warp, affine]
        inv = [affine, inverse_warp]
    for f in required:
        if not Path(f).exists():
            raise FileNotFoundError(f"Expected transform file not found: {f}")
    return {"fwdtransforms": fwd, "invtransforms": inv}
