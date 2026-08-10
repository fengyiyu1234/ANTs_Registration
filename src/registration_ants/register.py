"""Registration to the Allen CCF: one-shot SyNRA."""
import re
import shutil
from pathlib import Path

import ants

from .atlas_utils import get_allen_atlas

_FWD_WARP_RE = re.compile(r"\d+Warp\.nii\.gz$")


def _repair_invtransforms_for_initial(reg, initial_inverse, outprefix):
    """Rebuild reg['invtransforms'] when an external initial_transform field
    was used. Mutates reg in place.

    Two separate problems, both silent:

    1. ANTs cannot invert a displacement field handed to it via -r, so the
       initial deformation is simply absent from invtransforms. Everything
       atlas->sample (labels_in_sample, cell region assignment) would then
       apply only the Affine+SyN part and land systematically off. Measured
       on a phantom: atlas->sample Dice 0.86 with the link missing vs 0.99
       with the separately-fitted inverse field in front. Hence
       fit_initial_transform.py always emits a matching inverse field, and
       it belongs at the FRONT of invtransforms (ANTs applies a transform
       list back-to-front, so the initial deformation must be undone last).

    2. antspyx's own list-building is wrong for this file layout. It globs
       the outprefix and drops exactly ONE forward-warp entry from
       invtransforms (`idx != findfwd[0]`), but with an initial transform
       there are two forward warps on disk -- {prefix}0Warp.nii.gz (the
       initial field, which ANTs re-writes as stage 0) and
       {prefix}2Warp.nii.gz (the SyN result). Only the first is dropped, so
       the SyN FORWARD warp is left sitting in invtransforms. Verified by
       replaying antspyx's glob logic on a real 3-transform output:
       invtransforms came back as [1GenericAffine.mat, 2InverseWarp.nii.gz,
       2Warp.nii.gz]. fwdtransforms is unaffected and was verified correct
       by exact replay against warpedmovout (max diff 0.0).

    Also copies the inverse field next to the other transforms as
    {prefix}0InverseWarp.nii.gz so transforms.load_saved_transforms() can
    reconstruct the same chain later from disk alone. Done after
    ants.registration() returns, never before -- an extra *InverseWarp file
    present during the call would change antspyx's own glob result.
    """
    kept = [t for t in reg["invtransforms"] if not _FWD_WARP_RE.search(str(t))]
    if outprefix:
        staged = f"{outprefix}0InverseWarp.nii.gz"
        if Path(initial_inverse).resolve() != Path(staged).resolve():
            shutil.copyfile(initial_inverse, staged)
        initial_inverse = staged
    reg["invtransforms"] = [initial_inverse] + kept


def register_to_atlas(sample_img, atlas_template, atlas_annotation, atlas_structures=None,
                       type_of_transform="SyNRA", outprefix="", verbose=True, mask=None, moving_mask=None,
                       guide_regions=None, syn_sampling=4, reg_iterations=(100, 70, 50),
                       prealign_moving_mask=None, initial_transform=None, initial_inverse=None):
    """Register a preprocessed, isotropic sample image to an already-loaded
    atlas template/annotation (from either get_allen_atlas or
    atlas_utils.load_custom_atlas — this function doesn't care which).

    fixed=atlas, moving=sample, so on the returned dict:
      reg['fwdtransforms']: warps sample -> atlas space
      reg['invtransforms']: warps atlas -> sample space
    reg also carries 'atlas_template' / 'atlas_annotation' / 'atlas_structures'
    so downstream transform application doesn't need to re-fetch the atlas.

    outprefix: where ANTs writes the transform files (.mat / Warp.nii.gz).
    Leave empty to use a temp directory (fine for one-off exploration); pass
    a real path prefix (e.g. "output_dir/transforms/mouse01_") to keep the
    transforms so they can be reused without re-running registration.

    verbose: ANTs prints per-iteration metric values directly to the
    process's stdout (this bypasses Python's own stdout object, so it only
    shows up if you redirect the whole process's output — see pipeline.py).
    Turn off only if the output is too noisy for your purposes.

    mask / moving_mask: ANTsImage, atlas-space / sample-space binary masks
    passed straight through to ants.registration() -- nonzero = used in the
    metric, 0 = excluded. Use for real structural mismatches the intensity
    metric shouldn't be forced to explain away (missing structures on the
    atlas side via atlas_utils.build_region_exclusion_mask, tears/damage on
    the sample side via a hand-painted mask -- see ../GT_tool_for_registration/paint_mask.py's `mask` kind --
    or generated automatically via brain_mask.generate_brain_mask, wired in
    through config.yaml's mask.auto_brain_mask).
    Files produced by those two helpers are already in this "nonzero=use"
    convention, no inversion needed here.
    prealign_moving_mask: a sample-space mask used ONLY for the coarse
    Translation pre-alignment described below, never for the main
    registration. This is where a whole-brain silhouette mask belongs.
    A silhouette passed as moving_mask instead actively PREVENTS the
    sample from being stretched to fill the atlas: ANTs only evaluates
    the metric at points that land inside the moving mask, so every bit
    of atlas territory the sample does not cover yet is silently dropped
    from the objective -- there is then no "expand to fill me" term at
    all, and leaving the sample small and centered scores just as well as
    growing it. Measured on a phantom matching a real failing case (sample
    squashed to 61% of atlas height, hemisphere atlas flush against its
    cut face, sample buffered on all sides): silhouette as moving_mask
    recovered 9% of the missing extent, moving it to prealign_moving_mask
    recovered 101%, with the initial alignment just as good (centre-of-mass
    error 0.00 mm either way). Most of that difference is the Affine stage
    rather than SyN -- a masked Affine will not scale up, which forces the
    deformation field to do work a global scale factor should have done.
    Reserve mask/moving_mask for genuine correspondence failures (missing
    structures via atlas_utils.build_region_exclusion_mask, tears via a
    hand-painted damage mask); those exclude a specific region rather than
    hiding all not-yet-covered territory, so they do not have this effect.

    initial_transform / initial_inverse: paths to a matched pair of
    displacement fields produced by scripts/fit_initial_transform.py from
    hand-placed landmark pairs -- for the case parameter tuning cannot
    reach, where the intensity metric has no way to establish
    correspondence at all (tissue distorted past recognition, a structure's
    boundary that simply is not visible in the sample). The forward field
    replaces the Translation pre-alignment as the starting pose AND
    starting deformation, so Affine/SyN refine from something anatomically
    informed rather than from a rigid guess. initial_inverse is not
    optional when initial_transform is used: ANTs cannot invert a field it
    was handed, so without it every atlas->sample product is silently wrong
    -- see _repair_invtransforms_for_initial for the measurements.

    Whenever mask or the pre-alignment mask is given (outside the
    guide_regions branch), a coarse mask-constrained Translation pass runs
    first and its result is fed in as initial_transform for the main
    type_of_transform call below --
    see the inline comment where it's built for why (ANTs' own default
    initializer ignores these masks, which can land the optimizer nowhere
    near the true alignment for oddly-shaped mask pairs, e.g. a
    zero-padded hemisphere atlas against a symmetrically-buffered sample).

    guide_regions: optional list of (atlas_outline_img, sample_outline_img,
    weight) tuples -- for tissue that's genuinely present but too locally
    deformed for the intensity metric to align on its own (e.g. a bulged
    patch of cortex that keeps ending up mapped to background). This is the
    opposite of mask/moving_mask: instead of excluding a region, it adds an
    extra term that actively pulls the deformation to make the two outlines
    overlap, on top of the normal intensity metric -- see
    ../GT_tool_for_registration/paint_mask.py's `guide` kind and scripts/project_outline.py
    for how to produce the outline pair. When given, this forces a two-stage
    registration (plain Rigid+Affine, then SyNOnly + multivariate_extras)
    regardless of type_of_transform, because ANTs' multivariate_extras is
    only supported for SyNOnly/antsRegistrationSyN* transforms. Validated on
    synthetic data before wiring in (see PROGRESS_LOG.md): pulled two
    deliberately-offset regions from Dice 0.18 to 0.92.

    syn_sampling: ants.registration()'s syn_sampling means DIFFERENT things
    depending on syn_metric -- histogram bins for mattes, but the local
    neighborhood RADIUS (in voxels) for CC, which is what this function
    always uses. Its antspyx default of 32 is the sensible bin count for
    mattes and a disastrous radius for CC: a 65^3 = 275k-voxel correlation
    window (1.3 mm across on a 20um grid) averages away exactly the local
    structure SyN needs to see, so the deformation stays near-global while
    each iteration costs minutes. 4 is what antsRegistrationSyN.sh and
    antspyx's own SyNCC preset use. Only meaningful while syn_metric is CC.

    reg_iterations: max iterations per SyN resolution level, coarsest
    first. Its length also sets the pyramid depth (antspyx derives
    smoothing sigmas len-1...0 and shrink factors 2^(len-1)...1 from it),
    so three entries means 4x / 2x / full resolution. antspyx's default
    (40, 20, 0) leaves the full-resolution level at zero iterations -- the
    deformation is then only ever optimized on a 2x-downsampled grid and
    upsampled -- and cuts the coarser levels off while the metric is still
    descending. Fine for a sample already close to the atlas, not for one
    needing large, spatially non-uniform deformation. These are caps, not
    fixed costs: a level exits early once its convergence value drops
    below ANTs' 1e-7 threshold.
    """
    if guide_regions:
        reg_affine = ants.registration(
            fixed=atlas_template, moving=sample_img, type_of_transform="Affine",
            aff_metric="mattes", verbose=verbose, mask=mask, moving_mask=moving_mask,
            mask_all_stages=True,
        )
        extras = [("MeanSquares", atlas_outline, sample_outline, weight, 0)
                  for atlas_outline, sample_outline, weight in guide_regions]
        reg = ants.registration(
            fixed=atlas_template,
            moving=sample_img,
            type_of_transform="SyNOnly",
            initial_transform=reg_affine["fwdtransforms"][0],
            syn_metric="CC",
            syn_sampling=syn_sampling,
            reg_iterations=reg_iterations,
            outprefix=outprefix,
            verbose=verbose,
            mask=mask,
            moving_mask=moving_mask,
            mask_all_stages=True,
            multivariate_extras=extras,
        )
    else:
        prealign_mm = prealign_moving_mask if prealign_moving_mask is not None else moving_mask
        if initial_transform is not None:
            # A landmark-derived field already fixes both position and gross
            # shape, and ants.registration() takes only one initial_transform,
            # so the Translation pre-alignment below would have nothing to add
            # and nowhere to go.
            pass
        elif mask is not None or prealign_mm is not None:
            # ANTs' own default initializer (used whenever initial_transform
            # is left unset) aligns whole-image intensity centroids WITHOUT
            # respecting mask/moving_mask -- fine when sample and atlas have
            # similar background extent, but a hemisphere atlas with zero
            # padding on its cut face vs. a sample with buffer on all sides
            # pulls that centroid guess far enough off-target that the
            # mask-constrained Affine/SyN metric below barely overlaps
            # afterward, leaving the optimizer stuck near that bad initial
            # pose. A coarse, mask-constrained translation pass first fixes
            # that -- few enough DOF to have a large capture range, and
            # orientation is already handled separately via atlas.orientation
            # in config, so translation alone is what's left to correct.
            reg_translation = ants.registration(
                fixed=atlas_template, moving=sample_img, type_of_transform="Translation",
                aff_metric="mattes", verbose=verbose, mask=mask, moving_mask=prealign_mm,
                mask_all_stages=True,
            )
            initial_transform = reg_translation["fwdtransforms"][0]

        reg = ants.registration(
            fixed=atlas_template,
            moving=sample_img,
            type_of_transform=type_of_transform,
            initial_transform=initial_transform,
            aff_metric="mattes",
            syn_metric="CC",
            syn_sampling=syn_sampling,
            reg_iterations=reg_iterations,
            outprefix=outprefix,
            verbose=verbose,
            mask=mask,
            moving_mask=moving_mask,
            # ants.registration() only applies mask/moving_mask to a
            # transform's LAST internal stage unless this is set -- for a
            # combined multi-stage preset like the default "SyNRA"
            # (Rigid+Affine+SyN built as ONE antsRegistration call), that
            # means the Rigid+Affine stages run completely UNMASKED by
            # default (verified in a real run.log: `-x [NA,NA]` for the
            # Rigid/Affine metric stages, real mask pointers only on SyN's).
            # That silently undoes the whole point of passing mask/
            # moving_mask in the first place -- the unmasked Affine stage
            # can drag a good masked-Translation initial alignment (above)
            # right back into a bad pose using the full unmasked buffer/atlas
            # extent.
            mask_all_stages=True,
        )
        if initial_inverse is not None:
            _repair_invtransforms_for_initial(reg, initial_inverse, outprefix)
    reg["atlas_template"] = atlas_template
    reg["atlas_annotation"] = atlas_annotation
    reg["atlas_structures"] = atlas_structures
    return reg


def register_to_allen(sample_img, atlas_res_um=25, type_of_transform="SyNRA", outprefix="", verbose=True,
                       mask=None, moving_mask=None, guide_regions=None,
                       syn_sampling=4, reg_iterations=(100, 70, 50)):
    """Register to the Allen CCF, auto-fetched via BrainGlobe at atlas_res_um.
    See register_to_atlas() for what the returned dict contains and what
    mask/moving_mask/guide_regions/syn_sampling/reg_iterations mean.
    """
    template, annotation, structures = get_allen_atlas(atlas_res_um)
    return register_to_atlas(sample_img, template, annotation, structures, type_of_transform, outprefix, verbose,
                              mask, moving_mask, guide_regions, syn_sampling, reg_iterations)
