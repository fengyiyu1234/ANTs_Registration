"""Registration to the Allen CCF: one-shot SyNRA."""
import ants

from .atlas_utils import get_allen_atlas


def register_to_atlas(sample_img, atlas_template, atlas_annotation, atlas_structures=None,
                       type_of_transform="SyNRA", outprefix="", verbose=True, mask=None, moving_mask=None,
                       guide_regions=None):
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
    Whenever either mask is given (outside the guide_regions branch), a
    coarse mask-constrained Translation pass runs first and its result is
    fed in as initial_transform for the main type_of_transform call below --
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
            outprefix=outprefix,
            verbose=verbose,
            mask=mask,
            moving_mask=moving_mask,
            mask_all_stages=True,
            multivariate_extras=extras,
        )
    else:
        initial_transform = None
        if mask is not None or moving_mask is not None:
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
                aff_metric="mattes", verbose=verbose, mask=mask, moving_mask=moving_mask,
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
    reg["atlas_template"] = atlas_template
    reg["atlas_annotation"] = atlas_annotation
    reg["atlas_structures"] = atlas_structures
    return reg


def register_to_allen(sample_img, atlas_res_um=25, type_of_transform="SyNRA", outprefix="", verbose=True,
                       mask=None, moving_mask=None, guide_regions=None):
    """Register to the Allen CCF, auto-fetched via BrainGlobe at atlas_res_um.
    See register_to_atlas() for what the returned dict contains and what
    mask/moving_mask/guide_regions mean.
    """
    template, annotation, structures = get_allen_atlas(atlas_res_um)
    return register_to_atlas(sample_img, template, annotation, structures, type_of_transform, outprefix, verbose,
                              mask, moving_mask, guide_regions)
