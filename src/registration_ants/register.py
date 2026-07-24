"""Registration to the Allen CCF: one-shot SyNRA, or coarse-to-fine two-tier."""
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
    the sample side via a hand-painted mask -- see mask_tools/paint_mask.py's `mask` kind --
    or generated automatically via brain_mask.generate_brain_mask, wired in
    through config.yaml's mask.auto_brain_mask).
    Files produced by those two helpers are already in this "nonzero=use"
    convention, no inversion needed here.

    guide_regions: optional list of (atlas_outline_img, sample_outline_img,
    weight) tuples -- for tissue that's genuinely present but too locally
    deformed for the intensity metric to align on its own (e.g. a bulged
    patch of cortex that keeps ending up mapped to background). This is the
    opposite of mask/moving_mask: instead of excluding a region, it adds an
    extra term that actively pulls the deformation to make the two outlines
    overlap, on top of the normal intensity metric -- see
    mask_tools/paint_mask.py's `guide` kind and scripts/project_outline.py
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
            multivariate_extras=extras,
        )
    else:
        reg = ants.registration(
            fixed=atlas_template,
            moving=sample_img,
            type_of_transform=type_of_transform,
            aff_metric="mattes",
            syn_metric="CC",
            outprefix=outprefix,
            verbose=verbose,
            mask=mask,
            moving_mask=moving_mask,
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


def register_to_allen_coarse_to_fine(
    sample_coarse_img, sample_fine_img, coarse_res_um=50, fine_res_um=25, outprefix="", verbose=True
):
    """Two-tier strategy: Rigid+Affine on a small coarse resample (fast, robust
    global alignment), then SyN refinement on the finer resample initialized
    from that affine. Optional — use when a direct SyNRA at fine_res_um isn't
    converging cleanly; otherwise register_to_allen() alone is simpler.

    Both sample_*_img must already be isotropic-resampled and preprocessed at
    their respective resolutions (see io_utils.convert_to_isotropic_nifti and
    preprocess.preprocess_for_registration).

    outprefix: applies to the fine-stage (SyN) transform files, the ones
    actually used downstream; the coarse-stage affine is an intermediate and
    always goes through a temp prefix.
    """
    coarse_template, _, _ = get_allen_atlas(coarse_res_um)
    reg_coarse = ants.registration(
        fixed=coarse_template, moving=sample_coarse_img, type_of_transform="Affine", verbose=verbose,
    )

    fine_template, fine_annotation, structures = get_allen_atlas(fine_res_um)
    reg_fine = ants.registration(
        fixed=fine_template,
        moving=sample_fine_img,
        type_of_transform="SyNOnly",
        initial_transform=reg_coarse["fwdtransforms"][0],
        syn_metric="CC",
        flow_sigma=3,
        total_sigma=0,
        outprefix=outprefix,
        verbose=verbose,
    )
    reg_fine["atlas_template"] = fine_template
    reg_fine["atlas_annotation"] = fine_annotation
    reg_fine["atlas_structures"] = structures
    reg_fine["coarse_transform"] = reg_coarse["fwdtransforms"][0]
    return reg_fine
