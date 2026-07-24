"""Run the full registration pipeline from a YAML config.

Usage:
    conda activate antsreg
    python -m registration_ants.pipeline path/to/config.yaml

Logging: this module's own step markers go through `logging`, written to both
the console and <output_dir>/run.log. ANTs' own per-iteration convergence
output (from verbose=True in register.py) is printed directly by the
underlying C++ library and bypasses Python's logging/stdout objects, so it
only ends up in run.log if you also redirect the whole process's output, e.g.:
    python -m registration_ants.pipeline configs/s12t.yaml 2>&1 | tee -a run_s12t.log
(useful for nohup/background runs, so nothing is lost if the terminal closes)
"""
import logging
import sys
from pathlib import Path

import ants

from . import atlas_utils, brain_mask, cell_points, io_utils, preprocess, register, transforms
from .config import load_config

logger = logging.getLogger("registration_ants.pipeline")


def _setup_logging(output_dir):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def _preprocess(img, prep_cfg):
    if prep_cfg["n4_bias_correction"]:
        return preprocess.preprocess_for_registration(img)
    lo, hi = prep_cfg["intensity_clip_percentiles"]
    return preprocess.clip_and_normalize(img, lo, hi)


def run_pipeline(config_path):
    config = load_config(config_path)
    sample = config["sample"]
    reg_cfg = config["registration"]
    prep_cfg = config["preprocess"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_prefix = str(output_dir / "transforms" / f"{sample['name']}_")
    Path(transforms_prefix).parent.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_dir)
    logger.info("Starting pipeline for sample '%s', config=%s", sample["name"], config_path)
    logger.info("Log file: %s", output_dir / "run.log")

    logger.info("[1/6] Resampling fine level to %sum isotropic...", reg_cfg["fine_target_um"])
    fine_iso_path = output_dir / f"{sample['name']}_fine_{reg_cfg['fine_target_um']}um.nii.gz"
    # sample_fine is the UNCROPPED fine/isotropic image, always written to
    # this exact path -- napari viewers in ../ClearMap/stats_vis/ glob for
    # *_fine_*um.nii.gz as the "resample space" grid cell_registration.csv
    # coordinates are indexed into (see cell_points.py), so this must stay
    # the full grid even when crop_for_registration below only registers a
    # sub-region of it.
    sample_fine = io_utils.convert_to_isotropic_nifti(
        sample["raw_tiff"], tuple(sample["voxel_size_um"]), reg_cfg["fine_target_um"], fine_iso_path,
    )

    sample_fine_for_reg = sample_fine
    crop_cfg = reg_cfg.get("crop_for_registration")
    if crop_cfg:
        logger.info("Cropping fine image for registration: %s", crop_cfg)
        sample_fine_for_reg = io_utils.crop_to_bounds(
            sample_fine, x=crop_cfg.get("x"), y=crop_cfg.get("y"), z=crop_cfg.get("z"),
        )
        cropped_path = output_dir / f"{sample['name']}_fine_{reg_cfg['fine_target_um']}um_cropped.nii.gz"
        ants.image_write(sample_fine_for_reg, str(cropped_path))
        logger.info("Cropped: %s -> %s, wrote %s", sample_fine.shape, sample_fine_for_reg.shape, cropped_path)

    logger.info("[2/6] Preprocessing (N4 + intensity normalization)...")
    sample_fine_prep = _preprocess(sample_fine_for_reg, prep_cfg)

    logger.info("[3/6] Registering to atlas (source=%s)...", config["atlas"]["source"])
    if reg_cfg["use_coarse_to_fine"]:
        # Coarse-to-fine doesn't support masks yet (config.py already rejects
        # this combined with atlas.source: custom; brainglobe-only for now).
        coarse_iso_path = output_dir / f"{sample['name']}_coarse_{reg_cfg['coarse_target_um']}um.nii.gz"
        sample_coarse = io_utils.convert_to_isotropic_nifti(
            sample["raw_tiff_coarse"], tuple(sample["voxel_size_coarse_um"]),
            reg_cfg["coarse_target_um"], coarse_iso_path,
        )
        sample_coarse_prep = _preprocess(sample_coarse, prep_cfg)
        reg = register.register_to_allen_coarse_to_fine(
            sample_coarse_prep, sample_fine_prep,
            coarse_res_um=reg_cfg["coarse_atlas_res_um"], fine_res_um=reg_cfg["atlas_res_um"],
            outprefix=transforms_prefix,
        )
    else:
        # Resolve atlas objects the same way regardless of source, so mask
        # building below (which needs atlas_annotation/atlas_structures) and
        # the register_to_atlas() call don't need source-specific branches.
        atlas_cfg = config["atlas"]
        if atlas_cfg["source"] == "custom":
            # prepare_custom_atlas transparently falls back to
            # load_custom_atlas's plain-load behavior when orientation and
            # slicing are both unset (e.g. the DeMBA P5 files, already
            # pre-oriented/cropped elsewhere) -- only reorients/crops (and
            # caches the result) when those fields are actually given.
            atlas_template, atlas_annotation = atlas_utils.prepare_custom_atlas(
                atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
                orientation=atlas_cfg.get("orientation"), slicing=atlas_cfg.get("slicing"),
            )
            atlas_structures = (
                atlas_utils.load_ccf_ontology_json(atlas_cfg["ontology_path"])
                if "ontology_path" in atlas_cfg else None
            )
        else:
            atlas_template, atlas_annotation, atlas_structures = atlas_utils.get_allen_atlas(reg_cfg["atlas_res_um"])

        mask_cfg = config["mask"]
        atlas_mask = None
        exclude_regions = mask_cfg.get("atlas_exclude_regions") or []
        if exclude_regions:
            keep = atlas_utils.build_region_exclusion_mask(atlas_annotation.numpy(), atlas_structures, exclude_regions)
            atlas_mask = ants.from_numpy(keep.astype("float32"), spacing=atlas_annotation.spacing,
                                          origin=atlas_annotation.origin, direction=atlas_annotation.direction)
            logger.info("Atlas exclusion mask for %s: %d/%d voxels excluded",
                        exclude_regions, int((~keep).sum()), keep.size)

        sample_mask = None
        if mask_cfg.get("sample_damage_mask_path"):
            sample_mask = ants.image_read(mask_cfg["sample_damage_mask_path"])
            logger.info("Loaded sample damage mask: %s", mask_cfg["sample_damage_mask_path"])

        auto_brain_mask_cfg = mask_cfg.get("auto_brain_mask")
        if auto_brain_mask_cfg:
            params = auto_brain_mask_cfg if isinstance(auto_brain_mask_cfg, dict) else {}
            mask_arr, bbox = brain_mask.generate_brain_mask(sample_fine_prep.numpy(), **params)
            auto_mask_img = ants.from_numpy(mask_arr.astype("float32"), spacing=sample_fine_prep.spacing,
                                             origin=sample_fine_prep.origin, direction=sample_fine_prep.direction)
            mask_out_path = output_dir / f"{sample['name']}_brain_mask.nii.gz"
            ants.image_write(auto_mask_img, str(mask_out_path))

            suggestion = brain_mask.suggest_crop(bbox, mask_arr.shape)
            logger.info("Auto brain mask: coverage=%.1f%%, bbox=%s, wrote %s", 100 * mask_arr.mean(), bbox, mask_out_path)
            logger.info("Suggested registration.crop_for_registration (padded, on this fine_target_um grid): "
                        "x=%s y=%s z=%s", suggestion[0], suggestion[1], suggestion[2])

            if sample_mask is None:
                sample_mask = auto_mask_img
            else:
                # Combine with the hand-painted damage mask: a voxel must be
                # both inside the auto brain silhouette AND not excluded by
                # the damage mask to be used in the metric.
                combined = (sample_mask.numpy() > 0) & (mask_arr > 0)
                sample_mask = ants.from_numpy(combined.astype("float32"), spacing=sample_fine_prep.spacing,
                                               origin=sample_fine_prep.origin, direction=sample_fine_prep.direction)
                logger.info("Combined auto brain mask with sample_damage_mask_path")

        guide_regions = None
        if mask_cfg.get("guide_regions"):
            guide_regions = []
            for gr in mask_cfg["guide_regions"]:
                atlas_outline = ants.image_read(gr["atlas_outline_path"])
                sample_outline = ants.image_read(gr["sample_outline_path"])
                guide_regions.append((atlas_outline, sample_outline, gr.get("weight", 1.0)))
            logger.info("Loaded %d guide region(s): %s", len(guide_regions),
                        [gr["sample_outline_path"] for gr in mask_cfg["guide_regions"]])

        reg = register.register_to_atlas(
            sample_fine_prep, atlas_template, atlas_annotation, atlas_structures,
            type_of_transform=reg_cfg["type_of_transform"], outprefix=transforms_prefix,
            mask=atlas_mask, moving_mask=sample_mask, guide_regions=guide_regions,
        )

    logger.info("[4/6] Applying transforms (sample <-> atlas)...")
    sample_in_atlas = transforms.warp_sample_to_atlas(sample_fine_prep, reg)
    ants.image_write(sample_in_atlas, str(output_dir / f"{sample['name']}_in_atlas.nii.gz"))

    # Reference = the UNCROPPED sample_fine grid, not sample_fine_prep --
    # apply_transforms only uses its `fixed` argument as an output-grid
    # template (pixel values don't matter for a label warp), so this is free
    # to differ from whatever grid was actually registered against, and doing
    # so gives full-brain label coverage even when crop_for_registration only
    # registered a sub-region. This also keeps labels_in_sample.nii.gz on the
    # exact same grid as cell_registration.csv's "resample space" columns
    # (see cell_points.py) -- a constraint scripts/relabel_cells.py
    # already depends on (it looks up corrected labels via those columns with
    # no rescaling).
    labels_in_sample = transforms.warp_labels_to_sample(sample_fine, reg)
    ants.image_write(labels_in_sample, str(output_dir / f"{sample['name']}_labels_in_sample.nii.gz"))

    logger.info("[5/6] Warping additional channels (if any)...")
    for ch in sample.get("channels", []):
        ch_img = io_utils.convert_to_isotropic_nifti(
            ch["raw_tiff"], tuple(ch["voxel_size_um"]), reg_cfg["fine_target_um"],
            output_dir / f"{ch['name']}_fine_{reg_cfg['fine_target_um']}um.nii.gz",
        )
        ch_in_atlas = transforms.warp_sample_to_atlas(ch_img, reg)
        ants.image_write(ch_in_atlas, str(output_dir / f"{ch['name']}_in_atlas.nii.gz"))

    logger.info("[6/6] Assigning cell regions (if configured)...")
    if "cells" in config:
        cells_cfg = config["cells"]
        classes = cell_points.assign_cell_regions(
            cells_cfg["cell_centroids_dir"], output_dir, tuple(cells_cfg["voxel_size_um"]),
            sample_fine, reg, atlas_structures=reg.get("atlas_structures"), prefix=cells_cfg["prefix"],
        )
        logger.info("Cell regions assigned for %d class(es): %s", len(classes), classes)
    else:
        logger.info("No cells: block in config -- skipping.")

    logger.info("Done. Outputs written to %s", output_dir)
    return reg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m registration_ants.pipeline path/to/config.yaml")
        sys.exit(1)
    run_pipeline(sys.argv[1])
