"""Run the full registration pipeline from a YAML config.

Usage:
    conda activate antsreg
    ./run_pipeline.sh path/to/config.yaml

Logging: this module's own step markers go through `logging`, printed to the
console only (see _setup_logging below) -- NOT written directly to a file.
ANTs' own per-iteration convergence output (from verbose=True in register.py)
is printed directly by the underlying C++ library straight to the process's
stdout file descriptor, bypassing Python's logging/stdout objects entirely, so
neither this module's step markers nor ANTs' output can be captured by a
Python-side file handler alone -- both only end up in <output_dir>/run.log
because ../run_pipeline.sh redirects the whole process's output there via
`2>&1 | tee -a run.log` (useful for nohup/background runs too, so nothing is
lost if the terminal closes). Running `python -m registration_ants.pipeline`
directly (skipping the wrapper) still works but only prints to the console --
nothing is written to run.log.
"""
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import ants
import numpy as np

from . import atlas_utils, brain_mask, cell_points, io_utils, preprocess, register, transforms
from .config import load_config

logger = logging.getLogger("registration_ants.pipeline")

# Handlers are attached to the "registration_ants" package logger (not just
# .pipeline), so per-module loggers elsewhere (e.g. cell_points.py's
# per-class counts) propagate into the console too instead of only reaching
# it via print().
#
# No FileHandler here on purpose: run_pipeline.sh already redirects this
# process's whole stdout/stderr into <output_dir>/run.log via `tee`, which is
# also the only way to catch ANTs' own native stdout writes (see module
# docstring) -- adding a FileHandler here as well would duplicate every line
# this module logs (once written directly to the file, once via the tee
# capturing what got printed to the console).
_package_logger = logging.getLogger("registration_ants")


def _setup_logging():
    _package_logger.setLevel(logging.INFO)
    _package_logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    _package_logger.addHandler(stream_handler)


def _fmt_duration(seconds):
    return str(timedelta(seconds=round(seconds)))


def _preprocess(img, prep_cfg):
    lo, hi = prep_cfg["intensity_clip_percentiles"]
    if prep_cfg["n4_bias_correction"]:
        logger.info("Preprocessing: N4 bias correction + clip_and_normalize percentiles=(%s, %s), input shape=%s",
                    lo, hi, img.shape)
        return preprocess.preprocess_for_registration(img, lo, hi)
    logger.info("Preprocessing: intensity clip_and_normalize percentiles=(%s, %s), input shape=%s", lo, hi, img.shape)
    return preprocess.clip_and_normalize(img, lo, hi)


def _build_guide_regions_from_labels(guide_cfg, sample_ref, atlas_annotation, atlas_structures):
    """(atlas_region_img, sample_region_img, weight) triples from ONE
    hand-painted multi-label volume plus atlas region names.

    Only the sample side is drawn by hand; the atlas side comes out of the
    annotation, so a region never has to be painted twice.

    The atlas side is resolved by structure ID when `atlas_ids` is given
    (what paint_mask.py's ontology picker writes, alongside the names, into
    <output>.regions.json), and by substring-matched NAME otherwise. Prefer
    ids: name matching is a substring match, so it can quietly pull in an
    unrelated structure that merely contains the target as a substring --
    "Cerebellum" also matches "cerebellum related fiber tracts", a different
    top-level branch worth ~22% of Cerebellum's true descendant-voxel count
    (measured; see atlas_utils.region_mask_by_exact_name). With ids, the
    region paired here is exactly the one that was highlighted in the GUI
    while painting. `atlas_names` stays supported for hand-written configs
    and is the only option when there is no ontology id to hand.

    The painted volume lives on the RAW tiff's grid (that is the only grid
    where the structures are still resolvable by eye -- the isotropic
    resample throws away ~8x of the in-plane detail), and it carries no
    spacing in its header, so spacing is rebuilt from config and the volume
    is resampled onto whatever grid registration actually runs on. Physical
    space is shared across all of these (origin 0, spacing = microns,
    cropping only shifts origin), so the resample is a pure regrid.
    """
    regions = io_utils.load_nifti_stack_as_ants(guide_cfg["regions_mask"], guide_cfg["voxel_size_um"])
    if regions.shape != sample_ref.shape:
        regions = ants.resample_image_to_target(regions, sample_ref, interp_type="genericLabel")
    regions_arr = np.rint(regions.numpy()).astype(np.int32)
    painted = {int(v) for v in np.unique(regions_arr) if v != 0}

    # atlas_ids wins per-label where both are given, so a config can carry the
    # ids paint_mask.py exported for the regions it painted and still fall back
    # to hand-written names for anything added by hand afterwards.
    ids_cfg = guide_cfg.get("atlas_ids") or {}
    names_cfg = guide_cfg.get("atlas_names") or {}
    configured = sorted(set(ids_cfg) | set(names_cfg))

    unconfigured = painted - set(configured)
    if unconfigured:
        raise ValueError(
            f"mask.guide_regions.regions_mask contains painted label(s) {sorted(unconfigured)} with no "
            "atlas_ids/atlas_names entry -- they would be silently ignored. Add them, or remove them "
            "from the mask.")

    annotation_arr = atlas_annotation.numpy()
    weight_cfg = guide_cfg["weight"]
    out = []
    for label in configured:
        if label in ids_cfg:
            source = f"atlas_ids[{label}] = {ids_cfg[label]}"
            atlas_arr, matched = atlas_utils.build_region_inclusion_mask_by_ids(
                annotation_arr, atlas_structures, ids_cfg[label])
        else:
            source = f"atlas_names[{label}] = {names_cfg[label]}"
            atlas_arr, matched = atlas_utils.build_region_inclusion_mask(
                annotation_arr, atlas_structures, names_cfg[label])
        if not matched:
            # Matching that hits nothing yields an all-False mask and no error;
            # the registration would then run with a guide term that can never
            # be satisfied. Refuse instead. (An unknown *id* has already raised
            # inside descendant_ids_of by this point -- this is the case where
            # the structure is real but simply has no voxels in this
            # annotation, which is common: the DevCCF P04 annotation carries
            # 193 of the ontology's 2552 structures.)
            raise ValueError(
                f"mask.guide_regions.{source} matched no structure present in this atlas annotation. "
                "Names are case-insensitive substrings of ontology names (descendants included); ids "
                "are matched exactly, descendants included -- check against the atlas ontology.")
        sample_arr = regions_arr == label
        if not sample_arr.any():
            raise ValueError(
                f"mask.guide_regions.{source} is configured but nothing is painted with "
                f"label {label} in {guide_cfg['regions_mask']}.")
        weight = weight_cfg.get(label, 1.0) if isinstance(weight_cfg, dict) else weight_cfg
        logger.info("Guide region label %d (%s): sample %d voxels, atlas %d voxels from %d structure(s) "
                    "(%s), weight=%.2f",
                    label, source, int(sample_arr.sum()), int(atlas_arr.sum()), len(matched),
                    ", ".join(f"{n} ({c})" for n, c in sorted(matched.items(), key=lambda kv: -kv[1])[:6]),
                    weight)
        out.append((
            ants.from_numpy(atlas_arr.astype("float32"), spacing=atlas_annotation.spacing,
                            origin=atlas_annotation.origin, direction=atlas_annotation.direction),
            ants.from_numpy(sample_arr.astype("float32"), spacing=sample_ref.spacing,
                            origin=sample_ref.origin, direction=sample_ref.direction),
            weight,
        ))
    return out


def run_pipeline(config_path):
    config = load_config(config_path)
    sample = config["sample"]
    reg_cfg = config["registration"]
    prep_cfg = config["preprocess"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_prefix = str(output_dir / "transforms" / f"{sample['name']}_")
    Path(transforms_prefix).parent.mkdir(parents=True, exist_ok=True)

    _setup_logging()
    logger.info("Starting pipeline for sample '%s', config=%s", sample["name"], config_path)
    logger.info("Log file (only populated when run via run_pipeline.sh): %s", output_dir / "run.log")

    # (label, perf_counter() at the moment that step begins) -- consecutive
    # pairs give each step's wall-clock duration, logged in the timing
    # summary at the end so runs are easy to compare (e.g. after changing
    # type_of_transform or mask settings).
    step_marks = [("[1/6] resample", time.perf_counter())]
    logger.info("[1/6] Resampling fine level to %sum isotropic...", reg_cfg["fine_target_um"])
    fine_iso_path = output_dir / f"{sample['name']}_fine_{reg_cfg['fine_target_um']}um.nii.gz"
    # Load the raw stack once, at its own native (anisotropic) resolution --
    # reused below both for the full resample and, if crop_for_registration
    # is set, for the cropped one, instead of decoding the TIFF twice.
    raw_img = io_utils.load_tiff_stack_as_ants(sample["raw_tiff"], tuple(sample["voxel_size_um"]))

    # sample_fine is the UNCROPPED fine/isotropic image, always written to
    # this exact path -- napari viewers in ../ClearMap/stats_vis/ glob for
    # *_fine_*um.nii.gz as the "resample space" grid cell_registration.csv
    # coordinates are indexed into (see cell_points.py), so this must stay
    # the full grid even when crop_for_registration below only registers a
    # sub-region of it.
    sample_fine = io_utils.resample_to_isotropic(raw_img, reg_cfg["fine_target_um"])
    ants.image_write(sample_fine, str(fine_iso_path))
    logger.info("Resampled sample_fine: shape=%s spacing=%s, wrote %s",
                sample_fine.shape, sample_fine.spacing, fine_iso_path)

    sample_fine_for_reg = sample_fine
    crop_cfg = reg_cfg.get("crop_for_registration")
    if crop_cfg:
        # crop_cfg bounds are RAW TIFF voxel indices (x=column, y=row,
        # z=slice number -- same [x,y,z] convention as sample.voxel_size_um),
        # exactly what you'd read off the raw stack directly in an image
        # viewer, with no fine-grid resampling needed first to figure out
        # what to write in config. Cropped on raw_img BEFORE resampling
        # (not on sample_fine afterward), so there's no separate raw<->fine
        # index conversion to get wrong -- the numbers you read off the raw
        # stack are exactly the numbers that go in config.
        logger.info("Cropping raw stack for registration (raw-tiff voxel-index space): %s", crop_cfg)
        raw_img_cropped = io_utils.crop_to_bounds(
            raw_img, x=crop_cfg.get("x"), y=crop_cfg.get("y"), z=crop_cfg.get("z"),
        )
        sample_fine_for_reg = io_utils.resample_to_isotropic(raw_img_cropped, reg_cfg["fine_target_um"])
        cropped_path = output_dir / f"{sample['name']}_fine_{reg_cfg['fine_target_um']}um_cropped.nii.gz"
        ants.image_write(sample_fine_for_reg, str(cropped_path))
        logger.info("Cropped: raw shape=%s -> fine shape=%s, wrote %s",
                    raw_img_cropped.shape, sample_fine_for_reg.shape, cropped_path)

    step_marks.append(("[2/6] preprocess", time.perf_counter()))
    logger.info("[2/6] Preprocessing...")
    sample_fine_prep = _preprocess(sample_fine_for_reg, prep_cfg)

    step_marks.append(("[3/6] register", time.perf_counter()))
    logger.info("[3/6] Registering to atlas (source=%s)...", config["atlas"]["source"])
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
    prealign_moving_mask = None
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
        # suggestion is in sample_fine_prep's own voxel-index space (fine_target_um
        # grid, and already offset if crop_for_registration was set) -- convert
        # through physical space (sample_fine_prep.origin/spacing -> raw_img's,
        # which is always origin 0 at sample.voxel_size_um spacing) to get
        # raw-tiff voxel indices, since that's what crop_for_registration now
        # expects in config.
        raw_um = sample["voxel_size_um"]
        fine_origin, fine_spacing = sample_fine_prep.origin, sample_fine_prep.spacing
        raw_suggestion = [
            [round((fine_origin[axis] + lo * fine_spacing[axis]) / raw_um[axis]),
             round((fine_origin[axis] + hi * fine_spacing[axis]) / raw_um[axis])]
            for axis, (lo, hi) in enumerate(suggestion)
        ]
        logger.info("Suggested registration.crop_for_registration (raw-tiff voxel-index space, "
                    "paste directly into config): x=%s y=%s z=%s",
                    raw_suggestion[0], raw_suggestion[1], raw_suggestion[2])

        # The silhouette is handed to register_to_atlas as the PRE-ALIGNMENT
        # mask only, never as moving_mask -- see register_to_atlas's docstring
        # for the measurements, but in short: a silhouette used as moving_mask
        # hides every part of the atlas the sample does not cover yet, which
        # removes the only signal that would make the sample grow to fill it.
        # sample_damage_mask_path stays on moving_mask: it marks tissue that
        # genuinely has no counterpart, and is 1 nearly everywhere else, so it
        # does not hide un-covered atlas territory the way a silhouette does.
        prealign_moving_mask = auto_mask_img

    guide_regions = None
    guide_cfg = mask_cfg.get("guide_regions")
    if isinstance(guide_cfg, dict):
        guide_regions = _build_guide_regions_from_labels(
            guide_cfg, sample_fine_prep, atlas_annotation, atlas_structures)
    elif guide_cfg:
        guide_regions = []
        for gr in guide_cfg:
            atlas_outline = ants.image_read(gr["atlas_outline_path"])
            sample_outline = ants.image_read(gr["sample_outline_path"])
            guide_regions.append((atlas_outline, sample_outline, gr.get("weight", 1.0)))
        logger.info("Loaded %d guide region(s): %s", len(guide_regions),
                    [gr["sample_outline_path"] for gr in guide_cfg])

    init_cfg = reg_cfg.get("initial_transform") or {}
    initial_transform = init_cfg.get("path")
    initial_inverse = init_cfg.get("inverse_path")
    logger.info("Registration params: type_of_transform=%s, atlas_mask=%s, "
                "sample_mask=%s, prealign_mask=%s, initial_transform=%s, "
                "guide_regions=%d, syn_sampling(CC radius)=%d, reg_iterations=%s, outprefix=%s",
                reg_cfg["type_of_transform"], atlas_mask is not None, sample_mask is not None,
                prealign_moving_mask is not None, initial_transform or "none(auto prealign)",
                len(guide_regions or []),
                reg_cfg["syn_sampling"], reg_cfg["reg_iterations"], transforms_prefix)
    reg = register.register_to_atlas(
        sample_fine_prep, atlas_template, atlas_annotation, atlas_structures,
        type_of_transform=reg_cfg["type_of_transform"], outprefix=transforms_prefix,
        mask=atlas_mask, moving_mask=sample_mask, guide_regions=guide_regions,
        syn_sampling=reg_cfg["syn_sampling"], reg_iterations=reg_cfg["reg_iterations"],
        prealign_moving_mask=prealign_moving_mask,
        initial_transform=initial_transform, initial_inverse=initial_inverse,
    )

    logger.info("Registration complete: fwdtransforms=%s, invtransforms=%s",
                reg.get("fwdtransforms"), reg.get("invtransforms"))

    step_marks.append(("[4/6] apply_transforms", time.perf_counter()))
    logger.info("[4/6] Applying transforms (sample <-> atlas)...")
    sample_in_atlas = transforms.warp_sample_to_atlas(sample_fine_prep, reg)
    sample_in_atlas_path = output_dir / f"{sample['name']}_in_atlas.nii.gz"
    ants.image_write(sample_in_atlas, str(sample_in_atlas_path))
    logger.info("Wrote sample_in_atlas: shape=%s -> %s", sample_in_atlas.shape, sample_in_atlas_path)

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
    labels_in_sample_path = output_dir / f"{sample['name']}_labels_in_sample.nii.gz"
    ants.image_write(labels_in_sample, str(labels_in_sample_path))
    logger.info("Wrote labels_in_sample: shape=%s -> %s", labels_in_sample.shape, labels_in_sample_path)

    channels = sample.get("channels", [])
    step_marks.append(("[5/6] warp_channels", time.perf_counter()))
    logger.info("[5/6] Warping additional channels (%d configured)...", len(channels))
    for ch in channels:
        ch_img = io_utils.convert_to_isotropic_nifti(
            ch["raw_tiff"], tuple(ch["voxel_size_um"]), reg_cfg["fine_target_um"],
            output_dir / f"{ch['name']}_fine_{reg_cfg['fine_target_um']}um.nii.gz",
        )
        ch_in_atlas = transforms.warp_sample_to_atlas(ch_img, reg)
        ch_in_atlas_path = output_dir / f"{ch['name']}_in_atlas.nii.gz"
        ants.image_write(ch_in_atlas, str(ch_in_atlas_path))
        logger.info("Warped channel '%s': shape=%s -> %s", ch["name"], ch_in_atlas.shape, ch_in_atlas_path)

    step_marks.append(("[6/6] assign_cells", time.perf_counter()))
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

    step_marks.append(("done", time.perf_counter()))
    logger.info("Step timing summary (total %s):", _fmt_duration(step_marks[-1][1] - step_marks[0][1]))
    for (label, t), (_, t_next) in zip(step_marks, step_marks[1:]):
        logger.info("  %-24s %s", label, _fmt_duration(t_next - t))

    logger.info("Done. Outputs written to %s", output_dir)
    return reg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./run_pipeline.sh path/to/config.yaml "
              "(or: python -m registration_ants.pipeline path/to/config.yaml -- console-only, no run.log)")
        sys.exit(1)
    run_pipeline(sys.argv[1])
