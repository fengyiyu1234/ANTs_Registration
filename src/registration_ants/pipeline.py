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
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import ants
import numpy as np
import SimpleITK as sitk

from . import (atlas_utils, brain_mask, cell_points, io_utils, preprocess, register,
               reposition, transforms)
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


def _build_guide_regions_from_labels(guide_cfg, sample_ref, atlas_annotation, atlas_structures,
                                     reposition_plan=None, reposition_fragments=None):
    """(guide_triples, damage_hole): (atlas_region_img, sample_region_img,
    weight) triples from ONE hand-painted multi-label volume plus atlas region
    names, and a bool array (registration grid, or None) of voxels painted
    with a damage_labels label -- sample tissue with no atlas counterpart,
    for the caller to punch out of moving_mask.

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
    # The outlines were traced on the geometry the plan is about to change, so
    # they move with it -- otherwise every region painted over a flap would
    # keep pointing at where the flap used to be.
    regions = _reposition_volume(regions, reposition_plan, reposition_fragments,
                                 "labels", "the guide regions mask")
    if regions.shape != sample_ref.shape:
        regions = ants.resample_image_to_target(regions, sample_ref, interp_type="genericLabel")
    regions_arr = np.rint(regions.numpy()).astype(np.int32)
    painted = {int(v) for v in np.unique(regions_arr) if v != 0}

    # atlas_ids wins per-label where both are given, so a config can carry the
    # ids paint_mask.py exported for the regions it painted and still fall back
    # to hand-written names for anything added by hand afterwards.
    ids_cfg = guide_cfg.get("atlas_ids") or {}
    names_cfg = guide_cfg.get("atlas_names") or {}

    # Third, lowest-priority source: the ids paint_mask.py itself recorded in
    # <regions_mask>.regions.json when this label was painted. A label absent
    # from both atlas_ids and atlas_names falls back to this instead of
    # erroring, so a config only has to spell out atlas_ids for the labels it
    # wants to hand-override -- everything else is read straight from the
    # mask's own sidecar instead of a copy of it that can drift out of sync.
    sidecar_path = atlas_utils.regions_sidecar_path(guide_cfg["regions_mask"])
    sidecar_ids = atlas_utils.load_regions_sidecar_ids(guide_cfg["regions_mask"])
    for label, ids in sidecar_ids.items():
        if label in ids_cfg and set(ids_cfg[label]) != set(ids):
            logger.warning(
                "Guide region label %d: mask.guide_regions.atlas_ids=%s does not match "
                "%s's region_ids=%s for this label -- the mask was repainted/re-picked after "
                "the config was written, or the override is deliberate. Using the config value.",
                label, ids_cfg[label], sidecar_path.name, ids)

    # A painted label with no atlas pairing is normally a config omission, and
    # skipping it silently would mean hours of registration quietly ignoring
    # work that was drawn by hand. ignore_labels is the way to say "yes, that
    # one is deliberate" -- keeping the error for everything not named, so a
    # forgotten label still stops the run. Common reason to skip one: the
    # drawn extent is systematically off from the atlas structure's (e.g. a
    # thin sheet drawn over only part of its length), which pulls the
    # deformation the wrong way rather than helping.
    #
    # Subtracted out of `configured` too, not just used to excuse it from the
    # unconfigured check below: the sidecar now supplies a fallback id for
    # EVERY painted label (including ones ignore_labels deliberately opts
    # out of), so without this an ignored label would still get built into a
    # guide pair.
    ignored = {int(v) for v in (guide_cfg.get("ignore_labels") or [])}
    # damage_labels: painted regions that mark sample tissue with NO atlas
    # counterpart (e.g. a sliver of contralateral tissue past the midline on a
    # hemisphere sample). Not built into a guide pair; instead returned as a
    # hole for the caller to merge into moving_mask, same semantics as
    # mask.sample_damage_mask_path. Painting them here instead of a separate
    # damage file gets paint_mask.py's per-label keyframe interpolation and
    # keeps everything in one painting session. The sidecar's own
    # damage_labels (labels assigned to the GUI's "damage / no atlas
    # counterpart" pseudo-region at export) are unioned in, same
    # nothing-to-keep-in-sync rationale as the region_ids fallback above --
    # but an explicit config entry for a label (atlas_ids/atlas_names/
    # ignore_labels) beats the sidecar, mirroring the atlas_ids precedence.
    config_damage = {int(v) for v in (guide_cfg.get("damage_labels") or [])}
    sidecar_damage = set(atlas_utils.load_regions_sidecar_damage_labels(guide_cfg["regions_mask"]))
    overridden = sidecar_damage & (set(ids_cfg) | set(names_cfg) | ignored)
    if overridden:
        logger.warning(
            "Guide regions: label(s) %s are marked damage in %s but the config explicitly "
            "configures them (atlas_ids/atlas_names/ignore_labels) -- using the config value.",
            sorted(overridden), sidecar_path.name)
    damage = config_damage | (sidecar_damage - overridden)
    configured = sorted((set(ids_cfg) | set(names_cfg) | set(sidecar_ids)) - ignored - damage)
    unconfigured = painted - set(configured) - ignored - damage
    if unconfigured:
        raise ValueError(
            f"mask.guide_regions.regions_mask contains painted label(s) {sorted(unconfigured)} with no "
            f"atlas_ids/atlas_names entry and no matching region_ids in {sidecar_path} -- they would be "
            "silently ignored. Add them, list them under ignore_labels if that is deliberate, list them "
            "under damage_labels if they mark tissue with no atlas counterpart, or "
            "re-export the mask from paint_mask.py so its sidecar covers them.")
    if ignored & painted:
        logger.info("Guide regions: ignoring painted label(s) %s by config", sorted(ignored & painted))
    stale = ignored - painted
    if stale:
        raise ValueError(f"mask.guide_regions.ignore_labels lists label(s) {sorted(stale)} that are not "
                         "painted in regions_mask -- stale config, or the wrong mask file.")
    stale_damage = config_damage - painted
    if stale_damage:
        raise ValueError(f"mask.guide_regions.damage_labels lists label(s) {sorted(stale_damage)} that are "
                         "not painted in regions_mask -- stale config, or the wrong mask file.")
    damage_hole = None
    if damage:
        damage_hole = np.isin(regions_arr, sorted(damage))
        logger.info("Guide regions: damage label(s) %s mark %d voxel(s) as tissue with no atlas "
                    "counterpart -- excluded from the metric via moving_mask",
                    sorted(damage), int(damage_hole.sum()))

    annotation_arr = atlas_annotation.numpy()
    weight_cfg = guide_cfg["weight"]
    exclude_ids_cfg = guide_cfg.get("atlas_exclude_ids") or {}
    out = []
    for label in configured:
        if label in ids_cfg:
            source = f"atlas_ids[{label}] = {ids_cfg[label]}"
            atlas_arr, matched = atlas_utils.build_region_inclusion_mask_by_ids(
                annotation_arr, atlas_structures, ids_cfg[label])
        elif label in names_cfg:
            source = f"atlas_names[{label}] = {names_cfg[label]}"
            atlas_arr, matched = atlas_utils.build_region_inclusion_mask(
                annotation_arr, atlas_structures, names_cfg[label])
        else:
            source = f"{sidecar_path.name}.region_ids[{label}] = {sidecar_ids[label]}"
            atlas_arr, matched = atlas_utils.build_region_inclusion_mask_by_ids(
                annotation_arr, atlas_structures, sidecar_ids[label])
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

        if label in exclude_ids_cfg:
            # A label's atlas_ids/atlas_names match already pulled in every
            # descendant -- e.g. pallium's match includes the olfactory bulb,
            # since OB is a pallium descendant in DevCCF. If OB is ALSO guided
            # under its own label, the same atlas voxels would be pulled
            # toward two different sample outlines at once; subtracting here
            # is what keeps the two guide pairs disjoint.
            exclude_mask = np.isin(
                annotation_arr,
                list(atlas_utils.descendant_ids_of(atlas_structures, exclude_ids_cfg[label])))
            removed = int((atlas_arr & exclude_mask).sum())
            atlas_arr = atlas_arr & ~exclude_mask
            present, counts = np.unique(annotation_arr[atlas_arr], return_counts=True)
            matched = {
                atlas_structures.get(int(sid), {}).get("name", f"<unknown id {int(sid)}>"): int(count)
                for sid, count in zip(present, counts)
            }
            logger.info("Guide region label %d: atlas_exclude_ids=%s removed %d voxel(s)",
                        label, exclude_ids_cfg[label], removed)
            if not atlas_arr.any():
                raise ValueError(
                    f"mask.guide_regions.atlas_exclude_ids[{label}] removed every voxel {source} "
                    "matched -- the exclusion covers the whole region, leaving nothing to guide.")

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
    return out, damage_hole


def _log_atlas_face_clearance(atlas_annotation, configured_margin):
    """Log how many background voxels sit between atlas tissue and each face of
    the atlas grid, and warn about any face tissue is flush against.

    Worth its own check because the failure it catches is both silent and
    unbounded: SyN holds its displacement field at exactly zero on the fixed
    image's faces (measured -- see atlas_utils.background_pad_width), so tissue
    touching a face simply cannot move, and nothing in the run log would
    otherwise say so. The registration converges, the metric improves, and the
    frozen surface just quietly stays where it started.
    """
    tissue = atlas_annotation.numpy() > 0
    if not tissue.any():
        logger.warning("Atlas annotation has no nonzero voxels -- cannot check face clearance.")
        return
    clearances = []
    for axis in range(tissue.ndim):
        present = np.where(tissue.any(axis=tuple(a for a in range(tissue.ndim) if a != axis)))[0]
        clearances.append((int(present[0]), int(tissue.shape[axis] - 1 - int(present[-1]))))
    logger.info("Atlas face clearance (background voxels between tissue and each grid face), "
                "background_margin_voxels=%s: %s",
                configured_margin if configured_margin else "unset",
                ", ".join(f"axis{a}: lo={lo} hi={hi}" for a, (lo, hi) in enumerate(clearances)))
    flush = [f"axis{a} {'lo' if side == 0 else 'hi'}"
             for a, pair in enumerate(clearances) for side, gap in enumerate(pair) if gap == 0]
    if flush:
        logger.warning("Atlas tissue is FLUSH against grid face(s) %s -- SyN pins its displacement field to "
                        "exactly zero there, so that surface cannot deform at all (a hemisphere atlas cropped "
                        "at the midline hits this on the medial face, freezing the atlas midline no matter how "
                        "tilted the sample's is). Set atlas.background_margin_voxels to pad it away.",
                        ", ".join(flush))


# Runs take hours and write ~20 files under output_dir; silently reusing a
# directory that already holds a previous run mixes two runs' outputs (the
# ones whose names happen to differ survive, the rest get overwritten) and
# there is no way to tell afterwards which file came from which run. Set
# REGANTS_OVERWRITE=1 to deliberately write into an existing directory.
_OVERWRITE_ENV = "REGANTS_OVERWRITE"


def _guard_output_dir(output_dir):
    """Abort if output_dir already holds a previous run's outputs.

    run.log is ignored: run_pipeline.sh creates output_dir and starts `tee`
    into it *before* this process gets going, so a directory containing
    nothing but run.log is this run's own doing, not a previous run's.
    """
    if os.environ.get(_OVERWRITE_ENV) == "1":
        return
    if not output_dir.exists():
        return
    existing = sorted(p.name for p in output_dir.iterdir() if p.name != "run.log")
    if not existing:
        return
    raise SystemExit(
        f"Refusing to run: output_dir already exists and is not empty: {output_dir}\n"
        f"  contains: {', '.join(existing[:8])}"
        f"{' ...' if len(existing) > 8 else ''}\n"
        "Point config.output_dir (atlas_variants.<source>.output_dir) at a new "
        f"directory, move/delete the old one, or set {_OVERWRITE_ENV}=1 to "
        "overwrite it on purpose."
    )


def _load_reposition(sample_cfg):
    """(plan, fragments_zyx) for sample.reposition_plan, or (None, None).

    A reposition plan closes the gaps left by tissue that split open, and it
    has to be applied to EVERY input drawn on the raw grid -- the stack, the
    guide outlines, the damage mask, the cell centroids -- or those stop
    describing the same anatomy as each other. Doing it here, at load, rather
    than asking for a pre-repositioned copy of each file, is what keeps the
    config pointing at the original data and keeps the four from drifting
    apart: one key moves all of them or none of them.

    (scripts/apply_reposition.py writes the repositioned copies out instead,
    for looking at and for QC. Both go through registration_ants.reposition,
    so they cannot disagree.)
    """
    plan_path = sample_cfg.get("reposition_plan")
    if not plan_path:
        return None, None
    plan = reposition.read_plan(plan_path)
    fragments_path = plan.get("labels_path")
    if not fragments_path or not Path(fragments_path).exists():
        raise FileNotFoundError(
            f"{plan_path} names labels_path={fragments_path!r}, which does not exist. "
            f"That file is the fragment outlines the plan moves; without it the plan "
            f"says how far to move tissue but not which tissue.")
    fragments = sitk.GetArrayFromImage(sitk.ReadImage(str(fragments_path))).astype(np.uint8)
    if tuple(plan["image_shape_zyx"]) != fragments.shape:
        raise ValueError(f"{fragments_path} is {fragments.shape} but {plan_path} was drawn on "
                         f"{tuple(plan['image_shape_zyx'])}")
    n = sum(len(f["keyframes"]) for f in plan["fragments"])
    logger.info("Reposition: %d fragment(s), %d keyframe(s) from %s", len(plan["fragments"]), n,
                plan_path)
    for line in reposition.boundary_warnings(reposition.boundary_report(plan, fragments)):
        logger.warning("Reposition: %s", line)
    return plan, fragments


def _reposition_volume(img, plan, fragments, kind, what):
    """Apply a plan to one ANTs image loaded on the painted grid.

    Refuses rather than resamples when the shapes disagree: a plan's offsets
    are microns on the grid it was drawn on, so applying it to a volume of a
    different shape would move everything by a believable but wrong amount,
    and nothing downstream could tell.
    """
    if plan is None:
        return img
    expected = tuple(reversed(plan["image_shape_zyx"]))          # (z,y,x) -> ANTs (x,y,z)
    if img.shape != expected:
        raise ValueError(
            f"{what} is {img.shape} but sample.reposition_plan was drawn on {expected} "
            f"(x, y, z). A reposition plan can only be applied on the grid it was drawn on.")
    moved = reposition.apply_to_volume_xyz(img.numpy(), fragments, plan, kind=kind)
    out = ants.from_numpy(moved.astype(img.numpy().dtype), spacing=img.spacing,
                          origin=img.origin, direction=img.direction)
    logger.info("Reposition applied to %s", what)
    return out


def run_pipeline(config_path):
    config = load_config(config_path)
    sample = config["sample"]
    reg_cfg = config["registration"]
    prep_cfg = config["preprocess"]

    output_dir = Path(config["output_dir"])
    _guard_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_prefix = str(output_dir / "transforms" / f"{sample['name']}_")
    Path(transforms_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Snapshot the exact config this run used, next to its outputs -- config
    # values (crop bounds, guide region ids, atlas variant...) otherwise live
    # only in whatever the caller's working copy of the yaml says, which
    # keeps changing as the next run's edits land on the same file.
    shutil.copy2(config_path, output_dir / Path(config_path).name)

    _setup_logging()
    logger.info("Starting pipeline for sample '%s', config=%s", sample["name"], config_path)
    # Loaded once and threaded through every input drawn on the raw grid --
    # see _load_reposition for why they have to move together or not at all.
    reposition_plan, reposition_fragments = _load_reposition(sample)
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
    # Before the resample, not after: the plan's offsets are microns on the raw
    # grid, and a flap has to be closed while the sections it was drawn on are
    # still the sections in the array.
    raw_img = _reposition_volume(raw_img, reposition_plan, reposition_fragments,
                                 "image", "the raw stack")

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
            background_margin_voxels=atlas_cfg.get("background_margin_voxels"),
        )
        # SyN pins its displacement field to exactly zero on every face of the
        # fixed (atlas) grid, so any tissue flush against a face is frozen no
        # matter what the metric wants -- see atlas_utils.background_pad_width.
        # A hemisphere crop puts the whole midline on such a face, which is the
        # one case where this is silent AND large, so report the real clearance
        # rather than just echoing what was configured.
        _log_atlas_face_clearance(atlas_annotation, atlas_cfg.get("background_margin_voxels"))
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
        damage_path = mask_cfg["sample_damage_mask_path"]
        damage_um = mask_cfg.get("sample_damage_mask_voxel_size_um")
        # Same grid story as guide_regions' regions_mask, and it matters more
        # here: the void is only identifiable on the RAW tiff (that is the
        # only grid where you can see what is tissue and what is nothing), but
        # paint_mask.py's export copies the source's header verbatim, so the
        # file claims spacing (1,1,1) and covers the UNCROPPED volume.
        # ants.image_read alone would hand ANTs a mask in the wrong physical
        # space on the wrong grid -- and a wrong mask is applied silently,
        # there is no shape check inside ants.registration to catch it.
        # Rebuild spacing from config and regrid onto whatever registration
        # actually runs on; physical space is shared (origin 0, spacing in
        # microns, cropping only shifts origin), so this is a pure regrid.
        if damage_um:
            sample_mask = io_utils.load_nifti_stack_as_ants(damage_path, damage_um)
            sample_mask = _reposition_volume(sample_mask, reposition_plan, reposition_fragments,
                                             "labels", "the sample damage mask")
        else:
            sample_mask = ants.image_read(damage_path)
        if sample_mask.shape != sample_fine_prep.shape:
            if not damage_um:
                raise ValueError(
                    f"mask.sample_damage_mask_path is {sample_mask.shape} but registration runs on "
                    f"{sample_fine_prep.shape}. Add mask.sample_damage_mask_voxel_size_um (the voxel size "
                    "of the grid it was painted on, e.g. the raw tiff's [2.6, 2.6, 32.0]) so it can be "
                    "regridded -- without it the file's own header is trusted, and a damage mask in the "
                    "wrong physical space silently excludes the wrong voxels.")
            # Resample the HOLE, not the mask. resample_image_to_target fills
            # anything outside the source's extent with 0, and the registration
            # grid does stick out past the raw grid by a fraction of a voxel
            # (the crop's last plane sits at 4992um while the 20um grid's does
            # at 5000um, and likewise in x), so regridding the mask directly
            # comes back with a thin shell of 0 = EXCLUDED around the edges
            # -- measured 0.2% of the volume, all of it at the boundary, and
            # silent. Inverted, that same fill value reads as "not a hole",
            # which is the right default for territory the painted volume
            # never covered.
            hole = ants.from_numpy((sample_mask.numpy() == 0).astype("float32"),
                                    spacing=sample_mask.spacing, origin=sample_mask.origin,
                                    direction=sample_mask.direction)
            hole = ants.resample_image_to_target(hole, sample_fine_prep, interp_type="genericLabel")
            sample_mask = ants.from_numpy((hole.numpy() == 0).astype("float32"),
                                           spacing=sample_fine_prep.spacing,
                                           origin=sample_fine_prep.origin,
                                           direction=sample_fine_prep.direction)
        coverage = float((sample_mask.numpy() > 0).mean())
        logger.info("Loaded sample damage mask: %s (%.1f%% of the registration grid participates)",
                    damage_path, 100 * coverage)
        # A damage mask is a hole in an otherwise all-1 canvas. Anything near
        # the ~40% a tissue silhouette scores means the polarity is flipped,
        # which is the one mistake here that makes registration worse rather
        # than erroring -- see register_to_atlas's docstring for the measured
        # cost (a silhouette on moving_mask recovered 9% of a squashed
        # sample's missing extent; the same mask on the pre-alignment, 101%).
        if coverage < 0.8:
            logger.warning("Sample damage mask covers only %.1f%% -- a damage mask should be 1 nearly "
                            "everywhere with a hole in it. This is close to a tissue-silhouette mask, "
                            "which as moving_mask stops the Affine from scaling the sample up. Check the "
                            "polarity (nonzero = USED in the metric).", 100 * coverage)

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
    guide_damage_hole = None
    guide_cfg = mask_cfg.get("guide_regions")
    if isinstance(guide_cfg, dict):
        guide_regions, guide_damage_hole = _build_guide_regions_from_labels(
            guide_cfg, sample_fine_prep, atlas_annotation, atlas_structures,
            reposition_plan, reposition_fragments)
    elif guide_cfg:
        guide_regions = []
        for gr in guide_cfg:
            atlas_outline = ants.image_read(gr["atlas_outline_path"])
            sample_outline = ants.image_read(gr["sample_outline_path"])
            guide_regions.append((atlas_outline, sample_outline, gr.get("weight", 1.0)))
        logger.info("Loaded %d guide region(s): %s", len(guide_regions),
                    [gr["sample_outline_path"] for gr in guide_cfg])

    if guide_damage_hole is not None and guide_damage_hole.any():
        # Same moving_mask semantics as sample_damage_mask_path (see the
        # comment above prealign_moving_mask); ANDed with it when both exist.
        keep = (sample_mask.numpy() > 0) if sample_mask is not None \
            else np.ones(sample_fine_prep.shape, dtype=bool)
        newly_excluded = int((keep & guide_damage_hole).sum())
        keep &= ~guide_damage_hole
        sample_mask = ants.from_numpy(keep.astype("float32"), spacing=sample_fine_prep.spacing,
                                      origin=sample_fine_prep.origin,
                                      direction=sample_fine_prep.direction)
        logger.info("Merged guide-mask damage label(s) into moving_mask: %d voxel(s) newly excluded, "
                    "%.1f%% of the registration grid participates", newly_excluded, 100 * keep.mean())

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
    # to differ from whatever grid was actually registered against. Kept as
    # the full grid (rather than cropped) so labels_in_sample.nii.gz stays on
    # the exact same grid as cell_registration.csv's "resample space" columns
    # (see cell_points.py) -- a constraint scripts/relabel_cells.py already
    # depends on (it looks up corrected labels via those columns with no
    # rescaling). The part of this full grid outside crop_for_registration is
    # cleared below -- see the comment there for why it can't be trusted.
    labels_in_sample = transforms.warp_labels_to_sample(sample_fine, reg)
    if crop_cfg:
        # apply_transforms fills EVERY voxel of the (uncropped) reference grid
        # above, including the part outside crop_for_registration -- the
        # affine there is just evaluated past where it was fit (it has no
        # bounds), and the SyN field falls back to its boundary value once a
        # sample voxel maps outside the field's own support. Neither is backed
        # by real correspondence, so it's not a registration result out there,
        # just the fitted parameters extended past the region that fit them.
        # crop_for_registration exists specifically to say "don't register
        # this region" -- zeroing it out here is what makes that mean "don't
        # label it either" instead of "label it with an unconstrained
        # extrapolation and don't say so".
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = io_utils.crop_bounds_to_grid(
            crop_cfg, sample["voxel_size_um"], reg_cfg["fine_target_um"], labels_in_sample.shape)
        arr = labels_in_sample.numpy()
        keep = np.zeros(arr.shape, dtype=bool)
        keep[xlo:xhi, ylo:yhi, zlo:zhi] = True
        cleared = int((arr != 0).sum() - (arr[keep] != 0).sum())
        arr[~keep] = 0
        labels_in_sample = ants.from_numpy(arr, spacing=labels_in_sample.spacing,
                                            origin=labels_in_sample.origin, direction=labels_in_sample.direction)
        logger.info("Cleared %d labeled voxel(s) outside crop_for_registration bounds "
                    "(fine-grid indices x=[%d,%d) y=[%d,%d) z=[%d,%d) kept)",
                    cleared, xlo, xhi, ylo, yhi, zlo, zhi)
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
            reposition_plan=reposition_plan, reposition_fragments=reposition_fragments,
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
