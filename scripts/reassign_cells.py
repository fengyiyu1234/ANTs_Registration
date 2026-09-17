"""Re-run ONLY the pipeline's [6/6] assign_cells step against a registration
that already ran, using a (different) cell_centroids directory.

The registration itself never sees cell centroids -- they enter only at the
last step -- so re-detecting cells does NOT require re-running SyN. Doing so
would in fact make the result *less* comparable: ANTs is not run-to-run
reproducible here (see PROGRESS_LOG), so a fresh run of the same config lands
a measurably different transform. This script reuses the saved transforms
verbatim, so the only thing that changes between the old and new
cell_registration.csv is the centroid set.

Usage:
    conda activate antsreg
    # one run
    python scripts/reassign_cells.py <source_run_dir> <new_output_dir> \
            [--centroids-dir DIR] [--config YAML]
    # several runs, from a jobs file (see configs/reassign.example.yaml)
    python scripts/reassign_cells.py --jobs configs/reassign_local.yaml [-- only s10 s18]

<source_run_dir>: a completed run's output_dir -- must contain the config
yaml the pipeline snapshotted there, transforms/, and <name>_fine_<N>um.nii.gz.
--centroids-dir overrides cells.cell_centroids_dir from that config.

Unchanged volumes (fine/cropped/labels_in_sample/in_atlas/brain_mask) are
symlinked into the new directory rather than copied, so the output is a
drop-in sample dir for the stats/napari tooling that globs for them without
duplicating several hundred MB.
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import ants
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils, cell_points, transforms  # noqa: E402
from registration_ants.config import load_config  # noqa: E402
# Same loader the pipeline's own [6/6] uses, imported rather than reimplemented
# so a repositioned sample cannot be assigned two different ways depending on
# which entry point ran it.
from registration_ants.pipeline import _load_reposition  # noqa: E402

logger = logging.getLogger("reassign_cells")

# Everything a completed run writes that this step does not change -- linked
# through so <new_output_dir> works wherever <source_run_dir> did.
_LINKED_SUFFIXES = ("_fine_20um.nii.gz", "_fine_20um_cropped.nii.gz",
                    "_labels_in_sample.nii.gz", "_in_atlas.nii.gz", "_brain_mask.nii.gz")


def _find_config(run_dir):
    """The yaml run_pipeline snapshotted into its own output_dir."""
    yamls = sorted(p for p in run_dir.glob("*.yaml"))
    if len(yamls) != 1:
        raise SystemExit(f"Expected exactly one config yaml in {run_dir}, found {[p.name for p in yamls]}")
    return yamls[0]


def _load_jobs(jobs_path, only=None):
    """Read a jobs yaml -> [(label, source_run_dir, new_output_dir, centroids_dir, config_path)].

    Keeping the sample list in a file rather than in argv is what makes a
    six-sample re-assign reproducible: the exact source run each sample was
    re-assigned against is the thing that has to be looked up again later, and
    it is easy to get wrong by hand (several samples here have more than one
    dated run dir).
    """
    doc = yaml.safe_load(Path(jobs_path).read_text(encoding="utf-8")) or {}
    defaults = doc.get("defaults") or {}
    jobs = doc.get("jobs")
    if not jobs:
        raise SystemExit(f"{jobs_path} has no jobs: list.")
    out = []
    for i, job in enumerate(jobs):
        label = job.get("sample") or f"job{i}"
        if only and label not in only:
            continue
        merged = {**defaults, **job}
        for key in ("source_run_dir", "new_output_dir"):
            if not merged.get(key):
                raise SystemExit(f"{jobs_path}: job '{label}' is missing {key}.")
        out.append((
            label,
            Path(merged["source_run_dir"]).expanduser().resolve(),
            Path(merged["new_output_dir"]).expanduser().resolve(),
            merged.get("centroids_dir"),
            Path(merged["config"]).expanduser().resolve() if merged.get("config") else None,
        ))
    if only:
        missing = set(only) - {label for label, *_ in out}
        if missing:
            raise SystemExit(f"{jobs_path} has no job(s) named: {', '.join(sorted(missing))}")
    if not out:
        raise SystemExit(f"{jobs_path}: nothing selected.")
    return out


def reassign(src, out, centroids_dir=None, config_path=None):
    """Re-run assign_cells for one completed run. Returns the class list."""
    config_path = config_path or _find_config(src)
    config = load_config(str(config_path))
    name = config["sample"]["name"]
    fine_um = config["registration"]["fine_target_um"]

    if "cells" not in config:
        raise SystemExit(f"{config_path} has no cells: block -- nothing to assign.")
    centroids_dir = centroids_dir or config["cells"]["cell_centroids_dir"]

    # A repositioned sample was registered on the closed-up geometry, so its
    # cells have to be moved by the same plan before they are looked up --
    # exactly as pipeline's [6/6] does. Without this the fragments' cells are
    # queried at their pre-repositioning coordinates against an atlas mapping
    # built for the post-repositioning ones (s18: up to ~980 um off).
    reposition_plan, reposition_fragments = _load_reposition(config["sample"])
    if reposition_plan is not None:
        logger.info("Reposition plan: %s (%d fragment(s))",
                    config["sample"]["reposition_plan"], len(reposition_plan["fragments"]))

    out.mkdir(parents=True, exist_ok=True)
    logger.info("Source run: %s", src)
    logger.info("Config:     %s", config_path)
    logger.info("Centroids:  %s", centroids_dir)
    logger.info("Output:     %s", out)

    # sample_fine: read back off disk rather than re-resampling the raw tiff.
    # cell_points indexes its "resample space" columns into this grid, and the
    # file the source run wrote IS that grid, bit for bit.
    fine_path = src / f"{name}_fine_{fine_um}um.nii.gz"
    sample_fine = ants.image_read(str(fine_path))
    logger.info("sample_fine: shape=%s spacing=%s (from %s)",
                sample_fine.shape, sample_fine.spacing, fine_path.name)

    atlas_cfg = config["atlas"]
    if atlas_cfg["source"] != "custom":
        raise SystemExit("Only atlas.source resolving to 'custom' is supported here.")
    # Hits prepare_custom_atlas's on-disk cache written by the original run,
    # so this is a load, not a re-orient/re-crop.
    _, atlas_annotation = atlas_utils.prepare_custom_atlas(
        atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
        orientation=atlas_cfg.get("orientation"), slicing=atlas_cfg.get("slicing"),
        background_margin_voxels=atlas_cfg.get("background_margin_voxels"),
    )
    atlas_structures = (atlas_utils.load_ccf_ontology_json(atlas_cfg["ontology_path"])
                        if "ontology_path" in atlas_cfg else None)
    logger.info("atlas_annotation: shape=%s spacing=%s", atlas_annotation.shape, atlas_annotation.spacing)

    reg = transforms.load_saved_transforms(str(src / "transforms" / f"{name}_"))
    reg["atlas_annotation"] = atlas_annotation
    reg["atlas_structures"] = atlas_structures
    logger.info("Reusing saved transforms: %s", [Path(t).name for t in reg["fwdtransforms"]])

    classes = cell_points.assign_cell_regions(
        centroids_dir, out, tuple(config["cells"]["voxel_size_um"]),
        sample_fine, reg, atlas_structures=atlas_structures, prefix=config["cells"]["prefix"],
        reposition_plan=reposition_plan, reposition_fragments=reposition_fragments,
    )
    logger.info("Cell regions assigned for %d class(es): %s", len(classes), classes)

    shutil.copy2(config_path, out / config_path.name)
    for suffix in _LINKED_SUFFIXES:
        target = src / f"{name}{suffix}"
        link = out / f"{name}{suffix}"
        if target.exists() and not link.exists():
            link.symlink_to(target)
    link = out / "transforms"
    if not link.exists():
        link.symlink_to(src / "transforms")
    logger.info("Linked unchanged volumes + transforms from %s", src)
    logger.info("Done: %s", out)
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_run_dir", type=Path, nargs="?")
    ap.add_argument("new_output_dir", type=Path, nargs="?")
    ap.add_argument("--centroids-dir", default=None,
                    help="override cells.cell_centroids_dir from the source config")
    ap.add_argument("--config", type=Path, default=None,
                    help="use this yaml instead of the one snapshotted in <source_run_dir> -- "
                         "needed when a path the snapshot names (e.g. a guide mask that has since "
                         "been moved) no longer exists, since load_config validates those even "
                         "though this step does not use them")
    ap.add_argument("--jobs", type=Path, default=None,
                    help="yaml listing several (source_run_dir, new_output_dir) pairs to "
                         "re-assign in one go -- see configs/reassign.example.yaml")
    ap.add_argument("--only", nargs="+", default=None, metavar="SAMPLE",
                    help="with --jobs: run only these sample: entries")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.jobs:
        if args.source_run_dir or args.new_output_dir:
            raise SystemExit("--jobs takes the directories from the yaml -- don't also pass them.")
        jobs = _load_jobs(args.jobs, only=args.only)
        logger.info("%d job(s) from %s: %s", len(jobs), args.jobs,
                    ", ".join(label for label, *_ in jobs))
        for n, (label, src, out, centroids, config_path) in enumerate(jobs, 1):
            logger.info("===== [%d/%d] %s =====", n, len(jobs), label)
            reassign(src, out, centroids_dir=centroids or args.centroids_dir,
                     config_path=config_path)
        logger.info("All %d job(s) done.", len(jobs))
        return

    if not (args.source_run_dir and args.new_output_dir):
        raise SystemExit("Need <source_run_dir> <new_output_dir>, or --jobs <yaml>.")
    reassign(args.source_run_dir.resolve(), args.new_output_dir.resolve(),
             centroids_dir=args.centroids_dir,
             config_path=args.config.resolve() if args.config else None)


if __name__ == "__main__":
    main()
