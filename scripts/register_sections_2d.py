"""Register 2D sagittal immunofluorescence sections to a 3D atlas: find each
section's (oblique) sagittal plane with a Similarity grid search, then
Affine + SyN onto that plane. The method and its conventions are in
src/registration_ants/section2d.py's module docstring; the config format in
configs/sections2d.example.yaml.

Usage (antsreg env, headless):

    # every *.tif / *.tiff in a folder; results in <folder>/registration/
    python scripts/register_sections_2d.py configs/my_sections.yaml --input-dir J:/my_sections
    python scripts/register_sections_2d.py configs/my_sections.yaml --only sec03 --overwrite

The folder can also be set in the config (input_dir); without one, the
config's sections list names each image. In folder mode a section's name is
its file name without extension, and sections entries only override
per-section fields by that name.

Batch behaviour: a section whose plane.json already exists is skipped (rerun
the same command to resume after a crash; --overwrite redoes everything), and
a section that fails is reported and the rest still run.

Per section, <output_dir>/<name>/ gets qc.png (look at it first), plane.json,
search_candidates.csv, labels_in_section_{affine,syn}.nii.gz, the transforms,
and cells_registered.csv when cells_csv is given. <output_dir>/summary.csv
collects plane.json across sections.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Before ants/ITK is imported: ITK reads its thread count once, so setting it
# later (ants.config.set_ants_deterministic does, inside process_section)
# leaves the main process multithreaded and its Affine/SyN not reproducible
# run to run even with a fixed seed -- measured on the DevCCF validation: the
# plane search (workers, 1 thread) was bit-identical across two runs, the
# final SyN MI was not. 2D registrations are small; one thread costs little.
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils, config as config_mod, section2d, section_io  # noqa: E402

_HINTS = set(section2d._IMAGE_DIRS)


def _collect_sections(cfg):
    """Raw section entries: the config's sections list, or in folder mode one
    entry per image in input_dir with any sections entry of the same name
    merged over it. Entries naming no file in the folder must bring their own
    image and are appended."""
    listed = cfg.get("sections") or []
    if not cfg.get("input_dir"):
        return listed
    images = section_io.find_section_images(cfg["input_dir"], cfg.get("input_pattern"))
    if not images:
        raise FileNotFoundError(f"no images matching {cfg.get('input_pattern') or section_io.IMAGE_PATTERNS} "
                                f"in {cfg['input_dir']}")
    overrides = {}
    for i, raw in enumerate(listed):
        if "name" not in raw:
            raise ValueError(f"sections[{i}].name is required (in folder mode: the file name without extension)")
        overrides[raw["name"]] = raw
    found = [{"name": p.stem, "image": str(p), **overrides.pop(p.stem, {})} for p in images]
    for name, raw in overrides.items():
        if "image" not in raw:
            raise ValueError(f"sections[{name}]: no file {name}.* in {cfg['input_dir']} and no image given")
    return found + list(overrides.values())


def load_sections_config(path, input_dir=None, output_dir=None, pattern=None):
    """input_dir / output_dir / pattern override the config's input_dir /
    output_dir / input_pattern. A folder given here without an output_dir
    writes to <folder>/registration, not to the config's output_dir, so one
    config can be pointed at several folders without results landing together."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if input_dir:
        cfg["input_dir"] = input_dir
        cfg["output_dir"] = output_dir or str(Path(input_dir) / "registration")
    elif output_dir:
        cfg["output_dir"] = output_dir
    if pattern:
        cfg["input_pattern"] = pattern
    if cfg.get("input_dir") and not cfg.get("output_dir"):
        cfg["output_dir"] = str(Path(cfg["input_dir"]) / "registration")
    for required in ("output_dir", "atlas") if cfg.get("input_dir") else ("output_dir", "atlas", "sections"):
        if required not in cfg:
            raise ValueError(f"config.{required} is required")
    cfg["atlas"] = config_mod.resolve_atlas_preset(cfg["atlas"])
    cfg["search"] = {**section2d.SEARCH_DEFAULTS, **(cfg.get("search") or {})}
    cfg["registration"] = {**section2d.REGISTRATION_DEFAULTS, **(cfg.get("registration") or {})}
    if len(cfg["search"]["ml_range_um"]) != 2:
        raise ValueError("search.ml_range_um must be [low, high] (high may be null)")

    defaults = cfg.get("section_defaults") or {}
    sections, names = [], set()
    for i, raw in enumerate(_collect_sections(cfg)):
        sec = {**defaults, **raw}
        for required in ("name", "image", "pixel_size_um"):
            if required not in sec:
                raise ValueError(f"sections[{i}].{required} is required (or set it in section_defaults)")
        if sec["name"] in names:
            raise ValueError(f"duplicate section name {sec['name']!r}")
        names.add(sec["name"])
        for key in ("image", "tissue_mask", "damage_mask", "cells_csv"):
            if sec.get(key) and not Path(sec[key]).exists():
                raise FileNotFoundError(f"sections[{sec['name']}].{key} not found: {sec[key]}")
        channel = sec.get("channel")
        # A marker name is a channel name on multichannel greyscale files (checked
        # when the file is read) and an unmixing target on RGB composites -- only
        # the latter can be checked this early, and only when panel_colors is given.
        if isinstance(channel, str) and channel != "sum" and sec.get("panel_colors"):
            # Fails here, before the atlas loads, if the colours are not separable.
            section_io.rgb_unmix_weights(sec["panel_colors"], channel)
        hints = [sec.get("anterior"), sec.get("dorsal")]
        if not all(h in _HINTS for h in hints):
            raise ValueError(f"sections[{sec['name']}]: anterior and dorsal must both be one of {sorted(_HINTS)} "
                             "(set them in section_defaults)")
        if set(hints) in ({"left", "right"}, {"up", "down"}) or hints[0] == hints[1]:
            raise ValueError(f"sections[{sec['name']}]: anterior and dorsal must be perpendicular")
        sections.append(sec)
    cfg["sections"] = sections
    return cfg


def load_atlas(atlas_cfg):
    """(template, annotation, res_um, structures) in the canonical orientation
    section2d expects: axis0 left->right, axis1 anterior->posterior, axis2
    dorsal->ventral."""
    if atlas_cfg["source"] == "brainglobe":
        res = atlas_cfg["resolution_um"]
        template, annotation, structures = atlas_utils.get_allen_atlas(res)
        # BrainGlobe Allen is (AP, SI, RL) -> (RL, AP, SI).
        return (np.transpose(template.numpy(), (2, 0, 1)), np.transpose(annotation.numpy(), (2, 0, 1)),
                res, structures)
    template, annotation = atlas_utils.prepare_custom_atlas(
        atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
        orientation=atlas_cfg.get("orientation"))
    structures = (atlas_utils.load_ccf_ontology_json(atlas_cfg["ontology_path"])
                  if atlas_cfg.get("ontology_path") else None)
    return template.numpy(), annotation.numpy(), atlas_cfg["resolution_um"], structures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--input-dir", help="folder of section images; registers every image in it "
                                        "(overrides config input_dir)")
    ap.add_argument("--output-dir", help="default: config output_dir, or <input-dir>/registration with --input-dir")
    ap.add_argument("--pattern", help="file pattern inside the folder, e.g. '*_s01*.tif' "
                                      f"(default {' and '.join(section_io.IMAGE_PATTERNS)})")
    ap.add_argument("--only", action="append", default=[], help="section name to run (repeatable)")
    ap.add_argument("--overwrite", action="store_true", help="redo sections that already have results")
    args = ap.parse_args()

    cfg = load_sections_config(args.config, args.input_dir, args.output_dir, args.pattern)
    sections = [s for s in cfg["sections"] if not args.only or s["name"] in args.only]
    if not sections:
        sys.exit(f"--only matched no section (known: {[s['name'] for s in cfg['sections']]})")
    out_root = Path(cfg["output_dir"])
    todo = [s for s in sections if args.overwrite or not (out_root / s["name"] / "plane.json").exists()]
    print(f"{len(sections)} section(s), {len(sections) - len(todo)} already done, {len(todo)} to run "
          f"-> {out_root}", flush=True)

    summaries, failed = [], []
    for sec in sections:
        if sec not in todo:
            with open(out_root / sec["name"] / "plane.json") as f:
                summaries.append(json.load(f))
    if todo:
        print("loading atlas ...", flush=True)
        template, annotation, res, structures = load_atlas(cfg["atlas"])
        notes, bad_orientation = section2d.check_canonical_orientation(annotation, structures)
        for note in notes:
            print(f"  orientation check: {note}")
        if bad_orientation:
            sys.exit("atlas is not in the canonical orientation (see above) -- fix atlas.orientation")
        atlas = section2d.SagittalAtlas(template, annotation, res, midline_vox=cfg["atlas"].get("midline_voxel"))
        del template, annotation
        print(f"  atlas {res} um, midline at voxel {atlas.midline_vox:.1f}, "
              f"hemisphere half-width {atlas.ml_max_um:.0f} um", flush=True)

        for i, sec in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {sec['name']}", flush=True)
            t0 = time.time()
            try:
                # overwrite=True: without --overwrite only sections lacking plane.json
                # get here, and a folder without one is a run that died part way.
                summaries.append(section2d.process_section(sec, atlas, cfg["search"], cfg["registration"],
                                                           out_root, structures, overwrite=True))
                print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception:
                traceback.print_exc()
                failed.append(sec["name"])
                print(f"  FAILED: {sec['name']} -- continuing with the rest", flush=True)

    if summaries:
        out = out_root / "summary.csv"
        new = pd.DataFrame(summaries)
        if out.exists():
            old = pd.read_csv(out)
            new = pd.concat([old[~old["name"].isin(new["name"])], new], ignore_index=True)
        new.to_csv(out, index=False)
        print(f"\nwrote {out}")
    if failed:
        sys.exit(f"{len(failed)} section(s) failed: {', '.join(failed)} (tracebacks above)")


if __name__ == "__main__":
    main()
