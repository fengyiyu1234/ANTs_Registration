"""Register 2D sagittal immunofluorescence sections to a 3D atlas: find each
section's (oblique) sagittal plane with a Similarity grid search, then
Affine + SyN onto that plane. The method and its conventions are in
src/registration_ants/section2d.py's module docstring; the config format in
configs/sections2d.example.yaml.

Usage (antsreg env, headless):

    python scripts/register_sections_2d.py configs/my_sections.yaml
    python scripts/register_sections_2d.py configs/my_sections.yaml --only sec03 --overwrite

Per section, <output_dir>/<name>/ gets qc.png (look at it first), plane.json,
search_candidates.csv, labels_in_section_{affine,syn}.nii.gz, the transforms,
and cells_registered.csv when cells_csv is given. <output_dir>/summary.csv
collects plane.json across sections.
"""
import argparse
import os
import sys
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
from registration_ants import atlas_utils, config as config_mod, section2d  # noqa: E402

_HINTS = set(section2d._IMAGE_DIRS)


def load_sections_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for required in ("output_dir", "atlas", "sections"):
        if required not in cfg:
            raise ValueError(f"config.{required} is required")
    cfg["atlas"] = config_mod.resolve_atlas_preset(cfg["atlas"])
    cfg["search"] = {**section2d.SEARCH_DEFAULTS, **(cfg.get("search") or {})}
    cfg["registration"] = {**section2d.REGISTRATION_DEFAULTS, **(cfg.get("registration") or {})}
    if len(cfg["search"]["ml_range_um"]) != 2:
        raise ValueError("search.ml_range_um must be [low, high] (high may be null)")

    defaults = cfg.get("section_defaults") or {}
    sections, names = [], set()
    for i, raw in enumerate(cfg["sections"]):
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
        hints = [sec.get("anterior"), sec.get("dorsal")]
        if any(hints):
            if not all(h in _HINTS for h in hints):
                raise ValueError(f"sections[{sec['name']}]: anterior and dorsal must both be one of "
                                 f"{sorted(_HINTS)} (or both left out for a full orientation search)")
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
    ap.add_argument("--only", action="append", default=[], help="section name to run (repeatable)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing per-section outputs")
    args = ap.parse_args()

    cfg = load_sections_config(args.config)
    sections = [s for s in cfg["sections"] if not args.only or s["name"] in args.only]
    if not sections:
        sys.exit(f"--only matched no section (known: {[s['name'] for s in cfg['sections']]})")

    print("loading atlas ...", flush=True)
    template, annotation, res, structures = load_atlas(cfg["atlas"])
    notes, failed = section2d.check_canonical_orientation(annotation, structures)
    for note in notes:
        print(f"  orientation check: {note}")
    if failed:
        sys.exit("atlas is not in the canonical orientation (see above) -- fix atlas.orientation")
    atlas = section2d.SagittalAtlas(template, annotation, res, midline_vox=cfg["atlas"].get("midline_voxel"))
    del template, annotation
    print(f"  atlas {res} um, midline at voxel {atlas.midline_vox:.1f}, "
          f"hemisphere half-width {atlas.ml_max_um:.0f} um\n", flush=True)

    summaries = []
    for sec in sections:
        summaries.append(section2d.process_section(sec, atlas, cfg["search"], cfg["registration"],
                                                   cfg["output_dir"], structures, args.overwrite))
    out = Path(cfg["output_dir"]) / "summary.csv"
    new = pd.DataFrame(summaries)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old["name"].isin(new["name"])], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
