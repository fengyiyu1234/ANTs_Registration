"""Load and validate the pipeline YAML config (see config.example.yaml)."""
from pathlib import Path

import yaml

# Named atlas shortcuts, e.g. atlas.source: devccf_p04 instead of spelling out
# template_path/annotation_path/resolution_um/ontology_path/orientation every time --
# see configs/atlas_presets.example.yaml for the format. Real file is gitignored
# (machine-specific paths), same convention as mask_tools/paint_mask_local.yaml.
_ATLAS_PRESETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "atlas_presets_local.yaml"


def _load_atlas_presets():
    if not _ATLAS_PRESETS_PATH.exists():
        return {}
    with open(_ATLAS_PRESETS_PATH) as f:
        return yaml.safe_load(f) or {}

_DEFAULTS = {
    "atlas": {
        "source": "brainglobe",  # "brainglobe" (auto-fetch by resolution) or "custom" (your own files)
    },
    "registration": {
        "fine_target_um": 25,
        "atlas_res_um": 25,
        "use_coarse_to_fine": False,
        "coarse_target_um": 50,
        "coarse_atlas_res_um": 50,
        "type_of_transform": "SyNRA",
    },
    "preprocess": {
        "n4_bias_correction": True,
        "intensity_clip_percentiles": [0.5, 99.5],
    },
    "mask": {
        "atlas_exclude_regions": [],  # e.g. ["Olfactory bulb"] -- names matched against the atlas ontology
    },
}


def _merge_defaults(config, defaults):
    for key, value in defaults.items():
        if isinstance(value, dict):
            config[key] = _merge_defaults(config.get(key, {}), value)
        else:
            config.setdefault(key, value)
    return config


def load_config(path):
    """Load a pipeline config YAML file, fill in defaults, and validate that
    the referenced input files exist (fails fast before any expensive step)."""
    with open(path) as f:
        config = yaml.safe_load(f)

    config = _merge_defaults(config, _DEFAULTS)

    if "output_dir" not in config:
        raise ValueError("config.output_dir is required")

    sample = config.get("sample", {})
    for required in ("name", "raw_tiff", "voxel_size_um"):
        if required not in sample:
            raise ValueError(f"config.sample.{required} is required")
    if not Path(sample["raw_tiff"]).exists():
        raise FileNotFoundError(f"sample.raw_tiff not found: {sample['raw_tiff']}")

    if config["registration"]["use_coarse_to_fine"]:
        for required in ("raw_tiff_coarse", "voxel_size_coarse_um"):
            if required not in sample:
                raise ValueError(
                    f"config.sample.{required} is required when registration.use_coarse_to_fine is true"
                )
        if not Path(sample["raw_tiff_coarse"]).exists():
            raise FileNotFoundError(f"sample.raw_tiff_coarse not found: {sample['raw_tiff_coarse']}")
        if config["atlas"]["source"] == "custom":
            raise ValueError("atlas.source: custom + registration.use_coarse_to_fine: true isn't supported yet")

    crop_cfg = config["registration"].get("crop_for_registration")
    if crop_cfg:
        if config["registration"]["use_coarse_to_fine"]:
            raise ValueError(
                "registration.crop_for_registration isn't supported together with use_coarse_to_fine: true yet"
            )
        for axis in ("x", "y", "z"):
            if crop_cfg.get(axis) is not None and len(crop_cfg[axis]) != 2:
                raise ValueError(f"registration.crop_for_registration.{axis} must be a [start, end] pair or null")

    for ch in sample.get("channels", []):
        if not Path(ch["raw_tiff"]).exists():
            raise FileNotFoundError(f"channel '{ch['name']}' raw_tiff not found: {ch['raw_tiff']}")

    atlas_cfg = config["atlas"]
    if atlas_cfg["source"] not in ("brainglobe", "custom"):
        presets = _load_atlas_presets()
        if atlas_cfg["source"] not in presets:
            raise ValueError(
                f"config.atlas.source {atlas_cfg['source']!r} is not 'brainglobe'/'custom' "
                f"and not a known preset in {_ATLAS_PRESETS_PATH} (known: {sorted(presets)})"
            )
        atlas_cfg = {**presets[atlas_cfg["source"]],
                     **{k: v for k, v in atlas_cfg.items() if k != "source"}}
        atlas_cfg["source"] = "custom"
        config["atlas"] = atlas_cfg

    if atlas_cfg["source"] == "custom":
        for required in ("template_path", "annotation_path", "resolution_um"):
            if required not in atlas_cfg:
                raise ValueError(f"config.atlas.{required} is required when atlas.source is 'custom'")
        if not Path(atlas_cfg["template_path"]).exists():
            raise FileNotFoundError(f"atlas.template_path not found: {atlas_cfg['template_path']}")
        if not Path(atlas_cfg["annotation_path"]).exists():
            raise FileNotFoundError(f"atlas.annotation_path not found: {atlas_cfg['annotation_path']}")
        if "ontology_path" in atlas_cfg and not Path(atlas_cfg["ontology_path"]).exists():
            raise FileNotFoundError(f"atlas.ontology_path not found: {atlas_cfg['ontology_path']}")
        if atlas_cfg.get("orientation") is not None and len(atlas_cfg["orientation"]) != 3:
            raise ValueError("config.atlas.orientation must have exactly 3 entries")
        if atlas_cfg.get("slicing") is not None and len(atlas_cfg["slicing"]) != 3:
            raise ValueError("config.atlas.slicing must have exactly 3 entries (x,y,z), each [start,end] or null")
    elif atlas_cfg["source"] != "brainglobe":
        raise ValueError(f"config.atlas.source must be 'brainglobe' or 'custom', got {atlas_cfg['source']!r}")
    elif atlas_cfg.get("orientation") is not None or atlas_cfg.get("slicing") is not None:
        raise ValueError("atlas.orientation/slicing only apply when atlas.source is 'custom'")

    mask_cfg = config["mask"]
    if mask_cfg.get("atlas_exclude_regions") and atlas_cfg["source"] == "custom" and "ontology_path" not in atlas_cfg:
        raise ValueError("mask.atlas_exclude_regions requires atlas.ontology_path when atlas.source is 'custom'")
    if mask_cfg.get("sample_damage_mask_path") and not Path(mask_cfg["sample_damage_mask_path"]).exists():
        raise FileNotFoundError(f"mask.sample_damage_mask_path not found: {mask_cfg['sample_damage_mask_path']}")

    auto_brain_mask = mask_cfg.get("auto_brain_mask")
    if auto_brain_mask and not isinstance(auto_brain_mask, (bool, dict)):
        raise ValueError("mask.auto_brain_mask must be true/false or a dict of generate_brain_mask overrides "
                          "(sigma/closing_radius/dilate_radius/slice_axis)")

    if mask_cfg.get("guide_regions"):
        if config["registration"]["use_coarse_to_fine"]:
            raise ValueError("mask.guide_regions isn't supported together with registration.use_coarse_to_fine: true yet")
        for i, gr in enumerate(mask_cfg["guide_regions"]):
            for required in ("atlas_outline_path", "sample_outline_path"):
                if required not in gr:
                    raise ValueError(f"mask.guide_regions[{i}].{required} is required")
                if not Path(gr[required]).exists():
                    raise FileNotFoundError(f"mask.guide_regions[{i}].{required} not found: {gr[required]}")

    if "cells" in config:
        cells_cfg = config["cells"]
        for required in ("cell_centroids_dir", "voxel_size_um"):
            if required not in cells_cfg:
                raise ValueError(f"config.cells.{required} is required when the cells: block is present")
        if not Path(cells_cfg["cell_centroids_dir"]).is_dir():
            raise FileNotFoundError(f"cells.cell_centroids_dir not found: {cells_cfg['cell_centroids_dir']}")
        cells_cfg.setdefault("prefix", "ob_")

    return config
