"""Load and validate the pipeline YAML config (see config.example.yaml)."""
from pathlib import Path

import yaml

# Named atlas shortcuts, e.g. atlas.source: devccf_p04 instead of spelling out
# template_path/annotation_path/resolution_um/ontology_path/orientation every time --
# see configs/atlas_presets.example.yaml for the format. Real file is gitignored
# (machine-specific paths), same convention as ../GT_tool_for_registration/paint_mask_local.yaml.
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
        "type_of_transform": "SyNRA",
        # CC neighborhood radius in voxels (NOT histogram bins -- see
        # register.register_to_atlas's docstring for why antspyx's own
        # default of 32 is wrong here).
        "syn_sampling": 4,
        # Max SyN iterations per resolution level, coarsest first; the
        # length also sets the pyramid depth (3 entries = 4x/2x/full).
        "reg_iterations": [100, 70, 50],
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


# Where each atlas_variants field lands: (section, field). "top" means a
# top-level config key (just output_dir today) rather than a nested section.
_ATLAS_VARIANT_FIELDS = {
    "output_dir": ("top", "output_dir"),
    "fine_target_um": ("registration", "fine_target_um"),
    "atlas_res_um": ("registration", "atlas_res_um"),
    "orientation": ("atlas", "orientation"),
    "slicing": ("atlas", "slicing"),
}


def _apply_atlas_variant(config):
    """Optional per-sample config.atlas_variants: {source_name: {output_dir,
    fine_target_um, atlas_res_um, orientation, slicing}} -- fields that must
    track *which atlas* this sample is registered against (output_dir so
    results from different atlases don't overwrite each other, the two
    resolution knobs so they stay matched to that atlas's own resolution,
    orientation/slicing for atlases that need reorienting) but that
    atlas_presets_local.yaml can't own itself since output_dir is per-sample,
    not per-atlas. Switching config.atlas.source then flips all of these at
    once instead of hand-editing several commented-out lines in sync.

    A variant entry may be partial -- fields it doesn't mention are left as
    whatever's already in the config (e.g. a top-level `output_dir:` you
    still want to set directly). Pops atlas_variants off the returned config;
    it's consumed here, not part of the pipeline's own config shape.
    """
    variants = config.pop("atlas_variants", None)
    if variants is None:
        return config

    source = config.get("atlas", {}).get("source")
    if source not in variants:
        raise ValueError(
            f"config.atlas.source {source!r} has no entry under atlas_variants (known: {sorted(variants)})"
        )

    for key, value in variants[source].items():
        if key not in _ATLAS_VARIANT_FIELDS:
            raise ValueError(
                f"atlas_variants.{source}.{key} is not a recognized field (known: {sorted(_ATLAS_VARIANT_FIELDS)})"
            )
        section, field = _ATLAS_VARIANT_FIELDS[key]
        if section == "top":
            config[field] = value
        else:
            config.setdefault(section, {})[field] = value

    return config


def load_config(path):
    """Load a pipeline config YAML file, fill in defaults, and validate that
    the referenced input files exist (fails fast before any expensive step)."""
    with open(path) as f:
        config = yaml.safe_load(f)

    config = _apply_atlas_variant(config)
    config = _merge_defaults(config, _DEFAULTS)

    if "output_dir" not in config:
        raise ValueError("config.output_dir is required")

    sample = config.get("sample", {})
    for required in ("name", "raw_tiff", "voxel_size_um"):
        if required not in sample:
            raise ValueError(f"config.sample.{required} is required")
    if not Path(sample["raw_tiff"]).exists():
        raise FileNotFoundError(f"sample.raw_tiff not found: {sample['raw_tiff']}")

    reg_iters = config["registration"]["reg_iterations"]
    if not isinstance(reg_iters, (list, tuple)) or not reg_iters:
        raise ValueError("registration.reg_iterations must be a non-empty list, coarsest level first")
    if any(not isinstance(it, int) or it < 0 for it in reg_iters):
        raise ValueError(f"registration.reg_iterations must be non-negative integers, got {reg_iters}")
    if not isinstance(config["registration"]["syn_sampling"], int) or config["registration"]["syn_sampling"] < 1:
        raise ValueError("registration.syn_sampling must be a positive integer (CC neighborhood radius in voxels)")

    init_cfg = config["registration"].get("initial_transform")
    if init_cfg:
        for key in ("path", "inverse_path"):
            # inverse_path is NOT optional: ANTs cannot invert a displacement
            # field it was handed, so an initial transform without its matching
            # inverse silently corrupts every atlas->sample product
            # (labels_in_sample, cell region assignment) rather than failing.
            if not init_cfg.get(key):
                raise ValueError(
                    f"registration.initial_transform.{key} is required when initial_transform is set "
                    "(scripts/fit_initial_transform.py writes both files)")
            if not Path(init_cfg[key]).exists():
                raise FileNotFoundError(f"registration.initial_transform.{key} not found: {init_cfg[key]}")

    crop_cfg = config["registration"].get("crop_for_registration")
    if crop_cfg:
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

    guide_cfg = mask_cfg.get("guide_regions")
    if isinstance(guide_cfg, dict):
        # Multi-region form: one hand-painted multi-label volume for the
        # sample side, atlas side pulled from the annotation by ontology id
        # (atlas_ids, what paint_mask.py's region picker exports) or by name
        # (atlas_names). The older list form (one hand-painted file per side)
        # is still accepted below for regions no ontology entry can express.
        for required in ("regions_mask", "voxel_size_um"):
            if required not in guide_cfg:
                raise ValueError(f"mask.guide_regions.{required} is required")
        if not guide_cfg.get("atlas_ids") and not guide_cfg.get("atlas_names"):
            raise ValueError("mask.guide_regions needs atlas_ids or atlas_names (atlas_ids preferred -- "
                             "ids are matched exactly, names are substring-matched and can pull in "
                             "unrelated structures; paint_mask.py writes ids into <output>.regions.json)")
        if not Path(guide_cfg["regions_mask"]).exists():
            raise FileNotFoundError(f"mask.guide_regions.regions_mask not found: {guide_cfg['regions_mask']}")
        if len(guide_cfg["voxel_size_um"]) != 3:
            raise ValueError("mask.guide_regions.voxel_size_um must be [x, y, z] in microns "
                             "-- required because a mask painted on a raw TIFF carries no "
                             "spacing in its header (it reads back as 1,1,1)")

        def _normalize_label_map(key, coerce, describe):
            """{label: [entry, ...]} with int labels, or {} if the key is absent.
            A bare scalar value is accepted as a one-element list -- writing
            `1: cortex` instead of `1: [cortex]` is the obvious spelling and
            both YAML forms look identical at a glance."""
            raw = guide_cfg.get(key)
            if not raw:
                return {}
            if not isinstance(raw, dict):
                raise ValueError(f"mask.guide_regions.{key} must be a non-empty {{label: {describe}}} map")
            normalized = {}
            for raw_label, entries in raw.items():
                try:
                    label = int(raw_label)
                except (TypeError, ValueError):
                    raise ValueError(f"mask.guide_regions.{key} key must be a paint label (integer), "
                                     f"got {raw_label!r}") from None
                if label < 1:
                    raise ValueError(f"mask.guide_regions.{key} label must be >= 1 (0 is background), "
                                     f"got {label}")
                if isinstance(entries, (str, int)):
                    entries = [entries]
                try:
                    entries = [coerce(e) for e in entries]
                except (TypeError, ValueError):
                    raise ValueError(f"mask.guide_regions.{key}[{label}] must be a non-empty list of "
                                     f"{describe}") from None
                if not entries:
                    raise ValueError(f"mask.guide_regions.{key}[{label}] must be a non-empty list of "
                                     f"{describe}")
                normalized[label] = entries
            return normalized

        def _region_name(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(value)
            return value

        guide_cfg["atlas_ids"] = _normalize_label_map(
            "atlas_ids", int, "ontology structure ids (integers, descendants included)")
        guide_cfg["atlas_names"] = _normalize_label_map(
            "atlas_names", _region_name, "atlas region names (substring-matched against the ontology)")
        weight = guide_cfg.get("weight", 1.0)
        if isinstance(weight, dict):
            guide_cfg["weight"] = {int(k): float(v) for k, v in weight.items()}
        else:
            guide_cfg["weight"] = float(weight)
    elif guide_cfg:
        for i, gr in enumerate(guide_cfg):
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
