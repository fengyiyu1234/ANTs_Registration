"""A 2D section batch in the layout Registration_toolkit/single_sample.py
already reads, so the sections can be browsed interactively with the viewer
unchanged.

The viewer's Native view shows one 3D sample: <sample>_fine_<res>um.nii.gz +
<sample>_labels_in_sample.nii.gz, plus an optional full-resolution tiff it
aligns by the shape ratio. The sections of one batch share an image size, so
they are written as that sample with z = section: the z slider flips through
the sections, and hover for the region and its hierarchy, region search and
fill/outline all work on them. register_sections_2d.py writes this after
every batch into <output_dir>/viewer/; scripts/export_sections_for_viewer.py
rewrites it on its own.

    <sample>_fine_<res>um.nii.gz        section_prep of every section
    <sample>_labels_in_sample.nii.gz    labels_in_section_syn (affine where SyN is off)
    <sample>_raw_sections.tif           registration channel at full resolution, one page per
                                        section, uncompressed so the viewer memmaps it
    sections.csv                        z -> section name and its plane.json fields
    single_sample.yaml                  the viewer config

Sections without plane.json (still to run, or failed) get empty planes.
Sections of different sizes are padded at the bottom/right, which keeps pixel
(0, 0) where every grid puts it. The raw stack is only rewritten when an image
or its channel settings changed (sidecar <raw>.json): it is ~1 GB for a
typical batch and the images do not change between reruns.
"""
import json
import os
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import tifffile

from . import atlas_utils, section_io

_PLANE_FIELDS = ("ml_um", "yaw_deg", "roll_deg", "size_ratio", "far_gap", "syn_unlabelled_pct",
                 "affine_unlabelled_pct")


def _pad_stack(planes, dtype):
    """(rows, cols) arrays -> (n, rows, cols), zero-padded at the bottom/right."""
    out = np.zeros((len(planes), max(p.shape[0] for p in planes), max(p.shape[1] for p in planes)), dtype=dtype)
    for z, p in enumerate(planes):
        out[z, :p.shape[0], :p.shape[1]] = p
    return out


def _read_rc(path, labels=False):
    """A 2D ANTs image written by section2d (x = column, y = row) -> (rows, cols)."""
    return ants.image_read(str(path), pixeltype="unsigned int" if labels else "float").numpy().T


def _write_zyx(arr_zyx, path, res_um):
    # The viewer reads NIfTI with SimpleITK, which hands back (z, y, x) for an
    # (x, y, z) file: write the transpose.
    atlas_utils._write_atlas_array_xyz(np.ascontiguousarray(arr_zyx.transpose(2, 1, 0)), Path(path), res_um)


def _prepared_annotation_path(atlas_cfg):
    """The canonical-orientation annotation the registration used, as a file
    (prepare_custom_atlas's cache), for the viewer's Atlas view."""
    if atlas_cfg.get("source") == "brainglobe":
        return None
    src = Path(atlas_cfg["annotation_path"])
    if not (atlas_cfg.get("orientation") or atlas_cfg.get("slicing")):
        return src
    stem, suffix = atlas_utils._split_stem_suffix(src)
    path = src.parent / f"{stem}_{atlas_utils._atlas_prep_postfix(atlas_cfg.get('orientation'), atlas_cfg.get('slicing'))}{suffix}"
    if not path.exists():
        atlas_utils.prepare_custom_atlas(atlas_cfg["template_path"], atlas_cfg["annotation_path"],
                                         atlas_cfg["resolution_um"], orientation=atlas_cfg.get("orientation"),
                                         slicing=atlas_cfg.get("slicing"))
    return path


def _write_raw_stack(sections, path, log):
    sources = [[str(s["image"]), os.path.getmtime(s["image"]), s.get("channel"), s.get("panel_colors"),
                s.get("z_projection", "max")] for s in sections]
    sidecar = Path(f"{path}.json")
    if path.exists() and sidecar.exists():
        with open(sidecar) as f:
            if json.load(f) == json.loads(json.dumps(sources)):
                log(f"  {path.name}: images unchanged, kept")
                return

    def load(sec):
        return section_io.load_registration_image(sec["image"], sec.get("channel"), sec.get("panel_colors"),
                                                  sec.get("z_projection", "max"), log=lambda *_: None)

    def yx_shape(image):
        with tifffile.TiffFile(image) as tf:
            series = tf.series[0]
            return series.shape[series.axes.index("Y")], series.shape[series.axes.index("X")]

    first = load(sections[0])
    shapes = [first.shape] + [yx_shape(s["image"]) for s in sections[1:]]
    shape = (len(sections), max(s[0] for s in shapes), max(s[1] for s in shapes))
    peak = float(first.max())
    dtype = np.uint8 if peak <= 255 else np.uint16 if peak <= 65535 else np.float32
    high = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else None
    sidecar.unlink(missing_ok=True)       # a stack cut short must not look complete next time
    # One page at a time into a memmapped file: the stack never sits in memory whole.
    stack = tifffile.memmap(path, shape=shape, dtype=dtype)
    for z, sec in enumerate(sections):
        plane = first if z == 0 else load(sec)
        stack[z, :plane.shape[0], :plane.shape[1]] = np.clip(plane, 0, high)
    stack.flush()
    del stack
    with open(sidecar, "w") as f:
        json.dump(sources, f)
    log(f"  {path.name}: {' x '.join(map(str, shape))} {np.dtype(dtype).name}")


def export_for_viewer(cfg, raw=True, log=print):
    """cfg: register_sections_2d.load_sections_config's result. Returns the
    viewer config path, or None when no section has finished yet."""
    out_root = Path(cfg["output_dir"])
    sections = cfg["sections"]
    res = float(cfg["registration"]["register_res_um"])
    sample = Path(cfg["input_dir"]).name if cfg.get("input_dir") else out_root.name
    view_dir = out_root / "viewer"

    rows, images, labels = [], [], []
    for z, sec in enumerate(sections):
        d = out_root / sec["name"]
        row = {"z": z, "name": sec["name"], "registered": (d / "plane.json").exists()}
        image = lab = None
        if row["registered"]:
            with open(d / "plane.json") as f:
                plane = json.load(f)
            row.update({k: plane.get(k) for k in _PLANE_FIELDS})
            syn = d / "labels_in_section_syn.nii.gz"
            row["labels"] = "syn" if syn.exists() else "affine"
            image = _read_rc(d / "section_prep.nii.gz")
            lab = _read_rc(syn if syn.exists() else d / "labels_in_section_affine.nii.gz", labels=True)
        rows.append(row)
        images.append(image)
        labels.append(lab)
    done = [r["name"] for r in rows if r["registered"]]
    if not done:
        log("viewer export: no section has finished registering yet -- skipped")
        return None

    log(f"\nviewer export ({len(done)}/{len(rows)} sections registered) -> {view_dir}")
    view_dir.mkdir(parents=True, exist_ok=True)
    shape = next(i.shape for i in images if i is not None)
    images = [np.zeros(shape, np.float32) if i is None else i for i in images]
    labels = [np.zeros(shape, np.uint32) if lab is None else lab for lab in labels]
    _write_zyx(_pad_stack(images, np.float32), view_dir / f"{sample}_fine_{res:g}um.nii.gz", res)
    _write_zyx(_pad_stack(labels, np.uint32), view_dir / f"{sample}_labels_in_sample.nii.gz", res)
    pd.DataFrame(rows).to_csv(view_dir / "sections.csv", index=False)
    if len(done) < len(rows):
        log(f"  empty planes (not registered): {', '.join(r['name'] for r in rows if not r['registered'])}")

    raw_path = view_dir / f"{sample}_raw_sections.tif"
    if raw:
        _write_raw_stack(sections, raw_path, log)

    ann = _prepared_annotation_path(cfg["atlas"])
    lines = [
        "# Written by registration_ants.section_viewer after each 2D batch -- regenerated, do not edit.",
        "# Native view: z slider = section (z -> name in sections.csv). 2D sections have no cell",
        "# tables, so the cell panels stay empty.",
        f"sample_dir: '{view_dir}'",
        f"std_atlas_path: '{ann}'" if ann else "std_atlas_path: ''   # brainglobe atlas: no local file",
        f"ontology_json_path: '{cfg['atlas'].get('ontology_path', '')}'",
    ]
    if raw and raw_path.exists():
        lines.append(f"native_image_path: '{raw_path}'")
    config_path = view_dir / "single_sample.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  view: python <Registration_toolkit>/single_sample.py '{config_path}'")
    return config_path
