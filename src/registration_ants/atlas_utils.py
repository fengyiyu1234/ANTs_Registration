"""Atlas template/annotation sources: BrainGlobe (auto-fetch) or a custom
atlas already prepared elsewhere (e.g. a ClearMap-style pre-reoriented and
pre-cropped atlas), wrapped as ANTs images either way.

BrainGlobe's array axis order is (ap, si, rl) — anterior-posterior,
superior-inferior, right-left (see atlas.space.axes_description). Resolution
is always isotropic for the Allen CCF (10/25/50/100um), so spacing is
unambiguous regardless of axis order.

`ants` / `brainglobe_atlasapi` are imported lazily inside the two functions
that need them, not at module level -- the ontology-only helpers below
(load_ccf_ontology_json, structures_at_levels) have no such dependency, and
callers running in a display-only env without antspyx installed (e.g. the
interactive napari tools in scripts/) need to be able to import this module
for those alone.
"""
import json
from pathlib import Path

import numpy as np


def get_allen_atlas(resolution_um):
    """Fetch (downloads on first use) the Allen CCF template + annotation.

    resolution_um: one of 10, 25, 50, 100.
    Returns (template_img, annotation_img, structures) where the two images
    are ANTs images on the same grid, and structures maps label id -> dict
    with 'name', 'acronym', 'structure_id_path', etc.
    """
    import ants
    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(f"allen_mouse_{resolution_um}um")
    spacing = (float(resolution_um),) * 3
    template = ants.from_numpy(atlas.reference.astype("float32"), spacing=spacing)
    annotation = ants.from_numpy(atlas.annotation.astype("float32"), spacing=spacing)
    return template, annotation, atlas.structures


def _is_nifti(path):
    """.nii/.nii.gz check on the full filename, not Path.suffix -- .suffix on a
    double extension like '...20um.nii.gz' would wrongly return just '.gz'."""
    name = Path(path).name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _split_stem_suffix(path):
    """Like (Path.stem, Path.suffix), but treats '.nii.gz' as one suffix
    instead of splitting it into stem='...nii', suffix='.gz'."""
    if _is_nifti(path) and path.name.lower().endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")], ".nii.gz"
    return path.stem, path.suffix


def load_custom_atlas(template_path, annotation_path, resolution_um):
    """Load a template + annotation you've already reoriented/cropped
    yourself (e.g. exported from a ClearMap-style pipeline, or a NIfTI atlas
    like DevCCF) instead of fetching from BrainGlobe.

    resolution_um: isotropic voxel size of these files, in microns. Neither
    file's own embedded spacing is trusted (TIFF has none; NIfTI's is
    typically in mm and/or paired with a non-identity direction this codebase
    doesn't handle -- see io_utils.load_nifti_stack_as_ants) -- this value is
    what actually sets the physical grid ANTs registers on, get it right (see
    the elastix/ClearMap config, or the atlas's own published resolution, that
    produced these files).
    Returns (template_img, annotation_img).
    """
    from . import io_utils

    spacing = (float(resolution_um),) * 3
    loader = io_utils.load_nifti_stack_as_ants if _is_nifti(template_path) else io_utils.load_tiff_stack_as_ants
    template = loader(template_path, spacing)
    loader = io_utils.load_nifti_stack_as_ants if _is_nifti(annotation_path) else io_utils.load_tiff_stack_as_ants
    annotation = loader(annotation_path, spacing)
    return template, annotation


def _format_orientation(orientation):
    """Validate/normalize a ClearMap-style orientation spec: a 3-tuple of
    signed ints from {-3..-1, 1..3}, one entry per axis of an already
    (x,y,z)-ordered array (the same axis order io_utils.load_tiff_stack_as_ants
    produces) -- entry d names which source axis (1=x, 2=y, 3=z; negative =
    that axis gets flipped) ends up in destination axis d. Matches
    ClearMap.Alignment.Resampling.format_orientation's contract exactly, e.g.
    the DeMBA P5 atlas files already in use were produced with (1, 3, 2).
    """
    if orientation is None:
        return None
    orientation = tuple(int(o) for o in orientation)
    if len(orientation) != 3 or sorted(abs(o) for o in orientation) != [1, 2, 3]:
        raise ValueError(f"orientation must be a permutation of +-1,+-2,+-3, got {orientation}")
    return orientation


def reorient_volume(arr, orientation):
    """Permute + flip a 3D (x,y,z)-ordered array's axes per a ClearMap-style
    orientation spec (see _format_orientation). Reproduces
    ClearMap.Alignment.Annotation.prepare_annotation_files's permute-then-flip
    logic exactly (same order of operations), so an orientation tuple that
    worked there produces an identical result here.
    """
    orientation = _format_orientation(orientation)
    if orientation is None:
        return arr

    permutation = tuple(abs(o) - 1 for o in orientation)
    arr = arr.transpose(permutation)

    slicing = [slice(None)] * arr.ndim
    flipped = False
    for axis, o in enumerate(orientation):
        if o < 0:
            slicing[axis] = slice(None, None, -1)
            flipped = True
    return arr[tuple(slicing)] if flipped else arr


def _parse_slicing(slicing_spec):
    """[[x1,x2]|None, [y1,y2]|None, [z1,z2]|None] -> tuple of slice(), one per
    axis of an (x,y,z)-ordered array -- applied AFTER reorient_volume, same as
    ClearMap's `slicing` argument to prepare_annotation_files."""
    if slicing_spec is None:
        return None
    if len(slicing_spec) != 3:
        raise ValueError(f"slicing must have exactly 3 entries (x,y,z), got {len(slicing_spec)}")
    return tuple(slice(None) if s is None else slice(s[0], s[1]) for s in slicing_spec)


def _atlas_prep_postfix(orientation, slicing_spec):
    """Short, deterministic filename postfix derived from orientation/slicing
    (mirrors ClearMap.Alignment.Annotation.format_annotation_filename) -- so
    multiple sample configs sharing the same raw atlas + prep params reuse the
    same cached oriented/cropped file instead of redoing the permute/crop on
    every pipeline run."""
    orient_part = '_'.join(str(o) for o in orientation) if orientation else 'orig'
    if slicing_spec is None:
        slicing_part = 'full'
    else:
        parts = []
        for s in slicing_spec:
            parts.append('full' if s is None else f'{s[0] if s[0] is not None else 0}-{s[1]}')
        slicing_part = '_'.join(parts)
    return f'{orient_part}__{slicing_part}'


def prepare_custom_atlas(template_path, annotation_path, resolution_um, orientation=None, slicing=None,
                          cache_dir=None, overwrite=False):
    """Reorient + crop a raw (not-yet-prepared) atlas template/annotation pair,
    then load them as ANTs images -- the in-house equivalent of ClearMap's
    Annotation.prepare_annotation_files, so a fresh atlas source (e.g. a raw
    DeMBA/Allen download) can be pointed at directly from config instead of
    requiring a file already pre-baked by running ClearMap first.

    template_path/annotation_path: raw TIFF or NIfTI (.nii/.nii.gz) stacks in
    the source atlas's own orientation, read into (x,y,z) order the same way
    load_custom_atlas does per-format (TIFF: tifffile (z,y,x) on disk ->
    transposed; NIfTI: already (x,y,z) on disk, no transpose -- see
    io_utils.load_nifti_stack_as_ants) before any reorienting.
    orientation: see reorient_volume; None keeps the source's native axes.
    slicing: see _parse_slicing, applied after reorientation; None = no crop.

    If both are None this is equivalent to load_custom_atlas (no prep
    needed -- e.g. files already pre-oriented, like the current DeMBA P5
    trimmed files). Otherwise, results are cached next to the source file (or
    under cache_dir if given) with a postfix derived from orientation/slicing,
    and reused across runs/samples unless overwrite=True -- same
    skip-if-exists caching cellMap.py relies on to avoid redoing this every
    run.

    Returns (template_img, annotation_img) as ANTs images, same as
    load_custom_atlas.
    """
    if orientation is None and slicing is None:
        return load_custom_atlas(template_path, annotation_path, resolution_um)

    postfix = _atlas_prep_postfix(orientation, slicing)
    slicing_tuple = _parse_slicing(slicing)

    prepared_paths = []
    for src_path in (template_path, annotation_path):
        src_path = Path(src_path)
        nifti = _is_nifti(src_path)
        stem, suffix = _split_stem_suffix(src_path)
        out_dir = Path(cache_dir) if cache_dir else src_path.parent
        out_path = out_dir / f"{stem}_{postfix}{suffix}"
        if overwrite or not out_path.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            if nifti:
                import ants
                # Already (x,y,z) on disk (see io_utils.load_nifti_stack_as_ants) --
                # no transpose needed before reorienting, unlike the TIFF branch below.
                arr_xyz = ants.image_read(str(src_path)).numpy()
            else:
                import tifffile
                arr_xyz = np.transpose(tifffile.imread(str(src_path)), (2, 1, 0))
            arr_xyz = reorient_volume(arr_xyz, orientation)
            if slicing_tuple is not None:
                arr_xyz = arr_xyz[slicing_tuple]
            if nifti:
                # Spacing here is only what this cache file's own header claims;
                # load_custom_atlas always overrides it from config.atlas.resolution_um
                # when this cache is loaded back, same as for the TIFF branch.
                out_img = ants.from_numpy(arr_xyz.astype(np.float32), spacing=(float(resolution_um),) * 3)
                ants.image_write(out_img, str(out_path))
            else:
                # Write back out in (z,y,x) on-disk order, matching how every
                # other raw TIFF in this codebase is stored/read.
                tifffile.imwrite(str(out_path), np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0))))
        prepared_paths.append(out_path)

    return load_custom_atlas(str(prepared_paths[0]), str(prepared_paths[1]), resolution_um)


def load_ccf_ontology_json(path):
    """Parse an Allen-API-style ontology JSON (nested 'msg' tree, as shipped
    e.g. alongside ClearMap's atlas resources) into a flat dict keyed by
    structure id, matching the shape of BrainGlobe's atlas.structures:
    {id: {'id', 'name', 'acronym', 'structure_id_path'}}.
    """
    with open(path) as f:
        data = json.load(f)

    flat = {}

    def _walk(node, parent_path):
        path_here = parent_path + [node["id"]]
        flat[node["id"]] = {
            "id": node["id"],
            "name": node["name"],
            "acronym": node["acronym"],
            "structure_id_path": path_here,
        }
        for child in node.get("children", []):
            _walk(child, path_here)

    for root in data["msg"]:
        _walk(root, [])
    return flat


def structures_at_levels(structures, min_level, max_level):
    """Filter an ontology dict (id -> info with 'structure_id_path', as
    returned by get_allen_atlas or load_ccf_ontology_json) down to structures
    whose tree depth falls within [min_level, max_level] (root = level 1,
    via len(structure_id_path)) -- e.g. for a region picker that should offer
    major structures (level ~4-6) rather than every fine-grained leaf area.
    """
    return {sid: info for sid, info in structures.items()
            if min_level <= len(info["structure_id_path"]) <= max_level}


def collapse_labels_to_level(label_arr, structures, level):
    """Remap every voxel's (possibly fine-grained) label id to its ontology
    ancestor at tree depth `level` (root = level 1, via structure_id_path).
    Ids whose own path is already shorter than `level` (no descendant that
    deep) are left unchanged. Used to build a read-only reference view of a
    chosen level's regions without finer subdivisions cluttering it (see
    ../GT_tool_for_registration/edit_sample_labels.py's level-overview layer).
    """
    max_id = max(int(label_arr.max()) if label_arr.size else 0,
                 max(structures) if structures else 0)
    lut = np.arange(max_id + 1, dtype=label_arr.dtype)
    for sid, info in structures.items():
        path = info["structure_id_path"]
        lut[sid] = path[level - 1] if len(path) >= level else sid
    return lut[label_arr]


def build_region_exclusion_mask(annotation_arr, structures, exclude_names):
    """Binary mask over an annotation array: True = keep (use in
    registration), False = excluded region -- e.g. olfactory bulb, absent in
    a damaged/amputated sample.

    exclude_names are matched case-insensitively as substrings against each
    structure's name; every descendant of a matched structure is excluded
    too (via structure_id_path), so passing "Olfactory bulb" also excludes
    its subregions, not just the top-level structure itself.
    """
    exclude_root_ids = {
        sid for sid, info in structures.items()
        if any(name.lower() in info["name"].lower() for name in exclude_names)
    }
    exclude_ids = {
        sid for sid, info in structures.items()
        if set(info["structure_id_path"]) & exclude_root_ids
    }
    return ~np.isin(annotation_arr, list(exclude_ids))


def region_mask_by_exact_name(annotation_arr, structures, exact_name):
    """Binary inclusion mask (True = in this structure or a descendant of
    it) resolved by an EXACT (case/whitespace-insensitive) name match,
    unlike build_region_exclusion_mask's substring match. Substring matching
    can silently pull in unrelated structures whose name merely contains the
    target as a substring -- e.g. "Cerebellum" also matches "cerebellum
    related fiber tracts", a totally different top-level branch (verified
    against a real CCF ontology + segmentation: that alone is ~22% of
    Cerebellum's true descendant-voxel count). Raises if the name doesn't
    resolve to exactly one structure.
    """
    name_norm = exact_name.strip().lower()
    matches = [sid for sid, info in structures.items() if info["name"].strip().lower() == name_norm]
    if len(matches) != 1:
        found = [(sid, structures[sid]["name"]) for sid in matches]
        raise ValueError(f"expected exactly one structure named {exact_name!r}, found {len(matches)}: {found}")
    root_id = matches[0]
    include_ids = {sid for sid, info in structures.items() if root_id in info["structure_id_path"]}
    return np.isin(annotation_arr, list(include_ids))
