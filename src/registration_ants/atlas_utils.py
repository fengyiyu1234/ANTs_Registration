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
    # Annotation keeps BrainGlobe's own integer dtype -- these are the same
    # CCFv3 structure ids that float32 cannot hold (io_utils._LABEL_DTYPE_NOTE).
    annotation = ants.from_numpy(np.ascontiguousarray(atlas.annotation), spacing=spacing)
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
    # preserve_labels: structure IDs, not intensities -- float32 would round
    # every CCFv3 id above 2**24 and silently merge structures. The whole
    # story, with the measured damage, is in io_utils._LABEL_DTYPE_NOTE.
    annotation = loader(annotation_path, spacing, preserve_labels=True)
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


def _atlas_prep_postfix(orientation, slicing_spec, background_margin_voxels=None):
    """Short, deterministic filename postfix derived from orientation/slicing
    (mirrors ClearMap.Alignment.Annotation.format_annotation_filename) -- so
    multiple sample configs sharing the same raw atlas + prep params reuse the
    same cached oriented/cropped file instead of redoing the permute/crop on
    every pipeline run. The margin is appended only when actually requested, so
    caches written before it existed keep their names and stay valid."""
    orient_part = '_'.join(str(o) for o in orientation) if orientation else 'orig'
    if slicing_spec is None:
        slicing_part = 'full'
    else:
        parts = []
        for s in slicing_spec:
            parts.append('full' if s is None else f'{s[0] if s[0] is not None else 0}-{s[1]}')
        slicing_part = '_'.join(parts)
    margin_part = f'__pad{int(background_margin_voxels)}' if background_margin_voxels else ''
    return f'{orient_part}__{slicing_part}{margin_part}'


def _read_atlas_array_xyz(src_path, preserve_labels=False):
    """Raw atlas file -> (x,y,z)-ordered array, per format -- the read half of
    prepare_custom_atlas, split out so template and annotation can be loaded
    together (padding couples them: both must get the identical pad width).

    preserve_labels: True for an annotation, whose voxels are structure ids
    rather than intensities (io_utils._LABEL_DTYPE_NOTE)."""
    src_path = Path(src_path)
    if _is_nifti(src_path):
        import ants
        # Already (x,y,z) on disk (see io_utils.load_nifti_stack_as_ants) --
        # no transpose needed before reorienting, unlike the TIFF branch below.
        # pixeltype is forced here for the same reason the loader forces it:
        # image_read's default 'float' rounds structure ids (_LABEL_DTYPE_NOTE).
        pixeltype = "unsigned int" if preserve_labels else "float"
        return ants.image_read(str(src_path), pixeltype=pixeltype).numpy()
    import tifffile
    return np.transpose(tifffile.imread(str(src_path)), (2, 1, 0))


def _write_atlas_array_xyz(arr_xyz, out_path, resolution_um):
    """Write a prepared (x,y,z)-ordered array back out in the same format its
    source came in as -- the write half of prepare_custom_atlas."""
    if _is_nifti(out_path):
        import ants
        # Spacing here is only what this cache file's own header claims;
        # load_custom_atlas always overrides it from config.atlas.resolution_um
        # when this cache is loaded back, same as for the TIFF branch.
        # An integer array is written out as-is: casting a label volume to
        # float32 here would bake the id rounding this pipeline just went to
        # some trouble to undo into the cache (io_utils._LABEL_DTYPE_NOTE).
        # The TIFF branch below already preserves dtype for free.
        arr_out = arr_xyz if np.issubdtype(arr_xyz.dtype, np.integer) else arr_xyz.astype(np.float32)
        ants.image_write(ants.from_numpy(np.ascontiguousarray(arr_out),
                                         spacing=(float(resolution_um),) * 3), str(out_path))
    else:
        # Write back out in (z,y,x) on-disk order, matching how every other
        # raw TIFF in this codebase is stored/read.
        import tifffile
        tifffile.imwrite(str(out_path), np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0))))


def background_pad_width(annotation_xyz, margin_voxels):
    """Per-axis ((lo, hi), ...) zero-padding that leaves at least
    `margin_voxels` of all-background voxels between tissue and every face of
    the array. Faces that already have that much clearance get 0 -- this is a
    "pad UP TO this margin", not "add this much", so re-preparing an atlas
    that already has the margin is a no-op rather than growing it every time.

    Why this exists at all: ANTs/ITK's SyN holds the displacement field at
    EXACTLY zero on every face of the fixed image's grid (measured on a real
    run's 1Warp.nii.gz and 1InverseWarp.nii.gz -- max |displacement| was
    0.000000 on all six faces, against an interior median of 110um). Any
    tissue flush against a face is therefore frozen: no metric, no guide
    region, and no amount of weight or iterations can move it, because the
    constraint is imposed on the field itself rather than on the objective.

    That is exactly what a hemisphere crop does to the midline. Cropping the
    atlas at the anatomical midline (e.g. slicing [[320, 640], ...] on a
    640-voxel left-right axis whose midline is 320) puts the whole medial
    cut face -- 105259 tissue voxels in the measured case -- on the pinned
    boundary, so the atlas midline stays perfectly straight no matter how
    tilted the sample's own midline is, while the interior deforms normally.

    Padding with BACKGROUND is what fixes it, and it is strictly better than
    the obvious alternative of widening the crop to include contralateral
    tissue: the tissue/background step at the medial surface is the very
    edge the intensity metric locks onto, and real tissue on the far side
    erases that step. Zero padding moves the pinned face away from the
    midline while leaving the step intact and now free to move.

    Tissue extent is read off the ANNOTATION (>0 is unambiguous) and the same
    pad width is then applied to the template too -- an intensity template's
    own background is not reliably zero (LSFM noise, bias field).
    """
    tissue = annotation_xyz > 0
    if not tissue.any():
        raise ValueError("atlas annotation has no nonzero voxels -- cannot locate tissue to pad around "
                         "(wrong file, or an orientation/slicing that cropped the brain away entirely)")
    margin = int(margin_voxels)
    pad = []
    for axis in range(tissue.ndim):
        present = np.where(tissue.any(axis=tuple(a for a in range(tissue.ndim) if a != axis)))[0]
        lo_gap, hi_gap = int(present[0]), int(tissue.shape[axis] - 1 - int(present[-1]))
        pad.append((max(0, margin - lo_gap), max(0, margin - hi_gap)))
    return tuple(pad)


def prepare_custom_atlas(template_path, annotation_path, resolution_um, orientation=None, slicing=None,
                          background_margin_voxels=None, cache_dir=None, overwrite=False):
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
    background_margin_voxels: pad the prepared volumes with zero background so
    at least this many all-background voxels separate tissue from every face
    of the array -- see background_pad_width, which is where the measured
    reason lives. Short version: SyN pins its displacement field to exactly
    zero on the fixed image's faces, so tissue flush against a face cannot
    move at all, which is precisely what cropping a hemisphere atlas at the
    anatomical midline does to the midline. None/0 = no padding.

    If all three are None this is equivalent to load_custom_atlas (no prep
    needed -- e.g. files already pre-oriented, like the current DeMBA P5
    trimmed files). Otherwise, results are cached next to the source file (or
    under cache_dir if given) with a postfix derived from
    orientation/slicing/margin, and reused across runs/samples unless
    overwrite=True -- same skip-if-exists caching cellMap.py relies on to
    avoid redoing this every run. Template and annotation are prepared
    TOGETHER (rather than one file at a time as before) because padding
    couples them: the pad width is measured on the annotation and must be
    applied identically to the template, or the two would no longer share a
    grid.

    Returns (template_img, annotation_img) as ANTs images, same as
    load_custom_atlas.
    """
    if orientation is None and slicing is None and not background_margin_voxels:
        return load_custom_atlas(template_path, annotation_path, resolution_um)

    postfix = _atlas_prep_postfix(orientation, slicing, background_margin_voxels)
    slicing_tuple = _parse_slicing(slicing)

    src_paths = [Path(template_path), Path(annotation_path)]
    out_paths = []
    for src_path in src_paths:
        stem, suffix = _split_stem_suffix(src_path)
        out_dir = Path(cache_dir) if cache_dir else src_path.parent
        out_paths.append(out_dir / f"{stem}_{postfix}{suffix}")

    # All-or-nothing on the cache: the two files are only meaningful as a pair
    # on a shared grid, and with padding one cannot be rebuilt without the
    # other (the pad width comes from the annotation).
    if overwrite or not all(p.exists() for p in out_paths):
        # src_paths is [template, annotation] -- only the annotation is a
        # label volume, so only it keeps its integer dtype through the prep.
        arrays = [_read_atlas_array_xyz(p, preserve_labels=(i == 1))
                  for i, p in enumerate(src_paths)]
        arrays = [reorient_volume(a, orientation) for a in arrays]
        if slicing_tuple is not None:
            arrays = [a[slicing_tuple] for a in arrays]
        if background_margin_voxels:
            pad_width = background_pad_width(arrays[1], background_margin_voxels)
            arrays = [np.pad(a, pad_width, mode="constant", constant_values=0) for a in arrays]
        for arr_xyz, out_path in zip(arrays, out_paths):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_atlas_array_xyz(arr_xyz, out_path, resolution_um)

    return load_custom_atlas(str(out_paths[0]), str(out_paths[1]), resolution_um)


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


def regions_sidecar_path(regions_mask_path):
    """<regions_mask>.regions.json -- the sidecar Registration_toolkit's
    paint_mask.py (guide mode) writes next to its exported multi-label
    volume. Mirrors that tool's own _output_stem: .nii.gz is special-cased
    because it is a double suffix, so plain Path.stem would only strip the
    .gz and leave a bogus "<name>.nii.regions.json"."""
    path = Path(regions_mask_path)
    name = path.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem
    return path.with_name(stem + ".regions.json")


def load_regions_sidecar_ids(regions_mask_path):
    """{label: [ontology structure id, ...]} read back from a guide mask's
    own .regions.json sidecar, or {} if the mask has no such sidecar (hand-
    built mask, or role="atlas" side without one).

    This is what paint_mask.py's ontology picker recorded at export time --
    the ids actually highlighted in the GUI while a label was painted, not a
    copy of them. A pipeline config's own atlas_ids has to be hand-copied
    from the same sidecar and kept in sync by hand across repaints, which is
    exactly the kind of drift a hand-copy invites (a label repainted against
    a different ontology node, with the config's atlas_ids left stale). This
    reads the sidecar directly instead, so there is nothing to keep in sync.
    """
    sidecar = regions_sidecar_path(regions_mask_path)
    if not sidecar.exists():
        return {}
    with open(sidecar, encoding="utf-8") as f:
        data = json.load(f)
    return {int(label): [int(v) for v in ids] for label, ids in (data.get("region_ids") or {}).items()}


def load_regions_sidecar_damage_labels(regions_mask_path):
    """Sorted brush labels the guide mask's .regions.json sidecar marks as
    damage -- sample tissue with NO atlas counterpart (paint_mask.py's
    "damage / no atlas counterpart" pseudo-region), [] without a sidecar or
    the key. Same read-the-sidecar-instead-of-a-hand-copy rationale as
    load_regions_sidecar_ids: the pipeline unions these with the config's own
    mask.guide_regions.damage_labels, so a label marked damage in the GUI
    needs no config entry at all."""
    sidecar = regions_sidecar_path(regions_mask_path)
    if not sidecar.exists():
        return []
    with open(sidecar, encoding="utf-8") as f:
        data = json.load(f)
    return sorted(int(v) for v in (data.get("damage_labels") or []))


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
    return ~np.isin(annotation_arr, list(_structure_ids_matching(structures, exclude_names)))


def _structure_ids_matching(structures, names):
    """ids of every structure whose name substring-matches any of `names`
    (case-insensitive), plus all of their descendants via structure_id_path."""
    root_ids = {
        sid for sid, info in structures.items()
        if any(name.lower() in info["name"].lower() for name in names)
    }
    return {
        sid for sid, info in structures.items()
        if set(info["structure_id_path"]) & root_ids
    }


def descendant_ids_of(structures, root_ids):
    """Every structure id at or below any of `root_ids` in the ontology tree.

    Membership is decided by structure_id_path containment, never by name, so
    this cannot conflate two structures whose names merely share a substring
    (the "Cerebellum" / "cerebellum related fiber tracts" trap measured in
    region_mask_by_exact_name's docstring). Unknown ids raise rather than
    contributing nothing: an id that isn't in this ontology is a config typo
    or an id from a different atlas, and silently yielding an empty mask for
    it is exactly the failure mode ids were chosen to avoid.
    """
    root_ids = {int(r) for r in root_ids}
    unknown = sorted(root_ids - set(structures))
    if unknown:
        raise ValueError(
            f"structure id(s) {unknown} are not in this ontology -- wrong atlas, or a typo. "
            f"(This ontology has {len(structures)} structures.)")
    return {sid for sid, info in structures.items()
            if set(info["structure_id_path"]) & root_ids}


def _mask_and_matched(annotation_arr, structures, ids):
    """(binary mask, {structure name: voxel count}) for a set of structure
    ids -- the shared tail of the name- and id-based inclusion builders. Only
    structures actually PRESENT in this annotation appear in `matched`, which
    is what lets callers tell "resolved to nothing" apart from "resolved fine"."""
    mask = np.isin(annotation_arr, list(ids))
    present, counts = np.unique(annotation_arr[mask], return_counts=True)
    matched = {
        structures.get(int(sid), {}).get("name", f"<unknown id {int(sid)}>"): int(count)
        for sid, count in zip(present, counts)
    }
    return mask, matched


def build_region_inclusion_mask_by_ids(annotation_arr, structures, include_ids):
    """Binary inclusion mask from ontology structure IDS (plus descendants),
    the unambiguous counterpart of build_region_inclusion_mask's substring
    name matching. Returns (mask, matched) in the same shape, so callers can
    swap between the two without branching on the result.

    This is what paint_mask.py's region picker writes into its
    <output>.regions.json sidecar: the GUI highlights a region by id, so
    resolving by that same id downstream is the only way to guarantee the
    region used in registration is the one that was actually looked at while
    painting.
    """
    return _mask_and_matched(annotation_arr, structures, descendant_ids_of(structures, include_ids))


def build_region_inclusion_mask(annotation_arr, structures, include_names):
    """Binary mask over an annotation array: True = inside one of the named
    regions. The complement of build_region_exclusion_mask's convention --
    used to pull the atlas-side half of a guide region straight out of the
    annotation, so only the sample side has to be drawn by hand.

    include_names are matched the same way build_region_exclusion_mask
    matches: case-insensitive substring, descendants included. Several names
    are unioned, which is what real use needs -- e.g. DevCCF has no single
    "cortex" label, only 36 separate `layer N of <area>` structures.

    Returns (mask, matched) where matched is {structure name: voxel count}
    for every named structure actually present in this annotation. Callers
    must check it: substring matching that hits nothing produces an
    all-False mask and no error, which downstream looks like "the guide
    region simply did nothing".
    """
    return _mask_and_matched(annotation_arr, structures, _structure_ids_matching(structures, include_names))


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
