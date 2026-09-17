"""Shareable PNG overlays of a 2D section batch: the registered atlas drawn on
the original image, in the atlas' own region colours.

Two pictures per section, both written flat into <output_dir>/overlays/ so the
folder can be zipped and sent as is:

    <name>_atlas_outline.png    region boundaries only, the image stays visible
    <name>_atlas_filled.png     regions filled (alpha), boundaries on top

plus regions.csv (every region present, its colour and its area) and
legend.png (the colour key), so the colours are readable without this repo.

Colours are the ontology's own `color_hex_triplet` (BrainGlobe: `rgb_triplet`)
-- the Allen CCF palette, the same one the atlas is normally shown in. Ids the
ontology does not list keep a stable colour hashed from the id rather than
silently rendering as background.

The background is the ORIGINAL image (section_io reads it exactly as the
registration did), block-mean downsampled to max_px so a batch stays small
enough to mail; labels are placed on it by physical coordinate, not by shape,
so nothing depends on the raw and the 20 um label grid dividing evenly.
`source="prep"` uses section_prep.nii.gz instead -- the preprocessed image the
registration actually saw, at the label grid, no original file needed.

register_sections_2d.py writes this after every batch;
scripts/export_overlays.py rewrites it on its own (fast: no registration, and
--source prep does not even read the originals).
"""
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from scipy import ndimage

from . import atlas_utils, section_io

MAX_PX = 3000          # longest edge of a written PNG
ALPHA = 0.45           # region fill opacity on the filled picture
LINE_WIDTH = 2         # boundary width in output pixels


def _hashed_rgb(sid):
    """A stable, well-spread colour for an id the ontology does not list."""
    h = (int(sid) * 2654435761) % 2**32
    return (np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255], np.uint16) // 2 + 64).astype(np.uint8)


def region_rgb(sid, structures):
    """One region's (3,) uint8 colour, from the ontology or hashed from the id."""
    info = (structures or {}).get(int(sid))
    if info:
        if info.get("rgb_triplet"):
            return np.array(info["rgb_triplet"][:3], np.uint8)
        if info.get("color_hex_triplet"):
            hexed = str(info["color_hex_triplet"]).lstrip("#")
            return np.array([int(hexed[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)
    return _hashed_rgb(sid)


def encode_labels(lab, structures, colors=None):
    """(compact code image, (n, 3) uint8 palette) for a label image.

    A dense id -> colour table is not an option here: CCF ids run past 6e8, so
    one would be 1.8 GB of mostly-unused rows. The ids present in a section are
    a few hundred, so they are renumbered 0..n-1 (code 0 = background, always
    present) and the palette is that small. `colors` caches id -> rgb across
    sections."""
    ids = np.union1d(np.unique(lab), np.zeros(1, lab.dtype))
    code = np.searchsorted(ids, lab).astype(np.int32)
    palette = np.zeros((len(ids), 3), np.uint8)
    for k, sid in enumerate(ids.tolist()):
        if sid:
            palette[k] = colors.setdefault(int(sid), region_rgb(sid, structures)) if colors is not None \
                else region_rgb(sid, structures)
    return code, palette


def _read_2d(path, labels=False):
    """A 2D ANTs image written by section2d -> ((rows, cols) array, origin_xy,
    spacing_xy). section2d writes x = column, y = row, hence the transpose."""
    img = ants.image_read(str(path), pixeltype="unsigned int" if labels else "float")
    return img.numpy().T, np.asarray(img.origin, float), np.asarray(img.spacing, float)


def _block_mean(arr, g):
    if g <= 1:
        return arr.astype(np.float32)
    h, w = (arr.shape[0] // g) * g, (arr.shape[1] // g) * g
    return arr[:h, :w].reshape(h // g, g, w // g, g).mean(axis=(1, 3)).astype(np.float32)


def _to_gray(arr):
    """float image -> uint8, contrast stretched over its non-empty pixels."""
    body = arr[arr > 0]
    lo, hi = np.percentile(body, (1, 99.5)) if body.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _labels_on_grid(lab, lab_origin, lab_spacing, xs_um, ys_um):
    """Nearest-neighbour resample of a label image onto a display grid given by
    the physical (micron) centres of its columns and rows. Display pixels
    falling outside the label image become background instead of being clamped
    onto its border row/column."""
    cols = np.rint((xs_um - lab_origin[0]) / lab_spacing[0]).astype(int)
    rows = np.rint((ys_um - lab_origin[1]) / lab_spacing[1]).astype(int)
    col_ok, row_ok = (cols >= 0) & (cols < lab.shape[1]), (rows >= 0) & (rows < lab.shape[0])
    out = lab[np.clip(rows, 0, lab.shape[0] - 1)[:, None], np.clip(cols, 0, lab.shape[1] - 1)[None, :]]
    return np.where(row_ok[:, None] & col_ok[None, :], out, 0)


def _boundaries(lab, width):
    """(mask, label id at each boundary pixel) for the region outlines. The
    boundary is drawn just inside each region, so a border between two regions
    shows both their colours and the tissue outline is not thickened outwards."""
    edge = np.zeros(lab.shape, bool)
    diff_r, diff_c = lab[:-1, :] != lab[1:, :], lab[:, :-1] != lab[:, 1:]
    edge[:-1, :] |= diff_r
    edge[1:, :] |= diff_r
    edge[:, :-1] |= diff_c
    edge[:, 1:] |= diff_c
    edge &= lab > 0
    if width <= 1 or not edge.any():
        return edge, np.where(edge, lab, 0)
    grown = ndimage.binary_dilation(edge, iterations=int(width) - 1)
    # Each grown pixel takes the colour of the boundary pixel it grew from, so
    # a thick line stays one colour instead of picking up the neighbouring
    # region's (which a max filter over the ids would do).
    _, (ri, ci) = ndimage.distance_transform_edt(~edge, return_indices=True)
    return grown, np.where(grown, lab[ri, ci], 0)


def render_overlays(gray, code, palette, alpha=ALPHA, line_width=LINE_WIDTH):
    """(outline RGB, filled RGB) uint8 for one section. `code`/`palette` come
    from encode_labels: code 0 is background."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    edge, edge_code = _boundaries(code, line_width)
    edge_rgb = palette[edge_code]

    outline = rgb.copy()
    outline[edge] = edge_rgb[edge]

    inside = code > 0
    filled = rgb.astype(np.float32)
    filled[inside] = (1 - alpha) * filled[inside] + alpha * palette[code][inside]
    filled = filled.astype(np.uint8)
    # On the fill, a region's own colour would not separate it from a similarly
    # coloured neighbour: darken the boundary colour instead of drawing it flat.
    filled[edge] = (edge_rgb[edge] * 0.45).astype(np.uint8)
    return outline, filled


def _background(sec, out_dir, source, max_px):
    """(gray uint8, xs_um, ys_um) for one section's picture."""
    if source == "prep":
        arr, origin, spacing = _read_2d(out_dir / "section_prep.nii.gz")
        xs = origin[0] + np.arange(arr.shape[1]) * spacing[0]
        ys = origin[1] + np.arange(arr.shape[0]) * spacing[1]
        return _to_gray(arr), xs, ys
    px = float(sec["pixel_size_um"])
    raw = section_io.load_registration_image(sec["image"], sec.get("channel"), sec.get("panel_colors"),
                                             sec.get("z_projection", "max"), log=lambda *_: None)
    g = max(1, int(np.ceil(max(raw.shape) / max_px)))
    arr = _block_mean(raw, g)
    del raw
    # Physical coordinates are (col * px, row * px) of the ORIGINAL pixels
    # (section2d.downsample_section's convention), so a block's centre is the
    # centre of the original pixels it averages.
    xs = (np.arange(arr.shape[1]) * g + (g - 1) / 2) * px
    ys = (np.arange(arr.shape[0]) * g + (g - 1) / 2) * px
    return _to_gray(arr), xs, ys


def _write_legend(path, rows, colors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = max(len(rows), 1)
    cols = max(1, min(4, int(np.ceil(n / 45))))
    per_col = int(np.ceil(n / cols))
    fig, ax = plt.subplots(figsize=(4.6 * cols, max(2.0, per_col * 0.155 + 0.6)))
    ax.set_xlim(0, cols)
    ax.set_ylim(per_col, -1)
    ax.axis("off")
    for k, row in enumerate(rows):
        c, r = divmod(k, per_col)
        ax.add_patch(Rectangle((c + 0.02, r - 0.35), 0.06, 0.7, color=colors[row["id"]] / 255))
        ax.text(c + 0.10, r, f"{row['acronym']}  {row['name']}"[:62], fontsize=6.2, va="center")
    fig.suptitle("atlas regions in this batch (Allen CCF colours)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_overlays(cfg, source="raw", max_px=MAX_PX, alpha=ALPHA, line_width=LINE_WIDTH,
                    level=None, log=print):
    """cfg: register_sections_2d.load_sections_config's result. Writes
    <output_dir>/overlays/ and returns it, or None when no section has finished
    yet. `level` collapses labels to that ontology depth (root = 1) first."""
    from PIL import Image

    out_root = Path(cfg["output_dir"])
    over_dir = out_root / "overlays"
    ontology = cfg["atlas"].get("ontology_path")
    structures = atlas_utils.load_ccf_ontology_json(ontology) if ontology else None

    done = [sec for sec in cfg["sections"] if (out_root / sec["name"] / "plane.json").exists()]
    if not done:
        log("overlay export: no section has finished registering yet -- skipped")
        return None
    log(f"\noverlay export ({len(done)}/{len(cfg['sections'])} sections registered, "
        f"source={source}) -> {over_dir}")
    if not structures:
        log("  atlas.ontology_path is not set -- regions get hashed colours, not the atlas' own")
    over_dir.mkdir(parents=True, exist_ok=True)

    colors, area = {}, {}
    for sec in done:
        name = sec["name"]
        d = out_root / name
        syn = d / "labels_in_section_syn.nii.gz"
        lab, lab_origin, lab_spacing = _read_2d(syn if syn.exists() else d / "labels_in_section_affine.nii.gz",
                                                labels=True)
        if level:
            lab = atlas_utils.collapse_labels_to_level(lab, structures or {}, int(level))
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        px_mm2 = float(lab_spacing[0] * lab_spacing[1]) / 1e6
        for sid, count in zip(ids.tolist(), counts.tolist()):
            entry = area.setdefault(int(sid), {"area_mm2": 0.0, "n_sections": 0})
            entry["area_mm2"] += count * px_mm2
            entry["n_sections"] += 1

        gray, xs, ys = _background(sec, d, source, max_px)
        code, palette = encode_labels(_labels_on_grid(lab, lab_origin, lab_spacing, xs, ys),
                                      structures, colors)
        outline, filled = render_overlays(gray, code, palette, alpha, line_width)
        Image.fromarray(outline).save(over_dir / f"{name}_atlas_outline.png")
        Image.fromarray(filled).save(over_dir / f"{name}_atlas_filled.png")
        log(f"  {name}: {gray.shape[1]} x {gray.shape[0]} px, {len(ids)} regions")

    rows = [{"id": sid, "acronym": (structures or {}).get(sid, {}).get("acronym", ""),
             "name": (structures or {}).get(sid, {}).get("name", f"<id {sid}>"),
             "color_hex": "#%02X%02X%02X" % tuple(colors.setdefault(sid, region_rgb(sid, structures))),
             "n_sections": v["n_sections"], "area_mm2": round(v["area_mm2"], 4)}
            for sid, v in sorted(area.items())]
    rows.sort(key=lambda r: (r["name"] or "", r["id"]))
    pd.DataFrame(rows).to_csv(over_dir / "regions.csv", index=False)
    _write_legend(over_dir / "legend.png", rows, colors)
    (over_dir / "README.txt").write_text(
        f"Atlas overlays for {out_root.name} -- {len(done)} section(s).\n\n"
        "  <name>_atlas_outline.png   atlas region boundaries on the section\n"
        "  <name>_atlas_filled.png    atlas regions filled in, boundaries on top\n"
        "  legend.png / regions.csv   which colour is which region (Allen CCF palette)\n\n"
        f"source={source}, alpha={alpha}, line_width={line_width}, "
        f"level={level or 'leaf labels as registered'}\n"
        "Written by registration_ants.section_overlays -- regenerated on rerun.\n",
        encoding="utf-8")
    log(f"  {len(rows)} regions -> regions.csv + legend.png")
    return over_dir
