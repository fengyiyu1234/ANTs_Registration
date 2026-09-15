"""Reading 2D section images in whatever form a confocal export comes in --
greyscale, multichannel greyscale (CYX, usually 8 bit per channel), a Z stack,
or an RGB composite with each marker already colour-assigned -- plus the
property report to run BEFORE writing a sections config (inspect_image,
scripts/inspect_sections.py).

Axes come from tifffile's series axes, never from the array shape: an 8-bit
file with three planes is either an RGB composite (axes YXS, every marker's
colour spread over the planes) or three greyscale channels (CYX, one marker
per plane), and the two need opposite handling -- unmixing vs picking a plane.
Non-TIFF files (PNG/JPEG) are read with skimage as YX or YXS.

Pixel size is read from OME / ImageJ calibration or the TIFF resolution tags,
but a resolution in inches at 72/96/300 dpi is what image editors write by
default (seen on real Photoshop-saved sections: 72 dpi), not a calibration,
so it is reported as missing rather than turned into 352.8 um/pixel.
"""
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

_TIFF_SUFFIXES = {".tif", ".tiff", ".lsm", ".btf", ".tf8"}
_VENDOR_SUFFIXES = {".czi", ".nd2", ".lif", ".oib", ".oir", ".vsi", ".ims"}
_EDITOR_DPI = {72.0, 96.0, 150.0, 300.0, 600.0}
_UM_PER = {"µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0, "\\u00b5m": 1.0, "nm": 1e-3, "mm": 1e3}


# ----------------------------------------------------------------- reading

def _pixel_size(tf, page):
    """{'pixel_size_um': float | None, 'pixel_size_note': str}."""
    if tf.ome_metadata:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
        if m:
            u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tf.ome_metadata)
            scale = _UM_PER.get(u.group(1) if u else "µm")
            if scale:
                return {"pixel_size_um": float(m.group(1)) * scale, "pixel_size_note": "OME PhysicalSizeX"}
    if "XResolution" not in page.tags:
        return {"pixel_size_um": None, "pixel_size_note": "no resolution tag"}
    num, den = page.tags["XResolution"].value
    res = num / den if den else 0.0            # pixels per unit
    if res <= 0:
        return {"pixel_size_um": None, "pixel_size_note": "resolution tag is 0"}
    unit = str((tf.imagej_metadata or {}).get("unit", "")).strip().lower()
    if unit in _UM_PER:
        return {"pixel_size_um": _UM_PER[unit] / res, "pixel_size_note": f"ImageJ calibration ({unit})"}
    runit = int(page.tags["ResolutionUnit"].value) if "ResolutionUnit" in page.tags else 2
    if runit == 3:
        return {"pixel_size_um": 1e4 / res, "pixel_size_note": "TIFF resolution (per cm)"}
    if runit == 2:
        if round(res, 3) in _EDITOR_DPI:
            return {"pixel_size_um": None,
                    "pixel_size_note": f"{res:g} dpi -- an image editor's default, not a microscope calibration"}
        return {"pixel_size_um": 25400 / res, "pixel_size_note": "TIFF resolution (per inch) -- verify"}
    return {"pixel_size_um": None, "pixel_size_note": "resolution tag has no physical unit"}


def _channel_names(tf, n_channels):
    if tf.ome_metadata:
        names = re.findall(r'<Channel\b[^>]*\bName="([^"]*)"', tf.ome_metadata)
        if len(names) >= n_channels > 0:
            return names[:n_channels]
    labels = (tf.imagej_metadata or {}).get("Labels")
    if isinstance(labels, (list, tuple)) and n_channels > 0 and len(labels) >= n_channels \
            and len(labels) % n_channels == 0:
        return [str(x) for x in labels[:n_channels]]
    return []


def read_section(path):
    """(array, axes, info) with singleton axes dropped; axes in tifffile's
    letters (Y X S=RGB samples C=channel Z T, Q/I = unlabelled)."""
    path = Path(path)
    suffix = path.suffix.lower()
    info = {"file": str(path), "format": suffix.lstrip(".")}
    if suffix in _VENDOR_SUFFIXES:
        raise ValueError(f"{path.name}: {suffix} needs a vendor reader that is not installed in this env -- "
                         "export to TIFF first (Fiji/Bio-Formats: File > Save As > Tiff keeps every channel)")
    if suffix in _TIFF_SUFFIXES:
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            arr, axes, page = series.asarray(), series.axes, tf.pages[0]
            info.update(photometric=page.photometric.name, bits_per_sample=int(page.bitspersample),
                        software=str(page.tags["Software"].value) if "Software" in page.tags else "",
                        n_series=len(tf.series), pyramid_levels=len(series.levels), **_pixel_size(tf, page))
            n_ch = arr.shape[axes.index("C")] if "C" in axes else 0
            info["channel_names"] = _channel_names(tf, n_ch)
    else:
        from skimage import io
        arr = io.imread(str(path))
        axes = "YX" if arr.ndim == 2 else "YXS"
        info.update(photometric="RGB" if arr.ndim == 3 else "MINISBLACK", bits_per_sample=arr.dtype.itemsize * 8,
                    software="", n_series=1, pyramid_levels=1, channel_names=[], pixel_size_um=None,
                    pixel_size_note=f"{suffix.lstrip('.').upper()} carries no calibration")
    keep = [i for i, n in enumerate(arr.shape) if n > 1 or axes[i] in "YX"]
    arr = arr.reshape([arr.shape[i] for i in keep])
    return arr, "".join(axes[i] for i in keep), info


def to_planes(arr, axes, z_projection="max"):
    """-> (planes (Y, X, n), kind, axes_note) with kind 'gray' | 'channels' | 'rgb'.
    z_projection: 'max' | 'mean' | an integer plane index. An unlabelled
    extra axis (Q/I) of at most 10 planes is taken as channels; a longer one
    is ambiguous (channels? Z?) and raises."""
    axes = list(axes)
    notes = []
    if "T" in axes:
        raise ValueError(f"axes {''.join(axes)}: time series are not supported -- export one timepoint")
    if "Z" in axes:
        zi = axes.index("Z")
        nz = arr.shape[zi]
        if z_projection == "max":
            arr = arr.max(axis=zi)
        elif z_projection == "mean":
            arr = arr.mean(axis=zi, dtype=np.float32)
        elif isinstance(z_projection, (int, np.integer)) and not isinstance(z_projection, bool):
            arr = np.take(arr, int(z_projection), axis=zi)
        else:
            raise ValueError(f"z_projection must be 'max', 'mean' or a plane index, got {z_projection!r}")
        axes.pop(zi)
        notes.append(f"{nz} Z planes -> {z_projection} projection")
    for i, a in enumerate(axes):
        if a not in "YXSC":
            if arr.shape[i] <= 10 and "C" not in axes and "S" not in axes:
                notes.append(f"unlabelled axis {a} ({arr.shape[i]} planes) taken as channels")
                axes[i] = "C"
            else:
                raise ValueError(f"axes {''.join(axes)} shape {arr.shape}: cannot tell what axis {a} is -- "
                                 "re-export with channels labelled (Fiji: Image > Hyperstacks)")
    if "S" in axes and "C" in axes:
        raise ValueError(f"axes {''.join(axes)}: RGB samples inside each channel are not supported")
    order = [axes.index("Y"), axes.index("X")] + [i for i, a in enumerate(axes) if a in "SC"]
    arr = np.transpose(arr, order)
    if arr.ndim == 2:
        return arr[..., None], "gray", notes
    if "S" in axes and arr.shape[-1] in (3, 4):
        if arr.shape[-1] == 4:
            notes.append("alpha plane dropped")
        return arr[..., :3], "rgb", notes
    return arr, "channels", notes


def rgb_unmix_weights(panel_colors, target):
    """w (length 3) with  target marker = w . (R, G, B)  in an ADDITIVE RGB
    composite whose markers were displayed in panel_colors ({name: [r, g, b]},
    0-255, the colour at full intensity). E.g. DAPI blue + Sox9 magenta +
    GFP green gives DAPI = B - R.

    Such a w exists exactly when the target's colour is not a mix of the
    others' -- more than three markers can still work if the target's colour
    stays separable (DAPI blue with red/green/yellow markers: DAPI = B). When
    it is not (DAPI blue next to magenta AND red: magenta = red + blue), no
    per-pixel weighting can recover it, and this raises instead of returning
    a best fit that would silently mix markers into the registration image."""
    if target not in panel_colors:
        raise ValueError(f"channel {target!r} is not a key of panel_colors ({sorted(panel_colors)})")
    names = list(panel_colors)
    C = np.array([panel_colors[n] for n in names], dtype=float) / 255.0     # k x 3
    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError(f"panel_colors values must be [r, g, b] (0-255), got {panel_colors}")
    e = np.zeros(len(names))
    e[names.index(target)] = 1.0
    w = np.linalg.lstsq(C, e, rcond=None)[0]
    if np.abs(C @ w - e).max() > 1e-6:
        others = [n for n in names if n != target]
        raise ValueError(
            f"{target} {panel_colors[target]} cannot be separated from {others} in an RGB composite -- its "
            "colour is a mix of theirs, so every pixel is ambiguous. Use channel: sum (all markers together), "
            f"or export {target} as its own greyscale image.")
    return w


def load_registration_image(path, channel=None, panel_colors=None, z_projection="max", log=print):
    """(rows, cols) float32 registration image from any supported section file.

    channel:
      None   -- only for single-channel files
      int    -- that plane: a greyscale channel, or R/G/B of a composite
      "sum"  -- all planes summed (every marker together)
      name   -- multichannel greyscale: a channel name from the file's own
                OME/ImageJ metadata; RGB composite: a key of panel_colors,
                recovered by linear unmixing (rgb_unmix_weights). Unmixing
                assumes an additive merge; clipped pixels are counted in the log.
    """
    arr, axes, info = read_section(path)
    planes, kind, notes = to_planes(arr, axes, z_projection)
    n = planes.shape[-1]
    name = Path(path).name
    log(f"  {name}: axes {axes}, {kind}, {n} plane(s), {arr.dtype}" + "".join(f"; {x}" for x in notes))
    if kind == "gray":
        if isinstance(channel, str) and channel != "sum":
            raise ValueError(f"{name} is single-channel but channel is {channel!r}")
        return planes[..., 0].astype(np.float32)
    if channel is None:
        raise ValueError(f"{name} has {n} {'colour planes (RGB composite)' if kind == 'rgb' else 'channels'} "
                         "-- set channel (run scripts/inspect_sections.py to see which is which)")
    if isinstance(channel, (int, np.integer)) and not isinstance(channel, bool):
        if not 0 <= int(channel) < n:
            raise ValueError(f"channel {channel} out of range for {name} ({n} planes)")
        return planes[..., int(channel)].astype(np.float32)
    if channel == "sum":
        return planes.sum(axis=-1, dtype=np.float32)
    if kind == "channels":
        names = info.get("channel_names") or []
        if channel in names:
            return planes[..., names.index(channel)].astype(np.float32)
        raise ValueError(f"channel {channel!r}: {name} has {n} separate greyscale channels, so give an index "
                         f"0..{n - 1}" + (f" or one of its channel names {names}" if names else "")
                         + " -- panel_colors only applies to RGB composites")
    if not panel_colors:
        raise ValueError(f"channel {channel!r} on an RGB composite needs panel_colors (the colour each marker "
                         "was assigned when the channels were merged)")
    w = rgb_unmix_weights(panel_colors, channel)
    log(f"  {channel} = " + " ".join(f"{c:+.2f}*{k}" for c, k in zip(w, "RGB") if abs(c) > 1e-9))
    if np.issubdtype(planes.dtype, np.integer):
        saturated = float((planes == np.iinfo(planes.dtype).max).any(axis=-1).mean())
        if saturated > 0.001:
            log(f"  warning: {100 * saturated:.1f}% of pixels have a saturated colour plane -- the composite "
                "was clipped there, so unmixing is not exact on those pixels")
    return np.clip(np.tensordot(planes.astype(np.float32), w.astype(np.float32), axes=([-1], [0])), 0, None)


# -------------------------------------------------------------- inspection

def inspect_image(path, z_projection="max", max_side_px=2000):
    """Property report for one section file, computed on a stride-downsampled
    view. Returns (info, view (Y, X, n) float32, tissue mask on the view).

    Per plane: intensity percentiles, zero/saturated fractions, number of
    distinct levels in tissue (few levels = heavily compressed/stretched
    8-bit), coverage (fraction of tissue brighter than the background's 99th
    percentile) and sparseness (tissue p99 / p50). A registration channel
    wants high coverage and low sparseness -- DAPI / autofluorescence rather
    than a sparse marker; `suggested_channel` is that ranking, a hint to check
    against the thumbnails, not a decision."""
    arr, axes, info = read_section(path)
    planes, kind, notes = to_planes(arr, axes, z_projection)
    h, w, n = planes.shape
    info.update(axes=axes, kind=kind, notes=notes, height_px=h, width_px=w, n_planes=n, dtype=str(arr.dtype))
    stride = max(1, int(np.ceil(max(h, w) / max_side_px)))
    view = planes[::stride, ::stride].astype(np.float32)
    info["inspect_stride"] = stride
    top = float(np.iinfo(planes.dtype).max) if np.issubdtype(planes.dtype, np.integer) else None
    info["observed_max"] = float(view.max())
    if top and top > 255 and info["observed_max"] <= 4095:
        info["notes"].append(f"{planes.dtype} but max {info['observed_max']:.0f} -- 12-bit data in a 16-bit container")

    total = ndimage.gaussian_filter(view.sum(axis=-1), 2)
    logv = np.log1p(total - total.min())
    try:
        tissue = ndimage.binary_fill_holes(logv > threshold_otsu(logv))
    except ValueError:
        tissue = np.ones(total.shape, bool)
    info["tissue_pct"] = float(100 * tissue.mean())

    names = info.get("channel_names") or []
    stats = []
    for i in range(n):
        v = view[..., i]
        tv, bg = v[tissue], v[~tissue]
        thr = np.percentile(bg, 99) if bg.size else 0.0
        p50 = float(np.percentile(tv, 50)) if tv.size else 0.0
        p99 = float(np.percentile(tv, 99)) if tv.size else 0.0
        sample = tv if tv.size <= 1_000_000 else tv[:: tv.size // 1_000_000]
        stats.append({
            "plane": i,
            "label": names[i] if i < len(names) else ("RGB"[i] if kind == "rgb" else f"C{i}"),
            "min": float(v.min()), "p50_tissue": p50, "p99_tissue": p99, "max": float(v.max()),
            "bg_p99": float(thr), "zero_pct": float(100 * (v == 0).mean()),
            "saturated_pct": float(100 * (v == top).mean()) if top else 0.0,
            "levels_in_tissue": int(np.unique(sample).size),
            "coverage_pct": float(100 * (tv > thr).mean()) if tv.size else 0.0,
            "sparseness": float(p99 / max(p50, 1.0)),
        })
    info["planes"] = stats
    ranked = sorted(stats, key=lambda s: (-s["coverage_pct"] / np.log2(2 + s["sparseness"])))
    # No suggestion for an RGB composite: each colour plane mixes markers, so
    # ranking planes answers the wrong question (on two real Photoshop
    # composites the ranking picked B for one and G for the other).
    info["suggested_channel"] = ranked[0]["plane"] if kind == "channels" else None

    if kind == "rgb":
        rgb_t = view[tissue]
        info["rgb_is_gray"] = bool((np.ptp(rgb_t, axis=1) <= 2).mean() > 0.99) if rgb_t.size else False
        # "Lit" = the brightest 5% of each plane's tissue pixels. The background
        # 99th percentile is far too low for this: on real composites it is
        # 2-9 grey levels, so autofluorescence lit all three planes in >90% of
        # tissue and every combination read as R+G+B.
        on = rgb_t > np.array([max(s["bg_p99"], np.percentile(rgb_t[:, i], 95))
                               for i, s in enumerate(stats)])[None, :]
        code = on[:, 0] * 1 + on[:, 1] * 2 + on[:, 2] * 4
        lit = code > 0
        combos = {}
        for c in range(1, 8):
            label = "+".join(ch for ch, bit in zip("RGB", (1, 2, 4)) if c & bit)
            combos[label] = float(100 * (code[lit] == c).mean()) if lit.any() else 0.0
        info["rgb_cooccurrence_pct"] = combos
    return info, view, tissue


def render_inspection(png_path, info, view, tissue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = view.shape[-1]
    panels = ([("composite (as merged)", None)] if info["kind"] == "rgb" else []) + \
        [(f"plane {s['plane']} [{s['label']}]  cov {s['coverage_pct']:.0f}%  sparse {s['sparseness']:.1f}"
          + (f"  sat {s['saturated_pct']:.1f}%" if s["saturated_pct"] >= 0.1 else ""), s["plane"])
         for s in info["planes"]]
    aspect = view.shape[1] / view.shape[0]
    cols = min(len(panels), 3)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * aspect * cols, 4.2 * rows + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, plane) in zip(axes.ravel(), panels):
        if plane is None:
            ax.imshow(np.clip(view[..., :3] / max(view.max(), 1), 0, 1))
        else:
            v = view[..., plane]
            hi = np.percentile(v[tissue], 99.5) if tissue.any() else v.max()
            ax.imshow(v, cmap="gray", vmin=0, vmax=max(hi, 1))
            if plane == info.get("suggested_channel"):
                title = "* " + title
        ax.contour(tissue.astype(float), levels=[0.5], colors="c", linewidths=0.6)
        ax.set_title(title, fontsize=8)
    px = info.get("pixel_size_um")
    fig.suptitle(f"{Path(info['file']).name}   {info['axes']} {info['kind']} {info['dtype']}   "
                 f"{info['width_px']}x{info['height_px']} px   pixel size: "
                 f"{f'{px:.4g} um' if px else 'UNKNOWN'}   "
                 "(expected mounting: anterior left, dorsal up;  * = suggested registration channel)", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=90)
    plt.close(fig)
