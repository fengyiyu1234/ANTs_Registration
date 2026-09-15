"""2D sagittal sections -> a 3D atlas: first find which (possibly oblique)
sagittal plane of the atlas each section is, by running Similarity
registrations against a grid of candidate planes; then Affine + SyN the
section onto the plane that won.

Why a search at all: ANTs has no slice-to-volume registration, and a 2D
registration against the WRONG plane still converges -- SyN will bend a
1.2 mm-lateral section onto a 0.6 mm-lateral plane and the overlay looks
fine. So the plane has to be decided first, by a transform that cannot bend.
Similarity (rotation + ONE scale + translation) is used for that rather than
Affine on purpose: lateral sagittal planes are smaller than medial ones, and
with the section's pixel size known, a plane that needs the section shrunk or
blown up beyond search.scale_range is ruled out. A full Affine absorbs that
size difference with anisotropic stretch and flattens the score landscape, so
it runs only once, on the chosen plane, before SyN.

Atlas convention (CANONICAL): template/annotation arrays oriented so that
axis0 = left->right, axis1 = anterior->posterior, axis2 = dorsal->ventral --
the same thing the 3D pipeline's atlas.orientation produces, and what the
DevCCF preset's [1, -3, 2] already gives. A plane image is then a 2D ANTs
image with x = anterior->posterior, y = dorsal->ventral, in microns relative
to the plane centre. check_canonical_orientation tests A-P and D-V against
the ontology; left/right cannot be tested and does not matter here.

A sagittal section cannot tell which hemisphere it came from -- the mirror
plane on the other side looks the same -- so the search runs over the
distance from the midline (ml_um >= 0) and reports that distance, not a side.

Section coordinates: a section image (tifffile (rows, cols)) becomes a 2D
ANTs image with x = column, y = row and pixel (col, row) at physical
(col * pixel_size_um, row * pixel_size_um). Cell centroids given in original
pixels therefore map in by multiplying by the pixel size, at any
downsampling -- see downsample_section.
"""
import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import ants
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu

from . import atlas_utils, transforms

SEARCH_DEFAULTS = {
    "search_res_um": 40,
    # Distance of the plane centre from the midline; null upper bound = the
    # atlas tissue's own lateral edge.
    "ml_range_um": [0, None],
    "coarse_ml_step_um": 200,
    # Applied to yaw AND roll, so 5 values = 25 tilt combinations per ML step.
    "coarse_angles_deg": [-10, -5, 0, 5, 10],
    "fine_ml_step_um": 40,
    "fine_angle_step_deg": 2.5,
    "fine_top_k": 3,
    # sqrt(det) of the Similarity, section size / atlas size. Candidates
    # outside are not allowed to win. null disables the check.
    "scale_range": [0.75, 1.3],
    "n_workers": 8,
    # Candidates at least this far (ML) from the winner are the "far"
    # competitors reported as far_gap -- how decisively the ML position won.
    "far_um": 300,
    # Last stage: pattern search in (ml, yaw, roll) from the best few distinct
    # results so far -- all 26 neighbours at the current step in parallel,
    # move to the best, halve the steps when none improves, stop below the
    # minimum steps. Grids alone are not enough: the score basin is narrow
    # and couples ML with roll. Measured on a DevCCF phantom (truth ml 1500,
    # yaw 3.3, roll -6.2): the truth out-scored every other plane under every
    # scoring setting tried, yet the grid search returned 1400/5/-2.5 because
    # no grid point sat inside the basin -- 1480/5/-5, only 20 um and 1.2 deg
    # away, scored worse than the wrong answer. Out-of-plane error is what no
    # 2D registration afterwards can undo (that run: 160 um median cell error
    # for Affine and SyN alike). refine_starts: 0 skips the stage.
    "refine_starts": 3,
    "refine_min_ml_step_um": 10,
    "refine_min_angle_step_deg": 0.5,
}

REGISTRATION_DEFAULTS = {
    "register_res_um": 20,
    "syn": True,
    # "mattes" or "CC". CC assumes intensities correlate LOCALLY LINEARLY with
    # the template; immunofluorescence vs an LSFM/STPT template often does not
    # (e.g. fibre tracts bright in one, dark in the other). Measured on DevCCF
    # phantom sections with a non-monotonic contrast change (median 3D cell
    # error, ml 400 / ml 1500): Affine alone 26 / 18 um, SyN+CC 38 / 34 um
    # (worse than no SyN), SyN+mattes 26 / 18 um with in-plane error down to
    # 10-12 um. syn_sampling is the CC radius in pixels for CC (see
    # register.register_to_atlas), the histogram bin count for mattes.
    "syn_metric": "mattes",
    "syn_sampling": 32,
    "reg_iterations": [100, 70, 30],
    "grad_step": 0.1,
    "flow_sigma": 3,
    "random_seed": 42,
    "n4_bias_correction": True,
    "intensity_clip_percentiles": [0.5, 99.5],
}

# Where the anatomical direction points in a displayed image (row 0 at top).
_IMAGE_DIRS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}
_MIN_PLANE_TISSUE_PX = 50


@dataclass(frozen=True)
class PlaneParams:
    ml_um: float      # plane centre's distance from the midline
    yaw_deg: float    # tilt about the D-V axis: ML position changes along A-P
    roll_deg: float   # tilt about the A-P axis: ML position changes along D-V


class SagittalAtlas:
    """A canonical-orientation atlas (see module docstring) cropped to its
    tissue, that can be cut along arbitrary near-sagittal planes."""

    def __init__(self, template, annotation, res_um, midline_vox=None, margin_vox=10):
        tissue = annotation > 0
        if not tissue.any():
            raise ValueError("atlas annotation has no nonzero voxels")
        bbox = []
        for ax in range(3):
            idx = np.where(tissue.any(axis=tuple(a for a in range(3) if a != ax)))[0]
            bbox.append((int(idx[0]), int(idx[-1])))
        lo = [max(0, b[0] - margin_vox) for b in bbox]
        hi = [min(s, b[1] + 1 + margin_vox) for b, s in zip(bbox, annotation.shape)]
        crop = tuple(slice(a, b) for a, b in zip(lo, hi))

        self.res_um = float(res_um)
        self.offset_vox = np.array(lo, dtype=float)
        self.annotation = np.ascontiguousarray(annotation[crop])
        tmpl = np.asarray(template[crop], dtype=np.float32)
        inside = ndimage.binary_dilation(self.annotation > 0, iterations=1)
        hi_val = np.percentile(tmpl[inside], 99.5)
        # Background forced to exactly 0 so the atlas silhouette is as crisp as
        # the section's (whose background is zeroed by prepare_section_image).
        self.template = np.where(inside, np.clip(tmpl / max(hi_val, 1e-8), 0, 1), 0).astype(np.float32)

        mid = (bbox[0][0] + bbox[0][1]) / 2 if midline_vox is None else float(midline_vox)
        self.midline_vox = mid
        self.center_vox = np.array([mid, (bbox[1][0] + bbox[1][1]) / 2, (bbox[2][0] + bbox[2][1]) / 2])
        # 1.15: room for the tilted plane to still cover the tissue.
        self.half_extent_um = tuple(1.15 * (b[1] - b[0] + 1) / 2 * self.res_um for b in bbox[1:])
        self.ml_max_um = max(mid - bbox[0][0], bbox[0][1] - mid) * self.res_um
        self._smoothed = {}

    def _template_at(self, res_um):
        """Template anti-aliased for sampling at res_um (sigma rule as
        io_utils.resample_to_isotropic); unsmoothed at or finer than native."""
        if res_um <= self.res_um:
            return self.template
        if res_um not in self._smoothed:
            self._smoothed[res_um] = ndimage.gaussian_filter(self.template, 0.5 * res_um / self.res_um)
        return self._smoothed[res_um]

    def _frame(self, p):
        y, r = np.radians(p.yaw_deg), np.radians(p.roll_deg)
        e_u = np.array([-np.sin(y), np.cos(y), 0.0])
        e_v = np.array([np.cos(y) * np.sin(r), np.sin(y) * np.sin(r), np.cos(r)])
        centre_um = (self.center_vox + [p.ml_um / self.res_um, 0.0, 0.0]) * self.res_um
        return centre_um, e_u, e_v

    def plane_to_atlas_um(self, u, v, p):
        """Plane physical (u, v) in microns -> (..., 3) microns on the oriented,
        UNCROPPED atlas grid (voxel index * res_um)."""
        centre, e_u, e_v = self._frame(p)
        return centre + np.multiply.outer(np.asarray(u, float), e_u) + np.multiply.outer(np.asarray(v, float), e_v)

    def plane_grid(self, res_um):
        hu, hv = self.half_extent_um
        shape = (int(np.floor(2 * hu / res_um)) + 1, int(np.floor(2 * hv / res_um)) + 1)
        return shape, (-hu, -hv)

    def sample_plane(self, p, res_um):
        """(template, annotation, origin) of plane p on a res_um grid, arrays
        in (x = A->P, y = D->V) order. The annotation keeps its integer dtype:
        order-0 map_coordinates interpolates in float64, which holds every
        CCFv3 id exactly (io_utils._LABEL_DTYPE_NOTE is about float32)."""
        (nu, nv), origin = self.plane_grid(res_um)
        uu, vv = np.meshgrid(origin[0] + np.arange(nu) * res_um, origin[1] + np.arange(nv) * res_um,
                             indexing="ij")
        coords = np.moveaxis(self.plane_to_atlas_um(uu, vv, p) / self.res_um - self.offset_vox, -1, 0)
        tmpl = ndimage.map_coordinates(self._template_at(res_um), coords, order=1, cval=0.0)
        ann = ndimage.map_coordinates(self.annotation, coords, order=0, cval=0)
        return tmpl.astype(np.float32), ann, origin

    def lookup(self, xyz_um):
        """Annotation id at (N, 3) microns on the uncropped grid; 0 outside."""
        idx = np.rint(np.asarray(xyz_um) / self.res_um - self.offset_vox).astype(int)
        inside = np.all((idx >= 0) & (idx < self.annotation.shape), axis=1)
        ids = np.zeros(len(idx), dtype=self.annotation.dtype)
        ids[inside] = self.annotation[tuple(idx[inside].T)]
        return ids


def check_canonical_orientation(annotation, structures):
    """Best-effort A-P / D-V check against the ontology. Returns (notes,
    failed): failed only when both structures of a pair were found and sit
    the wrong way round -- a structure that is missing from this ontology
    is a skipped check, not a failure."""
    if not structures:
        return ["no ontology -- atlas orientation not checked"], False
    ann = annotation[::2, ::2, ::2]

    def centroid(names):
        mask = np.isin(ann, list(atlas_utils._structure_ids_matching(structures, names)))
        return np.argwhere(mask).mean(axis=0) if mask.any() else None

    notes, failed = [], False
    for axis, first, second, what in ((1, ["olfactory bulb"], ["cerebell"], "anterior->posterior"),
                                      (2, ["layer 1"], ["hypothalamus"], "dorsal->ventral")):
        a, b = centroid(first), centroid(second)
        if a is None or b is None:
            notes.append(f"axis{axis} ({what}): {first[0]!r}/{second[0]!r} not both in ontology -- skipped")
        elif a[axis] < b[axis]:
            notes.append(f"axis{axis} ({what}): ok ({first[0]} {a[axis]:.0f} < {second[0]} {b[axis]:.0f})")
        else:
            notes.append(f"axis{axis} is NOT {what}: {first[0]} at {2 * a[axis]:.0f}, "
                         f"{second[0]} at {2 * b[axis]:.0f} -- fix atlas.orientation")
            failed = True
    return notes, failed


# ----------------------------------------------------------------- section IO

def load_section_array(path, channel=None):
    """(rows, cols) float32 from a 2D TIFF, or one channel of a multichannel
    one (channel axis = the shortest axis)."""
    arr = np.squeeze(tifffile.imread(str(path)))
    if arr.ndim == 3:
        if channel is None:
            raise ValueError(f"{path} has shape {arr.shape} (multichannel) -- set `channel` (e.g. the DAPI index)")
        arr = np.take(arr, int(channel), axis=int(np.argmin(arr.shape)))
    elif arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image or 2D + channels, got shape {arr.shape}")
    return arr.astype(np.float32)


def downsample_section(arr_rc, pixel_size_um, target_um):
    """(rows, cols) raw pixels -> ANTs 2D image (x = column, y = row) at
    target_um. Integer block-mean first (anti-aliasing), then a linear
    resample for the remaining non-integer factor. The block-mean image's
    origin is set to its blocks' centres, so physical coordinates stay
    (col * px, row * px) of the ORIGINAL pixels."""
    f = max(1, int(target_um // pixel_size_um))
    arr = arr_rc
    if f > 1:
        h, w = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
        arr = arr[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    img = ants.from_numpy(np.ascontiguousarray(arr.T.astype(np.float32)),
                          spacing=(pixel_size_um * f,) * 2, origin=((f - 1) / 2 * pixel_size_um,) * 2)
    return ants.resample_image(img, (float(target_um),) * 2, use_voxels=False, interp_type=0)


def _load_mask_on(path, raw_shape_rc, pixel_size_um, target):
    """A hand-made 2D mask (nonzero = on) covering the same field of view as
    the raw section at ANY pixel scale -> boolean array on target's grid."""
    m = np.squeeze(tifffile.imread(str(path))) > 0
    if m.ndim != 2:
        raise ValueError(f"mask {path} must be 2D, got shape {m.shape}")
    sx = pixel_size_um * raw_shape_rc[1] / m.shape[1]
    sy = pixel_size_um * raw_shape_rc[0] / m.shape[0]
    img = ants.from_numpy(np.ascontiguousarray(m.T.astype(np.float32)), spacing=(sx, sy),
                          origin=((sx - pixel_size_um) / 2, (sy - pixel_size_um) / 2))
    return ants.resample_image_to_target(img, target, interp_type="nearestNeighbor").numpy() > 0.5


def auto_tissue_mask(img, threshold=None, min_component_fraction=0.02):
    """Tissue vs slide background. threshold=None: Otsu on the log of the
    smoothed image -- workable on a DAPI/autofluorescence section at tens of
    microns, where slide background is a separate mode. Give a fixed value
    (raw intensity units) if the QC outline says otherwise; on light-sheet
    data Otsu only picked out bright cells, so never assume it works."""
    sm = ndimage.gaussian_filter(img.numpy(), 1.0)
    if threshold is None:
        logv = np.log1p(sm - sm.min())
        mask = logv > threshold_otsu(logv)
    else:
        mask = sm > float(threshold)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = np.isin(lab, np.where(sizes >= min_component_fraction * sizes.max())[0] + 1)
    return mask


def prepare_section_image(img, tissue, n4=True, clip=(0.5, 99.5)):
    """N4 inside the tissue, percentile clip on tissue pixels, scale to
    [0, 1], background set to exactly 0 (matching SagittalAtlas.template)."""
    like = dict(spacing=img.spacing, origin=img.origin, direction=img.direction)
    if n4:
        arr = img.numpy()
        pos = ants.from_numpy(arr - arr.min() + 1.0, **like)
        img = ants.n4_bias_field_correction(pos, mask=ants.from_numpy(tissue.astype(np.float32), **like))
    arr = img.numpy()
    lo, hi = np.percentile(arr[tissue], clip)
    arr = np.where(tissue, np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1), 0)
    return ants.from_numpy(arr.astype(np.float32), **like)


def _resample_2d(img, res_um, nearest=False):
    if nearest:
        return ants.resample_image(img, (float(res_um),) * 2, use_voxels=False, interp_type=1)
    sigma = [0.5 * res_um / s if s < res_um else 0.0 for s in img.spacing]
    if any(sigma):
        img = ants.from_numpy(ndimage.gaussian_filter(img.numpy(), sigma), spacing=img.spacing,
                              origin=img.origin, direction=img.direction)
    return ants.resample_image(img, (float(res_um),) * 2, use_voxels=False, interp_type=0)


# ------------------------------------------------------------ linear helpers

def _rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _linear_part(tx):
    """(M, t) with tx(p) = M p + t, read off by applying tx to three points,
    so it does not matter which ITK linear type antsRegistration collapsed
    the stages into (Similarity and Affine lay out their parameters
    differently)."""
    o = np.array(tx.apply_to_point((0.0, 0.0)))
    ex = np.array(tx.apply_to_point((1.0, 0.0))) - o
    ey = np.array(tx.apply_to_point((0.0, 1.0))) - o
    return np.column_stack([ex, ey]), o


def _write_linear(M, t, path):
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="double", dimension=2,
                                    matrix=np.asarray(M, float), center=(0.0, 0.0),
                                    translation=np.asarray(t, float))
    ants.write_transform(tx, str(path))


def _orthogonal_part(M):
    from scipy.linalg import polar
    return polar(M)[0]


def describe_matrix(M):
    """size_ratio (section / atlas), in-plane rotation, mirrored -- for a
    plane->section matrix M = s * R(theta) [* diag(1, -1) if mirrored]."""
    det = np.linalg.det(M)
    R = _orthogonal_part(M)
    mirrored = det < 0
    if mirrored:
        R = R @ np.diag([1.0, -1.0])
    return {"size_ratio": float(np.sqrt(abs(det))),
            "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "mirrored": bool(mirrored)}


def orientation_inits(sec):
    """Starting plane->section matrices for the orientation stage. With
    `anterior` + `dorsal` hints (where they point in the displayed image)
    the mirror state is known and only a +-20 degree wobble is tried;
    without, both mirror states x 12 rotations."""
    anterior, dorsal = sec.get("anterior"), sec.get("dorsal")
    if anterior and dorsal:
        a, d = np.array(_IMAGE_DIRS[anterior]), np.array(_IMAGE_DIRS[dorsal])
        # Plane +x is posterior, +y ventral: the columns say where those two
        # directions point in the section image.
        base = np.column_stack([-a, -d])
        return [_rot(th) @ base for th in (-20, -10, 0, 10, 20)]
    return [_rot(th) @ np.diag([1.0, s]) for s in (1.0, -1.0) for th in range(0, 360, 30)]


# --------------------------------------------------------------- the search

_W = {}


def _worker_init(section_arr, spacing, origin, keep_arr, seed):
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    ants.config.set_ants_deterministic(True, seed)
    _W["section"] = ants.from_numpy(section_arr, spacing=spacing, origin=origin)
    _W["keep"] = None if keep_arr is None else ants.from_numpy(keep_arr, spacing=spacing, origin=origin)


def _similarity_task(task):
    plane = ants.from_numpy(task["plane"], spacing=(task["res"],) * 2, origin=task["origin"])
    with tempfile.TemporaryDirectory() as tmp:
        init_path = os.path.join(tmp, "init.mat")
        _write_linear(task["M"], task["t"], init_path)
        reg = ants.registration(
            fixed=plane, moving=_W["section"], type_of_transform="Similarity",
            initial_transform=init_path, aff_metric="mattes",
            aff_iterations=(500, 250, 100), aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0),
            moving_mask=_W["keep"], mask_all_stages=True, outprefix=os.path.join(tmp, "r_"), verbose=False)
        M, t = _linear_part(ants.read_transform(reg["fwdtransforms"][0]))
        fixed_mask = None
        if _W["keep"] is not None:
            fixed_mask = ants.apply_transforms(plane, _W["keep"], reg["fwdtransforms"],
                                               interpolator="nearestNeighbor")
        try:
            score, error = float(ants.image_similarity(plane, reg["warpedmovout"], "MattesMutualInformation",
                                                       fixed_mask=fixed_mask)), ""
        except RuntimeError as exc:
            # Mattes raises (ITK "Zero-valued spacing": a histogram bin width
            # of 0) when the intensities it sees have no range at all. Seen on
            # a DevCCF run for a plane at the atlas's lateral edge (1892 tissue
            # px) whose Similarity had diverged -- a candidate that could never
            # win, which uncaught took the whole search down with it.
            score, error = float("inf"), str(exc).strip().splitlines()[-1]
    return {"score": score, "error": error, "M": M, "t": t}


class _Runner:
    """Runs _similarity_task over a list of tasks, in a spawn process pool
    (one ITK thread each, fixed seed -- see the run-to-run-noise note in
    PROGRESS_LOG) or in-process when n_workers <= 1.

    spawn re-imports the calling script in every worker, so a script that
    calls process_section/search_plane at top level, without an
    `if __name__ == "__main__":` guard, makes each worker re-run the whole
    script and die -- and the parent then HANGS silently rather than
    raising (seen: parent blocked in a pipe write at 0% CPU). Jupyter is
    fine; bare scripts need the guard, or n_workers: 1."""

    def __init__(self, section, keep, n_workers, seed):
        self.initargs = (section.numpy(), section.spacing, section.origin,
                         None if keep is None else keep.numpy().astype(np.float32), seed)
        self.n_workers = int(n_workers)
        self.pool = None

    def __enter__(self):
        if self.n_workers > 1:
            self.pool = ProcessPoolExecutor(self.n_workers, mp_context=mp.get_context("spawn"),
                                            initializer=_worker_init, initargs=self.initargs)
        else:
            _worker_init(*self.initargs)
        return self

    def map(self, tasks):
        if self.pool is None:
            return [_similarity_task(t) for t in tasks]
        return list(self.pool.map(_similarity_task, tasks, chunksize=4))

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown()


def _evaluate(runner, atlas, section_centroid, candidates, res, scale_range, stage):
    """candidates: [(PlaneParams, initial plane->section 2x2)] -> result rows.
    Each start is centred by tissue centroids, so only rotation/mirror has to
    be supplied."""
    tasks, kept = [], []
    for params, M in candidates:
        tmpl, ann, origin = atlas.sample_plane(params, res)
        tissue = ann > 0
        if tissue.sum() < _MIN_PLANE_TISSUE_PX:
            continue
        c_plane = np.asarray(origin) + np.argwhere(tissue).mean(axis=0) * res
        tasks.append({"plane": tmpl, "origin": tuple(float(o) for o in origin), "res": float(res),
                      "M": M, "t": section_centroid - M @ c_plane})
        kept.append(params)
    rows = []
    for params, out in zip(kept, runner.map(tasks)):
        desc = describe_matrix(out["M"])
        valid = np.isfinite(out["score"]) and (
            scale_range is None or scale_range[0] <= desc["size_ratio"] <= scale_range[1])
        (m00, m01), (m10, m11) = out["M"]
        rows.append({"stage": stage, **asdict(params), "score": out["score"], "error": out["error"], **desc,
                     "valid": bool(valid),
                     "m00": m00, "m01": m01, "m10": m10, "m11": m11, "tx": out["t"][0], "ty": out["t"][1]})
    return rows


def _row_M(row):
    return np.array([[row["m00"], row["m01"]], [row["m10"], row["m11"]]])


def _row_params(row):
    return PlaneParams(row["ml_um"], row["yaw_deg"], row["roll_deg"])


def _pattern_search(runner, atlas, centroid, rows, res, scale_range, cfg, seen, log):
    """See SEARCH_DEFAULTS' refine_* comment. All starts advance together, so
    each round is one parallel batch. Returns the new rows (stage 'refine')."""
    ranked = sorted((r for r in rows if r["valid"] and r["stage"] != "orientation"), key=lambda r: r["score"])
    min_ml, min_ang = float(cfg["refine_min_ml_step_um"]), float(cfg["refine_min_angle_step_deg"])
    ml_sep = float(cfg["fine_ml_step_um"])
    starts = []
    for r in ranked:
        if all(abs(r["ml_um"] - s["ml_um"]) > ml_sep or abs(r["yaw_deg"] - s["yaw_deg"]) > 1
               or abs(r["roll_deg"] - s["roll_deg"]) > 1 for s in starts):
            starts.append(r)
        if len(starts) == cfg["refine_starts"]:
            break
    scored = {_row_params(r): r for r in rows if r["stage"] != "orientation"}
    # Start at half the fine grid's spacing: the fine grid already ruled out
    # moves that large, but not smaller ones in between its points.
    state = [{"best": s, "ml": ml_sep / 2, "ang": float(cfg["fine_angle_step_deg"]) / 2} for s in starts]
    new_rows, rounds = [], 0
    offsets = [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]
    while True:
        active = [st for st in state if st["ml"] >= min_ml or st["ang"] >= min_ang]
        if not active:
            break
        cands, owner = [], []
        for k, st in enumerate(active):
            b, R = st["best"], _orthogonal_part(_row_M(st["best"]))
            for o in offsets:
                p = PlaneParams(round(max(0.0, b["ml_um"] + (o[0] - 1) * st["ml"]), 3),
                                round(b["yaw_deg"] + (o[1] - 1) * st["ang"], 3),
                                round(b["roll_deg"] + (o[2] - 1) * st["ang"], 3))
                if p not in seen:
                    seen.add(p)
                    cands.append((p, R))
                    owner.append(k)
        batch = _evaluate(runner, atlas, centroid, cands, res, scale_range, "refine") if cands else []
        new_rows += batch
        for r in batch:
            scored[_row_params(r)] = r
        for k, st in enumerate(active):
            b = st["best"]
            neighbours = [scored.get(PlaneParams(round(max(0.0, b["ml_um"] + (o[0] - 1) * st["ml"]), 3),
                                                 round(b["yaw_deg"] + (o[1] - 1) * st["ang"], 3),
                                                 round(b["roll_deg"] + (o[2] - 1) * st["ang"], 3)))
                          for o in offsets]
            better = [r for r in neighbours if r is not None and r["valid"] and r["score"] < b["score"]]
            if better:
                st["best"] = min(better, key=lambda r: r["score"])
            else:
                st["ml"] /= 2
                st["ang"] /= 2
        rounds += 1
    final = min((st["best"] for st in state), key=lambda r: r["score"])
    log(f"  refine: {len(new_rows)} candidates in {rounds} rounds -> ml {final['ml_um']:.0f} "
        f"yaw {final['yaw_deg']:+.2f} roll {final['roll_deg']:+.2f} ({final['score']:.4f})")
    return new_rows


def search_plane(section, tissue, keep, atlas, sec, cfg, seed, log=print):
    """Orientation -> coarse grid -> fine grid around the best few. Returns
    (all rows as a DataFrame, the winning row as a dict). Lower score is
    better (ANTs' MI is negative)."""
    res = float(cfg["search_res_um"])
    centroid = np.asarray(section.origin) + np.argwhere(tissue).mean(axis=0) * np.asarray(section.spacing)
    lo, hi = cfg["ml_range_um"]
    hi = atlas.ml_max_um if hi is None else float(hi)
    scale_range = cfg["scale_range"]
    rows = []
    with _Runner(section, keep, cfg["n_workers"], seed) as runner:
        refs = [PlaneParams(lo + f * (hi - lo), 0.0, 0.0) for f in (0.3, 0.5, 0.7)]
        cands = [(p, M) for M in orientation_inits(sec) for p in refs]
        o_rows = _evaluate(runner, atlas, centroid, cands, res, None, "orientation")
        rows += o_rows
        best_o = min(o_rows, key=lambda r: r["score"])
        R = _orthogonal_part(_row_M(best_o))
        log(f"  orientation: rotation {best_o['rotation_deg']:.0f} deg, mirrored={best_o['mirrored']} "
            f"({len(o_rows)} starts)")

        angles = [float(a) for a in cfg["coarse_angles_deg"]]
        step = float(cfg["coarse_ml_step_um"])
        mls = np.arange(lo, hi + 1e-6, step)
        cands = [(PlaneParams(float(ml), y, r), R) for ml in mls for y in angles for r in angles]
        c_rows = _evaluate(runner, atlas, centroid, cands, res, scale_range, "coarse")
        rows += c_rows
        valid = sorted((r for r in c_rows if r["valid"]), key=lambda r: r["score"])
        if not valid:
            ratios = [r["size_ratio"] for r in c_rows]
            raise RuntimeError(
                f"no coarse candidate has size_ratio inside scale_range {scale_range} "
                f"(observed {min(ratios):.2f}-{max(ratios):.2f}). Check pixel_size_um, and that the "
                "atlas age matches the tissue -- or widen/disable search.scale_range.")
        seeds = []
        for r in valid:
            if all(abs(r["ml_um"] - s["ml_um"]) > step for s in seeds):
                seeds.append(r)
            if len(seeds) == cfg["fine_top_k"]:
                break
        log("  coarse: " + ", ".join(f"ml {s['ml_um']:.0f} yaw {s['yaw_deg']:+.0f} roll {s['roll_deg']:+.0f} "
                                     f"({s['score']:.4f})" for s in seeds))

        gap = max(np.diff(sorted(angles))) if len(angles) > 1 else 2 * cfg["fine_angle_step_deg"]
        a_step = float(cfg["fine_angle_step_deg"])
        a_offsets = np.arange(-gap, gap + 1e-6, a_step)
        m_offsets = np.arange(-step, step + 1e-6, float(cfg["fine_ml_step_um"]))
        seen, cands = set(), []
        for s in seeds:
            Rs = _orthogonal_part(_row_M(s))
            for dm in m_offsets:
                for dy in a_offsets:
                    for dr in a_offsets:
                        p = PlaneParams(round(max(0.0, s["ml_um"] + dm), 3), round(s["yaw_deg"] + dy, 3),
                                        round(s["roll_deg"] + dr, 3))
                        if p not in seen:
                            seen.add(p)
                            cands.append((p, Rs))
        rows += _evaluate(runner, atlas, centroid, cands, res, scale_range, "fine")

        if cfg["refine_starts"]:
            rows += _pattern_search(runner, atlas, centroid, rows, res, scale_range, cfg, seen, log)

    df = pd.DataFrame(rows)
    ranked = df[df["valid"] & (df["stage"] != "orientation")].sort_values("score")
    best = ranked.iloc[0].to_dict()
    far = ranked[(ranked["ml_um"] - best["ml_um"]).abs() >= cfg["far_um"]]
    best["far_gap"] = float(far["score"].iloc[0] - best["score"]) if len(far) else float("nan")
    best["n_candidates"] = int(len(df))
    return df, best


# ------------------------------------------------------ final registration

def register_to_plane(section, keep, atlas, best, cfg, prefix):
    """Affine (from the search's Similarity) then SyNOnly, fixed = the chosen
    atlas plane at register_res_um. Returns {'affine': reg, 'syn': reg or
    None}, each reg carrying atlas_template/atlas_annotation so the
    transforms.* helpers work on it unchanged."""
    res = float(cfg["register_res_um"])
    tmpl, ann, origin = atlas.sample_plane(_row_params(best), res)
    like = dict(spacing=(res, res), origin=tuple(float(o) for o in origin))
    plane = ants.from_numpy(tmpl, **like)
    plane_ann = ants.from_numpy(np.ascontiguousarray(ann), **like)

    init_path = f"{prefix}search_similarity.mat"
    _write_linear(_row_M(best), (best["tx"], best["ty"]), init_path)
    common = dict(fixed=plane, moving=section, moving_mask=keep, mask_all_stages=True, verbose=False)
    out = {}
    out["affine"] = ants.registration(type_of_transform="Affine", initial_transform=init_path,
                                      aff_metric="mattes", outprefix=f"{prefix}affine_", **common)
    out["syn"] = None
    if cfg["syn"]:
        out["syn"] = ants.registration(
            type_of_transform="SyNOnly", initial_transform=out["affine"]["fwdtransforms"][0],
            syn_metric=cfg["syn_metric"], syn_sampling=cfg["syn_sampling"],
            reg_iterations=tuple(cfg["reg_iterations"]),
            grad_step=cfg["grad_step"], flow_sigma=cfg["flow_sigma"], total_sigma=0,
            outprefix=f"{prefix}syn_", **common)
    for reg in out.values():
        if reg is not None:
            reg.update(atlas_template=plane, atlas_annotation=plane_ann)
    return out


def assign_cells(csv_path, x_col, y_col, pixel_size_um, reg, atlas, params, structures=None):
    """Cell centroids in ORIGINAL section pixels -> plane (u, v) -> 3D atlas
    microns -> annotation id. Coordinates are on the oriented, uncropped atlas
    grid; ml_from_midline_um is signed along axis0 but the side is arbitrary
    (see module docstring)."""
    df = pd.read_csv(csv_path)
    pts = pd.DataFrame({"x": df[x_col].to_numpy(float) * pixel_size_um,
                        "y": df[y_col].to_numpy(float) * pixel_size_um})
    uv = transforms.transform_cell_points(pts, reg, direction="sample_to_atlas", dim=2)
    xyz = atlas.plane_to_atlas_um(uv["x"].to_numpy(), uv["y"].to_numpy(), params)
    ids = atlas.lookup(xyz)
    df["plane_u_um"], df["plane_v_um"] = uv["x"].to_numpy(), uv["y"].to_numpy()
    df["atlas_x_um"], df["atlas_y_um"], df["atlas_z_um"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    df["ml_from_midline_um"] = xyz[:, 0] - atlas.midline_vox * atlas.res_um
    df["region_id"] = ids
    if structures:
        df["region_acronym"] = [structures[i]["acronym"] if i in structures else "" for i in ids.tolist()]
        df["region_name"] = [structures[i]["name"] if i in structures else "" for i in ids.tolist()]
    return df


def _coverage(labels, tissue):
    lab = labels.numpy() > 0
    return {"unlabelled_pct": float(100 * (tissue & ~lab).sum() / max(tissue.sum(), 1)),
            "outside_pct": float(100 * (lab & ~tissue).sum() / max(lab.sum(), 1))}


def _render_qc(path, name, section, tissue, df, best, regs, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, img, title):
        arr = img.numpy().T
        ax.imshow(arr, cmap="gray", vmin=0, vmax=np.percentile(arr[arr > 0], 99) if (arr > 0).any() else 1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def edges(ax, lab):
        a = lab.numpy().T
        e = ndimage.maximum_filter(a, 3) != ndimage.minimum_filter(a, 3)
        ax.imshow(np.ma.masked_where(~e, e), cmap="autumn", alpha=0.8, interpolation="nearest")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    show(axes[0, 0], section, f"{name}: section + tissue mask")
    axes[0, 0].contour(tissue.T.astype(float), levels=[0.5], colors="c", linewidths=1)
    show(axes[0, 1], regs["affine"]["atlas_template"],
         f"atlas plane  ml {best['ml_um']:.0f} um  yaw {best['yaw_deg']:+.1f}  roll {best['roll_deg']:+.1f}")

    ax = axes[0, 2]
    for stage, colour in (("coarse", "tab:blue"), ("fine", "tab:orange"), ("refine", "tab:green")):
        sub = df[(df["stage"] == stage) & df["valid"]]
        if len(sub):
            per_ml = sub.groupby("ml_um")["score"].min()
            ax.plot(per_ml.index, -per_ml.values, "-o", ms=3, color=colour, label=f"{stage} (best tilt)")
    bad = df[(df["stage"] != "orientation") & ~df["valid"] & np.isfinite(df["score"])]
    if len(bad):
        ax.plot(bad["ml_um"], -bad["score"], "x", color="0.7", ms=3, label="size_ratio out of range")
    ax.axvline(best["ml_um"], color="k", lw=0.8)
    ax.set_xlabel("plane distance from midline (um)")
    ax.set_ylabel("MI (higher = better)")
    ax.set_title(f"search: size_ratio {best['size_ratio']:.2f}, far_gap {best['far_gap']:.4f}", fontsize=9)
    ax.legend(fontsize=7)

    for col, key in ((0, "affine"), (1, "syn")):
        ax = axes[1, col]
        if key in labels:
            show(ax, section, f"{key}: atlas boundaries on section")
            edges(ax, labels[key])
        else:
            ax.axis("off")
    final = regs["syn"] or regs["affine"]
    show(axes[1, 2], final["atlas_template"], "section warped onto plane (red = section edges)")
    warped = final["warpedmovout"].numpy().T
    e = ndimage.sobel(warped, 0) ** 2 + ndimage.sobel(warped, 1) ** 2
    e = e > np.percentile(e[warped > 0], 90) if (warped > 0).any() else e > 0
    axes[1, 2].imshow(np.ma.masked_where(~e, e), cmap="autumn", alpha=0.5, interpolation="nearest")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process_section(sec, atlas, search_cfg, reg_cfg, out_dir, structures=None, overwrite=False, log=print):
    """One section end to end. Writes into <out_dir>/<name>/ and returns a
    summary dict (also written as plane.json)."""
    name = sec["name"]
    out = Path(out_dir) / name
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} exists and is not empty (pass overwrite to replace it)")
    (out / "transforms").mkdir(parents=True, exist_ok=True)
    seed = int(reg_cfg["random_seed"])
    ants.config.set_ants_deterministic(True, seed)

    px = float(sec["pixel_size_um"])
    raw = load_section_array(sec["image"], sec.get("channel"))
    img = downsample_section(raw, px, reg_cfg["register_res_um"])
    if sec.get("tissue_mask"):
        tissue = _load_mask_on(sec["tissue_mask"], raw.shape, px, img)
    else:
        tissue = auto_tissue_mask(img, sec.get("tissue_threshold"))
    keep = None
    if sec.get("damage_mask"):
        damaged = _load_mask_on(sec["damage_mask"], raw.shape, px, img)
        keep = ants.from_numpy((~damaged).astype(np.float32), spacing=img.spacing, origin=img.origin)
    del raw
    section = prepare_section_image(img, tissue, reg_cfg["n4_bias_correction"],
                                    reg_cfg["intensity_clip_percentiles"])
    tissue_img = ants.from_numpy(tissue.astype(np.float32), spacing=img.spacing, origin=img.origin)
    log(f"[{name}] section {tuple(img.shape)} px at {reg_cfg['register_res_um']} um, "
        f"tissue {tissue.mean() * 100:.0f}% of image")

    s_res = search_cfg["search_res_um"]
    df, best = search_plane(_resample_2d(section, s_res),
                            _resample_2d(tissue_img, s_res, nearest=True).numpy() > 0.5,
                            None if keep is None else _resample_2d(keep, s_res, nearest=True),
                            atlas, sec, search_cfg, seed, log)
    df.to_csv(out / "search_candidates.csv", index=False)
    log(f"  best plane: ml {best['ml_um']:.0f} um, yaw {best['yaw_deg']:+.1f}, roll {best['roll_deg']:+.1f}, "
        f"size_ratio {best['size_ratio']:.3f}, far_gap {best['far_gap']:.4f}")

    regs = register_to_plane(section, keep, atlas, best, reg_cfg, str(out / "transforms" / f"{name}_"))
    summary = {"name": name, **{k: best[k] for k in ("ml_um", "yaw_deg", "roll_deg", "score", "size_ratio",
                                                     "rotation_deg", "mirrored", "far_gap", "n_candidates")}}
    labels = {}
    for key, reg in regs.items():
        if reg is None:
            continue
        labels[key] = transforms.warp_labels_to_sample(section, reg)
        ants.image_write(labels[key], str(out / f"labels_in_section_{key}.nii.gz"))
        for metric, value in _coverage(labels[key], tissue).items():
            summary[f"{key}_{metric}"] = value
        summary[f"{key}_mi"] = float(ants.image_similarity(reg["atlas_template"], reg["warpedmovout"],
                                                           "MattesMutualInformation"))
    final_key = "syn" if regs["syn"] is not None else "affine"
    final = regs[final_key]
    ants.image_write(section, str(out / "section_prep.nii.gz"))
    ants.image_write(tissue_img, str(out / "section_tissue_mask.nii.gz"))
    ants.image_write(final["atlas_template"], str(out / "atlas_plane_template.nii.gz"))
    ants.image_write(final["atlas_annotation"], str(out / "atlas_plane_annotation.nii.gz"))
    ants.image_write(final["warpedmovout"], str(out / "section_in_plane.nii.gz"))
    summary["transforms_fwd"] = [str(t) for t in final["fwdtransforms"]]
    summary["transforms_inv"] = [str(t) for t in final["invtransforms"]]

    if sec.get("cells_csv"):
        cells = assign_cells(sec["cells_csv"], sec.get("cells_x_col", "x"), sec.get("cells_y_col", "y"), px,
                             final, atlas, _row_params(best), structures)
        cells.to_csv(out / "cells_registered.csv", index=False)
        summary["n_cells"] = int(len(cells))
        summary["cells_unassigned_pct"] = float(100 * (cells["region_id"] == 0).mean())

    _render_qc(out / "qc.png", name, section, tissue, df, best, regs, labels)
    with open(out / "plane.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"  {final_key}: unlabelled {summary[f'{final_key}_unlabelled_pct']:.1f}% of tissue, "
        f"outside {summary[f'{final_key}_outside_pct']:.1f}%  -> {out}")
    return summary
