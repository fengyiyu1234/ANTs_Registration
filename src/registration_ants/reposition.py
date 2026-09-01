"""Shared core: rigidly close the gaps left by hinged tissue that split open.

WHAT THIS IS FOR
----------------
A cleared brain that cracked during handling ends up with flaps of tissue --
often several across the cortical surface -- swung outward from where they
belong. They are still attached somewhere (the hinge), so they are NOT free
fragments: what happened is a rotation about the hinge, progressively larger
the further you get from it.

Left alone, each open crack is a dark gap where the atlas has tissue, and the
metric answers that by dragging neighbouring tissue in -- one spurious local
compression per crack. Repositioning closes the gaps in the raw data BEFORE
registration, so the pipeline downstream sees a geometrically ordinary brain
and needs no damage/guide special-casing for the cracks at all.

THE MODEL: A STACK OF IN-PLANE RIGID TRANSFORMS
-----------------------------------------------
Each fragment gets, per z plane, an in-plane (xy) rigid transform: rotate by
theta about a centre, then translate. Out-of-plane rotation is deliberately
NOT representable.

That restriction is not a simplification, it is a match to what the data can
support: these stacks are ~2.6 um in xy against 32 um in z (8x/64 um for the
coarser samples), so an out-of-plane tilt shows up only as sub-plane
differences between neighbouring sections -- fitting it produces a
confident-looking angle decided by noise. Three degrees of freedom that are
measurable beat six where three are not.

Working per plane is also what makes a HINGED flap safe to move. The hinge
runs along z: on the planes where the flap is open it is a free-standing
island within its own plane, with nothing in that plane to tear. A transform
defined per plane cannot reach the out-of-plane attachment, so it cannot
break it. Where a hinge does lie within a plane, put `center_um` ON the hinge
-- the tear a rigid move opens is exactly the displacement at the attachment
point, and rotating about that point makes it zero.

Because the transform varies with z, a stack of 2D rigids is not globally
rigid: it may shear the fragment along z. That is intended. The tissue opened
progressively -- wide far from the hinge, closed near it -- and a stack that
tapers the same way describes it better than one rigid body ever could.

KEYFRAMES, AND WHAT HAPPENS OUTSIDE THEM
----------------------------------------
You set a transform on a handful of z planes and the rest follow, the same
sparse-keyframe idiom paint_mask.py uses everywhere else. `interpolate`
picks between the two readings of "planes I did not draw":

  True   planes BETWEEN two keyframes interpolate (tx, ty, theta, centre
         linearly; dz rounded to whole planes). Outside the keyframe span,
         identity. Irregular cracks are answered by adding keyframes, which
         is strictly less work than drawing every plane for the same result.
  False  ONLY keyframe planes move; every other plane is identity.

Nothing tapers to identity on its own. A fragment whose z extent ends because
the fragment itself thins out to nothing leaves no step behind -- there is no
tissue at the boundary plane to be stepped. A fragment still carrying real
tissue at the edge of its keyframe span does leave one, so `boundary_report`
measures exactly that and says so, rather than a blanket rule forbidding it.
A step costs registration quality only: cells are looked up wherever they
land, so no count or region assignment is corrupted by one, and the plan is
recorded and invertible, so a bad one is re-run rather than repaired.

PHYSICAL MICRONS, NOT VOXELS
----------------------------
Every offset in a plan is in microns, and rotation centres too. The painted
grid (registration.tif, e.g. 5.2/5.2/64 um) and the grid cells were detected
on (0.65/0.65/8 um) are different sampling of ONE physical space, so a plan
expressed in microns applies to both without a per-grid conversion factor to
get wrong -- the same reason ../Registration_ants/src/registration_ants/cell_points.py
converts centroids through physical space rather than by a pixel ratio.
`dz_planes` is the sole exception and is deliberately in PAINTED planes: z
displacement is judged by dragging content from one section onto another, and
those sections are what the eye actually has to work with.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

FORMAT = "reposition/1"
KEYFRAME_FIELDS = ("z", "tx_um", "ty_um", "theta_deg", "dz_planes", "center_um")


# =====================================================================================
# The plan: what a fragment's transform is, and reading it off at any plane
# =====================================================================================

def make_keyframe(z, tx_um=0.0, ty_um=0.0, theta_deg=0.0, dz_planes=0, center_um=(0.0, 0.0)):
    """One keyframe: the in-plane rigid transform for source plane `z`.

    center_um is (x, y) in microns -- the point the rotation pivots about, and
    the one place a hinge inside the plane has to be put (see the module
    docstring). dz_planes is a whole number of PAINTED planes, everything else
    is microns.
    """
    return {"z": int(z), "tx_um": float(tx_um), "ty_um": float(ty_um),
            "theta_deg": float(theta_deg), "dz_planes": int(dz_planes),
            "center_um": [float(center_um[0]), float(center_um[1])]}


def make_fragment(label, keyframes, name=""):
    """One fragment: the paint label that marks it, plus its keyframes sorted
    by z. A label with no keyframes is legal and means "does not move" -- it
    still gets carried so the plan records that it was considered."""
    return {"label": int(label), "name": str(name),
            "keyframes": sorted((dict(k) for k in keyframes), key=lambda k: k["z"])}


def make_plan(image_shape_zyx, voxel_size_um, fragments, image_path="", labels_path="",
              interpolate=True, feather_um=0.0):
    """Assemble a plan. voxel_size_um is (x, y, z) of the PAINTED grid, the
    same order and grid as the pipeline's sample.voxel_size_um."""
    return {
        "format": FORMAT,
        "image_path": str(image_path),
        "labels_path": str(labels_path),
        "image_shape_zyx": [int(v) for v in image_shape_zyx],
        "voxel_size_um": [float(v) for v in voxel_size_um],
        "interpolate": bool(interpolate),
        "feather_um": float(feather_um),
        "fragments": [dict(f) for f in fragments],
    }


def write_plan(path, plan):
    path = Path(path)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def read_plan(path):
    """Parse a plan, checking the things a hand-edited file gets wrong: the
    format tag, the keyframe fields, and duplicate z within one fragment
    (two transforms for one plane has no defined answer)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reposition plan not found: {path}")
    plan = json.loads(path.read_text())
    if plan.get("format") != FORMAT:
        raise ValueError(f"{path}: expected format {FORMAT!r}, got {plan.get('format')!r}")
    for key in ("image_shape_zyx", "voxel_size_um", "fragments"):
        if key not in plan:
            raise ValueError(f"{path}: missing required key {key!r}")
    if len(plan["voxel_size_um"]) != 3:
        raise ValueError(f"{path}: voxel_size_um must have 3 entries (x, y, z), "
                         f"got {plan['voxel_size_um']}")
    seen = set()
    for frag in plan["fragments"]:
        label = int(frag["label"])
        if label in seen:
            raise ValueError(f"{path}: label {label} appears in more than one fragment")
        if label == 0:
            raise ValueError(f"{path}: label 0 is background and cannot be a fragment")
        seen.add(label)
        zs = [int(k["z"]) for k in frag["keyframes"]]
        dupes = sorted({z for z in zs if zs.count(z) > 1})
        if dupes:
            raise ValueError(f"{path}: fragment label {label} has two keyframes on "
                             f"plane(s) {dupes}; one plane can only have one transform")
        for kf in frag["keyframes"]:
            missing = [f for f in KEYFRAME_FIELDS if f not in kf]
            if missing:
                raise ValueError(f"{path}: fragment label {label}, keyframe z={kf.get('z')} "
                                 f"is missing {missing}")
    return plan


def plane_transform(fragment, z, interpolate=True):
    """The transform for source plane `z`, or None where the fragment does not
    move there.

    None and a zero transform are NOT the same thing downstream: None skips the
    plane entirely (no resample, no erase, no paste), which is what keeps an
    untouched plane bit-identical instead of blurred by an identity warp.
    """
    keyframes = fragment["keyframes"]
    if not keyframes:
        return None
    zs = [int(k["z"]) for k in keyframes]

    if z in zs:
        return dict(keyframes[zs.index(z)])
    if not interpolate or z < zs[0] or z > zs[-1]:
        return None

    hi = next(i for i, kz in enumerate(zs) if kz > z)
    lo = hi - 1
    a, b = keyframes[lo], keyframes[hi]
    span = zs[hi] - zs[lo]
    w = (z - zs[lo]) / span

    def lerp(key):
        return (1 - w) * float(a[key]) + w * float(b[key])

    # theta is interpolated as a plain angle rather than through a rotation
    # matrix: these are in-plane angles a hand set on adjacent sections, always
    # well under a quarter turn apart, so there is no wrap to be careful about
    # and the shortest-arc machinery slerp exists for would change nothing.
    return {"z": int(z),
            "tx_um": lerp("tx_um"), "ty_um": lerp("ty_um"),
            "theta_deg": lerp("theta_deg"),
            # dz is whole planes on both sides, so it is rounded rather than
            # blended: half a section is not a place content can be put.
            "dz_planes": int(round((1 - w) * a["dz_planes"] + w * b["dz_planes"])),
            "center_um": [(1 - w) * a["center_um"][0] + w * b["center_um"][0],
                          (1 - w) * a["center_um"][1] + w * b["center_um"][1]]}


def fragment_source_planes(fragment, n_planes, interpolate=True):
    """Every source plane this fragment actually moves on, in order."""
    return [z for z in range(int(n_planes))
            if plane_transform(fragment, z, interpolate) is not None]


# =====================================================================================
# The transform itself, in physical microns
# =====================================================================================

def _rotation_xy(theta_deg):
    """R acting on (x, y) column vectors, counter-clockwise in a right-handed
    xy frame. Screen sense depends on which way y points and is settled where
    the angle is entered, not here."""
    t = np.deg2rad(float(theta_deg))
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points_um(points_xy_um, tf):
    """Forward-map (N,2) xy microns: rotate about centre, then translate.

        p_out = R(theta) @ (p - c) + c + t
    """
    points_xy_um = np.asarray(points_xy_um, dtype=float).reshape(-1, 2)
    c = np.asarray(tf["center_um"], dtype=float)
    t = np.array([tf["tx_um"], tf["ty_um"]], dtype=float)
    return (points_xy_um - c) @ _rotation_xy(tf["theta_deg"]).T + c + t


def _plane_affine_inverse(tf, voxel_size_um):
    """(matrix, offset) for ndimage.affine_transform on one (y, x) plane.

    affine_transform PULLS -- it asks, for each OUTPUT index o, which input
    index to sample -- so what it needs is the inverse of the forward map. In
    voxel indices the forward map is

        i_out = A i_in + b,   A = S^-1 R S,   b = S^-1 (c - R c + t)

    with S the (y, x) voxel spacing. The inverse handed over is therefore
    matrix = A^-1 and offset = -A^-1 b.
    """
    sx, sy = float(voxel_size_um[0]), float(voxel_size_um[1])
    # Everything here is in ARRAY order (y, x), while the plan speaks (x, y):
    # the permutation is applied once, to R and to the vectors, rather than
    # being carried as a sign convention through the algebra.
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    r_yx = swap @ _rotation_xy(tf["theta_deg"]) @ swap
    s = np.diag([sy, sx])
    s_inv = np.diag([1.0 / sy, 1.0 / sx])

    c_yx = np.array([tf["center_um"][1], tf["center_um"][0]], dtype=float)
    t_yx = np.array([tf["ty_um"], tf["tx_um"]], dtype=float)

    a = s_inv @ r_yx @ s
    b = s_inv @ (c_yx - r_yx @ c_yx + t_yx)
    a_inv = np.linalg.inv(a)
    return a_inv, -a_inv @ b


# =====================================================================================
# Finding a fragment without painting one
# =====================================================================================

# =====================================================================================
# Sparse fragment outlines -> dense, at the moment they are used
# =====================================================================================
# A fragment is grabbed on a handful of planes, not traced on all of them, and
# the file that comes out of the GUI keeps it that way: the planes carrying
# voxels ARE the keyframes, and densifying before saving would erase the one
# record of which planes were actually decided. Reopening a dense file gives
# back a solid block with no way to tell a grabbed plane from a guessed one --
# the same reason mode: labels keeps a .keyframes.json beside its dense volume.
#
# So the filling happens here, against the volume in hand, every time a plan is
# applied. On an already-dense volume it is a no-op (every plane is a keyframe,
# so there is nothing between any two), which is what lets one code path serve
# both a sparse export and a hand-made outline traced the long way.
#
# What does NOT interpolate is the line segments. They are how a keyframe's
# transform was arrived at, and the transform is already what gets interpolated
# (plane_transform, above); blending the endpoints as well would be a second,
# disagreeing route to the same number -- linear interpolation of two endpoints
# does not even preserve the segment's length once there is rotation, which is
# the one quantity the whole copy-don't-redraw rule exists to keep fixed.

def _keyframe_planes(fragments_zyx, label):
    """{plane index: 2D bool} for the planes where `label` was actually put."""
    planes = {}
    for z in range(fragments_zyx.shape[0]):
        plane = fragments_zyx[z] == label
        if plane.any():
            planes[int(z)] = plane
    return planes


def densify_fragments(fragments_zyx):
    """Fill a sparse fragment volume between the planes each label appears on.

    Per label and independently, so two fragments whose grabbed planes
    interleave -- or which share an xy footprint at different z -- cannot bleed
    into one another the way a single merged interpolation would let them.
    """
    from . import mask_utils

    fragments_zyx = np.asarray(fragments_zyx)
    out = np.zeros_like(fragments_zyx)
    for label in sorted({int(v) for v in np.unique(fragments_zyx) if v != 0}):
        keyframes = _keyframe_planes(fragments_zyx, label)
        if not keyframes:
            continue
        dense = mask_utils.interpolate_sparse_mask(keyframes, fragments_zyx.shape)
        out[dense] = label
    return out


def densify_plane(fragments_zyx, z):
    """One filled plane of `densify_fragments`, without building the volume.

    Only the two keyframes bracketing z matter, so only those are interpolated
    -- through the same function the whole-volume path uses, on a stand-in
    stack, rather than a second copy of the arithmetic.
    """
    from . import mask_utils

    fragments_zyx = np.asarray(fragments_zyx)
    out = np.zeros(fragments_zyx.shape[1:], dtype=fragments_zyx.dtype)
    z = int(z)
    for label in sorted({int(v) for v in np.unique(fragments_zyx) if v != 0}):
        keyframes = _keyframe_planes(fragments_zyx, label)
        if z in keyframes:
            out[keyframes[z]] = label
            continue
        below = [k for k in keyframes if k < z]
        above = [k for k in keyframes if k > z]
        if not below or not above:
            continue                      # outside this label's span: stays empty
        lo, hi = max(below), min(above)
        span = mask_utils.interpolate_sparse_mask(
            {0: keyframes[lo], hi - lo: keyframes[hi]},
            (hi - lo + 1,) + fragments_zyx.shape[1:])
        out[span[z - lo]] = label
    return out


class threshold_planes:
    """`stack > value`, one plane at a time, for grab_fragment.

    A whole-volume comparison would be a boolean copy of the stack -- 1.7 GB
    for a 190 x 3967 x 2249 raw one -- to answer a question that only ever
    looks at a handful of planes. This defers it per plane and keeps the same
    `[z]` / `.shape` surface a bool array has, so callers with a small volume
    can still just pass the array.

    `exclude`, when given, is a same-shaped volume whose nonzero voxels are
    taken OUT of the tissue -- a few hand-drawn strokes across a crack too
    tight to threshold apart. That is far less work than tracing the piece
    they separate, and it is all the walk needs to get past a join: cutting
    the two apart in the mask makes them two components again.
    """

    def __init__(self, stack, value, exclude=None):
        self.stack, self.value, self.shape = stack, value, stack.shape
        self.exclude = exclude

    def __getitem__(self, z):
        plane = self.stack[z] > self.value
        if self.exclude is not None:
            plane = plane & (self.exclude[z] == 0)
        return plane


def otsu_threshold(stack, max_planes=24):
    """An Otsu threshold for `stack`, read off an evenly spaced subset of its
    planes -- a starting value for the control that tunes it, not a decision.
    Subsampled because this runs to fill in a slider's default, and a full
    histogram of a multi-gigabyte stack is not worth the wait for that."""
    from skimage.filters import threshold_otsu
    step = max(1, stack.shape[0] // max_planes)
    return float(threshold_otsu(np.asarray(stack[::step], dtype=np.float32)))


def grab_plane(tissue_zyx, seed_zyx):
    """The piece of tissue under `seed_zyx`, on that plane, found rather than traced.

    A split-open piece does not need outlining by hand: the crack is a gap, so
    on any plane that shows the piece open it is already its own 2D connected
    component. One click takes it.

    ONE PLANE, deliberately. An earlier version walked the component along z
    and stopped where it stopped being separate, which also located the hinge
    -- but that decides the extent, and on a sample carrying several pieces the
    extent is already known and better stated than inferred. Grabbing each
    piece's planes and letting densify_fragments fill between them is the same
    sparse-keyframe idiom the guide outlines use, and it keeps every piece's
    span exactly where it was put.

    tissue_zyx: anything where `tissue_zyx[z]` is a 2D boolean plane and
    `.shape` is (z, y, x) -- a bool array, or `threshold_planes(stack, value)`
    for a raw stack that would cost gigabytes to binarize whole. Threshold the
    raw stack for this rather than reusing a painted guide outline: a guide is
    sparse keyframes interpolated into a smooth blob, and interpolation closes
    exactly the thin crack this depends on.

    Returns (mask_zyx, note): a volume empty except on the seed plane, and a
    line saying what was taken.
    """
    z0, y0, x0 = (int(v) for v in seed_zyx)
    n_planes = tissue_zyx.shape[0]
    if not (0 <= z0 < n_planes):
        raise ValueError(f"seed plane {z0} is outside the volume's {n_planes} planes")
    plane = np.asarray(tissue_zyx[z0], dtype=bool)
    if not plane[y0, x0]:
        raise ValueError(f"the seed (z={z0}, y={y0}, x={x0}) is not on tissue -- click on the "
                         f"fragment itself, or lower the threshold until it shows up")

    labelled, n = ndimage.label(plane)
    out = np.zeros(tuple(tissue_zyx.shape), dtype=bool)
    if n:
        out[z0] = labelled == labelled[y0, x0]
    return out, f"plane {z0}, {int(out.sum())} voxels"


# =====================================================================================
# Applying a plan to the image
# =====================================================================================

def apply_to_image(image_zyx, labels_zyx, plan, fill_value=None):
    """Return a copy of `image_zyx` with every fragment moved per `plan`.

    Two passes, and the order is load-bearing: EVERY fragment's source voxels
    are erased before ANY fragment is pasted. Erasing as you go would let one
    fragment's source region wipe out another's already-pasted content wherever
    the two overlap -- and a flap swung back over its own former position
    overlaps itself, so this is the normal case, not a corner one.

    fill_value: what an emptied source region becomes. Defaults to the image's
    1st percentile, i.e. its own background level rather than a hard zero,
    which would read as a structure of its own to an intensity metric.
    """
    image_zyx = np.asarray(image_zyx)
    labels_zyx = np.asarray(labels_zyx)
    if image_zyx.shape != labels_zyx.shape:
        raise ValueError(f"image {image_zyx.shape} and labels {labels_zyx.shape} "
                         f"must be the same (z, y, x) volume")

    out = image_zyx.astype(np.float32, copy=True)
    if fill_value is None:
        fill_value = float(np.percentile(image_zyx, 1))
    interpolate = bool(plan.get("interpolate", True))
    voxel_um = plan["voxel_size_um"]
    feather_vox = float(plan.get("feather_um", 0.0)) / max(float(voxel_um[0]), 1e-9)
    n_planes = image_zyx.shape[0]

    pastes = []
    for frag in plan["fragments"]:
        label = int(frag["label"])
        for z in fragment_source_planes(frag, n_planes, interpolate):
            mask = labels_zyx[z] == label
            if not mask.any():
                continue
            tf = plane_transform(frag, z, interpolate)
            z_out = z + int(tf["dz_planes"])
            if not 0 <= z_out < n_planes:
                raise ValueError(
                    f"fragment label {label}: source plane {z} with dz_planes="
                    f"{tf['dz_planes']} lands on plane {z_out}, outside the "
                    f"volume's {n_planes} planes")
            matrix, offset = _plane_affine_inverse(tf, voxel_um)
            content = np.where(mask, image_zyx[z], fill_value).astype(np.float32)
            # order=1 on the content, order=0 on the mask: the mask decides
            # WHERE tissue is and must stay binary (a smoothed one would paste
            # a halo of background over the socket), the content is intensity
            # and is interpolated like any resample.
            warped = ndimage.affine_transform(content, matrix, offset=offset,
                                              order=1, mode="constant", cval=fill_value)
            warped_mask = ndimage.affine_transform(mask.astype(np.uint8), matrix, offset=offset,
                                                   order=0, mode="constant", cval=0) > 0
            out[z][mask] = fill_value          # erase pass
            pastes.append((z_out, warped, warped_mask))

    for z_out, warped, warped_mask in pastes:
        alpha = warped_mask.astype(np.float32)
        if feather_vox > 0:
            # Feathering the paste edge only: cosmetic, and deliberately weak.
            # A seam is a couple of voxels wide against a metric that smooths
            # far more than that before it ever looks, so this exists to keep
            # QC renders readable, not to change what registration sees.
            alpha = ndimage.gaussian_filter(alpha, sigma=feather_vox)
            alpha = np.clip(alpha, 0.0, 1.0)
        out[z_out] = alpha * warped + (1.0 - alpha) * out[z_out]

    if np.issubdtype(image_zyx.dtype, np.integer):
        info = np.iinfo(image_zyx.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(image_zyx.dtype)


def preview_plane(image_zyx, labels_zyx, plan, z_out, fill_value=None):
    """One OUTPUT plane exactly as apply_to_image would produce it.

    A GUI cannot resample the whole stack on every slider drag -- 190 x 3967 x
    2249 is a second and a half of work for a control that has to answer while
    a hand is still moving -- but it also must not show an approximation of
    what the export will do, or the thing being judged by eye is not the thing
    that gets written. So this reproduces apply_to_image's arithmetic for a
    single plane: erase whatever fragment sources sit on z_out, then paste
    every source plane whose dz lands it here.
    """
    image_zyx = np.asarray(image_zyx)
    labels_zyx = np.asarray(labels_zyx)
    if fill_value is None:
        fill_value = float(np.percentile(image_zyx[z_out], 1))
    interpolate = bool(plan.get("interpolate", True))
    voxel_um = plan["voxel_size_um"]
    feather_vox = float(plan.get("feather_um", 0.0)) / max(float(voxel_um[0]), 1e-9)
    n_planes = image_zyx.shape[0]

    out = image_zyx[z_out].astype(np.float32, copy=True)
    for frag in plan["fragments"]:
        label = int(frag["label"])
        if plane_transform(frag, z_out, interpolate) is not None:
            out[labels_zyx[z_out] == label] = fill_value

    for frag in plan["fragments"]:
        label = int(frag["label"])
        for z in fragment_source_planes(frag, n_planes, interpolate):
            tf = plane_transform(frag, z, interpolate)
            if z + int(tf["dz_planes"]) != z_out:
                continue
            mask = labels_zyx[z] == label
            if not mask.any():
                continue
            matrix, offset = _plane_affine_inverse(tf, voxel_um)
            content = np.where(mask, image_zyx[z], fill_value).astype(np.float32)
            warped = ndimage.affine_transform(content, matrix, offset=offset,
                                              order=1, mode="constant", cval=fill_value)
            alpha = (ndimage.affine_transform(mask.astype(np.uint8), matrix, offset=offset,
                                              order=0, mode="constant", cval=0) > 0
                     ).astype(np.float32)
            if feather_vox > 0:
                alpha = np.clip(ndimage.gaussian_filter(alpha, sigma=feather_vox), 0.0, 1.0)
            out = alpha * warped + (1.0 - alpha) * out
    return out


def orient_segments(line_a, line_b, fragments_zyx, label, samples=25, margin=0.25):
    """Of two drawn lines, which one is ON the fragment (the source) and which
    marks where it belongs (the target).

    Drawing order would answer this too -- the target is the copy, so it comes
    second -- but that is an invisible convention, and the failure when it is
    broken is the worst kind: reversing source and target produces a transform
    that is entirely well-formed and moves the flap further from home instead
    of onto it, at a magnitude that looks exactly right. Redrawing the source
    after the target, or drawing the target first, is enough to trip it.

    The fragment outline already answers it and can be checked: the source line
    lies on the fragment, the target lies off it, in the socket. So this
    samples both lines against the outline and decides by which one sits on it.

    Lines are (2, 3) arrays in (z, y, x) VOXEL indices -- what a napari Shapes
    layer holds. Returns (source, target, note); `note` says what it found, for
    showing to whoever drew them. Raises when the two cannot be told apart,
    rather than picking one: an ambiguous pair is a drawing to fix, not a coin
    to flip.
    """
    fragments_zyx = np.asarray(fragments_zyx)
    shape = np.array(fragments_zyx.shape) - 1

    def on_fragment(line):
        line = np.asarray(line, dtype=float).reshape(2, 3)
        t = np.linspace(0.0, 1.0, samples)[:, None]
        pts = np.rint(line[0] * (1 - t) + line[1] * t).astype(int)
        pts = np.clip(pts, 0, shape)
        return float((fragments_zyx[pts[:, 0], pts[:, 1], pts[:, 2]] == int(label)).mean())

    frac_a, frac_b = on_fragment(line_a), on_fragment(line_b)
    if max(frac_a, frac_b) < margin:
        raise ValueError(
            f"neither line lies on fragment {label} ({100 * frac_a:.0f}% and "
            f"{100 * frac_b:.0f}% of their length). Grab or paint the fragment first, "
            f"then draw one line across a feature ON it and move a copy to where that "
            f"feature belongs.")
    if abs(frac_a - frac_b) < margin:
        raise ValueError(
            f"both lines lie on fragment {label} ({100 * frac_a:.0f}% and "
            f"{100 * frac_b:.0f}%), so which one is the target is not decidable. Move the "
            f"copy off the fragment, onto the place it should end up.")
    if frac_a >= frac_b:
        source, target, sf, tf = line_a, line_b, frac_a, frac_b
    else:
        source, target, sf, tf = line_b, line_a, frac_b, frac_a
    note = (f"source line is {100 * sf:.0f}% on fragment {label}, target {100 * tf:.0f}% "
            f"-- read off the outline, not the drawing order")
    return np.asarray(source, dtype=float), np.asarray(target, dtype=float), note


def fit_from_segments(source_xy_um, target_xy_um, center_um=None):
    """The in-plane rigid transform carrying one drawn segment onto another.

    Two endpoints are exactly enough here and no more: an in-plane rigid
    transform has three degrees of freedom (tx, ty, theta) and a segment
    supplies four numbers, of which the length is already spent -- the target
    is a COPY of the source and cannot be stretched. So the fit is exact by
    construction rather than least-squares, and there is no residual to read
    (which is why boundary_report, not a residual, is what checks this work).

    center_um: the pivot to express the result about -- the hinge, when there
    is one in this plane, so the transform's fixed point is the place the
    tissue is still attached. It changes only how the same motion is written
    down, never the motion: the segment lands on its target either way.
    """
    a0, a1 = np.asarray(source_xy_um, dtype=float).reshape(2, 2)
    b0, b1 = np.asarray(target_xy_um, dtype=float).reshape(2, 2)
    da, db = a1 - a0, b1 - b0
    if np.hypot(*da) < 1e-9 or np.hypot(*db) < 1e-9:
        raise ValueError("a segment with zero length gives no direction to rotate to")
    theta = np.degrees(np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0]))
    theta = (theta + 180.0) % 360.0 - 180.0
    c = np.asarray(a0 if center_um is None else center_um, dtype=float)
    # Solve p_out = R(p - c) + c + t for t at the segment's start point.
    t = b0 - (_rotation_xy(theta) @ (a0 - c) + c)
    return float(t[0]), float(t[1]), float(theta), (float(c[0]), float(c[1]))


def apply_to_labels(labels_zyx, plan):
    """The same moves applied to the label volume itself -- so a repositioned
    stack can be reopened with its fragments still outlined where they now sit,
    and so `boundary_report` and the GUI preview have something to draw."""
    labels_zyx = np.asarray(labels_zyx)
    out = labels_zyx.copy()
    interpolate = bool(plan.get("interpolate", True))
    voxel_um = plan["voxel_size_um"]
    n_planes = labels_zyx.shape[0]

    pastes = []
    for frag in plan["fragments"]:
        label = int(frag["label"])
        for z in fragment_source_planes(frag, n_planes, interpolate):
            mask = labels_zyx[z] == label
            if not mask.any():
                continue
            tf = plane_transform(frag, z, interpolate)
            matrix, offset = _plane_affine_inverse(tf, voxel_um)
            warped = ndimage.affine_transform(mask.astype(np.uint8), matrix, offset=offset,
                                              order=0, mode="constant", cval=0) > 0
            out[z][mask] = 0
            pastes.append((z + int(tf["dz_planes"]), label, warped))

    for z_out, label, warped in pastes:
        out[z_out][warped] = label
    return out


def apply_to_volume_xyz(arr_xyz, labels_zyx, plan, kind="image", fill_value=None):
    """apply_to_image / apply_to_labels for a volume in ANTs' (x, y, z) order.

    Two array orders meet here and neither is wrong. Plans, painted volumes and
    everything in this module are (z, y, x) -- numpy's own order for a stack of
    sections, and what tifffile and SimpleITK hand back. ANTs images are
    (x, y, z), because that is the order their (sx, sy, sz) spacing is in (see
    io_utils.load_tiff_stack_as_ants, which transposes on the way in). Mixing
    them up does not raise: a plan applied to a transposed volume moves
    tissue along the wrong axes by plausible-looking amounts. So the transpose
    lives here, once, instead of at each of the four call sites in pipeline.py.
    """
    arr_zyx = np.transpose(np.asarray(arr_xyz), (2, 1, 0))
    if kind == "labels":
        out_zyx = apply_to_labels(arr_zyx, plan)
    elif kind == "image":
        out_zyx = apply_to_image(arr_zyx, labels_zyx, plan, fill_value=fill_value)
    else:
        raise ValueError(f"kind must be 'image' or 'labels', got {kind!r}")
    return np.ascontiguousarray(np.transpose(out_zyx, (2, 1, 0)))


# =====================================================================================
# Applying a plan to detected cells
# =====================================================================================

def apply_to_cells(cx, cy, cz, labels_zyx, plan, cells_voxel_um):
    """Move the cells that sit on a fragment; leave every other cell alone.

    cx/cy/cz are centroid coordinates in the ORIGINAL full-resolution pixel
    grid cells were detected on -- exactly the cx,cy,z columns of a
    cell_centroids CSV -- and cells_voxel_um is (x, y, z) microns of THAT grid.
    Returns (new_cx, new_cy, new_cz, moved_label), the last being the fragment
    label that moved each cell and 0 for cells left where they were.

    Which fragment a cell belongs to is read from the painted label volume at
    the cell's own position, so the flap outline drawn once decides both what
    the image move applies to and what the cell move applies to. There is no
    second boundary to keep in sync.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    cz = np.asarray(cz, dtype=float)
    labels_zyx = np.asarray(labels_zyx)
    interpolate = bool(plan.get("interpolate", True))
    paint_um = np.asarray(plan["voxel_size_um"], dtype=float)
    cells_um = np.asarray(cells_voxel_um, dtype=float)

    # Physical microns is the only frame the two grids share; index into the
    # painted volume through it rather than by a pixel ratio.
    x_um, y_um, z_um = cx * cells_um[0], cy * cells_um[1], cz * cells_um[2]
    ix = np.rint(x_um / paint_um[0]).astype(int)
    iy = np.rint(y_um / paint_um[1]).astype(int)
    iz = np.rint(z_um / paint_um[2]).astype(int)
    nz, ny, nx = labels_zyx.shape
    inside = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz))

    at_cell = np.zeros(len(cx), dtype=labels_zyx.dtype)
    at_cell[inside] = labels_zyx[iz[inside], iy[inside], ix[inside]]

    # Seeded from the INPUT coordinates, not from x_um/cells_um: a cell that
    # does not move must come back bit-identical, and a multiply-then-divide
    # round trip through microns perturbs the last bits of every row. That
    # difference is far below a voxel and harmless numerically, but it would
    # make every output CSV differ from its input everywhere, which is what
    # tells you at a glance that a rerun changed only what it should have.
    new_cx, new_cy, new_cz = cx.copy(), cy.copy(), cz.copy()
    moved = np.zeros(len(cx), dtype=int)

    for frag in plan["fragments"]:
        label = int(frag["label"])
        on_frag = inside & (at_cell == label)
        if not on_frag.any():
            continue
        # Grouped by painted plane, because the transform is per plane: every
        # cell whose painted plane is the same gets the same rigid move,
        # whatever its sub-plane z within it.
        for z in np.unique(iz[on_frag]):
            tf = plane_transform(frag, int(z), interpolate)
            if tf is None:
                continue
            sel = on_frag & (iz == z)
            pts = transform_points_um(np.column_stack([x_um[sel], y_um[sel]]), tf)
            new_cx[sel] = pts[:, 0] / cells_um[0]
            new_cy[sel] = pts[:, 1] / cells_um[1]
            new_cz[sel] = cz[sel] + tf["dz_planes"] * paint_um[2] / cells_um[2]
            moved[sel] = label

    return new_cx, new_cy, new_cz, moved


# =====================================================================================
# Diagnostics
# =====================================================================================

def boundary_report(plan, labels_zyx):
    """How much tissue is left standing at each z edge of a fragment's moved
    span, and how far that edge plane moves.

    This is the check that replaces a residual. With one segment pair driving
    a 3-DOF in-plane transform the fit is exact by construction, so its
    residual is always zero and says nothing; what can still go wrong is a
    step in z, and a step only matters where the boundary plane still carries
    real tissue whose neighbour stays put. Both numbers are reported per
    fragment per edge and the caller decides -- see the module docstring for
    why this is a measurement rather than a rule.
    """
    labels_zyx = np.asarray(labels_zyx)
    interpolate = bool(plan.get("interpolate", True))
    voxel_um = plan["voxel_size_um"]
    n_planes = labels_zyx.shape[0]
    rows = []

    for frag in plan["fragments"]:
        label = int(frag["label"])
        planes = fragment_source_planes(frag, n_planes, interpolate)
        if not planes:
            continue
        for edge, z in (("low", planes[0]), ("high", planes[-1])):
            neighbour = z - 1 if edge == "low" else z + 1
            mask = labels_zyx[z] == label
            voxels = int(mask.sum())
            # The step is what the tissue AT that plane actually does, so it is
            # measured on the fragment's own voxels rather than at its centroid
            # -- a rotation about a far hinge moves the far edge much further
            # than the middle, and the far edge is where a tear would show.
            if voxels:
                tf = plane_transform(frag, z, interpolate)
                iy, ix = np.nonzero(mask)
                pts = np.column_stack([ix * voxel_um[0], iy * voxel_um[1]])
                step_um = float(np.abs(transform_points_um(pts, tf) - pts).max())
            else:
                step_um = 0.0
            neighbour_moves = (0 <= neighbour < n_planes
                               and plane_transform(frag, neighbour, interpolate) is not None)
            rows.append({
                "label": label, "name": frag.get("name", ""), "edge": edge,
                "z": int(z), "voxels": voxels, "step_um": step_um,
                # A neighbour that also moves means this is not really an edge
                # (interpolate=False makes every keyframe plane look like one),
                # so it is recorded and left out of the warning below.
                "neighbour_moves": bool(neighbour_moves),
            })
    return rows


def boundary_warnings(rows, voxel_threshold=500, step_um_threshold=20.0):
    """The subset of boundary_report worth printing, as ready-made lines."""
    out = []
    for r in rows:
        if r["neighbour_moves"]:
            continue
        if r["voxels"] >= voxel_threshold and r["step_um"] >= step_um_threshold:
            out.append(
                f"label {r['label']}{(' (' + r['name'] + ')') if r['name'] else ''}: "
                f"{r['edge']} edge of the moved span is z={r['z']}, which still carries "
                f"{r['voxels']} fragment voxels moving up to {r['step_um']:.0f} um while "
                f"z={r['z'] - 1 if r['edge'] == 'low' else r['z'] + 1} stays put -- that is "
                f"a step in z. Fine if the fragment simply ends here; add a smaller "
                f"keyframe past this plane if it does not.")
    return out


def invert_plan(plan):
    """The plan that undoes this one -- for putting a repositioned result back
    onto the original data (e.g. overlaying atlas labels on the untouched
    stack for QC).

    Inverting p_out = R(p - c) + c + t gives p = R^-1(p_out - c - t) + c. Put
    in the same rotate-about-centre-then-translate form, that is a rotation by
    -theta about the MOVED centre c + t, translated by -t: the centre absorbs
    the forward translation and the inverse translation then has to take it
    back off again. (Dropping that -t leaves a result displaced by exactly t --
    correct for a pure rotation and silently wrong for everything else, which
    is what test_invert_plan_round_trips pins down.) Keyframes move to the
    planes their content landed on, since that is where the inverse has to be
    applied from.
    """
    inverted = []
    for frag in plan["fragments"]:
        keyframes = []
        for kf in frag["keyframes"]:
            cx, cy = kf["center_um"]
            keyframes.append(make_keyframe(
                z=kf["z"] + kf["dz_planes"],
                tx_um=-kf["tx_um"], ty_um=-kf["ty_um"],
                theta_deg=-kf["theta_deg"],
                dz_planes=-kf["dz_planes"],
                center_um=(cx + kf["tx_um"], cy + kf["ty_um"])))
        inverted.append(make_fragment(frag["label"], keyframes, frag.get("name", "")))
    out = dict(plan)
    out["fragments"] = inverted
    return out
