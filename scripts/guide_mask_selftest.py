"""Leave-one-out self-check of a hand-painted guide mask's slice interpolation:
which painted labels/planes would the pipeline's own SDF interpolation
reproduce well, and which ones is it guessing at?

`mask_utils.interpolate_sparse_mask` fills the planes between hand-drawn
keyframes by blending their signed-distance fields. This tool asks, for every
INTERIOR keyframe of every label: if that plane had NOT been drawn, how close
would the interpolation between its two neighbours have come? Drawn vs
predicted are compared by Dice and by mean surface distance, so a label whose
cross-section shifts or changes topology between planes (a thin fibre sheet
seen in-plane is the classic one) shows up as a near-zero Dice rather than
silently producing a wrong guide.

Read it as "where should I spend drawing time on the NEXT sample", not as a
pass/fail on this one. Two things to keep in mind:

  * It is deliberately PESSIMISTIC. Dropping a keyframe doubles the gap the
    interpolation has to span, so the reported error is roughly what you would
    get at twice the current spacing -- real interpolation error at the
    spacing actually drawn is about half of it.
  * Registration accuracy is NOT proportional to this. Measured on s12t
    (PROGRESS_LOG 2026-08-28), the registered outline's distance to an
    image-derived brain mask was flat against distance-to-nearest-keyframe
    (6.8 / 6.9 / 6.4 / 6.5 / 6.9 voxels at 0 / <2 / <4 / <7 / >7 planes away)
    -- i.e. planes that were hand-drawn came out no better than interpolated
    ones. A poor score here means "the interpolation is inventing this shape",
    which is worth knowing; it does not by itself mean the registration
    suffered.

Surface distances are reported in the mask's own in-plane pixels and, when
--voxel-size-um is given, in microns too (the sidecar deliberately does not
record voxel size -- see its voxel_size_um_note).

Usage (numpy/scipy/nibabel only, no ANTs -- any env with those works):
    python scripts/guide_mask_selftest.py atlas/mask/s12t_DeMBAguide7.nii.gz \\
        --voxel-size-um 2.6 2.6 32.0
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils  # noqa: E402
from registration_ants.mask_utils import _signed_distance  # noqa: E402

# Below these, the interpolation is not reproducing the drawn shape so much as
# inventing one. Chosen against the s12t guide7 numbers: everything the eye
# calls "fine" scores >0.90 there, the label whose cross-section genuinely
# cannot be interpolated (corpus callosum, a thin arc in-plane) scores ~0.00,
# and the ones worth redrawing land in between.
DICE_WARN = 0.90
DICE_BAD = 0.75

# A keyframe holding this little of the label's typical area is a stray brush
# mark, not an annotation. Worth separating out because such a plane does not
# just score badly itself -- it sits in the middle of the SDF blend and drags
# BOTH its neighbours' reconstructions to near-zero, so it reads as three
# broken keyframes when only one is actually wrong. Seen on s12t_guide6:
# label 3 planes 119 and 126 both collapsed (Dice 0.041 / 0.002) around a
# 4-pixel plane 124; s12t_DeMBAguide7, the repainted version, has none.
STRAY_AREA_FRACTION = 0.02


def _mean_surface_distance(pred, true):
    """Mean of (each drawn boundary pixel -> nearest predicted boundary) and
    the reverse, in pixels. Symmetric, so a prediction that is merely too
    small scores the same as one that is too large by as much."""
    if not pred.any() or not true.any():
        return float("nan")
    d_pred = ndimage.distance_transform_edt(~pred)
    d_true = ndimage.distance_transform_edt(~true)
    edge_pred = pred ^ ndimage.binary_erosion(pred)
    edge_true = true ^ ndimage.binary_erosion(true)
    return 0.5 * (d_true[edge_pred].mean() + d_pred[edge_true].mean())


def load_mask_planes(mask_path):
    """(volume as (plane, row, col), sidecar dict).

    The sidecar's slice indices are along SimpleITK's axis 0 -- the imaging
    planes a person scrolled through while painting -- which is nibabel's LAST
    axis, hence the transpose. Getting this backwards silently indexes the
    wrong axis and every label looks broken, so it is done once, here.
    """
    sidecar_path = atlas_utils.regions_sidecar_path(mask_path)
    if not sidecar_path.exists():
        raise SystemExit(f"no sidecar next to the mask: {sidecar_path}\n"
                         "This tool needs the annotated_slices/regions it records; re-export the "
                         "mask from paint_mask.py if it predates the sidecar.")
    with open(sidecar_path, encoding="utf-8") as f:
        sidecar = json.load(f)
    arr = np.asanyarray(nib.load(str(mask_path)).dataobj)
    return np.transpose(arr, (2, 1, 0)), sidecar


def leave_one_out(volume, label, keyframes):
    """[(z, gap, dice, surf_dist, drawn_px, predicted_px), ...] over the
    keyframes that HAVE two neighbours. First and last are skipped because
    there is nothing to interpolate them from -- which is also why they are
    the planes worth drawing most carefully: interpolate_sparse_mask leaves
    everything outside [min, max] empty, so an end keyframe placed short of
    where the structure really ends truncates it outright."""
    planes = {z: (volume[z] == label) for z in keyframes}
    rows = []
    for prev_z, z, next_z in zip(keyframes, keyframes[1:], keyframes[2:]):
        sdf_prev = _signed_distance(planes[prev_z])
        sdf_next = _signed_distance(planes[next_z])
        t = (z - prev_z) / (next_z - prev_z)
        predicted = ((1 - t) * sdf_prev + t * sdf_next) <= 0
        drawn = planes[z]
        denom = predicted.sum() + drawn.sum()
        dice = 2 * (predicted & drawn).sum() / denom if denom else 1.0
        rows.append((z, next_z - prev_z, dice, _mean_surface_distance(predicted, drawn),
                     int(drawn.sum()), int(predicted.sum())))
    return rows


def find_stray_keyframes(volume, results):
    """[(label, z, drawn_px, typical_px), ...] for keyframes painted with a
    negligible fraction of the label's usual area -- see STRAY_AREA_FRACTION.
    Compared against the MEDIAN keyframe area rather than the mean, so one
    speck cannot drag the reference down far enough to hide the next one."""
    strays = []
    for label, (keyframes, _) in sorted(results.items()):
        areas = {z: int((volume[z] == label).sum()) for z in keyframes}
        typical = float(np.median(list(areas.values())))
        if typical <= 0:
            continue
        for z in keyframes:
            if areas[z] < STRAY_AREA_FRACTION * typical:
                strays.append((label, z, areas[z], typical))
    return strays


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("regions_mask", help="the painted multi-label .nii.gz (its .regions.json "
                                         "sidecar must sit next to it)")
    ap.add_argument("--voxel-size-um", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="physical voxel size, to also report surface distances in microns "
                         "(the mask header carries none, by design)")
    ap.add_argument("--labels", type=int, nargs="+",
                    help="only check these painted labels (default: all in the sidecar)")
    ap.add_argument("--dice-warn", type=float, default=DICE_WARN,
                    help=f"flag keyframes below this Dice (default {DICE_WARN})")
    args = ap.parse_args()

    volume, sidecar = load_mask_planes(Path(args.regions_mask))
    annotated = sidecar.get("annotated_slices") or {}
    if not annotated:
        raise SystemExit("sidecar has no annotated_slices -- nothing to leave out")
    names = sidecar.get("regions") or {}
    # The in-plane axes are the mask's x/y; the keyframe axis is z, which the
    # surface distances never cross, so only the in-plane scale is meaningful.
    px_um = None
    if args.voxel_size_um:
        px_um = 0.5 * (args.voxel_size_um[0] + args.voxel_size_um[1])

    wanted = set(args.labels) if args.labels else None
    results = {}
    print(f"mask     {args.regions_mask}  {volume.shape} (plane, row, col)")
    print(f"sidecar  {atlas_utils.regions_sidecar_path(Path(args.regions_mask)).name}")
    if px_um:
        print(f"in-plane pixel {px_um:g} um")
    print()
    header = f"{'lbl':<4} {'region':<32} {'n':>3}  {'Dice mean':>9} {'min':>6}  {'surf-dist mean':>14} {'max':>8}"
    print(header)
    print("-" * len(header))
    for label_str in sorted(annotated, key=int):
        label = int(label_str)
        if wanted and label not in wanted:
            continue
        keyframes = sorted(annotated[label_str])
        rows = leave_one_out(volume, label, keyframes)
        results[label] = (keyframes, rows)
        name = (names.get(label_str) or ["?"])[0]
        if not rows:
            print(f"{label:<4} {name[:32]:<32} {0:>3}  (only {len(keyframes)} keyframe(s) -- "
                  "nothing has two neighbours)")
            continue
        dice = np.array([r[2] for r in rows])
        surf = np.array([r[3] for r in rows])
        unit = f" ({np.nanmean(surf) * px_um:.0f} um)" if px_um else ""
        print(f"{label:<4} {name[:32]:<32} {len(rows):>3}  {dice.mean():>9.3f} {dice.min():>6.3f}  "
              f"{np.nanmean(surf):>10.1f} px{unit} {np.nanmax(surf):>7.1f}")

    strays = find_stray_keyframes(volume, results)
    if strays:
        print("\nprobable stray brush marks -- these are almost certainly painting mistakes, and "
              "each one also wrecks its two neighbours' reconstruction:")
        for label, z, drawn_px, typical in strays:
            print(f"  {label:<4} plane {z:<5} {drawn_px} px drawn vs {typical:.0f} px typical for "
                  f"this label -- erase the plane, or finish painting it")

    stray_planes = {(label, z) for label, z, _, _ in strays}
    print(f"\nkeyframes the interpolation would not have reproduced (Dice < {args.dice_warn}):")
    print(f"  {'lbl':<4} {'plane':>6} {'gap':>4}  {'Dice':>6} {'surf-dist':>9}  drawn -> predicted")
    flagged = False
    for label, (keyframes, rows) in sorted(results.items()):
        for z, gap, dice, surf, drawn_px, pred_px in rows:
            if dice >= args.dice_warn:
                continue
            flagged = True
            neighbours = [n for n in keyframes if abs(keyframes.index(z) - keyframes.index(n)) == 1]
            if (label, z) in stray_planes:
                note = "  <- stray mark, not an interpolation failure"
            elif any((label, n) in stray_planes for n in neighbours):
                note = "  <- neighbour is a stray mark; fix that first"
            else:
                note = ""
            mark = "!!" if dice < DICE_BAD else " *"
            delta = 100 * (pred_px - drawn_px) / max(drawn_px, 1)
            surf_s = f"{surf:.1f}px" if surf == surf else "n/a"
            print(f"{mark} {label:<4} {z:>6} {gap:>4}  {dice:>6.3f} {surf_s:>9}  "
                  f"{drawn_px} -> {pred_px} ({delta:+.0f}%){note}")
    if not flagged:
        print("  (none)")

    print("\nends -- these are never leave-one-out testable and truncate the structure if placed "
          "short of where it really starts/stops:")
    for label, (keyframes, _) in sorted(results.items()):
        name = (names.get(str(label)) or ["?"])[0]
        print(f"  {label:<4} {name[:32]:<32} first={keyframes[0]:<4} last={keyframes[-1]:<4} "
              f"({len(keyframes)} keyframes, gaps {min(np.diff(keyframes)) if len(keyframes) > 1 else '-'}"
              f"-{max(np.diff(keyframes)) if len(keyframes) > 1 else '-'})")


if __name__ == "__main__":
    main()
