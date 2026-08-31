"""Non-interactive: apply a reposition plan to a stack, its labels, and its
detected cells.

The plan (shared/reposition.py's JSON) is drawn by hand -- fragment outlines
and a few keyframe transforms per fragment; this script is what turns it into
files. Splitting it out of the GUI is the same division paint_mask.py and
scripts/relabel_cells.py already use: the interactive
tool decides WHAT, a plain script does it, so the numeric work is re-runnable
against a corrected plan without reopening napari, and testable without one.

    conda activate antsreg
    python scripts/apply_reposition.py <plan.json> --output-dir <dir> \\
        [--cell-centroids-dir DIR --cells-voxel-size-um 0.65 0.65 8.0]

Outputs, into --output-dir:
    <stem>_repositioned.tif          the stack to register, fragments closed up
    <stem>_repositioned_labels.nii.gz  the fragment outlines where they now sit
    cell_centroids/<class>.csv       centroids with the same move applied
    <stem>_reposition_applied.json   the plan actually used, next to its output

WHAT MOVES WHAT
---------------
A cell is moved when the painted label volume says it sits on a fragment, so
the outline drawn once decides both the image move and the cell move and there
is no second boundary to keep in sync. Everything else -- every column of the
CSV, every cell off a fragment -- passes through untouched, and the original
files are never written to.

The centroid CSVs are in the full-resolution pixel grid cells were detected on
(cx,cy,z), which is finer than the painted grid, so --cells-voxel-size-um is
required alongside the plan's own voxel_size_um: the two grids meet only in
physical microns. Those are the pipeline config's `cells.voxel_size_um` and
`sample.voxel_size_um` respectively -- 0.65/0.65/8 against 2.6/2.6/32 (s12q,
s12t) or against 5.2/5.2/64 (s18). Getting them backwards does not raise, it
silently scales every move, so both are printed in the header for checking
against the config before the outputs get used.

--invert applies the plan backwards instead, taking a repositioned result back
onto the original geometry. That is for QC and figures -- overlaying atlas
labels on the untouched stack -- and is NOT needed for cell counting: cells
were moved and looked up in the same space, and each row carries its region
assignment home by its own identity.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import reposition as rp  # noqa: E402

CENTROID_COORD_COLUMNS = ("cx", "cy", "z")


def read_volume(path):
    """(array_zyx, writer) for either a .tif stack or anything SimpleITK reads.

    The writer closes over whatever geometry the source carried (a NIfTI's
    spacing/origin/direction), so a label volume round-trips through this
    script with its header intact instead of silently reverting to identity.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))

        def write(out_path, data):
            tifffile.imwrite(str(out_path), data)
        return arr, write

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)

    def write(out_path, data):
        out = sitk.GetImageFromArray(data)
        out.CopyInformation(image)
        sitk.WriteImage(out, str(out_path))
    return arr, write


def move_centroid_csv(csv_path, out_path, labels_zyx, plan, cells_voxel_um):
    """Rewrite one cell_centroids CSV with the plan applied. Returns
    (n_rows, n_moved)."""
    df = pd.read_csv(csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    missing = [c for c in CENTROID_COORD_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {missing}; found {list(df.columns)}. "
                         f"Expected the run_inference.py layout "
                         f"cx,cy,z,score,slice_name,tile_name.")

    cx, cy, cz, moved = rp.apply_to_cells(
        df[cols["cx"]].to_numpy(float), df[cols["cy"]].to_numpy(float),
        df[cols["z"]].to_numpy(float), labels_zyx, plan, cells_voxel_um)

    out = df.copy()
    out[cols["cx"]], out[cols["cy"]], out[cols["z"]] = cx, cy, cz
    # The originals ride along rather than being replaced outright: a cell's
    # pre-move position is the only way to tell afterwards which rows this
    # script touched and to check one by eye against the raw stack. Extra
    # columns are ignored by cell_points.read_centroid_csv, which reads
    # cx/cy/z and the named provenance columns by name.
    out["reposition_label"] = moved
    out["orig_cx"] = df[cols["cx"]].to_numpy(float)
    out["orig_cy"] = df[cols["cy"]].to_numpy(float)
    out["orig_z"] = df[cols["z"]].to_numpy(float)
    out.to_csv(out_path, index=False)
    return len(df), int((moved > 0).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="reposition plan JSON (shared/reposition.py format)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image", help="override the plan's image_path")
    ap.add_argument("--labels", help="override the plan's labels_path")
    ap.add_argument("--cell-centroids-dir",
                    help="directory of <class>.csv centroid files; omit to move only the image")
    ap.add_argument("--cells-voxel-size-um", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="microns per pixel of the grid cells were detected on "
                         "(the pipeline config's cells.voxel_size_um)")
    ap.add_argument("--feather-um", type=float,
                    help="override the plan's paste-edge feather (cosmetic; see reposition.py)")
    ap.add_argument("--invert", action="store_true",
                    help="apply the plan backwards, to take a repositioned result "
                         "back onto the original geometry (QC/figures only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan summary and boundary report, write nothing")
    args = ap.parse_args(argv)

    plan = rp.read_plan(args.plan)
    if args.feather_um is not None:
        plan["feather_um"] = float(args.feather_um)
    image_path = Path(args.image or plan["image_path"])
    labels_path = Path(args.labels or plan["labels_path"])
    for role, path in (("image", image_path), ("labels", labels_path)):
        if not str(path):
            raise SystemExit(f"no {role} path: the plan's {role}_path is empty and "
                             f"--{role} was not given")
        if not path.exists():
            raise SystemExit(f"{role} not found: {path}")

    image, write_image = read_volume(image_path)
    labels, write_labels = read_volume(labels_path)
    if image.shape != labels.shape:
        raise SystemExit(f"image {image.shape} and labels {labels.shape} are different "
                         f"volumes; they must be the same (z, y, x) grid")
    if tuple(plan["image_shape_zyx"]) != image.shape:
        raise SystemExit(f"the plan was drawn on a {tuple(plan['image_shape_zyx'])} volume "
                         f"but {image_path} is {image.shape}. Every offset in a plan is "
                         f"microns on the grid it was drawn on -- applying it to a "
                         f"different one would move everything by the wrong amount.")

    if args.invert:
        plan = rp.invert_plan(plan)

    print(f"plan            {args.plan}{'  (INVERTED)' if args.invert else ''}")
    print(f"image           {image_path}  {image.shape} {image.dtype}")
    print(f"labels          {labels_path}")
    print(f"painted grid    {tuple(plan['voxel_size_um'])} um per voxel (x, y, z)")
    if args.cells_voxel_size_um:
        print(f"cell grid       {tuple(args.cells_voxel_size_um)} um per pixel (x, y, z)")
    print(f"interpolate     {plan.get('interpolate', True)}   feather_um "
          f"{plan.get('feather_um', 0.0)}")
    print("")
    for frag in plan["fragments"]:
        planes = rp.fragment_source_planes(frag, image.shape[0], plan.get("interpolate", True))
        voxels = int((labels == int(frag["label"])).sum())
        span = f"z {planes[0]}..{planes[-1]} ({len(planes)} planes)" if planes else "does not move"
        print(f"  label {frag['label']:<3} {frag.get('name', '') or '(unnamed)':<24} "
              f"{len(frag['keyframes'])} keyframe(s), {span}, {voxels} voxels")

    warnings = rp.boundary_warnings(rp.boundary_report(plan, labels))
    if warnings:
        print("")
        for w in warnings:
            print(f"WARNING: {w}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    suffix = "_restored" if args.invert else "_repositioned"

    moved_image = rp.apply_to_image(image, labels, plan)
    image_out = out_dir / f"{stem}{suffix}.tif"
    write_image(image_out, moved_image)

    moved_labels = rp.apply_to_labels(labels, plan)
    labels_out = out_dir / f"{stem}{suffix}_labels{''.join(labels_path.suffixes)}"
    write_labels(labels_out, moved_labels)

    print(f"\nwrote {image_out}")
    print(f"wrote {labels_out}")

    if args.cell_centroids_dir:
        if not args.cells_voxel_size_um:
            raise SystemExit("--cell-centroids-dir needs --cells-voxel-size-um: the centroid "
                             "grid and the painted grid meet only in physical microns, and "
                             "guessing a ratio between them is the one error that would "
                             "scale every cell move without raising.")
        cells_dir = Path(args.cell_centroids_dir)
        csvs = sorted(cells_dir.glob("*.csv"))
        if not csvs:
            raise SystemExit(f"no *.csv under {cells_dir}")
        cells_out = out_dir / "cell_centroids"
        cells_out.mkdir(parents=True, exist_ok=True)
        total, total_moved = 0, 0
        for csv_path in csvs:
            n, n_moved = move_centroid_csv(csv_path, cells_out / csv_path.name,
                                           labels, plan, args.cells_voxel_size_um)
            total += n
            total_moved += n_moved
            print(f"  {csv_path.name:<36} {n_moved:>7} / {n} cells moved")
        print(f"wrote {cells_out}  ({total_moved} / {total} cells moved in total)")

    applied = out_dir / f"{stem}{suffix}_plan.json"
    rp.write_plan(applied, plan)
    print(f"wrote {applied}   (the plan as applied -- rerun or invert from this)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
