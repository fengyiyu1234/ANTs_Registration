"""Thumbnails + an Excel sheet for marking which end of each stack is dorsal.

For every sample in batch_samples.SAMPLES whose soma image is finished:
  - 6 horizontal planes (z from first to last slice), downsampled 8x in xy,
  - 3 cross-sections along z at 30/50/70 % of the anterior-posterior extent
    (x = medial -> lateral across, z = slice 0 at the TOP), drawn at roughly
    isotropic scale so the dorsal dome vs. the flat ventral base is visible.
Writes <REVIEW_DIR>/thumbs/<sample>.png and REVIEW_XLSX. Rerunning keeps
whatever is already filled in the user columns and only redraws missing
thumbnails (--redraw to force).

    python scripts/orientation_review.py [--only NAME ...] [--redraw]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_samples as bs  # noqa: E402

XY_STEP = 8          # 2.6 um -> 20.8 um
USER_COLS = ("z0_user", "xy_ok", "notes")


def block_mean(img, f):
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def draw(tif, out_png, title):
    vol = tifffile.memmap(str(tif))                      # (Z, Y, X)
    nz, ny, nx = vol.shape
    zs = np.linspace(0.04, 0.96, 6) * (nz - 1)
    axial = [(int(z), block_mean(np.asarray(vol[int(z)], dtype=np.float32), XY_STEP)) for z in zs]
    z_step = max(1, round(XY_STEP * bs.IMAGE_VOXEL_UM[0] / bs.IMAGE_VOXEL_UM[2]))  # ~same um as xy
    cross = []
    for frac in (0.3, 0.5, 0.7):
        y = int(frac * (ny - 1))
        rows = np.asarray(vol[:, y - 2:y + 2, :], dtype=np.float32).mean(axis=1)   # (Z, X)
        rows = rows[: (nz // z_step) * z_step].reshape(-1, z_step, nx).mean(axis=1)
        cols = (nx // XY_STEP) * XY_STEP
        rows = rows[:, :cols].reshape(rows.shape[0], -1, XY_STEP).mean(axis=2)
        cross.append((y, rows))
    sample = np.concatenate([a.ravel() for _, a in axial])
    lo, hi = np.percentile(sample[sample > 0], [1, 99.5]) if np.any(sample > 0) else (0, 1)

    fig = plt.figure(figsize=(18, 11), dpi=90)
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1.4])
    for i, (z, img) in enumerate(axial):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"z = {z} / {nz - 1}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("top = olfactory bulb?\nleft = midline?", fontsize=10)
    for i, (y, img) in enumerate(cross):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        # z_step makes a z row ~24 um vs 20.8 um per x column: close to true shape
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
        ax.set_title(f"section at y = {y} ({int(100 * y / (ny - 1))}% A->P);  "
                     f"z = 0 at TOP, left = midline", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([0, img.shape[0] - 1])
        ax.set_yticklabels(["z=0", f"z={nz - 1}"], fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def existing_user_values(path):
    if not path.exists():
        return {}
    import openpyxl
    ws = openpyxl.load_workbook(path)["orientation"]
    header = [c.value for c in ws[1]]
    keep = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        rec = dict(zip(header, row))
        keep[rec["sample"]] = {k: rec.get(k) for k in USER_COLS + ("z0_claude", "claude_reason")}
    return keep


def write_xlsx(rows, path):
    import openpyxl
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from PIL import Image

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "orientation"
    header = ["sample", "group", "image", "shape_zyx", "z0_default", "z0_claude", "claude_reason",
              "z0_user", "xy_ok", "notes", "thumbnail"]
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
    user_fill = PatternFill("solid", fgColor="FFF2CC")
    dv_z = DataValidation(type="list", formula1='"dorsal,ventral,unsure"', allow_blank=True)
    dv_xy = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv_z)
    ws.add_data_validation(dv_xy)
    widths = dict(A=11, B=8, C=24, D=17, E=11, F=11, G=40, H=11, I=8, J=30, K=95)
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for r, row in enumerate(rows, start=2):
        ws.append([row.get(k) for k in header[:-1]] + [None])
        for col in "HIJ":
            ws[f"{col}{r}"].fill = user_fill
        dv_z.add(f"H{r}")
        dv_xy.add(f"I{r}")
        for c in ws[r]:
            c.alignment = Alignment(vertical="top", wrap_text=True)
        png = row.get("png")
        if png and Path(png).exists():
            small = Path(png).with_name(Path(png).stem + "_xlsx.png")
            with Image.open(png) as im:
                im.thumbnail((680, 420))
                im.save(small)
            img = XLImage(str(small))
            ws.add_image(img, f"K{r}")
            ws.row_dimensions[r].height = img.height * 0.75 + 4
            ws[f"C{r}"].hyperlink = str(png)
    notes = wb.create_sheet("how_to")
    for line in [
        "z0_user: which side of the brain is at z = 0 (the first slice).",
        "  dorsal  = cortex dome at the top of the cross-sections (z = 0 row); first horizontal planes are cortex only.",
        "  ventral = flat base / brainstem / hypothalamus at z = 0; the dome is at the bottom.",
        "  dorsal -> DeMBA orientation [1, 3, 2] (right hemisphere); ventral -> [-1, 3, -2] (left hemisphere).",
        "xy_ok: 'no' if the olfactory bulb is not at the top or the midline is not on the left.",
        "Yellow columns are yours; rerunning orientation_review.py keeps them.",
        "Then: python scripts/make_batch_configs.py   (reads z0_user, falls back to z0_default).",
    ]:
        notes.append([line])
    notes.column_dimensions["A"].width = 120
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--redraw", action="store_true")
    args = p.parse_args()
    keep = existing_user_values(bs.REVIEW_XLSX)
    rows = []
    for s in bs.SAMPLES:
        tif, manifest = bs.soma_image(s["name"])
        png = bs.REVIEW_DIR / "thumbs" / f"{s['name']}.png"
        row = dict(sample=s["name"], group=s["group"], z0_default=bs.default_z_first(s),
                   image=tif.name if tif else "not stitched yet")
        if tif:
            row["shape_zyx"] = f"{manifest['output_slices']} x {manifest['output_height']} x {manifest['output_width']}"
            wanted = not args.only or s["name"] in args.only
            if wanted and (args.redraw or not png.exists()):
                print(f"drawing {s['name']} ...", flush=True)
                draw(tif, png, f"{s['name']}  ({s['group']})  {tif.name}  default z=0: {row['z0_default']}")
            row["png"] = str(png) if png.exists() else None
        row.update({k: v for k, v in keep.get(s["name"], {}).items() if v not in (None, "")})
        rows.append(row)
    write_xlsx(rows, bs.REVIEW_XLSX)
    print(f"wrote {bs.REVIEW_XLSX}")


if __name__ == "__main__":
    main()
