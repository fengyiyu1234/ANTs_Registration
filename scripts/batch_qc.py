"""Quick QC of one batch registration: atlas boundaries over the sample, and
where the cells landed.

Reads <output_dir>/<name>_fine_20um.nii.gz and <name>_labels_in_sample.nii.gz
(same grid), and cell_registration/*/cell_registration.csv. Writes into
<output_dir>/qc/:
  overlay.png        6 horizontal planes + 3 cross-sections with region edges
  cells_by_region.csv  counts per atlas region (all classes pooled) + per class
and prints the share of cells outside the atlas (background / no label).

    python scripts/batch_qc.py TSC_FW11 [TSC_FF11 ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_samples as bs  # noqa: E402


def edges(lab):
    e = np.zeros(lab.shape, bool)
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    return e


def panel(ax, img, lab, title, lo, hi, tissue):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    rgba = np.zeros(lab.shape + (4,))
    rgba[edges(lab)] = (1.0, 0.25, 0.1, 0.9)
    rgba[(lab == 0) & tissue & ~edges(lab)] = (0.1, 0.6, 1.0, 0.35)  # tissue outside atlas
    ax.imshow(rgba)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def qc(name):
    out = bs.OUTPUT_ROOT / name / "DeMBA_P5"
    img = np.asarray(nib.load(str(next(out.glob(f"{name}_fine_*um.nii.gz")))).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(str(out / f"{name}_labels_in_sample.nii.gz")).dataobj)
    mask_path = out / f"{name}_brain_mask.nii.gz"
    tissue = (np.asarray(nib.load(str(mask_path)).dataobj) > 0 if mask_path.exists()
              else np.zeros(img.shape, bool))
    if img.shape != lab.shape:
        raise ValueError(f"{name}: image {img.shape} vs labels {lab.shape}")
    nx, ny, nz = img.shape            # (x = midline->lateral, y = A->P, z = slice)
    lo, hi = np.percentile(img[img > 0], [2, 99.5])
    qc_dir = out / "qc"
    qc_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 11), dpi=90)
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1.5])
    for i, z in enumerate(np.linspace(0.1, 0.9, 6) * (nz - 1)):
        z = int(z)
        panel(fig.add_subplot(gs[0, i]), img[:, :, z].T, lab[:, :, z].T, f"z = {z}/{nz - 1}", lo, hi, tissue[:, :, z].T)
    for i, f in enumerate((0.3, 0.5, 0.7)):
        y = int(f * (ny - 1))
        panel(fig.add_subplot(gs[1, 2 * i:2 * i + 2]), img[:, y, :].T, lab[:, y, :].T,
              f"cross-section y = {y} ({int(f * 100)}% A->P), z = 0 at top", lo, hi, tissue[:, y, :].T)
    fig.suptitle(f"{name}: atlas region edges (red) on sample, 20 um; blue = auto brain mask with no atlas label",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(qc_dir / "overlay.png")
    plt.close(fig)

    rows = []
    for csv in sorted((out / "cell_registration").glob("*/cell_registration.csv")):
        df = pd.read_csv(csv, header=None, usecols=[9, 10])
        df.columns = ["id", "region"]
        df["cls"] = csv.parent.name
        rows.append(df)
    cells = pd.concat(rows, ignore_index=True)
    outside = cells["region"].isin(["background", "no label"])
    by_region = (cells.groupby(["id", "region"]).size().rename("cells")
                 .reset_index().sort_values("cells", ascending=False))
    by_region["share_pct"] = 100 * by_region["cells"] / len(cells)
    by_region.to_csv(qc_dir / "cells_by_region.csv", index=False)
    per_class = cells.assign(outside=outside).groupby("cls")["outside"].agg(["size", "mean"])
    per_class.columns = ["cells", "outside_frac"]
    per_class.to_csv(qc_dir / "cells_per_class.csv")
    print(f"== {name}: {len(cells):,} cells, {100 * outside.mean():.1f}% outside the atlas "
          f"(background/no label); {by_region['region'].nunique()} regions hit")
    print(by_region.head(15).to_string(index=False))
    print(f"   -> {qc_dir}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        qc(n)
