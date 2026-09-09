"""Burn a scale bar (bottom-right) and the sample name (top-right) into
every PNG in a folder, writing annotated copies next to the originals.

The sample name is taken from the file name itself, so "s18_RFP.png" is
labelled "s18_RFP". Annotated files get a suffix (default "_annot") so a
re-run never reads its own output back in.

Usage (any env with Pillow, e.g. the antsreg env):
    conda activate antsreg
    python scripts/annotate_scalebar.py <folder> [--um-per-px 2.6]

Common options:
    --um-per-px 2.6     pixel size in micrometres (default 2.6)
    --bar-um 500        force a bar length instead of picking a round one
    --suffix _annot     output name suffix
    --outline           add a thin dark outline behind the white text
    --font-frac 0.035   text height as a fraction of image height
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Bar lengths we are willing to draw, in micrometres.
NICE_UM = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def load_font(px):
    """A bold TrueType face at `px` pixels, falling back to matplotlib's copy."""
    paths = list(FONT_CANDIDATES)
    try:
        import matplotlib
        paths.append(
            str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
        )
    except ImportError:
        pass
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def pick_bar_um(width_px, um_per_px, target_frac=0.18):
    """Round bar length whose drawn width is closest to `target_frac` of the image."""
    target_um = width_px * um_per_px * target_frac
    return min(NICE_UM, key=lambda um: abs(um - target_um))


def fmt_um(um):
    if um >= 1000 and um % 1000 == 0:
        return f"{um // 1000} mm"
    return f"{um:g} µm"


def annotate(path, out_path, um_per_px, bar_um, font_frac, outline):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    margin = max(8, round(min(w, h) * 0.04))
    font = load_font(max(10, round(h * font_frac)))
    stroke = dict(stroke_width=max(1, round(h * font_frac * 0.08)),
                  stroke_fill=(0, 0, 0)) if outline else {}

    # --- sample name, top right ---
    name = path.stem
    box = draw.textbbox((0, 0), name, font=font)
    draw.text((w - margin - (box[2] - box[0]), margin - box[1]), name,
              fill=(255, 255, 255), font=font, **stroke)

    # --- scale bar, bottom right ---
    if bar_um is None:
        bar_um = pick_bar_um(w, um_per_px)
    bar_px = round(bar_um / um_per_px)
    if bar_px >= w - 2 * margin:
        raise ValueError(f"{path.name}: {bar_um} um bar is wider than the image")
    bar_h = max(3, round(h * 0.007))

    label = fmt_um(bar_um)
    lbox = draw.textbbox((0, 0), label, font=font)
    label_h = lbox[3] - lbox[1]
    gap = max(4, round(h * 0.012))

    bar_x1 = w - margin
    bar_x0 = bar_x1 - bar_px
    bar_y1 = h - margin
    bar_y0 = bar_y1 - bar_h
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

    label_x = bar_x0 + (bar_px - (lbox[2] - lbox[0])) / 2
    label_y = bar_y0 - gap - label_h - lbox[1]
    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font, **stroke)

    img.save(out_path)
    return bar_um, bar_px


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--um-per-px", type=float, default=2.6)
    ap.add_argument("--bar-um", type=float, default=None)
    ap.add_argument("--suffix", default="_annot")
    ap.add_argument("--font-frac", type=float, default=0.035)
    ap.add_argument("--outline", action="store_true")
    args = ap.parse_args()

    folder = args.folder
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    pngs = sorted(p for p in folder.glob("*.png")
                  if not p.stem.endswith(args.suffix))
    if not pngs:
        sys.exit(f"no PNGs to annotate in {folder}")

    for p in pngs:
        out = p.with_name(p.stem + args.suffix + ".png")
        bar_um, bar_px = annotate(p, out, args.um_per_px, args.bar_um,
                                  args.font_frac, args.outline)
        print(f"{p.name} -> {out.name}  ({fmt_um(bar_um)} = {bar_px} px)")


if __name__ == "__main__":
    main()
