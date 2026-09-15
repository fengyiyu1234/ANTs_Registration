"""Property report for 2D section images -- run this BEFORE writing a
sections config. For each file: axes as stored (RGB composite vs separate
greyscale channels vs Z stack), bit depth, pixel size and whether the file's
calibration can be trusted, per-plane intensity / saturation / coverage, a
suggested registration channel, and for RGB composites which colours light
up together (how markers were colour-assigned). Reading rules live in
src/registration_ants/section_io.py.

Usage (antsreg env):

    python scripts/inspect_sections.py /path/to/sections/*.tif --png-dir /tmp/inspect
    python scripts/inspect_sections.py --config configs/my_sections.yaml --png-dir /tmp/inspect

--png-dir writes one thumbnail sheet per file (every plane, tissue outline,
suggested channel starred) -- look at it to confirm which plane is DAPI.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import section_io  # noqa: E402


def _report(info):
    px = info.get("pixel_size_um")
    lines = [f"\n== {info['file']}",
             f"   axes {info['axes']}  ->  {info['kind']}, {info['n_planes']} plane(s), {info['dtype']} "
             f"({info['bits_per_sample']} bits/sample, photometric {info['photometric']}), "
             f"{info['width_px']} x {info['height_px']} px",
             f"   pixel size: {f'{px:.4g} um' if px else 'UNKNOWN'}  ({info['pixel_size_note']})"]
    if info.get("software"):
        lines.append(f"   software: {info['software']}")
    if info["n_series"] > 1 or info["pyramid_levels"] > 1:
        lines.append(f"   {info['n_series']} series, {info['pyramid_levels']} pyramid level(s) -- only series 0, "
                     "full resolution is read")
    for note in info["notes"]:
        lines.append(f"   note: {note}")
    lines.append(f"   tissue: {info['tissue_pct']:.0f}% of the image (Otsu on all planes, inspection only)")
    lines.append("   plane  label        p50   p99   max  bg_p99  sat%  zero%  levels  coverage%  sparseness")
    for s in info["planes"]:
        star = "*" if s["plane"] == info.get("suggested_channel") else " "
        lines.append(f"  {star}{s['plane']:>4}  {s['label'][:10]:<10} {s['p50_tissue']:>5.0f} {s['p99_tissue']:>5.0f} "
                     f"{s['max']:>5.0f} {s['bg_p99']:>7.0f} {s['saturated_pct']:>5.1f} {s['zero_pct']:>6.1f} "
                     f"{s['levels_in_tissue']:>7} {s['coverage_pct']:>10.0f} {s['sparseness']:>11.1f}")
    if info["kind"] == "rgb":
        lines.append(f"   RGB really greyscale (R=G=B): {info['rgb_is_gray']}")
        combos = "  ".join(f"{k} {v:.0f}%" for k, v in info["rgb_cooccurrence_pct"].items())
        lines.append(f"   brightest-5%-per-plane colour combinations: {combos}")
        lines.append("   (mostly single letters = each marker got a primary colour; a large R+B share = a magenta "
                     "marker, R+G = yellow, G+B = cyan; confirm on the thumbnails)")
    return "\n".join(lines)


def _warnings(info):
    out = []
    if not info.get("pixel_size_um"):
        out.append("pixel size unknown -- take it from the microscope's original file and set pixel_size_um")
    if any(s["saturated_pct"] > 1 for s in info["planes"]):
        out.append("a plane is >1% saturated -- fine for registration, but unmixing/intensity measurements "
                   "are not exact there")
    if info["dtype"] == "uint8" and any(s["levels_in_tissue"] < 64 for s in info["planes"]):
        out.append("a plane uses <64 grey levels in tissue -- heavily stretched or compressed 8-bit export")
    if info["kind"] == "rgb" and not info.get("rgb_is_gray"):
        out.append("RGB composite: registration needs channel: sum, or channel: <marker> + panel_colors "
                   "(the colour each marker was assigned) -- a single R/G/B plane mixes markers")
    return out


def _snippet(info):
    lines = [f"  - name: {Path(info['file']).stem}", f"    image: {info['file']}",
             f"    pixel_size_um: {info['pixel_size_um']:.4g}" if info.get("pixel_size_um")
             else "    pixel_size_um: FILL_IN   # not in the file"]
    if info["kind"] == "channels":
        names = info.get("channel_names") or []
        sug = info["suggested_channel"]
        lines.append(f"    channel: {names[sug] if sug < len(names) else sug}   # suggested -- confirm on the PNG")
    elif info["kind"] == "rgb":
        lines.append("    channel: sum   # or a marker name + panel_colors, see sections2d.example.yaml")
    if info["notes"] and any("Z planes" in x for x in info["notes"]):
        lines.append("    z_projection: max")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="image files or globs")
    ap.add_argument("--config", help="a sections config: inspect every section image it lists")
    ap.add_argument("--png-dir", help="write a thumbnail sheet per file here")
    ap.add_argument("--z-projection", default="max")
    ap.add_argument("--json", help="also write all reports to this JSON file")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += sorted(glob.glob(p)) or [p]
    if args.config:
        import register_sections_2d as cli
        files += [s["image"] for s in cli.load_sections_config(args.config)["sections"]]
    if not files:
        sys.exit("no input files")
    if args.png_dir:
        Path(args.png_dir).mkdir(parents=True, exist_ok=True)

    reports = []
    for f in files:
        try:
            info, view, tissue = section_io.inspect_image(f, z_projection=args.z_projection)
        except Exception as exc:          # report and keep going through the rest
            print(f"\n== {f}\n   ERROR: {exc}")
            reports.append({"file": f, "error": str(exc)})
            continue
        print(_report(info))
        for w in _warnings(info):
            print(f"   ! {w}")
        if args.png_dir:
            png = Path(args.png_dir) / f"{Path(f).stem}_inspect.png"
            section_io.render_inspection(png, info, view, tissue)
            print(f"   thumbnails: {png}")
        reports.append(info)

    ok = [r for r in reports if "error" not in r]
    if len({(r["axes"], r["dtype"], r["n_planes"]) for r in ok}) > 1:
        print("\n! files differ in axes/dtype/plane count -- they cannot all share one section_defaults.channel")
    if ok:
        print("\nconfig starting point (sections:)\n" + "\n".join(_snippet(r) for r in ok))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(reports, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
