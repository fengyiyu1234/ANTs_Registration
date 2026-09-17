"""Rewrite a 2D section batch's shareable atlas overlays (<output_dir>/overlays/,
see src/registration_ants/section_overlays.py): one PNG per section with the
registered atlas as coloured outlines, one with the regions filled in, plus the
colour key. register_sections_2d.py already writes them after every batch; this
regenerates them on their own -- for results from before that, or to change how
they look without repeating the registration.

Usage (antsreg env), with the same config (and options) the registration ran with:

    python scripts/export_overlays.py configs/my_sections.yaml
    python scripts/export_overlays.py configs/my_sections.yaml --level 6 --alpha 0.6
    python scripts/export_overlays.py configs/my_sections.yaml --source prep   # no original files read
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_sections_2d as cli  # noqa: E402  (also puts src/ on sys.path)
from registration_ants import section_overlays  # noqa: E402


def add_overlay_args(ap):
    """The rendering options. register_sections_2d.py deliberately does not
    take them (it only has --no-overlays): the batch writes the defaults, and
    rerunning this script is cheap, so tuning them never costs a re-run."""
    ap.add_argument("--source", choices=("raw", "prep"), default="raw",
                    help="background image: the original section (default) or the preprocessed "
                         "section_prep.nii.gz the registration saw (no original file needed)")
    ap.add_argument("--max-px", type=int, default=section_overlays.MAX_PX,
                    help=f"longest edge of a written PNG (default {section_overlays.MAX_PX})")
    ap.add_argument("--alpha", type=float, default=section_overlays.ALPHA,
                    help=f"region fill opacity on the filled picture (default {section_overlays.ALPHA})")
    ap.add_argument("--line-width", type=int, default=section_overlays.LINE_WIDTH,
                    help=f"boundary width in output pixels (default {section_overlays.LINE_WIDTH})")
    ap.add_argument("--level", type=int,
                    help="collapse labels to this ontology depth first (root = 1; e.g. 6 for major "
                         "structures instead of every leaf area). Default: the labels as registered")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the sections config the registration ran with")
    ap.add_argument("--input-dir", help="same as register_sections_2d.py --input-dir")
    ap.add_argument("--output-dir", help="same as register_sections_2d.py --output-dir")
    ap.add_argument("--pattern", help="same as register_sections_2d.py --pattern")
    add_overlay_args(ap)
    args = ap.parse_args()
    cfg = cli.load_sections_config(args.config, args.input_dir, args.output_dir, args.pattern)
    if section_overlays.export_overlays(cfg, source=args.source, max_px=args.max_px, alpha=args.alpha,
                                        line_width=args.line_width, level=args.level) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
