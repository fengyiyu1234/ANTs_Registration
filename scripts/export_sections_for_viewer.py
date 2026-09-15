"""Rewrite a 2D section batch's viewer copy (<output_dir>/viewer/, see
src/registration_ants/section_viewer.py) without registering anything.
register_sections_2d.py already writes it after every batch; this is for
results from before that, or after editing outputs by hand.

Usage (antsreg env), with the same config (and options) the registration ran with:

    python scripts/export_sections_for_viewer.py configs/my_sections.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_sections_2d as cli  # noqa: E402  (also puts src/ on sys.path)
from registration_ants import section_viewer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the sections config the registration ran with")
    ap.add_argument("--input-dir", help="same as register_sections_2d.py --input-dir")
    ap.add_argument("--output-dir", help="same as register_sections_2d.py --output-dir")
    ap.add_argument("--pattern", help="same as register_sections_2d.py --pattern")
    ap.add_argument("--no-raw", action="store_true", help="skip the full-resolution image stack")
    args = ap.parse_args()
    cfg = cli.load_sections_config(args.config, args.input_dir, args.output_dir, args.pattern)
    if section_viewer.export_for_viewer(cfg, raw=not args.no_raw) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
