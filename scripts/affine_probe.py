"""Run ONLY the Translation + Affine stages of the pipeline, for one or more
atlas croppings, and report what the Affine actually did -- ~30 s per variant
instead of the 1.5-3 h a full SyN run costs.

Why this is worth a script of its own: on this data the Affine turned out to be
the thing that decides whether the atlas covers the sample at all, and it is
extremely sensitive to how much background surrounds the atlas tissue. Measured
on s12t, same sample and same preprocessing, ONLY the atlas cropping differing:

    atlas.slicing                    det    rotation   stretch            unlabelled
    [[320,536],[108,690],[129,482]]  0.390  21.1 deg   [0.53 0.74 1.00]   40.6%
    [[320,536],[108,690],[151,371]]  1.000   0.0 deg   [1.00 1.00 1.00]   14.8%
    [[320,640], null,     null    ]  0.980   0.1 deg   [0.98 1.00 1.00]   21.0%

Three columns matter and they say different things:

* `rotation`/`stretch` come from a polar decomposition of the linear part, not
  from the raw matrix -- a determinant near 1 can hide a large rotation paired
  with an anisotropic squash, and those are exactly the solutions that misplace
  every structure while looking harmless in det alone.
* `rotation` and `stretch` BOTH at identity means the Affine stage did nothing
  and the result is whatever the Translation pre-alignment produced. That can
  still be the right answer (if the sample is already correctly posed, identity
  IS correct), but it is not evidence the Affine found it -- the same no-op
  would come back for a sample that genuinely needed correcting.
* `unlabelled` is the fraction of sample brain tissue that ends up with no
  atlas label. This is the number that matters for cell assignment, and it does
  NOT move together with det: the framings that make the Affine engage here are
  the ones with the worst coverage.

`outside` (atlas labels landing on non-tissue) is reported alongside because
coverage alone can be bought by an oversized atlas smothering everything.

Usage (antsreg env, headless):

    python scripts/affine_probe.py configs/s12t.yaml
    python scripts/affine_probe.py configs/s12t.yaml \\
        --slicing '[[320,536],[108,690],[151,371]]' \\
        --slicing '[[320,640],null,null]'
    python scripts/affine_probe.py configs/s12t.yaml --slicing ... --png /tmp/probe.png

With no --slicing, probes the config's own atlas.slicing. Each --slicing adds a
variant (JSON, `null` for "no crop on this axis"); the config's own value is
always probed first as the baseline. --png writes a coronal overlay strip per
variant, which is what actually settles ties the numbers leave open.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ants  # noqa: E402
from registration_ants import (  # noqa: E402
    atlas_utils, brain_mask, config as config_mod, io_utils, pipeline,
)


def prepare_sample(cfg):
    """The exact grid registration runs on: raw stack -> crop_for_registration
    -> isotropic resample -> preprocess, matching pipeline.run()."""
    s, reg = cfg["sample"], cfg["registration"]
    raw = io_utils.load_tiff_stack_as_ants(s["raw_tiff"], tuple(s["voxel_size_um"]))
    crop = reg.get("crop_for_registration")
    if crop:
        raw = io_utils.crop_to_bounds(raw, x=crop.get("x"), y=crop.get("y"), z=crop.get("z"))
    img = pipeline._preprocess(io_utils.resample_to_isotropic(raw, reg["fine_target_um"]),
                               cfg["preprocess"])
    mask_arr, _ = brain_mask.generate_brain_mask(img.numpy())
    mask_img = ants.from_numpy(mask_arr.astype("float32"), spacing=img.spacing,
                               origin=img.origin, direction=img.direction)
    return img, mask_arr.astype(bool), mask_img


def probe(sample_img, brain_arr, brain_img, atlas_cfg, slicing):
    """Translation pre-align + Affine, exactly as register.register_to_atlas
    does it (unmasked Affine -- a silhouette mask on the Affine is its own
    known failure, see that function's comments), then warp the annotation
    back to measure coverage."""
    template, annotation = atlas_utils.prepare_custom_atlas(
        atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
        orientation=atlas_cfg.get("orientation"), slicing=slicing,
        background_margin_voxels=atlas_cfg.get("background_margin_voxels"))
    prealign = ants.registration(
        fixed=template, moving=sample_img, type_of_transform="Translation",
        aff_metric="mattes", moving_mask=brain_img, mask_all_stages=True)
    affine = ants.registration(
        fixed=template, moving=sample_img, type_of_transform="Affine",
        initial_transform=prealign["fwdtransforms"][0],
        aff_metric="mattes", mask_all_stages=True)
    # whichtoinvert is NOT optional here: an Affine-only registration's
    # invtransforms is a bare single .mat, which apply_transforms' auto-default
    # falls through to "invert nothing" for -- applying the forward matrix and
    # returning a silently empty (or silently wrong) label volume. Same trap
    # transforms._mat_entries_to_invert exists to avoid.
    labels = ants.apply_transforms(
        fixed=sample_img, moving=annotation, transformlist=affine["invtransforms"],
        interpolator="genericLabel", whichtoinvert=[True]).numpy() > 0
    M = np.array(ants.read_transform(affine["fwdtransforms"][0]).parameters)[:9].reshape(3, 3)
    return M, labels, annotation, template.shape


def decompose(M):
    """(rotation angle in degrees, stretch eigenvalues) via polar decomposition.
    Reading scale off the raw matrix's diagonal or off det alone hides the
    rotation+anisotropic-squash solutions, which are the dangerous ones."""
    from scipy.linalg import polar
    R, S = polar(M)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return angle, np.linalg.eigvalsh(S)


def render(sample_img, results, out_path, n_slices=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    img = sample_img.numpy()
    ys = np.linspace(0.15, 0.85, n_slices) * img.shape[1]
    ys = ys.round().astype(int)
    rows = [("sample only", None)] + [(nm, r["labels"]) for nm, r in results.items()]
    fig, axes = plt.subplots(len(rows), len(ys), figsize=(3.5 * len(ys), 3.3 * len(rows)),
                             squeeze=False)
    for k, y in enumerate(ys):
        plane = img[:, y, :].T
        finite = plane[plane > 0]
        vmin, vmax = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
        for r, (nm, lab) in enumerate(rows):
            ax = axes[r][k]
            ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
            if lab is not None:
                m = lab[:, y, :].T
                ax.contour(m.astype(float), levels=[0.5], colors="r", linewidths=1.3)
                edges = (ndimage.maximum_filter(m, 3) != ndimage.minimum_filter(m, 3)) & m
                ax.imshow(np.ma.masked_where(~edges, edges), cmap="autumn", alpha=0.5)
            ax.set_title(f"y={y}  {nm}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--slicing", action="append", default=[],
                    help="extra atlas.slicing to probe, as JSON e.g. '[[320,536],null,[151,371]]'")
    ap.add_argument("--png", help="write a coronal overlay strip for every variant here")
    args = ap.parse_args()

    cfg = config_mod.load_config(args.config)
    atlas_cfg = cfg["atlas"]
    if atlas_cfg.get("source") != "custom":
        sys.exit("affine_probe only supports custom atlases (atlas.slicing has no meaning otherwise).")

    variants = [("config", atlas_cfg.get("slicing"))]
    for raw in args.slicing:
        variants.append((raw, json.loads(raw)))

    print("preparing sample (raw -> crop -> isotropic -> preprocess) ...", flush=True)
    sample_img, brain_arr, brain_img = prepare_sample(cfg)
    print(f"  sample grid {sample_img.shape}, brain {brain_arr.sum() * 8e-6:.1f} mm3"
          f" at {cfg['registration']['fine_target_um']}um\n", flush=True)

    results = {}
    hdr = (f"{'atlas.slicing':<34} {'grid':>16} {'det':>7} {'rot':>7} "
           f"{'stretch':>20} {'unlab':>7} {'outside':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, slicing in variants:
        t0 = time.perf_counter()
        M, labels, annotation, grid = probe(sample_img, brain_arr, brain_img, atlas_cfg, slicing)
        angle, stretch = decompose(M)
        unlab = 100 * (brain_arr & ~labels).sum() / brain_arr.sum()
        outside = 100 * (labels & ~brain_arr).sum() / max(labels.sum(), 1)
        label = json.dumps(slicing) if name == "config" else name
        results[label] = {"labels": labels, "det": np.linalg.det(M),
                          "rot": angle, "stretch": stretch, "unlab": unlab}
        print(f"{label[:34]:<34} {str(grid):>16} {np.linalg.det(M):>7.3f} {angle:>6.2f}d "
              f"{str(np.round(stretch, 3)):>20} {unlab:>6.1f}% {outside:>7.1f}%"
              f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    print()
    noop = [k for k, v in results.items() if v["rot"] < 0.5 and np.allclose(v["stretch"], 1, atol=0.03)]
    if noop:
        print("NOTE: the Affine stage was a no-op (identity rotation AND identity stretch) for:")
        for k in noop:
            print(f"  {k}")
        print("  The result there is whatever the Translation pre-alignment gave. That can still")
        print("  be the correct answer -- but it is not evidence the Affine found it, and a")
        print("  sample that genuinely needed rotating would get the same no-op.")
        print()
    best = min(results, key=lambda k: results[k]["unlab"])
    print(f"lowest unlabelled fraction: {best}  ({results[best]['unlab']:.1f}%)")
    print("Coverage alone does not prove correct correspondence -- an oversized atlas also")
    print("covers everything. Use --png and look before committing to a 3-hour run.")

    if args.png:
        render(sample_img, results, args.png)
        print(f"\nwrote {args.png}")


if __name__ == "__main__":
    main()
