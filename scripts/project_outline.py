"""Non-interactive helper: warp a hand-painted sample-space outline into
atlas space using an existing registration's forward transforms, giving a
starting "best guess" for ../GT_tool_for_registration/paint_mask.py's `guide` kind (ROLE="atlas") --
rather than asking a human to locate the corresponding un-deformed region
in the atlas purely from anatomical memory. The guess only needs to be
"roughly right" -- the human refines it afterward.

Usage (needs antspyx -- run in the antsreg env, this is headless/no display):
    conda activate antsreg
    python scripts/project_outline.py \\
        <sample_outline.nii.gz> <atlas_template_path> <resolution_um> \\
        <transforms_prefix> <output_guess.nii.gz>

<atlas_template_path>: the same atlas template file used for the existing
registration (e.g. the DeMBA reference .tif, or an Allen CCF .nii.gz) --
must be the exact same file, so the physical grid matches what the
transforms were computed in.
<resolution_um>: only used if atlas_template_path is a raw TIFF with no
embedded spacing (e.g. a ClearMap-style custom atlas) -- must match the
`atlas.resolution_um` from the config that produced this registration.
Ignored (but still required as a positional arg) for NIfTI/NRRD atlases,
which carry their own spacing.
<transforms_prefix>: the outprefix used for that registration, e.g.
"output_dir/transforms/tsc12t_" -- this looks for "<prefix>1Warp.nii.gz"
and "<prefix>0GenericAffine.mat" (ANTs' standard fwdtransforms naming for
a SyN-family registration).
"""
import sys
from pathlib import Path

import ants

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import io_utils, transforms  # noqa: E402


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    outline_path, atlas_template_path, resolution_um, transforms_prefix, out_path = sys.argv[1:6]

    sample_outline = ants.image_read(outline_path)

    if atlas_template_path.lower().endswith((".tif", ".tiff")):
        atlas_template = io_utils.load_tiff_stack_as_ants(atlas_template_path, (float(resolution_um),) * 3)
    else:
        atlas_template = ants.image_read(atlas_template_path)

    fwd_warp = f"{transforms_prefix}1Warp.nii.gz"
    fwd_affine = f"{transforms_prefix}0GenericAffine.mat"
    if not Path(fwd_warp).exists() or not Path(fwd_affine).exists():
        print(f"Expected transform files not found: {fwd_warp}, {fwd_affine}")
        sys.exit(1)

    reg_like = {"atlas_template": atlas_template, "fwdtransforms": [fwd_warp, fwd_affine]}
    guess = transforms.warp_sample_to_atlas(sample_outline, reg_like, interpolator="genericLabel")
    ants.image_write(guess, out_path)
    print(f"Wrote atlas-space guess -> {out_path}")


if __name__ == "__main__":
    main()
