"""Relabel an existing labels_in_sample.nii.gz (voxel ids in Allen adult CCFv3 id
space, e.g. from a DeMBA-atlas registration) into Kim Lab DevCCF ids, using the
published voxel-overlap crosswalk from the DevCCF paper's Supplementary Data 3
(41467_2024_53254_MOESM4_ESM.xlsx, sheet "SupplementaryData3" -- "DevCCF vs CCFv3
Voxel Mapping", computed at P56 by warping CCFv3 into DevCCF-P56 space and counting
per-region-pair voxel overlap).

This does NOT re-register anything -- it's a same-geometry relabel of ids already in
the volume, translating each CCFv3 structure to whichever DevCCF structure captures
the largest share of its voxels (majority vote; the crosswalk is inherently
many-to-many since the two atlases draw different boundaries, see the dominance_pct
column in the output CSV). The volume's actual boundaries stay exactly the ones the
original registration produced -- for genuinely different, developmentally-accurate
boundaries, register against DevCCF's own P04/P14 annotation instead (see
configs/atlas_presets.example.yaml's devccf_p04 preset).

Usage:
    python scripts/relabel_labels_to_devccf.py \\
        <labels_in_sample.nii.gz> <crosswalk.xlsx> <output.nii.gz> \\
        [--ccfv3-ontology-json CCF_v3_ontology.json] \\
        [--devccf-ontology-json DevCCFv1_ontology.json]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils  # noqa: E402


def build_crosswalk(xlsx_path):
    """Majority-vote {ccfv3_id: devccf_id} lookup from Supplementary Data 3, plus a
    per-ccfv3_id dominance fraction (share of that CCFv3 structure's overlap voxels
    captured by the chosen DevCCF id -- many structures split across several DevCCF
    labels, this is how confident any single pick is). Returns (crosswalk, dominance).
    """
    df = pd.read_excel(xlsx_path, sheet_name="SupplementaryData3", header=3)
    df["CCFv3 Label ID"] = df["CCFv3 Label ID"].astype(int)
    df["DevCCF Label ID"] = df["DevCCF Label ID"].astype(int)

    crosswalk = {}
    dominance = {}
    for ccfv3_id, group in df.groupby("CCFv3 Label ID"):
        best = group.loc[group["Overlapping Voxels"].idxmax()]
        crosswalk[int(ccfv3_id)] = int(best["DevCCF Label ID"])
        total = group["Overlapping Voxels"].sum()
        dominance[int(ccfv3_id)] = float(best["Overlapping Voxels"]) / total if total else 0.0
    return crosswalk, dominance


def relabel(labels_arr, crosswalk):
    """Apply {ccfv3_id: devccf_id} to labels_arr (any shape/dtype) via
    np.unique(..., return_inverse=True) rather than a np.arange(max_id+1)-sized LUT --
    this volume's ids can run past 6*10^8 (some are genuine large Allen ids, some are
    float32-rounding noise from DeMBA's own elastix warp -- see its data descriptor),
    so a dense LUT array would be multi-GB for a few hundred distinct values.

    ids not in crosswalk (including background id 0) map to 0. Returns
    (relabeled_arr, present_ids, voxel_counts, unmapped_ids) -- voxel_counts is
    {id: count} for every present id (one full-array pass, reused for the CSV sidecar
    instead of re-scanning); unmapped_ids is present_ids minus background minus
    whatever crosswalk covered, for the caller to diagnose.
    """
    present_ids, inverse, counts = np.unique(labels_arr, return_inverse=True, return_counts=True)
    lut = np.array([crosswalk.get(int(i), 0) for i in present_ids], dtype=np.int64)
    relabeled = lut[inverse].reshape(labels_arr.shape)
    voxel_counts = {int(i): int(c) for i, c in zip(present_ids, counts)}
    unmapped_ids = [int(i) for i in present_ids if i != 0 and int(i) not in crosswalk]
    return relabeled, present_ids, voxel_counts, unmapped_ids


def diagnose_unmapped(unmapped_ids, ccfv3_structures):
    """Split unmapped ids into (a) not a real CCFv3 id at all -- float32-rounding
    noise, safe to ignore -- vs (b) a real CCFv3 structure the crosswalk just doesn't
    cover (e.g. it only spans 669 of the full ontology's 1327 structures). Returns
    (noise_ids, gap_ids_with_names)."""
    noise_ids = [i for i in unmapped_ids if i not in ccfv3_structures]
    gap_ids = [(i, ccfv3_structures[i]["name"]) for i in unmapped_ids if i in ccfv3_structures]
    return noise_ids, gap_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_in_sample_path")
    parser.add_argument("crosswalk_xlsx_path", help="DevCCF paper's Supplementary Data xlsx (Supplementary Data 3 sheet)")
    parser.add_argument("output_path")
    parser.add_argument("--ccfv3-ontology-json", default=None,
                         help="CCF_v3_ontology.json, to split unmapped ids into noise vs. real coverage gaps")
    parser.add_argument("--devccf-ontology-json", default=None,
                         help="DevCCFv1_ontology.json (scripts/convert_devccf_ontology.py output), "
                              "to print DevCCF names in the sidecar CSV")
    args = parser.parse_args()

    print(f"Reading {args.labels_in_sample_path} ...")
    labels_sitk = sitk.ReadImage(args.labels_in_sample_path)
    labels_arr = np.rint(sitk.GetArrayFromImage(labels_sitk)).astype(np.int64)

    print(f"Building crosswalk from {args.crosswalk_xlsx_path} ...")
    crosswalk, dominance = build_crosswalk(args.crosswalk_xlsx_path)
    print(f"  {len(crosswalk)} CCFv3 ids covered")

    relabeled_arr, present_ids, voxel_counts, unmapped_ids = relabel(labels_arr, crosswalk)
    n_present = int((present_ids != 0).sum())
    n_mapped = n_present - len(unmapped_ids)
    print(f"Sample has {n_present} distinct nonzero ids; {n_mapped} matched the crosswalk, "
          f"{len(unmapped_ids)} did not")

    if args.ccfv3_ontology_json:
        ccfv3_structures = atlas_utils.load_ccf_ontology_json(args.ccfv3_ontology_json)
        noise_ids, gap_ids = diagnose_unmapped(unmapped_ids, ccfv3_structures)
        print(f"  of the {len(unmapped_ids)} unmapped ids: {len(noise_ids)} aren't valid CCFv3 ids at all "
              f"(likely float32-rounding noise), {len(gap_ids)} are real structures the crosswalk doesn't cover:")
        for i, name in sorted(gap_ids, key=lambda x: x[1]):
            print(f"    {i}: {name}")
    else:
        ccfv3_structures = {}
        print("  (pass --ccfv3-ontology-json to split unmapped ids into noise vs. real coverage gaps)")

    devccf_structures = atlas_utils.load_ccf_ontology_json(args.devccf_ontology_json) if args.devccf_ontology_json else {}

    out_sitk = sitk.GetImageFromArray(relabeled_arr.astype(np.uint32))
    out_sitk.CopyInformation(labels_sitk)
    sitk.WriteImage(out_sitk, args.output_path)
    print(f"Wrote {args.output_path}")

    out_path = Path(args.output_path)
    out_stem = out_path.name[: -len(".nii.gz")] if out_path.name.endswith(".nii.gz") else out_path.stem
    sidecar_path = out_path.with_name(out_stem + "_crosswalk_applied.csv")
    rows = []
    for ccfv3_id in sorted(i for i in present_ids if i != 0):
        ccfv3_id = int(ccfv3_id)
        devccf_id = crosswalk.get(ccfv3_id)
        rows.append({
            "ccfv3_id": ccfv3_id,
            "ccfv3_name": ccfv3_structures.get(ccfv3_id, {}).get("name", ""),
            "devccf_id": devccf_id if devccf_id is not None else "",
            "devccf_name": devccf_structures.get(devccf_id, {}).get("name", "") if devccf_id is not None else "",
            "dominance_pct": round(100 * dominance[ccfv3_id], 1) if ccfv3_id in dominance else "",
            "voxel_count_in_this_sample": int(voxel_counts.get(ccfv3_id, 0)),
        })
    pd.DataFrame(rows).to_csv(sidecar_path, index=False)
    print(f"Wrote {sidecar_path}")


if __name__ == "__main__":
    main()
