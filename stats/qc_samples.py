"""Pre-flight QC: does anything about the DATA, rather than the biology,
already separate the two groups?

Run this before reading any group comparison. With n=3 vs 3, a nuisance
variable that happens to rank the six samples the same way the grouping does
is indistinguishable from the effect being tested. One is already known to
exist in this dataset: the out-of-atlas ("background") rate ranks the controls
above the experimentals with no overlap at all (see stats/README.md).

For every per-sample quantity it computes, this reports whether the two groups
separate completely (no overlap between their ranges), and whether the same
quantity separates the optional `batch` labels declared in the config. A "yes"
is not proof of a confound, but it means an absolute-count comparison on that
axis cannot be interpreted until it is explained.

Note the `GFP` vs `3_GFP` class-name split is NOT a batch and should not be
declared as one -- it is a phantom marker from a tile-folder naming slip, with
identical detection parameters and unaffected counts (stats/README.md).

Usage:
    conda activate antsreg
    python -m stats.qc_samples --config stats/configs/tsc_marker.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.group_stats import load_config  # noqa: E402
from stats.ontology import Ontology  # noqa: E402
from stats.region_volumes import sample_region_volumes  # noqa: E402


def separates(values, labels):
    """True if the label groups' value ranges do not overlap at all -- i.e.
    every member of one group is above every member of the other. With three
    samples a side this is the only 'significance' a nuisance variable can
    show, and it is what makes it dangerous."""
    groups = {}
    for v, lab in zip(values, labels):
        if not np.isnan(v):
            groups.setdefault(lab, []).append(v)
    if len(groups) != 2 or any(len(v) == 0 for v in groups.values()):
        return None
    (a,), (b,) = [[v] for v in groups.values()]
    return bool(max(a) < min(b) or max(b) < min(a))


def _sep_flag(df, col, label_col):
    if label_col not in df.columns or df[label_col].nunique() != 2:
        return None
    return separates(df[col].to_numpy(dtype=float), df[label_col].tolist())


def per_sample_qc(cfg):
    ontology = Ontology.from_json(cfg["ontology_json"])
    groups = cfg["groups"]
    group_of = {}
    for key in ("a", "b"):
        for s in groups[key]["samples"]:
            group_of[s] = groups[key].get("name", key.upper())
    samples = list(group_of)

    rows, comp_rows = [], []
    for sample in samples:
        sdir = cfg["samples"][sample]["dir"]
        batch = cfg["samples"][sample].get("batch", "")
        totals = {"total": 0, "valid": 0, "background": 0, "unknown": 0}
        by_marker, by_soma = {}, {"neuron": 0, "glia": 0}

        for cls_dir in cell_tables.list_sample_classes(sdir):
            _, _, n_valid, n_total, n_unknown = cell_tables.class_counts(
                cell_tables.class_csv_path(sdir, cls_dir), ontology)
            totals["total"] += n_total
            totals["valid"] += n_valid
            totals["unknown"] += n_unknown
            totals["background"] += n_total - n_valid - n_unknown
            marker = cell_tables.marker_signature(cls_dir)
            by_marker[marker] = by_marker.get(marker, 0) + n_valid
            st = cell_tables.soma_type(cls_dir)
            if st:
                by_soma[st] += n_valid

        vols = sample_region_volumes(sdir, ontology, verbose=False)
        root = ontology.root_order
        n_soma = by_soma["neuron"] + by_soma["glia"]
        rows.append({
            "sample": sample, "group": group_of[sample], "batch": batch,
            "n_cells_total": totals["total"],
            "n_assigned": totals["valid"],
            "pct_background": totals["background"] / totals["total"] * 100 if totals["total"] else np.nan,
            "pct_unknown_id": totals["unknown"] / totals["total"] * 100 if totals["total"] else np.nan,
            "pct_yolo_glia": by_soma["glia"] / n_soma * 100 if n_soma else np.nan,
            "brain_volume_mm3": vols["volume_mm3"].iloc[root],
            "brain_coverage": vols["coverage"].iloc[root],
            "n_classes_on_disk": len(cell_tables.list_sample_classes(sdir)),
        })
        total_valid = sum(by_marker.values())
        for marker, n in sorted(by_marker.items()):
            comp_rows.append({
                "sample": sample, "group": group_of[sample], "batch": batch,
                "marker": marker, "n": n,
                "pct_of_sample": n / total_valid * 100 if total_valid else np.nan})

    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def separation_report(qc, comp):
    """One row per quantity: does it separate the groups, and does it separate
    the batches?"""
    out = []
    for col in ("n_cells_total", "n_assigned", "pct_background", "pct_yolo_glia",
                "brain_volume_mm3", "brain_coverage"):
        out.append({"quantity": col,
                    "separates_group": _sep_flag(qc, col, "group"),
                    "separates_batch": _sep_flag(qc, col, "batch")})
    for marker, g in comp.groupby("marker"):
        out.append({"quantity": f"composition:{marker}",
                    "separates_group": _sep_flag(g, "pct_of_sample", "group"),
                    "separates_batch": _sep_flag(g, "pct_of_sample", "batch")})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    qc, comp = per_sample_qc(cfg)
    rep = separation_report(qc, comp)

    pd.set_option("display.width", 200)
    print("\n=== per-sample QC ===")
    print(qc.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== marker composition (% of assigned cells in that sample) ===")
    print(comp.pivot(index="marker", columns="sample", values="pct_of_sample")
          .reindex(columns=qc["sample"]).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n=== complete separation check ===")
    print("(True = that group's/batch's samples do not overlap the other's at all;")
    print(" at n=3 vs 3 this is indistinguishable from a real effect)")
    print(rep.to_string(index=False))

    flagged = rep[(rep["separates_group"] == True)]  # noqa: E712
    if len(flagged):
        print(f"\n[!] {len(flagged)} quantity(ies) separate the groups completely:")
        for q in flagged["quantity"]:
            print(f"      {q}")
        print("    Explain these before interpreting absolute Count or Density differences.")

    out_dir = (cfg.get("output") or {}).get("dir", "./stats_output")
    os.makedirs(out_dir, exist_ok=True)
    qc.to_csv(os.path.join(out_dir, "qc_per_sample.csv"), index=False)
    comp.to_csv(os.path.join(out_dir, "qc_marker_composition.csv"), index=False)
    rep.to_csv(os.path.join(out_dir, "qc_separation.csv"), index=False)
    print(f"\nWrote QC tables to {out_dir}")


if __name__ == "__main__":
    main()
