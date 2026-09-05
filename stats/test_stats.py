"""Self-contained smoke tests for stats/. Builds a tiny synthetic ontology,
synthetic cell_registration.csv files and synthetic NIfTI volumes in a temp
directory, then runs the whole group comparison end to end.

No real data, no ANTs, no ClearMap:
    conda activate antsreg
    python stats/test_stats.py
"""
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables, group_stats  # noqa: E402
from stats.ontology import Ontology  # noqa: E402
from stats.region_volumes import (build_per_sample_volumes, rollup_volumes,
                                  sample_region_volumes)  # noqa: E402

# root(1) > A(10) > A1(100), A2(200);  root > B(20) > B1(300)
ONTOLOGY = {"msg": [{"id": 1, "name": "root", "acronym": "root", "children": [
    {"id": 10, "name": "Area A", "acronym": "A", "children": [
        {"id": 100, "name": "Area A1", "acronym": "A1", "children": []},
        {"id": 200, "name": "Area A2", "acronym": "A2", "children": []}]},
    {"id": 20, "name": "Area B", "acronym": "B", "children": [
        {"id": 300, "name": "Area B1", "acronym": "B1", "children": []}]}]}]}

CLASSES = ["neuron_GFP", "neuron_GFP_Sox9", "glia_GFP", "glia_GFP_Sox9"]


def _write_ontology(root):
    path = os.path.join(root, "ontology.json")
    with open(path, "w") as f:
        json.dump(ONTOLOGY, f)
    return path


def _write_cells(sample_dir, class_name, region_ids, names):
    d = os.path.join(sample_dir, "cell_registration", class_name)
    os.makedirs(d, exist_ok=True)
    rows = [[1.0, 2.0, 3.0, 1, 2, 3, 1, 2, 3, rid, nm, "sl", "ti", 0.9]
            for rid, nm in zip(region_ids, names)]
    pd.DataFrame(rows).to_csv(os.path.join(d, "cell_registration.csv"),
                              header=False, index=False)


def _write_volumes(sample_dir, name, counts, mask_fraction=1.0):
    """A 1-D label volume: `counts[id]` voxels carrying each id, laid out in
    order, plus a brain mask covering the first `mask_fraction` of it."""
    labels = np.concatenate([np.full(n, rid, dtype=np.uint32)
                             for rid, n in counts.items()])
    arr = labels.reshape(-1, 1, 1)
    aff = np.diag([20.0, 20.0, 20.0, 1.0])
    nib.save(nib.Nifti1Image(arr, aff),
             os.path.join(sample_dir, f"{name}_labels_in_sample.nii.gz"))
    mask = np.zeros_like(arr, dtype=np.float32)
    mask[: int(round(len(labels) * mask_fraction))] = 1.0
    nib.save(nib.Nifti1Image(mask, aff),
             os.path.join(sample_dir, f"{name}_brain_mask.nii.gz"))


def test_ontology():
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
    assert o.n == 6, o.n
    assert o.ids[o.root_order] == 1
    assert list(o.levels[o.order_of_ids([1, 10, 100, 20, 300])]) == [0, 1, 2, 1, 2]
    # parent precedes child, which is what makes the reverse-sweep rollup valid
    assert all(o.parent_order[i] < i for i in range(o.n) if o.parent_order[i] >= 0)

    direct = np.zeros(o.n)
    direct[o.order_of_ids([100])[0]] = 5
    direct[o.order_of_ids([200])[0]] = 3
    direct[o.order_of_ids([300])[0]] = 2
    rolled = o.rollup(direct)
    assert rolled[o.order_of_ids([10])[0]] == 8, rolled
    assert rolled[o.order_of_ids([20])[0]] == 2
    assert rolled[o.root_order] == 10
    assert list(o.order_of_ids([999])) == [-1]
    assert len(o.descendants_of([10])) == 3
    print("  ok: ontology (order, levels, rollup, unknown ids, subtree)")


def test_marker_recoding():
    assert cell_tables.marker_signature("neuron_3_GFP_RFP_Sox9") == "GFP_RFP_Sox9"
    assert cell_tables.marker_signature("glia_GFP_RFP_Sox9") == "GFP_RFP_Sox9"
    assert cell_tables.marker_signature("glia_RFP") == "RFP"
    assert cell_tables.soma_type("glia_3_GFP") == "glia"
    # the '3' naming variant must not split one class into two
    assert (cell_tables.normalize_class_key("glia_3_GFP")
            == cell_tables.normalize_class_key("glia_GFP"))
    print("  ok: marker recoding and class-name normalisation")


def test_real_naming_conventions_map_1to1():
    """The two naming conventions this dataset once carried must collapse onto
    the same six marker signatures. The '3' in s11/s12q/s12t came from a
    'GFP_3' tile folder name that brain_detector's _merge_class split into a
    marker token; it rode on exactly the GFP-positive classes and nowhere
    else, so dropping numeric tokens is an exact 1:1 mapping.

    Those files were renamed on disk on 2026-09-04, so the fixture below is
    historical -- the rule it covers is kept because the retired ClearMap
    outputs still carry the old spelling, and because anything reprocessed
    through the same _merge_class path can reintroduce it."""
    plain = ["neuron_GFP", "neuron_GFP_RFP", "neuron_GFP_RFP_Sox9", "neuron_GFP_Sox9",
             "neuron_RFP", "neuron_RFP_Sox9",
             "glia_GFP", "glia_GFP_RFP", "glia_GFP_RFP_Sox9", "glia_GFP_Sox9",
             "glia_RFP", "glia_RFP_Sox9"]
    phantom = ["neuron_3_GFP", "neuron_3_GFP_RFP", "neuron_3_GFP_RFP_Sox9",
               "neuron_3_GFP_Sox9", "neuron_RFP", "neuron_RFP_Sox9",
               "glia_3_GFP", "glia_3_GFP_RFP", "glia_3_GFP_RFP_Sox9",
               "glia_3_GFP_Sox9", "glia_RFP", "glia_RFP_Sox9"]
    assert len(plain) == len(phantom) == 12
    sig_plain = sorted({cell_tables.marker_signature(c) for c in plain})
    sig_phantom = sorted({cell_tables.marker_signature(c) for c in phantom})
    assert sig_plain == sig_phantom == [
        "GFP", "GFP_RFP", "GFP_RFP_Sox9", "GFP_Sox9", "RFP", "RFP_Sox9"], sig_phantom
    # ...and pairwise, not just as sets
    for a, b in zip(plain, phantom):
        assert (cell_tables.marker_signature(a) == cell_tables.marker_signature(b)), (a, b)
    print("  ok: GFP / 3_GFP naming conventions map 1:1 onto six signatures")


def test_class_resolution_guard():
    """Two distinct folders collapsing into one class in one sample must be
    reported -- that is the case where dropping numeric tokens would double
    count."""
    with tempfile.TemporaryDirectory() as root:
        good = os.path.join(root, "good")
        for c in ("neuron_3_GFP", "glia_3_GFP"):
            _write_cells(good, c, [100], ["Area A1"])
        other = os.path.join(root, "other")
        for c in ("neuron_GFP", "glia_GFP"):
            _write_cells(other, c, [100], ["Area A1"])
        assert cell_tables.check_class_resolution(
            {"good": good, "other": other}, ["GFP"], "marker") == []

        # same sample holding both spellings for the same soma type
        bad = os.path.join(root, "bad")
        for c in ("neuron_GFP", "neuron_3_GFP"):
            _write_cells(bad, c, [100], ["Area A1"])
        problems = cell_tables.check_class_resolution({"bad": bad}, ["GFP"], "marker")
        assert len(problems) == 1 and "2 folders" in problems[0], problems

        # one sample contributing neuron+glia, another only neuron
        half = os.path.join(root, "half")
        _write_cells(half, "neuron_GFP", [100], ["Area A1"])
        problems = cell_tables.check_class_resolution(
            {"good": good, "half": half}, ["GFP"], "marker")
        assert any("different number of folders" in p for p in problems), problems
    print("  ok: class-resolution guard catches collapsed and asymmetric classes")


def test_background_filter():
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        _write_cells(sdir, "neuron_GFP",
                     [100, 100, 200, 0, 0, 999],
                     ["Area A1", "Area A1", "Area A2", "background", "no label", "Area X"])
        roll, direct, n_valid, n_total, n_unknown, n_excl = cell_tables.class_counts(
            cell_tables.class_csv_path(sdir, "neuron_GFP"), o)
    assert n_total == 6 and n_valid == 3 and n_unknown == 1, (n_total, n_valid, n_unknown)
    assert n_excl == 0
    assert roll[o.order_of_ids([10])[0]] == 3
    assert direct[o.order_of_ids([100])[0]] == 2
    print("  ok: background / no-label / unknown-id filtering")


def test_exclusion_before_rollup():
    """Excluding a subtree must remove it from every ANCESTOR's rollup and
    from n_valid -- i.e. from the Percentage denominator -- not merely from
    the list of regions tested."""
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        _write_cells(sdir, "neuron_GFP", [100, 100, 200, 300, 300, 300],
                     ["Area A1", "Area A1", "Area A2", "Area B1", "Area B1", "Area B1"])
        csv = cell_tables.class_csv_path(sdir, "neuron_GFP")

        roll, _, n_valid, _, _, n_excl = cell_tables.class_counts(csv, o)
        assert n_valid == 6 and n_excl == 0
        assert roll[o.root_order] == 6

        # exclude the whole B subtree (id 20 -> B and B1)
        mask = o.subtree_mask(o.order_of_ids([20]))
        roll2, _, n_valid2, _, _, n_excl2 = cell_tables.class_counts(csv, o, exclude_mask=mask)
        assert n_excl2 == 3 and n_valid2 == 3, (n_excl2, n_valid2)
        assert roll2[o.order_of_ids([300])[0]] == 0
        assert roll2[o.order_of_ids([20])[0]] == 0
        # the ROOT must have lost them too -- this is the part a post-rollup
        # filter would get wrong
        assert roll2[o.root_order] == 3, roll2[o.root_order]
        assert roll2[o.order_of_ids([10])[0]] == 3
    print("  ok: exclusion applied before rollup reaches every ancestor")


def test_relative_volume_excludes():
    """relative_pct must be a share of the retained regions, so excluding a
    structure changes every other region's relative volume."""
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        os.makedirs(sdir)
        _write_volumes(sdir, "s1", {100: 10, 200: 10, 300: 20})
        direct = sample_region_volumes(sdir, o, verbose=False)
        direct.insert(0, "sample", "s1")

        full = rollup_volumes(direct, o)
        a_full = full.set_index("order")["relative_pct"][o.order_of_ids([10])[0]]
        assert np.isclose(a_full, 20 / 40 * 100), a_full

        mask = o.subtree_mask(o.order_of_ids([20]))
        cut = rollup_volumes(direct, o, exclude_mask=mask)
        c = cut.set_index("order")
        assert np.isclose(c["volume_mm3"][o.order_of_ids([300])[0]], 0.0)
        assert np.isclose(c["relative_pct"][o.order_of_ids([10])[0]], 100.0)
        assert np.isclose(c["volume_mm3"][o.root_order],
                          c["volume_mm3"][o.order_of_ids([10])[0]])
    print("  ok: relative volume is a share of the retained regions only")


def test_volumes_and_coverage():
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        os.makedirs(sdir)
        _write_volumes(sdir, "s1", {100: 10, 200: 10, 300: 20}, mask_fraction=0.5)
        v = rollup_volumes(sample_region_volumes(sdir, o, verbose=False).assign(sample="s1"), o)
        vox = 20.0 ** 3 * 1e-9
        assert np.isclose(v["volume_mm3"].iloc[o.order_of_ids([10])[0]], 20 * vox)
        assert np.isclose(v["volume_mm3"].iloc[o.root_order], 40 * vox)
        # mask covers the first 20 of 40 voxels: all of A1+A2, none of B1
        assert np.isclose(v["coverage"].iloc[o.order_of_ids([10])[0]], 1.0)
        assert np.isclose(v["coverage"].iloc[o.order_of_ids([300])[0]], 0.0)
        assert np.isclose(v["coverage"].iloc[o.root_order], 0.5)
    print("  ok: per-sample volumes and tissue coverage")


def test_grid_offset():
    """A brain mask cropped in physical space is offset from the label grid by
    a non-integer number of voxels (real case: s11, 0.4 voxel)."""
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        os.makedirs(sdir)
        _write_volumes(sdir, "s1", {100: 10, 200: 10, 300: 20})
        aff = np.diag([20.0, 20.0, 20.0, 1.0])
        aff[0, 3] = 104.0  # 5.2 voxels -- rounds to 5, residual 0.2
        mask = np.ones((30, 1, 1), dtype=np.float32)
        nib.save(nib.Nifti1Image(mask, aff),
                 os.path.join(sdir, "s1_brain_mask.nii.gz"))
        v = rollup_volumes(sample_region_volumes(sdir, o, verbose=False).assign(sample="s1"), o)
        # labels: A1 = voxels 0-9, A2 = 10-19, B1 = 20-39; the mask lands on
        # 5-34, so A1 is half covered, A2 fully, B1 three quarters
        assert np.isclose(v["coverage"].iloc[o.order_of_ids([100])[0]], 0.5)
        assert np.isclose(v["coverage"].iloc[o.order_of_ids([200])[0]], 1.0)
        assert np.isclose(v["coverage"].iloc[o.order_of_ids([300])[0]], 0.75)
    print("  ok: non-integer grid offset between mask and labels")


def _build_dataset(root, effect=3.0):
    ont = _write_ontology(root)
    samples = {}
    rng = np.random.default_rng(0)
    for i, s in enumerate(["c1", "c2", "c3", "e1", "e2", "e3"]):
        sdir = os.path.join(root, s)
        os.makedirs(sdir, exist_ok=True)
        _write_volumes(sdir, s, {100: 10, 200: 10, 300: 20})
        is_exp = s.startswith("e")
        for cls in CLASSES:
            # A1 carries the planted effect; A2 and B1 are noise only
            n_a1 = int(round((20 if not is_exp else 20 * effect) + rng.normal(0, 1)))
            rows_id = [100] * n_a1 + [200] * (10 + i) + [300] * 15
            rows_nm = (["Area A1"] * n_a1 + ["Area A2"] * (10 + i) + ["Area B1"] * 15)
            _write_cells(sdir, cls, rows_id, rows_nm)
        samples[s] = {"dir": sdir, "batch": "b1" if i % 2 == 0 else "b2"}
    return ont, samples


def test_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        ont, samples = _build_dataset(root)
        cfg = {
            "ontology_json": ont,
            "samples": samples,
            "groups": {"a": {"name": "Control", "samples": ["c1", "c2", "c3"]},
                       "b": {"name": "Exp", "samples": ["e1", "e2", "e3"]}},
            "classify_by": "marker",
            "classes": None,
            "combined_categories": {"all_cells": [{"sign": "+", "class": "GFP"},
                                                  {"sign": "+", "class": "GFP_Sox9"}]},
            "metrics": ["Count", "Density", "RegionProportion", "Percentage", "Volume"],
            "region_filter": {"min_coverage": 0.0, "levels": [1, 2]},
            "output": {"dir": os.path.join(root, "out")},
        }
        cfg_path = os.path.join(root, "cfg.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        r = group_stats.run_all(group_stats.load_config(cfg_path))
        df = r["result"]

        # marker mode must collapse 4 folders into 2 classes...
        assert sorted(r["classes"]) == ["GFP", "GFP_Sox9"], r["classes"]
        # ...and sum, not average: 2 folders x the same synthetic counts
        one = r["counts_df"]
        a1 = r["ontology"].order_of_ids([100])[0]
        got = one[(one["sample"] == "c1") & (one["class_name"] == "GFP")
                  & (one["order"] == a1)]["count"].iloc[0]
        assert got > 30, f"neuron+glia should have been summed, got {got}"

        # the planted effect must come out on top for A1, and only for A1
        cnt = df[(df["class_name"] == "GFP") & (df["metric"] == "Count")]
        top = cnt.sort_values("p_value").iloc[0]
        assert top["name"] == "Area A1", top[["name", "p_value"]].to_dict()
        assert top["log2fc"] > 1.0, top["log2fc"]
        assert not np.isnan(top["hedges_g"]) and top["hedges_g"] > 2

        b1 = cnt[cnt["name"] == "Area B1"]
        assert len(b1) == 1 and b1["p_value"].iloc[0] > 0.05, "planted no effect in B1"

        # RegionProportion of the two mutually-exclusive classes must close
        prop = df[(df["metric"] == "RegionProportion") & (df["formula"] == "")]
        tot = prop.groupby(["order"])["mean_a"].sum()
        assert np.allclose(tot.to_numpy(), 100.0), tot.to_dict()

        # combined category present and equal to the sum of its parts
        allc = df[(df["class_name"] == "all_cells") & (df["metric"] == "Count")
                  & (df["name"] == "Area A1")]["mean_a"].iloc[0]
        parts = df[(df["class_name"].isin(["GFP", "GFP_Sox9"])) & (df["metric"] == "Count")
                   & (df["name"] == "Area A1")]["mean_a"].sum()
        assert np.isclose(allc, parts), (allc, parts)

        # only the requested levels were tested
        assert set(df["level"]) <= {1, 2}, set(df["level"])

        group_stats.write_outputs(r)
        group_stats.write_per_sample_tables(r)
        group_stats.write_per_sample_trees(r)
        out = cfg["output"]["dir"]
        for f in ["region_stats.csv", "region_stats_by_level.xlsx",
                  "per_sample/c1_region_summary.xlsx", "per_sample/c1_region_tree.xlsx"]:
            assert os.path.exists(os.path.join(out, f)), f
    print("  ok: end-to-end run, marker collapse, planted effect, outputs")


def test_coverage_masking():
    """A region below min_coverage in one sample must drop that sample only,
    not the whole region."""
    with tempfile.TemporaryDirectory() as root:
        ont, samples = _build_dataset(root)
        # cripple B1's coverage in one control sample
        _write_volumes(os.path.join(root, "c1"), "c1",
                       {100: 10, 200: 10, 300: 20}, mask_fraction=0.5)
        cfg = {
            "ontology_json": ont, "samples": samples,
            "groups": {"a": {"name": "Control", "samples": ["c1", "c2", "c3"]},
                       "b": {"name": "Exp", "samples": ["e1", "e2", "e3"]}},
            "classify_by": "marker", "metrics": ["Density"],
            "region_filter": {"min_coverage": 0.8, "levels": [2]},
            "output": {"dir": os.path.join(root, "out2")},
        }
        cfg_path = os.path.join(root, "cfg2.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        r = group_stats.run_all(group_stats.load_config(cfg_path))
        df = r["result"]
        b1 = df[(df["name"] == "Area B1") & (df["class_name"] == "GFP")]
        assert len(b1) == 1, "region should still be tested on the remaining samples"
        assert b1["n_a"].iloc[0] == 2, b1["n_a"].iloc[0]
        assert b1["n_b"].iloc[0] == 3
        assert np.isnan(b1["c1"].iloc[0]), "the low-coverage sample must be NaN, not 0"
        a1 = df[(df["name"] == "Area A1") & (df["class_name"] == "GFP")]
        assert a1["n_a"].iloc[0] == 3, "unaffected region must keep all samples"
    print("  ok: coverage masking drops a sample, not a region")


def test_degenerate_ttest():
    """Two constant groups have no measurable difference. scipy returns
    t=inf, p=0.0 there, which would sail through FDR -- the real run produced
    three such rows ("1 cell in every control, 0 in every experimental")
    before this guard."""
    a = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    b = np.array([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    p = group_stats.welch_ttest(a, b)
    assert p[0] == 1.0, p[0]
    assert 0.0 < p[1] < 1.0, p[1]
    # one constant group is still testable against a varying one
    p2 = group_stats.welch_ttest(np.array([[1.0, 1.0, 1.0]]), np.array([[4.0, 5.0, 6.0]]))
    assert 0.0 < p2[0] < 1.0, p2[0]
    print("  ok: two constant groups give p=1, not p=0")


def test_min_total_count():
    with tempfile.TemporaryDirectory() as root:
        ont, samples = _build_dataset(root)
        base = {"ontology_json": ont, "samples": samples,
                "groups": {"a": {"name": "Control", "samples": ["c1", "c2", "c3"]},
                           "b": {"name": "Exp", "samples": ["e1", "e2", "e3"]}},
                "classify_by": "marker", "metrics": ["Count", "Density"]}

        def _run(min_total_count, out):
            cfg = dict(base)
            cfg["region_filter"] = {"min_coverage": 0.0, "levels": [1, 2],
                                    "min_total_count": min_total_count}
            cfg["output"] = {"dir": os.path.join(root, out)}
            path = os.path.join(root, f"{out}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            return group_stats.run_all(group_stats.load_config(path))["result"]

        loose = _run(0, "loose")
        # B1 holds 15 cells per sample per class -> 90 across the six samples
        strict = _run(200, "strict")
        assert "Area B1" in set(loose["name"])
        assert "Area B1" not in set(strict["name"]), "gate should drop the small region"
        assert "Area A1" in set(strict["name"]), "gate must not drop the big region"
        # the gate applies to every metric, not just Count
        assert set(strict["metric"]) == {"Count", "Density"}
    print("  ok: min_total_count gate drops sparse regions from every metric")


def test_bh_and_effect_size():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    got = group_stats.benjamini_hochberg(p)
    assert np.all(np.diff(got) >= -1e-12), "BH must be monotone"
    assert np.isclose(got[0], 0.008), got[0]
    assert np.all(got >= p - 1e-12)

    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[4.0, 5.0, 6.0]])
    g, lo, hi = group_stats.hedges_g(a, b)
    assert lo[0] < g[0] < hi[0]
    d = 3.0 / 1.0                      # (5-2)/pooled sd of 1
    assert np.isclose(g[0], d * (1 - 3 / (4 * 6 - 9))), g[0]
    # constant groups have no measurable effect, not an infinite one
    g2, _, _ = group_stats.hedges_g(np.ones((1, 3)), np.full((1, 3), 2.0))
    assert np.isnan(g2[0])
    print("  ok: BH monotonicity and Hedges' g small-sample correction")


def test_corrections():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    m = len(p)
    bh = group_stats.adjust_pvalues(p, "bh")
    hm = group_stats.adjust_pvalues(p, "holm")
    bf = group_stats.adjust_pvalues(p, "bonferroni")
    none = group_stats.adjust_pvalues(p, "none")

    assert np.allclose(none, p), "none must pass p through untouched"
    assert np.allclose(bf, np.clip(p * m, 0, 1))
    # Holm is uniformly at least as powerful as Bonferroni, and never more
    # powerful than BH -- so bh <= holm <= bonferroni everywhere.
    assert np.all(bh <= hm + 1e-12), (bh, hm)
    assert np.all(hm <= bf + 1e-12), (hm, bf)
    for adj in (bh, hm, bf):
        assert np.all(np.diff(adj) >= -1e-12), "adjusted p must stay monotone"
        assert np.all(adj >= p - 1e-12), "adjustment must never lower a p-value"
    assert np.isclose(hm[0], min(1.0, p[0] * m)), "Holm's first step is Bonferroni"
    print("  ok: bh / holm / bonferroni / none orderings and identities")


def test_gatekeeping():
    """A deep level may only be tested inside branches that were significant
    at the previous tested level. That restriction is what makes running the
    deep level uncorrected defensible."""
    with tempfile.TemporaryDirectory() as root:
        ont, samples = _build_dataset(root, effect=6.0)
        base = {"ontology_json": ont, "samples": samples,
                "groups": {"a": {"name": "Control", "samples": ["c1", "c2", "c3"]},
                           "b": {"name": "Exp", "samples": ["e1", "e2", "e3"]}},
                "classify_by": "marker", "classes": ["GFP"], "metrics": ["Count"],
                "region_filter": {"min_coverage": 0.0, "levels": [1, 2]}}

        def _run(gate, out):
            cfg = dict(base)
            cfg["stats"] = {"correction": "bh", "gatekeeping": gate,
                            "correction_by_level": {2: "none"}}
            cfg["output"] = {"dir": os.path.join(root, out)}
            path = os.path.join(root, f"{out}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f)
            return group_stats.run_all(group_stats.load_config(path))["result"]

        wide = _run(False, "nogate")
        narrow = _run(True, "gate")

        l1 = wide[wide["level"] == 1].set_index("acronym")
        assert set(l1.index) == {"A", "B"}, set(l1.index)
        assert l1.loc["A", "p_adj"] < 0.05, l1.loc["A", "p_adj"]
        assert l1.loc["B", "p_adj"] > 0.05, l1.loc["B", "p_adj"]

        assert set(wide[wide["level"] == 2]["acronym"]) == {"A1", "A2", "B1"}
        got = set(narrow[narrow["level"] == 2]["acronym"])
        assert got == {"A1", "A2"}, got

        deep = narrow[narrow["level"] == 2]
        assert deep["exploratory"].all() and (deep["correction"] == "none").all()
        assert np.allclose(deep["p_adj"], deep["p_value"])
        assert deep["gated"].all()
        assert not narrow[narrow["level"] == 1]["exploratory"].any()
    print("  ok: gatekeeping restricts deep levels to significant branches")


def test_welch_vs_student():
    """stats.test must actually change the test, not just be recorded."""
    # group b far more variable than a -- exactly where the pooled-variance
    # assumption bites
    a = np.array([[10.0, 10.1, 9.9]])
    b = np.array([[20.0, 5.0, 35.0]])
    pw = group_stats.two_sample_ttest(a, b, test="welch")[0]
    ps = group_stats.two_sample_ttest(a, b, test="student")[0]
    assert not np.isclose(pw, ps), (pw, ps)
    assert np.isclose(group_stats.welch_ttest(a, b)[0], pw), "alias must match welch"
    try:
        group_stats.two_sample_ttest(a, b, test="mann-whitney")
    except ValueError as e:
        assert "unknown test" in str(e)
    else:
        raise AssertionError("an unknown test name must be rejected, not silently ignored")
    print("  ok: welch vs student differ, unknown test name rejected")


def test_methods_summary():
    """methods.md must be generated from the settings in force, so that it
    cannot drift away from the numbers it sits next to."""
    with tempfile.TemporaryDirectory() as root:
        ont, samples = _build_dataset(root)
        cfg = {"ontology_json": ont, "samples": samples,
               "groups": {"a": {"name": "Ctrl", "samples": ["c1", "c2", "c3"]},
                          "b": {"name": "Exp", "samples": ["e1", "e2", "e3"]}},
               "classify_by": "marker", "metrics": ["Count"],
               "stats": {"test": "welch", "alpha": 0.05, "correction": "bh",
                         "correction_by_level": {2: "none"}, "gatekeeping": True},
               "region_filter": {"min_coverage": 0.0, "levels": [1, 2],
                                 "exclude_ids": [20]},
               "output": {"dir": os.path.join(root, "out")}}
        path = os.path.join(root, "cfg.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        r = group_stats.run_all(group_stats.load_config(path))
        text = group_stats.describe_methods(r)

        for expected in ["Welch's unequal-variance t-test", "Ctrl (n=3", "Exp (n=3",
                         "Benjamini-Hochberg", "no correction", "Gatekeeping",
                         "exploratory = TRUE", "C(6,3) = 20",
                         "Excluded before any aggregation", "Area B"]:
            assert expected in text, f"methods text is missing {expected!r}"

        # switching the test must change the text, i.e. it is generated not canned
        cfg["stats"]["test"] = "student"
        cfg["stats"]["gatekeeping"] = False
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        r2 = group_stats.run_all(group_stats.load_config(path))
        text2 = group_stats.describe_methods(r2)
        assert "Student's pooled-variance t-test" in text2
        assert "Welch" not in text2
        assert "Gatekeeping**: off" in text2

        group_stats.write_methods(r)
        assert os.path.exists(os.path.join(root, "out", "methods.md"))
    print("  ok: methods summary is generated from the settings actually used")


def test_atlas_subdivides():
    """An effective leaf is decided by what the ANNOTATION uses, not by what
    the ontology tree lists. Getting this backwards inflated the measured cell
    loss on the real data from 3-5% to 12-16%, because CCFv3's tree is deeper
    than the DeMBA P5 annotation ever labels."""
    from stats import qc_depth
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        os.makedirs(sdir)
        # annotation uses A1/A2 (children of A) but only B itself, never B1
        _write_volumes(sdir, "s1", {100: 10, 200: 10, 20: 20})
        sub = qc_depth.atlas_subdivides(sdir, o)

        assert sub[o.order_of_ids([10])[0]], "A's children are used -> A subdivides"
        assert not sub[o.order_of_ids([20])[0]], \
            "B1 is never used, so B is an effective leaf however the tree looks"
        assert sub[o.root_order], "root has used descendants"
        assert not sub[o.order_of_ids([100])[0]], "a true leaf never subdivides"
        # the ontology tree would have said otherwise for B:
        assert len(o.children[o.order_of_ids([20])[0]]) == 1
    print("  ok: effective leaves come from the annotation, not the ontology tree")


def test_depth_profile():
    from stats import qc_depth
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        sdir = os.path.join(root, "s1")
        os.makedirs(sdir)
        _write_volumes(sdir, "s1", {100: 10, 200: 10, 20: 20})
        # 2 cells resolve to A1 (a real leaf), 2 stop on B (effective leaf),
        # 1 stops on A -- which DOES subdivide, so that one is the only loss
        _write_cells(sdir, "neuron_GFP", [100, 100, 20, 20, 10],
                     ["Area A1", "Area A1", "Area B", "Area B", "Area A"])
        df, sinks, total = qc_depth.depth_profile(
            sdir, o, np.zeros(o.n, dtype=bool))
        assert total == 5, total
        by_level = df.set_index("level")
        assert np.isclose(by_level.loc[1, "stop_unresolved_pct"], 20.0), by_level
        assert np.isclose(by_level.loc[1, "stop_atlas_leaf_pct"], 40.0), by_level
        assert np.isclose(df["cum_unresolved_pct"].iloc[-1], 20.0)
        # the loss is attributed to A, the node that actually subdivides
        assert np.isclose(sinks[o.order_of_ids([10])[0]], 20.0)
        assert np.isclose(sinks[o.order_of_ids([20])[0]], 0.0)
    print("  ok: depth profile separates atlas granularity from real loss")


def test_region_maps():
    """Painting a statistic onto the atlas: the level collapse is the part
    that has to be right. A voxel labelled with a deep structure must read out
    its ancestor's value at the requested level, otherwise a level-5 map would
    only colour the few voxels whose own label happens to sit at level 5."""
    from stats import region_maps
    with tempfile.TemporaryDirectory() as root:
        import tifffile
        o = Ontology.from_json(_write_ontology(root))
        # A1 (level 2) on the left half, B (level 1) on the right half
        annot = np.zeros((4, 4, 8), dtype=np.uint32)
        annot[..., :4] = 100
        annot[..., 4:] = 20
        path = os.path.join(root, "annot.tif")
        tifffile.imwrite(path, annot)

        rv = region_maps.RegionVolume(region_maps.load_annotation(path), o)
        vals = np.full(o.n, np.nan)
        vals[o.order_of_ids([10])[0]] = 2.0     # Area A, level 1
        vals[o.order_of_ids([20])[0]] = -1.0    # Area B, level 1

        painted = rv.paint(vals, level=1)
        assert np.allclose(painted[..., :4], 2.0), "A1's voxels must read A's value"
        assert np.allclose(painted[..., 4:], -1.0)

        # at level 2 only A1 has a level-2 ancestor; B is shallower -> no value
        vals2 = np.full(o.n, np.nan)
        vals2[o.order_of_ids([100])[0]] = 5.0
        p2 = rv.paint(vals2, level=2)
        assert np.allclose(p2[..., :4], 5.0)
        assert np.isnan(p2[..., 4:]).all(), "a region shallower than the level gets no colour"

        # label 0 is outside the brain and must never be coloured
        annot0 = annot.copy()
        annot0[0, 0, 0] = 0
        tifffile.imwrite(path, annot0)
        rv0 = region_maps.RegionVolume(region_maps.load_annotation(path), o)
        assert np.isnan(rv0.paint(vals, level=1)[0, 0, 0])

        lo, hi = region_maps.symmetric_limits(painted)
        assert lo == -hi and hi > 0, (lo, hi)

        ids = rv.region_id_volume(level=1)
        assert (ids[..., :4] == 10).all() and (ids[..., 4:] == 20).all()
    print("  ok: region maps collapse to the requested level and leave gaps NaN")


def test_hemisphere_slice():
    from stats import region_maps
    full = np.zeros((2, 2, region_maps.MIDLINE_ML * 2), dtype=np.uint32)
    right, off = region_maps.hemisphere_slice(full, "right")
    assert right.shape[2] == region_maps.MIDLINE_ML and off == region_maps.MIDLINE_ML
    left, off_l = region_maps.hemisphere_slice(full, "left")
    assert left.shape[2] == region_maps.MIDLINE_ML and off_l == 0
    both, off_b = region_maps.hemisphere_slice(full, "both")
    assert both.shape == full.shape and off_b == 0
    # an already-cropped hemisphere must pass through untouched, not be halved again
    already = np.zeros((2, 2, 100), dtype=np.uint32)
    out, off2 = region_maps.hemisphere_slice(already, "right")
    assert out.shape == already.shape and off2 == 0
    print("  ok: hemisphere slicing, including an already-cropped volume")


def test_values_by_order():
    from stats import region_maps
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        df = pd.DataFrame([
            {"level": 1, "id": 10, "class_name": "X", "metric": "M",
             "log2fc": 2.0, "p_adj": 0.01},
            {"level": 1, "id": 20, "class_name": "X", "metric": "M",
             "log2fc": -1.0, "p_adj": 0.5},
            {"level": 2, "id": 100, "class_name": "X", "metric": "M",
             "log2fc": 9.0, "p_adj": 0.01},
        ])
        vals, n = region_maps.values_by_order(df, o, 1, "X", "M", "log2fc")
        assert n == 2
        assert vals[o.order_of_ids([10])[0]] == 2.0
        assert np.isnan(vals[o.order_of_ids([100])[0]]), "a different level must not leak in"
        vals_s, n_s = region_maps.values_by_order(df, o, 1, "X", "M", "log2fc",
                                                  significant_only=True, alpha=0.05)
        assert n_s == 1 and np.isnan(vals_s[o.order_of_ids([20])[0]])
    print("  ok: values pulled per (level, class, metric), significance filter applied")


def test_volume_cache_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        o = Ontology.from_json(_write_ontology(root))
        dirs = {}
        for s in ("s1", "s2"):
            sdir = os.path.join(root, s)
            os.makedirs(sdir)
            _write_volumes(sdir, s, {100: 10, 200: 10, 300: 20})
            dirs[s] = sdir
        cache = os.path.join(root, "vol.csv")
        v1 = build_per_sample_volumes(dirs, o, cache_path=cache)
        v2 = build_per_sample_volumes(dirs, o, cache_path=cache)
        assert np.allclose(v1["volume_mm3"], v2["volume_mm3"])
        assert "volume_ratio_to_median" in v1.columns
        # cache built for a different sample set must be rebuilt, not reused
        v3 = build_per_sample_volumes({"s1": dirs["s1"]}, o, cache_path=cache)
        assert set(v3["sample"]) == {"s1"}
    print("  ok: volume cache round-trip and sample-set invalidation")


def main():
    print("stats/ smoke tests")
    for fn in (test_ontology, test_marker_recoding,
               test_real_naming_conventions_map_1to1, test_class_resolution_guard,
               test_background_filter, test_exclusion_before_rollup,
               test_relative_volume_excludes,
               test_volumes_and_coverage, test_grid_offset, test_bh_and_effect_size,
               test_degenerate_ttest, test_welch_vs_student, test_methods_summary,
               test_corrections, test_gatekeeping,
               test_atlas_subdivides, test_depth_profile,
               test_region_maps, test_hemisphere_slice, test_values_by_order,
               test_volume_cache_roundtrip, test_end_to_end,
               test_coverage_masking, test_min_total_count):
        fn()
    print("all passed")


if __name__ == "__main__":
    main()
