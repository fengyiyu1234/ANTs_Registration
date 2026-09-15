"""Isocortex laminar distribution: every area's layers merged into depth bins,
then compared between groups.

WHY LAYERS AND NOT AREAS
------------------------
Cells reach the cortex by radial migration, and what migration decides is the
DEPTH a cell stops at, not the area it sits in. The per-region runs had
already shown that the Isocortex Sox9+ shift is one isocortex-wide shift seen
through every subregion (0913_01): dividing each area by its own animal's
isocortex removed most of the per-area effect. So the question is asked on
layers pooled across all isocortical areas.

WHY THESE BINS (measured 2026-09-14 on DeMBA P5 + the six TSC brains)
----------------------------------------------------------------------
* 6b is merged into L6. It is ~57 um thick in the atlas and 83% of its voxels
  sit within one voxel of its own boundary -- thinner than this dataset's
  registration residual. A layer that is all boundary cannot be counted.
* L1 is never a bin on its own. The L1/L2-3 boundary is the least reliable
  line in these registrations, in both directions: s10's cortex is partly
  compressed and pushes L2/3 cells into the L1 label (25.6% of its isocortical
  cells, rising to 52% along raw z), while in s12t the L1 label lies on the
  bright pial rim where almost nothing is detected (4.6%). The other four
  brains sit at 11.5-15.7%. That spread is tissue and detection, not biology.
* L4 is never a bin on its own either. Only 25 of 43 isocortical areas carry
  an L4 label; the agranular ones (MO, ACA, PL, ILA, ORB, AI, RSP, FRP, ECT,
  PERI) do not, so a pooled "L4" is sensory cortex only and the agranular
  areas' would-be L4 cells are already inside their L2/3 and L5.
* L5 vs L6 is taken as reliable: both are thick (~270 / ~500 um) and the
  corpus callosum below them registers well.

Hence the default: upper (1 + 2/3 + 4), L5, L6 (6a + 6b), plus superficial vs
deep as the coarsest summary. Every scheme must place every layer label the
atlas uses inside the root, or the run stops -- a layer silently left out would
shrink every denominator without anyone noticing.

METRICS
-------
  LaminarShare    cells of a class in a bin / cells of that class in all bins
                  of the same animal. PRIMARY. Labelling efficiency, dose and
                  brain size are in numerator and denominator and cancel; what
                  is left is only depth. Shares sum to 1 across a scheme's bins,
                  so a rise in one bin is a fall in the others: read the bins of
                  one readout together, never as independent findings.
  BinComposition  cells of a class in a bin / all labelled cells in that bin,
                  e.g. the Sox9+ fraction of L5. Complementary classes
                  (Sox9_pos / Sox9_neg) are the same test.
  Count           raw cells per bin, for reference only; it carries the
                  per-animal labelling efficiency (L1 cells range 7k-49k).

Two poolings of the proportions:
  pooled          counts summed over areas, then divided. Big areas (SS, MO)
                  weigh most. The primary estimator.
  area_mean       the proportion inside each area, then averaged over areas.
                  Every area weighs the same. Areas below `area_min_cells` in
                  ANY animal are dropped so all animals average the same set,
                  and a bin an area has no atlas label for is undefined there,
                  not zero.

Density is not offered: pooled bins in sample space carry the ~20% brain-size
difference that has already been shown to make up half of every Density effect.

    conda activate antsreg
    python -m stats.laminar --config stats/configs/tsc_laminar.yaml
"""
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.composition import readouts_from_config  # noqa: E402
from stats.group_stats import (load_config, samples_beside_means,  # noqa: E402
                               save_config_copy)
from stats.ontology import Ontology  # noqa: E402
from stats.summary_workbook import (_assert_ascii_text, _loo,  # noqa: E402
                                    _permutation_rank, _test_rows)

ISOCORTEX_ID = 315

# "Primary motor area, Layer 1", "..., layer 2/3", and the ACAv labels that lost
# the word: "Anterior cingulate area, ventral part, 6a".
LAYER_NAME = re.compile(r"(?:,\s*layer\s*|\s+layer\s+|,\s*)(1|2/3|2|3|4|5|6a|6b)\s*$",
                        re.IGNORECASE)
TOKEN_ALIAS = {"2": "2/3", "3": "2/3"}

DEFAULT_SCHEMES = {
    "three_bin": {"upper": ["1", "2/3", "4"], "L5": ["5"], "L6": ["6a", "6b"]},
    "two_bin": {"superficial": ["1", "2/3", "4"], "deep": ["5", "6a", "6b"]},
}
UNASSIGNED = "unassigned"


def layer_token(name):
    m = LAYER_NAME.search(str(name))
    if not m:
        return None
    t = m.group(1).lower()
    return TOKEN_ALIAS.get(t, t)


def layer_map(ontology, root_id, schemes):
    """-> one row per structure under `root_id`: its layer token, the area it
    belongs to (the layer label's parent), and one `bin_<scheme>` column.

    Structures without a layer token (the root itself, area parents, a layer
    label the regex does not know) get no bin; cells on them are reported as
    `unassigned`, never dropped silently."""
    root = int(ontology.order_of_ids([root_id])[0])
    if root < 0:
        raise ValueError(f"root id {root_id} is not in the ontology")
    rows = []
    for o in ontology.descendant_orders([root]):
        tok = layer_token(ontology.names[o]) if o != root else None
        p = ontology.parent_order[o]
        rows.append({"order": o, "id": int(ontology.ids[o]), "name": ontology.names[o],
                     "acronym": ontology.acronyms[o],
                     "area": ontology.acronyms[p] if tok else "",
                     "token": tok, "leaf": not ontology.children[o]})
    df = pd.DataFrame(rows)

    # A layer label with children would be counted once itself and again through
    # them. CCF has none; fail loudly if an ontology ever does.
    bad = df[df["token"].notna() & ~df["leaf"]]
    if len(bad):
        raise ValueError(f"layer labels with children: {bad['name'].tolist()}")

    tokens = set(df["token"].dropna())
    if not tokens:
        raise ValueError(f"no layer labels under root id {root_id}")
    for scheme, bins in schemes.items():
        assigned = {}
        for b, toks in bins.items():
            for t in toks:
                t = TOKEN_ALIAS.get(str(t), str(t))
                if t in assigned:
                    raise ValueError(f"scheme '{scheme}': layer {t} is in both "
                                     f"'{assigned[t]}' and '{b}'")
                assigned[t] = b
        missing = tokens - set(assigned)
        if missing:
            raise ValueError(f"scheme '{scheme}' puts layer(s) {sorted(missing)} in no bin; "
                             "every layer the atlas uses must be placed")
        unknown = set(assigned) - tokens
        if unknown:
            raise ValueError(f"scheme '{scheme}' names layer(s) {sorted(unknown)}, "
                             f"which the atlas does not use under root {root_id}")
        df[f"bin_{scheme}"] = df["token"].map(assigned)
    return df


def count_cells(cfg, ontology, lmap, class_map):
    """-> long [group, sample, class_name, order, count] for every structure under
    the root that holds at least one cell."""
    in_root = np.zeros(ontology.n, dtype=bool)
    in_root[lmap["order"].to_numpy()] = True
    frames = []
    for key in ("a", "b"):
        for s in cfg["groups"][key]["samples"]:
            sdir = cfg["samples"][s]["dir"]
            for cls in class_map:
                dirs = cell_tables.resolve_class_dirs(sdir, cls, class_map=class_map)
                if not dirs:
                    print(f"  [warn] {s}: class '{cls}' 没有对应的文件夹，按 0 计")
                bins = np.zeros(ontology.n)
                for d in dirs:
                    path = cell_tables.class_csv_path(sdir, d)
                    if not os.path.exists(path):
                        continue
                    ids = cell_tables.valid_region_ids(cell_tables.read_cell_registration(path))
                    orders = ontology.order_of_ids(ids)
                    orders = orders[orders >= 0]
                    bins += np.bincount(orders[in_root[orders]], minlength=ontology.n)
                nz = np.flatnonzero(bins)
                frames.append(pd.DataFrame({"group": key, "sample": s, "class_name": cls,
                                            "order": nz, "count": bins[nz]}))
    return pd.concat(frames, ignore_index=True)


def share_readouts_from_config(cfg):
    """Every base class plus every all-'+' combined category, INCLUDING the one
    that is all cells: its laminar share is the most basic readout there is."""
    base = list(cfg["class_map"].keys())
    out = {c: [c] for c in base}
    for name, terms in (cfg.get("combined_categories") or {}).items():
        parts = [t["class"] if isinstance(t, dict) else t for t in terms]
        if all((t.get("sign", "+") if isinstance(t, dict) else "+") == "+" for t in terms):
            out[name] = parts
    return base, out


def sample_values(long, lmap, scheme, bin_names, share_readouts, comp_readouts, base,
                  samples, area_min_cells):
    """-> tidy [scheme, pooling, metric, readout, bin, <one column per sample>]."""
    col = f"bin_{scheme}"
    d = long.merge(lmap[["order", "area", col]], on="order", how="inner")
    d = d[d[col].notna()].rename(columns={col: "bin"})
    areas = sorted(lmap.loc[lmap[col].notna(), "area"].unique())
    full = pd.MultiIndex.from_product([samples, areas], names=["sample", "area"])
    present = (lmap[lmap[col].notna()].groupby(["area", col]).size().unstack(col)
               .reindex(index=areas, columns=bin_names).notna())

    def agg(parts):
        return (d[d["class_name"].isin(parts)]
                .groupby(["sample", "area", "bin"])["count"].sum()
                .unstack("bin").reindex(index=full, columns=bin_names).fillna(0.0))

    rows = []

    def emit(metric, pooling, readout, wide):
        wide = wide.reindex(index=samples, columns=bin_names)
        for b in bin_names:
            rows.append({"scheme": scheme, "pooling": pooling, "metric": metric,
                         "readout": readout, "bin": b,
                         **{s: float(wide.loc[s, b]) for s in samples}})

    def area_mean(num, den, keep_area):
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = num / den.replace(0, np.nan)
        mask = present.reindex(frac.index.get_level_values("area")).to_numpy()
        frac = frac.where(mask)
        frac = frac[frac.index.get_level_values("area").isin(keep_area)]
        return frac.groupby(level="sample").mean()

    for r, parts in share_readouts.items():
        c = agg(parts)
        pooled = c.groupby(level="sample").sum()
        emit("Count", "pooled", r, pooled)
        emit("LaminarShare", "pooled", r,
             pooled.div(pooled.sum(axis=1).replace(0, np.nan), axis=0))
        area_tot = c.sum(axis=1).unstack("area")
        keep = area_tot.columns[(area_tot >= area_min_cells).all(axis=0)]
        emit("LaminarShare", "area_mean", r,
             area_mean(c, pd.concat([c.sum(axis=1)] * len(bin_names), axis=1,
                                    keys=bin_names), keep))

    total = agg(base)
    for r, parts in comp_readouts.items():
        c = agg(parts)
        pooled_t = total.groupby(level="sample").sum()
        emit("BinComposition", "pooled", r,
             c.groupby(level="sample").sum() / pooled_t.replace(0, np.nan))
        tot_area = total.sum(axis=1).unstack("area")
        keep = tot_area.columns[(tot_area >= area_min_cells).all(axis=0)]
        emit("BinComposition", "area_mean", r, area_mean(c, total, keep))
    return pd.DataFrame(rows)


def run_tests(values, samples_a, samples_b, test, correction, alpha, primary_scheme):
    """One correction family per (scheme, pooling, metric, readout): the bins of
    one readout. Families are small on purpose -- that is the point of pooling."""
    keys = ["scheme", "pooling", "metric", "readout"]
    samples = list(samples_a) + list(samples_b)
    out = []
    for _, sub in values.groupby(keys, sort=False):
        sub = sub.reset_index(drop=True)
        res = _test_rows(sub[samples_a].to_numpy(float), sub[samples_b].to_numpy(float),
                         sub[keys + ["bin"]].to_dict("records"), test, correction, alpha)
        extra = []
        for _, r in sub.iterrows():
            rank, of, p_perm, floor = _permutation_rank(r[samples_a].to_numpy(float),
                                                        r[samples_b].to_numpy(float))
            # A row whose group means are equal has log2fc 0, so every
            # leave-one-out fraction is 0/0; that NaN is the answer, not news.
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                worst, keeps = _loo(r, samples_a, samples_b)
            row = {"perm_rank": rank, "perm_of": of, "p_perm": p_perm, "p_perm_floor": floor,
                   "loo_worst_frac": worst, "loo_keeps_sign": keeps}
            for s in samples:
                ma = np.nanmean([r[x] for x in samples_a if x != s])
                mb = np.nanmean([r[x] for x in samples_b if x != s])
                row[f"log2fc_without_{s}"] = (float(np.log2(mb / ma))
                                              if ma > 0 and mb > 0 else np.nan)
            extra.append(row)
        out.append(pd.concat([res, pd.DataFrame(extra), sub[samples]], axis=1))
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "role", np.select(
        [(df["scheme"] == primary_scheme) & (df["pooling"] == "pooled")
         & (df["metric"] == "LaminarShare"),
         df["metric"] == "Count"],
        ["primary", "reference"], "secondary"))
    return samples_beside_means(df, samples)


def token_qc(long, lmap, samples):
    """Share of each animal's labelled cells on every ORIGINAL layer label, plus
    the unassigned ones. This is where a registration problem in a thin layer
    shows up (s10's L1), before any bin hides it."""
    d = long.merge(lmap[["order", "token"]], on="order", how="left")
    d["token"] = d["token"].fillna(UNASSIGNED)
    t = d.groupby(["token", "sample"])["count"].sum().unstack("sample").reindex(columns=samples)
    t = t.fillna(0.0)
    return t / t.sum(axis=0)


def run(cfg, out_dir=None):
    lam = cfg.get("laminar") or {}
    root_id = int(lam.get("root_id", ISOCORTEX_ID))
    schemes = lam.get("schemes") or DEFAULT_SCHEMES
    primary = lam.get("primary_scheme", next(iter(schemes)))
    if primary not in schemes:
        raise ValueError(f"primary_scheme '{primary}' is not one of {list(schemes)}")
    area_min = int(lam.get("area_min_cells", 50))
    st = cfg.get("stats") or {}
    test = str(st.get("test", "welch")).lower()
    correction = str(st.get("correction", "bh")).lower()
    alpha = float(st.get("alpha", 0.05))

    if not cfg.get("class_map"):
        raise ValueError("laminar 需要 class_map：比例的分母是它的基础类之和")
    class_map = cell_tables.normalize_class_map(cfg["class_map"])
    samples_a = list(cfg["groups"]["a"]["samples"])
    samples_b = list(cfg["groups"]["b"]["samples"])
    samples = samples_a + samples_b

    ontology = Ontology.from_json(cfg["ontology_json"])
    lmap = layer_map(ontology, root_id, schemes)
    long = count_cells(cfg, ontology, lmap, class_map)

    base, share_ro = share_readouts_from_config(cfg)
    _, comp = readouts_from_config(cfg)
    comp_ro = {re.sub(r"_fraction$", "", k): v for k, v in comp.items()}

    values = pd.concat([
        sample_values(long, lmap, scheme, list(bins), share_ro, comp_ro, base, samples, area_min)
        for scheme, bins in schemes.items()], ignore_index=True)
    tests = run_tests(values, samples_a, samples_b, test, correction, alpha, primary)
    qc = token_qc(long, lmap, samples)

    out_dir = out_dir or (cfg.get("output") or {}).get("dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        lab_counts = (long.groupby(["order", "sample"])["count"].sum()
                      .unstack("sample").reindex(columns=samples).fillna(0).reset_index())
        layer_table = lmap.merge(lab_counts, on="order", how="left").fillna(
            {s: 0 for s in samples})
        tables = {
            "laminar_tests.csv": tests,
            "laminar_per_sample.csv": values,
            "laminar_counts_long.csv": long.merge(
                lmap[["order", "name", "area", "token"]], on="order", how="left"),
            "laminar_layer_map.csv": layer_table,
            "laminar_token_qc.csv": qc.reset_index(),
        }
        for name, frame in tables.items():
            _assert_ascii_text(frame, name)
            frame.to_csv(os.path.join(out_dir, name), index=False)
        save_config_copy(cfg, out_dir)
    return {"layer_map": lmap, "long": long, "values": values, "tests": tests, "token_qc": qc,
            "samples_a": samples_a, "samples_b": samples_b, "primary": primary,
            "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="覆盖 config 里的 output.dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = run(cfg, args.out_dir)
    a, b = r["samples_a"], r["samples_b"]
    groups = cfg["groups"]
    print(f"{groups['a'].get('name', 'A')} {', '.join(a)}   "
          f"{groups['b'].get('name', 'B')} {', '.join(b)}\n")

    print("原始层标签上的细胞占比（所有标记细胞）。薄层的配准问题在这里看，分箱之后就看不见了：")
    print(r["token_qc"].to_string(float_format=lambda v: f"{v:6.3f}"))

    t = r["tests"]
    show = t[t["role"] == "primary"][["readout", "bin", "mean_a", "mean_b", *a, *b,
                                      "log2fc", "hedges_g", "p_value", "p_adj",
                                      "perm_rank", "loo_keeps_sign"]]
    print(f"\n主分析：{r['primary']}，pooled LaminarShare"
          f"（每类细胞在各箱的占比，同一类的几个箱加起来是 1，要一起读）")
    print(show.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print("\nperm_rank 是 10 种分法里的排名，1 最强；置换 p 的下限是 0.10，不是显著性判定")
    if r["out_dir"]:
        print(f"写出 -> {r['out_dir']}")


if __name__ == "__main__":
    main()
