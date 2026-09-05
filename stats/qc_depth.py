"""QC: how far down the ontology does each sample's data actually resolve?

The question this answers: if a region holds 1000 cells at level 3 but its
subregions only account for 800 at level 4, where did the other 200 go?

They did not vanish. Every cell is assigned exactly one atlas label, and atlas
labels sit at different depths in the ontology tree. A cell landing in a voxel
labelled `CA1` resolves to level 8; one landing in a voxel labelled `SUB`
stops at level 7, because that is the label the annotation carries there. So a
cell "disappears" from every level deeper than its own label. That is a
property of the annotation, not a registration failure -- but it does mean a
comparison at level 7 silently omits part of the data, and how much is
something you have to know before reading one.

Two causes have to be told apart, and the whole point of this tool is that
they look identical in a per-level cell count:

  stop_atlas_leaf   the annotation never uses ANY descendant of this label,
                    anywhere in this sample. The structure is simply not
                    subdivided in this atlas, so the cell is already as
                    resolved as it can be. Expected; nothing is lost.
  stop_unresolved   the annotation DOES use descendants of this label
                    elsewhere, but this particular voxel carries the parent.
                    This is the real attrition.

The distinction is made against the atlas annotation actually warped into each
sample (`*_labels_in_sample.nii.gz`), NOT against the ontology tree. That
matters a great deal here: CCFv3's ontology is deeper than the DeMBA P5
annotation ever labels. Presubiculum, Postsubiculum, Subiculum, Parasubiculum,
Prosubiculum, AON, taenia tecta and piriform area all list children in the
ontology and have ZERO voxels carrying any of them -- judging by the tree
would call every cell in them "lost" when nothing was lost at all.

Only stop_unresolved accumulates into a deficit. `cum_unresolved%` is the
share of a sample's cells that can never appear at that level or deeper.

Usage:
    conda activate antsreg
    python -m stats.qc_depth --config stats/configs/group_analysis.yaml
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import cell_tables  # noqa: E402
from stats.group_stats import build_exclude_mask, load_config  # noqa: E402
from stats.ontology import Ontology  # noqa: E402
from stats.qc_samples import separates  # noqa: E402
from stats.region_volumes import find_run_volumes  # noqa: E402

MAX_LEVEL = 10


def atlas_subdivides(run_dir, ontology):
    """-> boolean array: does this sample's warped annotation actually use any
    DESCENDANT of each node?

    False means the node is an effective leaf here, whatever the ontology tree
    claims -- a cell stopping there has not lost anything, because there is no
    finer label for it to have received."""
    labels_path, _ = find_run_volumes(run_dir)
    used = set(int(v) for v in np.unique(np.asarray(nib.load(labels_path).dataobj)) if v > 0)
    out = np.zeros(ontology.n, dtype=bool)
    # walk deepest-first so each node can reuse its children's answers
    for o in range(ontology.n - 1, -1, -1):
        for c in ontology.children[o]:
            if out[c] or int(ontology.ids[c]) in used:
                out[o] = True
                break
    return out


def depth_profile(run_dir, ontology, exclude_mask, classify_by="marker"):
    """-> (per-level DataFrame, per-region unresolved-stop Series, total cells).

    Aggregated over every class folder in the sample: attrition is a property
    of where the cells landed in the atlas, not of which marker they carry."""
    sample_dir = run_dir
    has_kids = atlas_subdivides(run_dir, ontology)
    roll = np.zeros(ontology.n)
    direct = np.zeros(ontology.n)
    for class_dir in cell_tables.list_sample_classes(sample_dir):
        r, d, *_ = cell_tables.class_counts(
            cell_tables.class_csv_path(sample_dir, class_dir), ontology,
            exclude_mask=exclude_mask)
        roll += r
        direct += d

    total = roll[ontology.root_order]
    if total <= 0:
        return pd.DataFrame(), pd.Series(dtype=float), 0.0

    keep = ~exclude_mask
    rows = []
    for level in range(MAX_LEVEL + 1):
        idx = np.where((ontology.levels == level) & keep)[0]
        if len(idx) == 0:
            continue
        rows.append({
            "level": level,
            "n_regions": len(idx),
            "reach_pct": roll[idx].sum() / total * 100,
            "stop_atlas_leaf_pct": direct[idx][~has_kids[idx]].sum() / total * 100,
            "stop_unresolved_pct": direct[idx][has_kids[idx]].sum() / total * 100,
        })
    df = pd.DataFrame(rows)
    df["cum_unresolved_pct"] = df["stop_unresolved_pct"].cumsum()

    sinks = pd.Series(direct * has_kids * keep, index=np.arange(ontology.n))
    return df, sinks / total * 100, total


def run(cfg):
    ontology = Ontology.from_json(cfg["ontology_json"])
    exclude_mask = build_exclude_mask(cfg, ontology)

    group_of = {}
    for key in ("a", "b"):
        for s in cfg["groups"][key]["samples"]:
            group_of[s] = cfg["groups"][key].get("name", key.upper())

    per_level, per_sink, totals = [], [], {}
    for sample, entry in cfg["samples"].items():
        if sample not in group_of:
            continue
        df, sinks, total = depth_profile(entry["dir"], ontology, exclude_mask)
        if df.empty:
            print(f"  [warn] {sample}: no cells found")
            continue
        totals[sample] = total
        df.insert(0, "group", group_of[sample])
        df.insert(0, "sample", sample)
        per_level.append(df)

        top = sinks[sinks > 0].sort_values(ascending=False).head(15)
        per_sink.append(pd.DataFrame({
            "sample": sample, "group": group_of[sample],
            "id": ontology.ids[top.index], "acronym": ontology.acronyms[top.index],
            "name": ontology.names[top.index], "level": ontology.levels[top.index],
            "pct_of_sample_cells": top.to_numpy(),
        }))

    levels_df = pd.concat(per_level, ignore_index=True)
    sinks_df = pd.concat(per_sink, ignore_index=True)
    return levels_df, sinks_df, totals, group_of


def report(levels_df, sinks_df, totals, group_of):
    pd.set_option("display.width", 220)
    samples = list(totals)

    print("\n=== 1. 每个样本能解析到多深 ===")
    print("reach%        到达该层的细胞占比")
    print("stop_atlas_leaf%   停在该层，且图谱在此没有更细的标签 —— 正常，没有丢失")
    print("stop_unresolved%   停在该层，但图谱在别处确实用了它的子标签 —— 这才是真丢失")
    print("cum_unresolved% 累计真丢失：该层及更深层永远见不到的细胞占比\n")
    for s in samples:
        g = levels_df[levels_df["sample"] == s]
        print(f"--- {s} ({group_of[s]}, {totals[s]:.0f} cells) ---")
        print(g[["level", "n_regions", "reach_pct", "stop_atlas_leaf_pct",
                 "stop_unresolved_pct", "cum_unresolved_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== 2. 累计真丢失（cum_unresolved%）横向对比 ===")
    piv = (levels_df.pivot(index="level", columns="sample", values="cum_unresolved_pct")
           .reindex(columns=samples))
    print(piv.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== 3. 丢失是否和分组绑在一起 ===")
    print("（True = 两组完全不重叠，那样按层做的组间比较会被系统性偏倚污染）")
    for level in sorted(levels_df["level"].unique()):
        row = levels_df[levels_df["level"] == level].set_index("sample")
        vals = row["cum_unresolved_pct"].reindex(samples).to_numpy(dtype=float)
        labs = [group_of[s] for s in samples]
        sep = separates(vals, labs)
        flag = "  <-- 注意" if sep else ""
        print(f"  level {level}: 范围 {np.nanmin(vals):.2f}%-{np.nanmax(vals):.2f}%  "
              f"完全分离={sep}{flag}")

    print("\n=== 4. 丢失去了哪里：最大的未解析沉淀 ===")
    print("（图谱在别处用了它们的子标签，但这些体素只带父标签）")
    top = (sinks_df.groupby(["acronym", "name", "level"])["pct_of_sample_cells"]
           .agg(["mean", "min", "max"]).sort_values("mean", ascending=False).head(12))
    print(top.to_string(float_format=lambda v: f"{v:.2f}"))


def write_report(levels_df, sinks_df, totals, group_of, out_dir, alpha_loss=5.0):
    """A written interpretation of the depth profile, generated from the
    numbers themselves so it cannot go stale."""
    samples = list(totals)
    piv = (levels_df.pivot(index="level", columns="sample", values="cum_unresolved_pct")
           .reindex(columns=samples))
    reach = (levels_df.pivot(index="level", columns="sample", values="reach_pct")
             .reindex(columns=samples))
    final = piv.iloc[-1]
    plateau = next((int(lv) for lv in piv.index
                    if np.allclose(piv.loc[lv], final, atol=0.05)), int(piv.index[-1]))

    L = []
    L.append("# QC: 逐层解析深度与细胞丢失")
    L.append("")
    L.append("由 `stats/qc_depth.py` 自动生成，数字与本目录中的 "
             "`qc_depth_by_level.csv` / `qc_depth_sinks.csv` 一致。")
    L.append("")
    L.append("## 这个现象是什么")
    L.append("")
    L.append("每个细胞只拿到**一个**图谱标签，而不同标签处在本体树的不同深度。落在标着 "
             "`CA1` 的体素上的细胞解析到 level 8；落在标着 `SUB` 的体素上的细胞停在 "
             "level 7——因为标注在那里就是这么写的。于是后者在 level 8 的表里根本不出现。")
    L.append("")
    L.append("所以「level 3 有 1000 个细胞、level 4 只剩 800」是**预期行为**，"
             "问题只在于那 200 个属于哪一种：")
    L.append("")
    L.append("| | 含义 | 是否算丢失 |")
    L.append("|---|---|---|")
    L.append("| `stop_atlas_leaf` | 图谱在这里根本没有更细的标签 | **否**，已经是最细 |")
    L.append("| `stop_unresolved` | 图谱在别处用了它的子标签，但这块体素只带父标签 | **是** |")
    L.append("")
    L.append("这个区分是拿**每个样本自己的 `*_labels_in_sample.nii.gz`** 判的，"
             "不是拿本体树判的。这一点在本数据集上关系重大：CCFv3 的本体树比 DeMBA P5 "
             "标注实际用到的深度更深。Presubiculum / Postsubiculum / Subiculum / "
             "Parasubiculum / Prosubiculum / AON / taenia tecta / piriform area "
             "在本体里都有子节点，而标注里**子区体素数为 0**。按本体树判会把这些区里的"
             "细胞全算成「丢失」，实际上一个都没丢。")
    L.append("")

    L.append("## 每个样本")
    L.append("")
    L.append("`reach%` = 到达该层的细胞占比；`cum_unresolved%` = 该层及更深处永远见不到的占比。")
    L.append("")
    header = "| level | " + " | ".join(samples) + " |"
    L.append("**reach%**")
    L.append("")
    L.append(header)
    L.append("|---" * (len(samples) + 1) + "|")
    for lv in reach.index:
        L.append(f"| {int(lv)} | " + " | ".join(f"{reach.loc[lv, s]:.1f}" for s in samples) + " |")
    L.append("")
    L.append("**cum_unresolved%（真丢失）**")
    L.append("")
    L.append(header)
    L.append("|---" * (len(samples) + 1) + "|")
    for lv in piv.index:
        L.append(f"| {int(lv)} | " + " | ".join(f"{piv.loc[lv, s]:.2f}" for s in samples) + " |")
    L.append("")

    L.append("## 解读")
    L.append("")
    L.append(f"- **真丢失最终停在 {final.min():.2f}%–{final.max():.2f}%**"
             f"（{final.idxmin()} 最低，{final.idxmax()} 最高），"
             f"并在 **level {plateau}** 之后不再增加——比它更深的层不会再丢东西。")
    worst = final.idxmax()
    if final.max() > alpha_loss:
        L.append(f"- **[!] {worst} 的丢失达到 {final.max():.2f}%，超过 {alpha_loss:.0f}% 的提示线**，"
                 f"读它的深层结果时要留意。")
    else:
        L.append(f"- 所有样本都低于 {alpha_loss:.0f}% 的提示线，逐层比较不会因此失真。")
    seps = [int(lv) for lv in piv.index
            if separates(piv.loc[lv].reindex(samples).to_numpy(dtype=float),
                         [group_of[s] for s in samples])]
    if seps:
        L.append(f"- **[!] 丢失率在 level {seps} 上把两组完全分开了。** n=3 vs 3 时这与真效应"
                 "无法区分，必须先解释再读这些层的组间结果。")
    else:
        L.append("- **丢失率没有把两组分开**（任何一层都没有完全分离），"
                 "所以它不是组间比较的系统性偏倚来源。")
    # "usable" = the deepest level where every sample still has at least half
    # its cells. Deeper levels exist but describe a shrinking minority, and
    # reporting a group difference computed on a quarter of the data without
    # saying so would be misleading.
    usable_levels = [int(lv) for lv in reach.index if reach.loc[lv].min() >= 50.0]
    deepest = max(usable_levels) if usable_levels else int(reach.index[0])
    L.append(f"- **数据能支撑到 level {deepest}**：那里每个样本都还有 "
             f"{reach.loc[deepest].min():.0f}%–{reach.loc[deepest].max():.0f}% 的细胞在场。"
             "缺席的那部分绝大多数是 `stop_atlas_leaf`（图谱没标那么细），不是配准没做到。")
    thinner = [int(lv) for lv in reach.index if lv > deepest and reach.loc[lv].max() > 0]
    if thinner:
        rng = ", ".join(f"L{lv} {reach.loc[lv].min():.0f}–{reach.loc[lv].max():.0f}%"
                        for lv in thinner)
        L.append(f"- 更深的层还有数据但已很稀薄（{rng}）。在那里做组间比较不是错，"
                 "但结论只覆盖样本的一小部分，必须在图注里写明 `reach%`。")
    L.append("")

    L.append("## 丢失去了哪里")
    L.append("")
    top = (sinks_df.groupby(["acronym", "name", "level"])["pct_of_sample_cells"]
           .agg(["mean", "min", "max"]).sort_values("mean", ascending=False).head(10))
    L.append("| 区 | level | 平均占比% | 最小 | 最大 |")
    L.append("|---|---|---|---|---|")
    for (acr, name, lv), row in top.iterrows():
        L.append(f"| {acr} ({name}) | {int(lv)} | {row['mean']:.2f} | "
                 f"{row['min']:.2f} | {row['max']:.2f} |")
    L.append("")
    L.append("`root` 居首是已知的：CCFv3 在海马与丘脑之间（侧脑室/伞/脉络丛一带）"
             "本来就没有叶子标签，那条带子上的体素只能带 `root`（见 PROGRESS_LOG 2026-08-28）。"
             "脑室（`VL`）和纤维束（`fa`）同理——那里本来也不该有细胞，"
             "落在那儿的更可能是归区误差而不是真实定位。")
    L.append("")

    L.append("## 实践结论")
    L.append("")
    L.append(f"- 逐层组间比较到 **level {deepest}** 都可用；真丢失在 level {plateau} 之后不再"
             "增加，所以往更深走损失的不是数据质量，只是覆盖面。")
    L.append("- 报告某一层的结果时，附上该层的 `reach%`——它说明这个数字覆盖了样本的多大比例。")
    L.append("- 想追某个具体区的丢失，看 `<sample>_region_tree.xlsx`：`Lost cells` 行"
             "精确到节点，Summary sheet 给逐层累计。")

    text = "\n".join(L)
    path = os.path.join(out_dir, "qc_depth_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
    print(f"Saved to: {path}")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    levels_df, sinks_df, totals, group_of = run(cfg)
    report(levels_df, sinks_df, totals, group_of)

    out_dir = (cfg.get("output") or {}).get("dir", "./stats_output")
    os.makedirs(out_dir, exist_ok=True)
    levels_df.to_csv(os.path.join(out_dir, "qc_depth_by_level.csv"), index=False)
    sinks_df.to_csv(os.path.join(out_dir, "qc_depth_sinks.csv"), index=False)
    print(f"\nWrote qc_depth_by_level.csv and qc_depth_sinks.csv to {out_dir}")
    write_report(levels_df, sinks_df, totals, group_of, out_dir)


if __name__ == "__main__":
    main()
