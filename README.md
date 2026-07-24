# scripts/ 与 mask_tools/ 工具说明

5 个手动运行的小工具，配合 `registration_ants` 主流程（ANTs 配准）和 `registration_eval.py`（配准质量评估）使用。都在 `antsreg` env 下跑：

```bash
conda activate antsreg
```

## 配准前：产出喂给 ANTs 的 mask / 引导结构

| 命令 | 做什么 | 输出给谁 |
|---|---|---|
| `mask_tools/paint_mask.py`（`KIND="mask"`） | 一份二值 mask 上直接密集画/擦：label 1=包含、label 0=排除。裂缝/损伤和"漏检的脑组织/误检的杂散区域"本质是同一种编辑，都在这一个模式里改，不插值不精修，改完导出一个文件。`EXISTING_MASK_PATH` 不填就是空白画布（默认全 1，擦掉的地方就是排除区），填了就是在已有 mask（比如自动脑轮廓）上接着改 | `mask.sample_damage_mask_path` |
| `mask_tools/paint_mask.py`（`KIND="guide"`，`ROLE="sample"`/`"atlas"`） | 画"真实存在但形变对不上"的引导结构（跟上面完全是两回事：不是 1=包含/0=排除的二值 mask，是喂给 `multivariate_extras` 的一对轮廓，主动拉形变场） | `mask.guide_regions` |
| `scripts/project_outline.py <outline> <atlas> <res> <transforms_prefix> <out>` | 把 guide 的样本侧轮廓投影到图谱空间，当 atlas 侧画图的起点 | `paint_mask.py`（`KIND="guide"`，`ROLE="atlas"`）的 `EXISTING_MASK_PATH` |

**guide 的用法**：`paint_mask.py`（`ROLE="sample"`）→ `project_outline.py` → `paint_mask.py`（`ROLE="atlas"`，`EXISTING_MASK_PATH=<上一步输出>`）。

**注意**：`paint_mask.py` 不用命令行传参，每次用之前直接改脚本顶部 `KIND`/`IMAGE_PATH`/`OUTPUT_PATH`/`EXISTING_MASK_PATH`/`ROLE` 这几个变量再 `python mask_tools/paint_mask.py` 运行；`project_outline.py`/`edit_sample_labels.py`/`relabel_cells.py`/`place_landmarks.py` 还是命令行传参，没变。

## 配准后：手动订正结果，喂给 registration_eval.py 做 ground truth

| 命令 | 做什么 | 输出给谁 |
|---|---|---|
| `scripts/edit_sample_labels.py <sample> <labels> <out>` | 直接改 `labels_in_sample.nii.gz` 里错的区域（挑真实 CCF id，几个关键层重画，其余插值），不重跑配准 | `registration_eval.py` 的 Dice/HD95；`scripts/relabel_cells.py` |
| `scripts/relabel_cells.py <corrected_labels> <sample_dir>` | 用订正后的标签，重算 `cell_registration.csv` 里每个细胞的所属脑区 | 下游细胞计数统计 |
| `scripts/place_landmarks.py --role {sample,atlas} <img> <out.csv>` | 在样本/图谱上手点一批对应的解剖标志点 | `registration_eval.py` 的 landmark TRE |

三个互相独立，做了哪个就用哪个——`registration_eval.py` 按对应文件是否存在自动跳过没做的部分。

## 共同点

- 都用 `SimpleITK.GetArrayFromImage` 读图（不是 `ants.image_read().numpy()`，两者轴序相反，用错了会画到错的切面）。
- 交互工具（`paint_mask.py`/`edit_sample_labels.py`/`place_landmarks.py`）是 napari + PyQt5，画完点右侧 Export 按钮导出。
- 非交互工具（`project_outline.py`/`relabel_cells.py`）不需要显示环境。
- 完整参数说明看各脚本自己的 docstring。

---
旧文件名对照（2026-07-23 改名 + 合并，功能不变）：`paint_damage_mask.py`/`refine_brain_mask.py`/`paint_guide_outline.py` → 合并进 `mask_tools/paint_mask.py`；`edit_labels_in_sample.py` → `edit_sample_labels.py`；`relabel_cells_from_corrected_atlas.py` → `relabel_cells.py`；`project_outline_to_atlas.py` → `project_outline.py`。详见 `PROGRESS_LOG.md`。
