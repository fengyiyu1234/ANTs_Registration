# stats/ — 组间统计

配准之后的组间比较，**不依赖 ClearMap**。前身是
`../ClearMap/stats_vis/stats_group_compare.py`，那份脚本绑死在
`ClearMap.Alignment.Annotation`、`ClearMap.IO` 和 elastix 的 `volume/result.mhd`
上；这里全部换成本项目自己的产物。

```
ontology.py        CCF 本体树（替代 ClearMap 的 ano）
cell_tables.py     读 cell_registration.csv、marker 重编码
region_volumes.py  每样本分区体积 + 组织覆盖率（密度的分母）
group_stats.py     主入口：指标、Welch + BH、效应量、输出
qc_samples.py      上机前体检：有没有非生物学变量把两组分开了
qc_depth.py        逐层解析深度：细胞在哪一层"消失"，哪些是真丢失
region_maps.py     把逐区统计量画到图谱体积上（level 折叠在这里）
plot_heatmaps.py   静态热图：冠状面 + 矢状面各若干张
test_stats.py      合成数据自测，不碰真实数据
configs/           样本/分组/区域集配置
```

## 跑法

```bash
conda activate antsreg
cp stats/configs/group_analysis.example.yaml stats/configs/group_analysis.yaml
python -m stats.qc_samples  --config stats/configs/group_analysis.yaml   # 先跑这两个
python -m stats.qc_depth    --config stats/configs/group_analysis.yaml
python -m stats.group_stats --config stats/configs/group_analysis.yaml
python -m stats.plot_heatmaps --config stats/configs/group_analysis.yaml --batch
python stats/test_stats.py                                              # 自测
```

`configs/group_analysis.yaml` 是日常用的配置，开头三节就是需要改的全部内容：

| 节 | 改什么 |
|---|---|
| `samples:` | 每个样本指向它的配准 run 目录 |
| `groups:` | 两组分别有哪些样本 |
| `stats:` | 检验方法、alpha、每层的校正方式、门控开关 |

下面还有一份 `tsc_marker.yaml`，字段相同但注释更详尽，当参考手册用。
两者都被 gitignore（规则 `stats/configs/*.yaml` + `!*.example.yaml`）：跟踪的是
`*.example.yaml` 模板，工作副本留在本地，改配置不会产生 git diff。

依赖只有 numpy / pandas / scipy / nibabel / pyyaml / openpyxl（+ tifffile 仅在
配了 `reference_annotation` 时）。**不 import ants**，所以任何装了这几个包的环境
都能跑，不必是 `antsreg`。

## 和 ClearMap 版的四处实质差异

| | ClearMap 版 | 这里 |
|---|---|---|
| 本体 | `ano`（ClearMap 内部 `order`/`graph_order`） | `ontology.py`，直接读 `atlas/DeMBA/CCF_v3_ontology.json` |
| 细胞的区域列 | 第 9 列当 `graph_order` | 第 9 列是**原始 CCF id**（见 `cell_points.py`）。拿旧脚本读 ANTs 的输出会静默归错区 |
| 分区体积 | `volume/result.mhd` | `<run>/*_labels_in_sample.nii.gz`，配 `*_brain_mask.nii.gz` 算覆盖率 |
| 细胞分类 | neuron/glia × marker 共 12 类 | `classify_by: marker`，丢掉 YOLO 的 neuron/glia，只留 6 个 marker 签名 |

`order` 是本模块自己 DFS 出来的稠密下标，**和 ClearMap 的 `order` 不是一回事**。
跨工具对照请 join `id`，不要 join `order`。

## marker 重编码

检测端（`brain_detector`）给每个物理细胞写**一个**复合类名
`{soma_type}_{soma channels}_{TF}`，所以 6 个 marker 签名互斥，
neuron_X 和 glia_X 直接相加即可，不会重复计数。实测验证过：同一样本 12 个类文件
两两之间，5 µm 内有对应细胞的比例都低于 0.3%。

```
GFP | RFP | GFP_RFP | GFP_Sox9 | RFP_Sox9 | GFP_RFP_Sox9
```

两条必须写进 methods 的语义限制：

* 细胞只在两个 soma 通道（RFP/GFP）上被检出，Sox9 只作为共定位属性存在，
  **不存在 Sox9 单阳细胞**。所以 `Sox9_any` 的含义是"reporter 阳性细胞中的
  Sox9+"，不能外推成组织里的星形胶质总数。
* `Sox9−` 不等于"确认阴性"，只等于"没匹配上 Sox9 核"，受 730nm 通道成像质量和
  `max_center_dist_ratio` 影响。

`classify_by: full` 可以退回 12 类原样比较。

### 关于类名里的 `3`（已在磁盘上改名，此节留作背景）

曾经 s11/s12q/s12t 的类名是 `neuron_3_GFP`，s8/s10/s18 是 `neuron_GFP`。原因是那三个
样本的 GFP tile 目录名叫 `GFP_3`，而 `brain_detector` 的 `stitcher._merge_class`
按 `_` 切分再 `sorted(markers_a | markers_b)` 拼类名，于是 `3` 被当成一个独立 marker
并排到了 GFP 前面——**是文件夹命名失误，不是两批检测**。

它只挂在 GFP 阳性的类上（`RFP` / `RFP_Sox9` 完全没有 `3`），检测端 Single Marker
Positivity 表里 `3` 的计数和 `GFP` 一模一样（s11 都是 233,174，s12t 都是 129,180）
——同一列重复了一行，不是多出一群细胞。计数本身正确，不需要重跑。

**2026-09-04 已把 `raw_data/` 和 `TSC_ants/` 下所有 `*_3_GFP*` 改名为 `*_GFP*`**
（200 项：24 个 centroid CSV + 176 个 `cell_registration/` 子目录）。改名前验证过
两种拼法从未在同一样本共存、无目标冲突、每个 `cell_registration` 改完仍是 12 个类；
改名后重跑 `group_stats.py`，5271 行结果**逐行完全一致**。
回滚记录在 `<Registration>/rename_3_GFP_to_GFP.json`。
旧的 `clearmap/TSC_clearmap/` 下还有 32 个未改（已弃用的管线，没动）。

**丢弃纯数字 token 的规则保留**，因为改名只清理了当前这批数据：旧 ClearMap 目录、
以及以后任何再次踩到 `_merge_class` 这个坑的数据仍然靠它兜底。
`test_stats.py` 里有对着当年那两套 12 个类名逐一比对的用例。

`check_class_resolution()` 是这条规则的防呆：如果某个样本里同时存在 `neuron_GFP` 和
`neuron_3_GFP` 两个**真的不同**的类，丢数字 token 就会把它们加起来重复计数——那种
情况会在 `group_stats.py` 启动时打 `[WARN]`；同一个类在不同样本解析到的文件夹数量
不一致也会报。

## 指标

| 指标 | 定义 |
|---|---|
| `Count` | 该区 + 其所有子区的细胞数 |
| `Percentage` | 占该类全脑细胞的 % |
| `Density` | 每 mm³ 细胞数，分母是**该样本自己的**分区体积 |
| `RegionProportion` | 占同一区内全部 base 类细胞的 %，免疫于样本间检出效率差异 |
| `Volume` | 分区体积本身，mm³ | 形态学表型 |
| `RelativeVolume` | 该区体积 / 分析范围内总体积 × 100 | 消掉整体脑大小和视野差异 |

`RegionProportion` 的分母只取 6 个互斥 base 类，**不含** combined 类——后者互相
重叠，加进去会重复计数、比例不收敛到 100%。

### 密度的分母为什么要过 brain mask

样本是半脑，切割面不是每只都正好落在解剖中线上（PROGRESS_LOG 2026-08-29）。
切短了的样本，中线附近的区在图谱里体积照旧、组织却没了，用全体积当分母会把密度
系统性压低。所以：

* `coverage` = 该区 warp 后落在 brain mask 内的体素比例
* `density_denominator: covered`（默认）→ 分母只算真正成像到的那部分
* `region_filter.min_coverage` → 覆盖率不够的区，**只在该样本上**置 NaN，其余样本
  照常参与检验，`n_a`/`n_b` 记录实际贡献了几只

另外 `labels_in_sample` 本身已经在 `crop_for_registration` 之外清零，所以裁进组织
也会让分区体积缩水。`volume_ratio_to_median`（某样本该区体积 / 各样本中位数）是
不依赖 brain mask 的第二重信号，在 `per_sample_region_volumes.csv` 里。

## n=3 vs 3 怎么读

这一条决定了上面所有数字的读法：

* **精确置换检验在这个设计下不可用**。6 只分成 3+3 只有 C(6,3)/2 = 10 种划分，
  双侧置换 p 的下限是 0.1，**永远到不了 0.05**。所以只能用 Welch t（代价是要接受
  正态假设），并把效应量摆在旁边。
* Welch 在 n=3v3 时 df 在 2.4~4 之间（取决于两组方差比）。等方差时 raw p<0.05 需要
  |Hedges' g| ≈ 1.8；但 BH 之后阈值随 family 大小陡升——m=17 需要 |g| ≈ 4.2，
  m=98 需要 |g| ≈ 6.7。**决定成败的是 family 大小，不是检验本身。**
* 因此 `region_filter` 里那两项才是真正决定有没有结果的开关：
  * `levels` — 每一层是独立的 FDR family，level 7 单层就有 324 个区。先粗后细。
  * `include_ids` — 预先指定要检验哪些区。**这是这里能拿到的最大一笔 power**，
    见下面一节。
  * `min_total_count` — 细胞数太少的区连噪声都算不上，只会撑大 family。
* 输出里 `hedges_g` / `g_ci_lo` / `g_ci_hi` 用的是小样本校正（n=3 时 Cohen's d 会
  高估 25%，J = 1-3/(4N-9) = 0.8）。**按效应量排序，把 p_fdr 当筛子，不要当结论。**
* **要看 CI**：n=3v3 时 g 的标准误是 sqrt(6/9 + g²/8)，即使 g=4，95% CI 也有
  [0.8, 7.2]。所有"显著"结果的效应量本身都只有数量级精度。

### 排除区域：必须在 rollup 之前

`region_filter.exclude_ids` 和 `include_ids` 作用完全不同：

| | 作用点 | 效果 |
|---|---|---|
| `include_ids` | 只挑"检验哪些区" | 不影响任何分母 |
| `exclude_ids` | **rollup 之前**打在逐区原始计数/体素上 | 该子树从所有祖先的 Count/Volume 里消失，也从 `Percentage` 和 `RelativeVolume` 的分母里消失 |

制备时有缺损的结构（小脑、嗅球）必须走 `exclude_ids`。否则每个别的区的"占比"都会
取决于那一只样本的小脑碰巧留下了多少——那不是生物学差异。

实现上就是在 `direct`（未 rollup 的逐区值）上把被排除子树清零再 rollup，于是
root 处的数自然变成"分析范围内的总量"（`cell_tables.class_counts` /
`region_volumes.rollup_volumes`，各有一个测试盯着祖先节点是否也被扣掉）。

当前配置排除 `[512 Cerebellum, 507 MOB, 151 AOB, 1016 onl]`，共 98 个区。依据是实测
覆盖率（六样本 min/mean）：CB 0.33/0.75、MOB 0.57/0.93、onl 0.01/0.66，
对比健康的 CTX 0.98/1.00、HPF 0.99/1.00。config 注释里还列了两个候选
（`354 Medulla` 0.39/0.76、`73 ventricular systems` 0.63/0.85）没有默认排除。

**注意小脑相关的纤维束不在 512 子树下**（`arb` 小脑白质、`cbc` 小脑连合挂在
fiber tracts 底下），如果也要去掉需要单独列。

### 分层校正与门控

`stats.correction_by_level` 给每层单独指定校正方法，`stats.gatekeeping` 打开
固定序检验：**只有当某区在上一个被检验的层级上通过了，才继续检验它的子树。**

当前配置：

`stats.test` 可选 `welch`（默认，不等方差）或 `student`（合并方差，恒定 df=4）。
没有非参数选项不是遗漏——3v3 下置换检验和 Mann-Whitney 的双侧 p 下限都是 0.1。

| 层级 | 区数 | 校正 | 角色 |
|---|---|---|---|
| L2–L4 | 8–36 | `holm`（FWER） | 确证性主假设 |
| L5 | 46 | `bh`（FDR） | 确证性 |
| L6–L8 | 82–153 | `none` | **描述性**，只在已通过的分支内 |

为什么这样成立、以及为什么门控是必需的：

* 粗层级区数少，FWER 负担得起，拿到的是强结论。Holm 一致优于 Bonferroni，没有理由
  用后者。
* 细层级几百个区，FDR 实测一个都过不了（L6/L7/L8 各 0）。当成假设生成、报原始 p +
  效应量是站得住的。
* **但 153 个区不校正意味着期望有 ~8 个纯靠运气 p<0.05。** 门控是这件事的正当性
  来源：不校正的层不是自由搜索，而是在少数几个已通过强校正的分支内部做描述。
  没开门控时 `group_stats.py` 会打 `[WARN]`。

输出里 `correction` / `m_family` / `p_adj` / `exploratory` / `gated` 五列记录了每行
走的是哪条路。**`exploratory=TRUE` 的行措辞用"提示"，不要写"显著"。**

门控在真实数据上的收敛效果（每层实际检验的行数）：
L2 558 → L3 9 → L4 10 → L5 12 → L6 19 → L7 33 → L8 41。
注意 L7 的 family 从无门控时的 98 降到 33——门控本身也在提高功效。

### `region_filter.include_ids` 是什么，为什么它是最重要的那个开关

一串 CCF structure id 的白名单，**子树自动包含**（`ontology.descendants_of`），
写 `[1089]` 就等于"海马结构及其下面所有子区"。留 `null` 表示不限制。

它之所以关键，是因为 BH 校正的严格程度只取决于**同一个 family 里检验了多少个区**：
family 有 m 个区时，最显著的那个必须 `p < 0.05/m` 才能过。这里一个 family =
一个 (class, metric, level) 组合。所以白名单不是"少看几个区"，是直接改校正阈值。

用本数据实测（GFP / Density / level 7，同一批 p 值，只改白名单）：

| `include_ids` | m | BH 阈值 | 最小实测 p | 过 FDR |
|---|---|---|---|---|
| `null`（全部） | 98 | 0.00051 | 0.00115 | **0** |
| `[688,623,549,1097,512]` 灰质五支 | 97 | 0.00052 | 0.00115 | **0** |
| `[315,1089,477]` Isocortex+海马+纹状体 | 81 | 0.00062 | 0.00115 | **0** |
| `[1089,477]` 海马+纹状体 | 17 | 0.00294 | 0.00115 | **4** |

同样的数据、同样的 p 值，白名单窄下来结论就从"什么都没有"变成"4 个区显著"。

两条实践要点：

* **粗白名单在深层级没用**。灰质五支在 level 4~5 能把 family 从 36→8、54→26
  （砍掉的几乎全是纤维束和脑室：`aco cing cpd fa fp int or st alv ec fi ml sm` …
  以及覆盖率本来就不够的中脑/脑桥），但到 level 7~8 几乎不起作用——那个深度上本来
  就全是灰质。要在皮层区/海马亚区这一层拿到 power，白名单必须窄到**十几个具体结构**。
* **必须先定后跑。** 上面那张表也是"怎么把没有的结论做出来"的示范：先看 p 值再挑
  白名单就是 p-hacking。白名单要按课题假设写死在 config 里，跑之前定。

顺带一个和白名单相关的观察：目前唯一过 FDR 的几条里，最强的是
`lateral forebrain bundle system`（纤维束）。纤维束里的"细胞密度"在生物学上不好解释，
更像归区误差；灰质白名单会把这类结果一并挡掉，这本身也是它的价值。

层级深度参考（DeMBA/CCFv3 各支深浅不一，所以按 id 圈比按 level 圈自然）：
`Cerebral cortex` L3 · `Cerebellum` L2 · `Thalamus` L4 · `Isocortex` / `海马结构` L5 ·
`Caudoputamen` L6 · `Primary motor area` L7 · `Field CA1` / 皮层分层 L8。
**注意 `levels: [2,3,4,5]` 根本够不到皮层分区**，要看皮层区得开到 6~8。

## 先跑 qc_samples.py

n=3 vs 3 时，任何把六只样本排成和分组一样顺序的干扰变量，都和真效应无法区分。
`qc_samples.py` 对每个逐样本量报告"两组是否完全不重叠"，以及同一个量是否把
config 里声明的 `batch` 分开。当前数据上它会报出两件事：

1. **未归区（background）率和分组完全分离**：对照 7.6/10.7/14.4%，实验 3.0/4.9/5.7%，
   没有重叠。**已查清，是预期行为不是混杂**：部分样本切割时越过中线，检测到了对侧
   半球的细胞，画 mask 时被刻意排除了那部分组织（`damage_labels`，见 PROGRESS_LOG
   2026-08-29）。这些细胞本来就没有图谱对应物，归到 background 是正确的，越线越多的
   样本 background 率自然越高。

   为什么不影响结论：`Percentage` / `Density` / `RegionProportion` 的分子分母都只用
   **已归区**细胞（`total_valid`，见 `cell_tables.valid_region_ids`），被排除的对侧
   细胞压根不进任何分母。唯一要留意的是 `n_cells_total` 这类原始检出数在样本间不可比，
   不要拿它当分母。
2. **`GFP_RFP` 双阳占比按（当年的）`GFP` / `3_GFP` 命名分界完全分开**（21~23% vs
   28~30%）。这个分界**不是检测批次**——见上面"关于类名里的 3"，那批命名已改掉。
   两边检测参数一致、计数准确，所以这条分离没有已知的技术解释；它也不沿着实验
   分组走（s11 是对照，s12q/s12t 是实验），所以也不是组间效应。n=3 vs 3 时一次
   完美的 3/3 分割本身就有 1/10 的概率偶然出现，目前按巧合处理，记在这里免得
   下次看到又重新惊讶一遍。

## 热图

**静态图**（`stats/plot_heatmaps.py`）：把某个 (level, 类别, 指标) 的统计量画到图谱
切面上，冠状面和矢状面各一行。

```bash
# 单张
python -m stats.plot_heatmaps --config ... --class-name GFP_any --metric Density \
       --level 5 --value hedges_g
# 所有有显著结果的组合，一次出全
python -m stats.plot_heatmaps --config ... --batch --value hedges_g
```

`--value` 可选 `log2fc` / `hedges_g` / `neglog10p` / `mean_a` / `mean_b`；
signed 的量用发散色标并以 0 为中心。默认只画 `p_adj < alpha` 的区，`--all-regions`
画出全部被检验的区。

**交互式 3D**（`../Registration_toolkit/tools/stats_view.py`，napari）：滑动查看，
右侧面板随时切 level / 类别 / 指标 / 统计量。右半球满分辨率加载 0.6 秒、重绘 0.15 秒。

```bash
cp configs/stats_view.example.yaml configs/stats_view.yaml   # 在 toolkit 里
python tools/stats_view.py
```

两边共用 `stats/region_maps.py`，所以颜色含义不会各说各话。

### 读图必须知道的三件事

1. **颜色是逐脑区的，不是逐体素的。** 一个区是一整片同色，图说明差异落在哪个解剖
   结构上，不是结构内部的分布。**这不是体素级统计图。**
2. **切 level 靠"每个体素读它在该层的祖先"**：标着 CA1（L8）的体素在 L5 图上显示
   HPF 的值。否则 L5 图只会点亮那些自身标签恰好在 L5 的零星体素。
3. **灰色 = 没有结果，不是没有差异。** 可能是没被检验、被覆盖率/最小计数过滤、
   或门控下祖先不显著所以根本没往下测。未校正层的图会在标题里标 `EXPLORATORY`。

## 输出

```
<output.dir>/
├── heatmaps/                     静态热图 PNG，按 <类别>_<指标>_L<层>_<统计量> 命名
├── methods.md                    自动生成的统计方法说明，按本次实际生效的设置
│                                 逐条写出（检验、每层校正、门控、排除、样本量
│                                 下限），也作为 Methods sheet 进 xlsx。跑完会
│                                 打印在终端上
├── qc_depth_report.md            逐层解析深度的解读文档（自动生成）
├── qc_depth_by_level.csv         每样本 × 每层：reach% / 图谱粒度停留 / 真丢失
├── qc_depth_sinks.csv            真丢失沉淀在哪些区
├── qc_per_sample.csv / qc_marker_composition.csv / qc_separation.csv
├── region_stats.csv              长表，全部结果
├── region_stats_by_level.xlsx    ReadMe + 每层一个 sheet + 分区体积
├── region_volumes.csv / .xlsx    逐区体积报告：每样本绝对 mm³ + 相对 % + 覆盖率，
│                                 加两组均值/SD。被排除的区不在其中
├── per_sample_region_direct_volumes.csv
│                                 逐区**未 rollup** 的体素数缓存。存直接值而不是
│                                 汇总值，所以改 exclude_ids 不会让缓存失效
└── per_sample/
    ├── <sample>_region_summary.xlsx   每类 × 每区的 count/density/coverage
    └── <sample>_region_tree.xlsx      树形，含 "Lost cells" 行
```

### 逐层"丢失"：ClearMap 里见过的那个现象

`python -m stats.qc_depth --config ...` 专门量这件事，并生成 `qc_depth_report.md`。

"level 3 有 1000 个细胞、level 4 只剩 800" 是**预期行为**：每个细胞只拿一个图谱标签，
不同标签处在本体树的不同深度，落在标着 `SUB` 的体素上的细胞就停在 level 7。关键是
把两种情况分开：

| | 含义 | 算丢失吗 |
|---|---|---|
| `stop_atlas_leaf` | 图谱在这里根本没有更细的标签 | **否**，已是最细 |
| `stop_unresolved` | 图谱在别处用了它的子标签，这块体素只带父标签 | **是** |

**判定依据必须是标注本身，不是本体树。** CCFv3 的树比 DeMBA P5 标注实际用到的深度更深
——PRE / POST / SUB / PAR / ProS / AON / TTv / TTd / PIR 在本体里都有子节点，标注里
子区体素数**为 0**。按树判会把这些区的细胞全算成丢失（本数据上是 12–16%），
按标注判才是真数（**3.1–4.6%**）。`qc_depth.atlas_subdivides()` 用每个样本自己的
`*_labels_in_sample.nii.gz` 来判。

当前数据的结论：真丢失 3.1–4.6%，**level 6 之后不再增加**，没有把两组分开；
数据能支撑到 level 8（各样本仍有 71–74% 细胞在场），level 9 只剩 23–28%。
最大的沉淀是 `root`（0.94%，即 CCFv3 在海马-丘脑之间没有叶子标签的那条带子，
见 PROGRESS_LOG 2026-08-28）、`OLF`、`VL`、`fa`。

`_region_tree.xlsx` 的 `Lost cells` 是"归到某个区、但没能落到它任何子区"的细胞。
它解释了为什么一个类在 level 2 看着齐全、到 level 6 就少了一截——那部分细胞在更细的
层级上根本不出现。`Summary` sheet 按层给出累计流失比例，用它来决定能往下测到第几层。
