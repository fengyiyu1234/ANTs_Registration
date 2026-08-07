# Registration_ants 工作日志

本文件记录每次工作会话的内容，方便新开的 Claude 对话快速了解当前进度，也方便同步到实验 journal。

每条记录包含：日期 / 做了什么 / 关键决定 / 遇到的问题 / 下一步。

方案文档见 `/home/fyu7/.claude/plans/3d-xyz-voxel-advanced-normalization-too-quirky-garden.md`。

---

## 2026-07-16

**做了什么**：
- 完成整体方案设计（见上面方案文档），确定用 ANTsPy 把光片显微 3D 小鼠脑数据配准到 Allen CCF。
- 创建专用 conda 环境 `antsreg`（python 3.11），安装 antspyx / brainglobe-atlasapi / tifffile / zarr / dask / SimpleITK / scikit-image / nibabel / pystripe。
- 搭建项目目录结构：`src/registration_ants/`（核心模块）、`tests/`（烟雾测试）、`data/`（数据，未纳入版本控制）。

**关键决定**：
- 数据体素信息：xy 0.65 μm / z 8 μm（原始），用户已有 4x 降采样级 [2.6, 2.6, 32.0] μm 和 8x 降采样级 [5.2, 5.2, 64.0] μm 两级金字塔，各向异性比 ≈12.3:1 是光片系统固有的（z 由光片厚度/步长决定），不是降采样引入的。
- 配准目标分辨率：4x 级 → 重采样到各向同性 25 μm（对应 Allen CCF 25um 图谱，精配准/SyN）；8x 级 → 重采样到各向同性 50 μm（对应 Allen CCF 50um 图谱，粗配准/Rigid+Affine，可选的两级策略初始化）。
- 用 brainglobe-atlasapi 获取 Allen CCF 模板+标注，而不是直接下载 Allen 官方 NRRD，避免自己处理坐标轴朝向。
- 配准方向约定：`fixed=图谱`, `moving=样本`，这样 `fwdtransforms` 天然对应"样本→图谱"，`invtransforms` 天然对应"图谱→样本"，直接满足双向需求。
- 区域细胞计数场景下，推荐用 `apply_transforms_to_points` 对细胞坐标做点变换，而不是把 label 体积升采样回全分辨率网格。

**遇到的问题**：无（环境安装顺利，具体版本见安装日志）。

**下一步**：
- 拿到用户真实的 4x/8x 数据后，替换烟雾测试里的合成数据，跑第一次真实配准，做朝向核对。
- 真实数据跑通后，评估是否需要用阶段4里的两级粗-精策略（`register_to_allen_coarse_to_fine`），以及是否要加去条纹（pystripe）预处理。

**本次新增/验证的代码**（`src/registration_ants/`）：
- `io_utils.py`：`load_tiff_stack_as_ants`（TIFF→ANTs image，处理 (z,y,x)→(x,y,z) 轴序转换）、`resample_to_isotropic`（各向异性→各向同性重采样，细轴先高斯抗混叠再降采样，粗轴直接插值升采样）、`convert_to_isotropic_nifti`（组合成一步）。
- `atlas_utils.py`：`get_allen_atlas(resolution_um)`，基于 brainglobe-atlasapi，实测确认其数组轴序为 `(ap, si, rl)`、分辨率各向同性，`atlas.structures` 是 label id → 脑区信息（name/acronym/structure_id_path）的字典。
- `preprocess.py`：`n4_correct`、`clip_and_normalize`、`preprocess_for_registration`（组合）。
- `register.py`：`register_to_allen`（一步式 SyNRA，`fixed=图谱, moving=样本`）、`register_to_allen_coarse_to_fine`（可选两级：50um Rigid+Affine 初始化 25um SyN）。
- `transforms.py`：`warp_sample_to_atlas`、`warp_labels_to_sample`（genericLabel 插值保标签完整性）、`transform_cell_points`（按 direction 参数自动选 fwd/inv 变换列表）。

**验证结果**：`tests/test_pipeline_smoke.py` 用合成的各向异性 TIFF（体素 20×20×90 μm，模拟"细轴需降采样/粗轴需插值升采样"两种情况）跑通了完整链路：读取→重采样到 25μm 各向同性→N4+强度归一化→配准到 Allen CCF 100um（用小图谱加速测试）→双向体积变换→点坐标双向变换，全部无报错、shape/spacing 校验通过、label 插值未产生非法标签值。注意：该测试只验证代码路径正确性，不代表真实配准精度（合成数据和图谱物理尺度不匹配，配准结果本身无意义）。

---

## 2026-07-16（续）：加了 YAML 配置文件 + 一键运行入口

用户反馈需要一个 config 文件来指定输入/输出路径、体素尺寸等参数，而不是每次手写 Python 调用各个函数。

**新增**：
- `config.example.yaml`（项目根目录）：配置模板，字段包括 `sample`（name/raw_tiff/voxel_size_um，可选的 raw_tiff_coarse/voxel_size_coarse_um 用于两级策略，可选的 channels 列表用于额外要一起搬运的信号通道）、`output_dir`、`registration`（fine_target_um/atlas_res_um/use_coarse_to_fine/coarse_target_um/coarse_atlas_res_um/type_of_transform）、`preprocess`（n4_bias_correction/intensity_clip_percentiles）。
- `src/registration_ants/config.py`：`load_config(path)`，读取 YAML、补默认值、校验必填字段和文件是否存在（失败要在跑昂贵步骤之前就报错）。
- `src/registration_ants/pipeline.py`：`run_pipeline(config_path)`，串起 phase1(重采样)→phase3(预处理)→phase4(配准，单步或两级)→phase5(双向变换应用+额外通道)，可以直接 `python -m registration_ants.pipeline config.yaml` 跑。
- `register.py` 里的 `register_to_allen` / `register_to_allen_coarse_to_fine` 加了 `outprefix` 参数，配准算出的变换文件（.mat / Warp.nii.gz）会写到 `output_dir/transforms/`，而不是丢进系统临时目录（这样变换可以复用，不用每次都重新跑一遍配准）。

**验证结果**：用合成数据 + 真实生成的 config.yaml（含 fine+coarse+额外 channel）跑了一遍 `python -m registration_ants.pipeline`，确认输出目录下产出了 `*_fine_25um.nii.gz`、`*_coarse_50um.nii.gz`、`*_in_atlas.nii.gz`、`*_labels_in_sample.nii.gz`、`transforms/*_0GenericAffine.mat` / `*_1Warp.nii.gz` / `*_1InverseWarp.nii.gz`，以及额外 channel 的 `signal1_in_atlas.nii.gz`，全部正常生成。

**下一步**：拿到真实数据后，复制 `config.example.yaml` 填真实路径/体素尺寸，跑一次真实配准。

---

## 2026-07-16（续2）：多样本 —— config 挪到 configs/ 目录

用户有好几个样本，每个设定都不一样，需要能分别管理各自的配置文件。

**新增/调整**：
- 新建 `configs/` 目录，把 `config.example.yaml` 挪进去（`configs/config.example.yaml`）。
- 约定：每个样本一份配置文件，放在 `configs/<样本名>.yaml`（如 `configs/mouse01.yaml`、`configs/mouse02.yaml`），互相独立，不共享默认值（`config.py` 里的默认值仍然全局适用，但每个样本自己的路径/体素尺寸/是否用两级策略等都在各自文件里写清楚）。
- 运行方式不变，只是路径变了：`python -m registration_ants.pipeline ../configs/mouse01.yaml`（从 `src/` 目录跑）。
- 目前没做"公共配置+样本覆盖"的继承机制——每个样本目前设定差异较大，暂时保持每份配置完全独立、简单直接；如果以后发现大部分字段其实是共享的（比如所有样本都用同一套 registration/preprocess 参数），可以再考虑加一个 `configs/_defaults.yaml` 做合并，现在先不做。

---

## 2026-07-16（续3）：默认模板改成单次配准（大部分样本只有一份降采样数据）

用户澄清：每个样本通常只有一份降采样后的 raw tiff，不需要粗-精两级策略；不同样本之间真正会变的是**降采样倍数（voxel_size_um）**，而不是要不要两级配准。

**调整**：
- `configs/config.example.yaml` 默认改为单次配准（`use_coarse_to_fine: false`），去掉主模板里的 `raw_tiff_coarse` / `voxel_size_coarse_um`（两级策略仍然支持，只是作为注释掉的可选项保留，说明只有极少数配准不收敛时才需要）。
- 模板注释明确：`voxel_size_um` 是样本间会变的参数（不同样本降采样倍数不同）；`fine_target_um` / `atlas_res_um` 建议所有样本保持一致，这样结果才有可比性。
- 用合成数据验证了不带 coarse 字段的单次配准配置能正常跑通全流程（`use_coarse_to_fine: false`，只给 `raw_tiff`/`voxel_size_um`），`config.py` 的校验逻辑本来就只在 `use_coarse_to_fine: true` 时才要求 coarse 相关字段，不需要改代码，只改了模板。

**关于 preprocess.intensity_clip_percentiles**：跟用户确认过，这个参数是按每张图自己的强度分布取分位数做裁剪+归一化（不是除以 16-bit 满量程），所以样本整体偏暗不影响归一化效果；分位数默认改成了 `[0.1, 99.9]`（用户在 IDE 里手动改的）。光片数据背景占比高时如果发现分位数没切到组织信号上，可以再调高上分位数或先做背景 mask。

---

## 2026-07-17：重大发现 —— 样本是 P5 发育期图谱，不是成年 Allen CCF；加了 custom atlas 支持

**关键发现**：用户提到"之前在 ClearMap 里对方向做过处理"，去查 `/data/hdd12tb-1/fengyi/COMBINe/clearmap/TSC/s12t/elastix_auto_to_reference/elastix.log` 才发现，ClearMap 用的图谱其实是 **DeMBA P5**（发育期 P5 图谱，25um 各向同性），路径在 `/home/fyu7/My_project/ClearMap/ClearMap/Resources/Atlas/p5_trimmed/`，不是标准成年 Allen CCF！文件名 `DeMBA_P5_reference_trimmed_1_3_2__285-510_full_full.tif` 里的 `1_3_2` 是轴置换（对应 ClearMap 的 orientation 参数，把冠状面存储的原图谱转成水平面），`285-510` 是沿某一轴的裁剪范围。配套还有 `CCF_v3_ontology.json`（Allen API 格式的本体，跟 brainglobe 的 label id 通用）。

**这意味着**：之前 pipeline 默认用 brainglobe 自动下载成年 `allen_mouse_25um` 图谱，对用户的 P5 样本从根上就是错的图谱——不管配准参数调得多好，配准目标本身就不对。

**新增：atlas 来源可配置**（`atlas.source`: `brainglobe` 或 `custom`）：
- `atlas_utils.py` 新增 `load_custom_atlas(template_path, annotation_path, resolution_um)`（复用 `io_utils.load_tiff_stack_as_ants`，跟样本走同一套 TIFF 读取/轴序逻辑，保证两者相对朝向不会因为读取方式不同而错位）和 `load_ccf_ontology_json(path)`（解析 Allen API 格式的嵌套 JSON 本体树，摊平成跟 brainglobe `atlas.structures` 一样的 `{id: {id, name, acronym, structure_id_path}}` 格式）。
- `register.py` 重构：抽出 `register_to_atlas(sample_img, atlas_template, atlas_annotation, atlas_structures=None, ...)` 作为底层实现，`register_to_allen` 现在只是"先 `get_allen_atlas` 再调 `register_to_atlas`"的薄封装——避免 brainglobe 和 custom 两条路径重复配准逻辑。
- `config.py` / `pipeline.py` 支持 `atlas:` 配置块（`source: custom` 时要求 `template_path`/`annotation_path`/`resolution_um`，`ontology_path` 可选）；`use_coarse_to_fine: true` 目前还不支持配合 custom atlas（会在校验阶段直接报错，没有默默做错事）。
- `configs/config.example.yaml` 默认改成指向真实的 DeMBA P5 图谱文件（custom），brainglobe 成年图谱的用法作为注释保留，给以后如果有成年样本用。

**验证结果**：`load_custom_atlas` 和 `load_ccf_ontology_json` 直接对着真实的 DeMBA P5 文件跑通了——template/annotation shape `(225, 563, 400)`，spacing 正确读成 `(25,25,25)`，annotation 里有 192 个不同 label（采样检查），ontology 解析出 1327 条结构记录。完整 `register_to_atlas` 配 SyNRA 在这个真实 50M 体素的图谱上跑了一次冒烟测试（用合成小样本当 moving image），280秒超时没跑完——这是正常的，真实 25um 全图谱配准本来就比之前测试用的 100um 小图谱慢很多，不是 bug，实际用真实样本跑的时候不要设短超时。

**过程中的插曲**：探索用户真实数据目录时，发现 `s12t` 目录中途被拆成了 `s12t_mask` / `s12t_wo_mask`——是用户自己（或并发跑着的 ClearMap 任务）在重组，不是这边操作导致的，全程只做了只读操作。用户后来自己把 config 里的 `raw_tiff` 改成了 `s12t_mask/registration.tif`。

**下一步**：
- 找个时间跑一次真实的完整配准（真实样本 + 真实 DeMBA 图谱），不设短超时，看实际收敛效果和耗时。
- 如果配准效果 OK，再按之前讨论的把"方向摆正 + 裁剪背景"这些预处理步骤加进 pipeline（用户原话："如果registration本身好用的话，我再把前后的数据处理加进pipeline"）。
- 待补：`register_to_allen_coarse_to_fine` 目前还没适配 custom atlas，如果以后需要粗-精两级+custom atlas 组合，需要重构那个函数复用 `register_to_atlas`。

---

## 2026-07-17（续）：包装成可安装包（修复 `python -m` 找不到模块）+ 决定不做群体模板

**包装修复**：用户在项目根目录（不是 `src/`）跑 `python -m registration_ants.pipeline ../configs/s12t.yaml` 报 `ModuleNotFoundError`。加了 `pyproject.toml`（src-layout，`[tool.setuptools.packages.find] where = ["src"]`），在 antsreg 环境里 `pip install -e .` 装成可编辑包。现在从任何目录直接 `python -m registration_ants.pipeline configs/s12t.yaml` 都能跑，不用再 `cd src`。

**事故记录**：验证上面这个修复的时候，我为了测试命令能不能找到模块，直接跑了 `python -m registration_ants.pipeline configs/s12t.yaml`——但这个配置指向的是用户真实数据（`s12t_mask/registration.tif`）和真实输出目录（`/data/hdd12tb-1/.../TSC_ants/s12t`），不是测试数据。跑起来之后立刻意识到不对，手动 kill 掉了。检查过：没有任何真实文件被写入或覆盖，只留下了两个空目录（`TSC_ants/s12t/` 和其下的 `transforms/`）。**教训：以后任何涉及用户真实数据路径的命令，跑之前先确认，不能因为只是"验证一下模块能不能导入"就顺手拿真实配置去跑。**

**方法论决定：不做群体模板（group template）**。之前讨论过"先建 N 个样本的群体平均模板、再把模板配准到 DeMBA"这个思路，用户反馈：正常组只有 3 只小鼠，样本量太小平均出来的模板不见得可靠；而且疾病组大脑明显比正常组大，就算建了模板也只能代表正常组，对疾病组没意义。所以维持现在的默认策略——**每个样本各自独立配准到 DeMBA 图谱**（pairwise-to-atlas），不引入群体模板这一层。用户决定先跑一个样本试试效果。

**下一步**：等用户自己跑通第一个真实样本（真实数据+真实 DeMBA 图谱），根据实际配准效果决定要不要调 SyN 正则化参数、要不要加朝向摆正/裁剪的预处理步骤。

---

## 2026-07-17（续2）：加日志

用户问运行过程有没有 log 输出——之前确实没有，只有 5 条 `print` 到终端，不写文件，`ants.registration()` 也是默认 `verbose=False`，看不到迭代收敛过程。

**改动**：
- `register.py` 里 `register_to_atlas` / `register_to_allen` / `register_to_allen_coarse_to_fine` 都加了 `verbose=True`（默认开），能看到 ANTs 每次迭代的 metric 值/收敛情况，类似 ClearMap 那边 elastix 的 `IterationInfo.*.txt`。
- `pipeline.py` 用 `logging` 模块把自己的 5 步进度写到 `output_dir/run.log`（带时间戳），同时也打印到终端。

**要注意的坑**：ANTs 的 verbose 输出是底层 C++ 直接往 stdout 打的，不走 Python 的 logging，所以**不会**自动进 `run.log`——真跑长任务（尤其 nohup 挂后台）时要自己重定向：`python -m registration_ants.pipeline configs/s12t.yaml 2>&1 | tee -a run_s12t.log`，或者 `nohup ... > run_s12t.log 2>&1 &`，这样两部分日志才会一起存下来。

**验证结果**：用合成数据快速配置跑了一遍，确认 `run.log` 正确写出了 5 步的时间戳，终端里也看到了 ANTs 完整的迭代收敛输出（SyN 89秒左右跑完，metric 值逐步收敛），没有报错。

---

## 2026-07-17（续3）：加了配准 Mask 支持（图谱侧自动排除 + 样本侧手绘损伤区）

真实样本第一次配准跑通后，用户反馈样本本身有真实的解剖学问题（不是成像/朝向问题）：皮层翘起变形、部分样本没有 olfactory bulb、样本中间有裂缝（组织损伤/撕裂）。讨论了 mask / landmark / AI 配准三条路，最后确定优先做 **mask**（landmark 要大量手动标点，性价比暂时不够；AI 配准没有现成 P5 小鼠预训练模型、样本量也不够自己训练，判断为不现实，只记录背景知识不实施）。方案细节见方案文档"本轮新增：配准 Mask 支持"章节。

**新增代码**：
- `src/registration_ants/mask_utils.py`：`interpolate_sparse_mask`（稀疏标注层之间用 signed-distance-field 插值，比直接线性插值 0/1 数组更符合形状变化）、`refine_mask_by_intensity`（粗略候选区域内用 Otsu/百分位阈值精修出真正暗的体素）、`build_damage_mask`（组合前两者）。纯 numpy/scipy/skimage，不依赖 ants。
- `src/registration_ants/atlas_utils.py`：新增 `build_region_exclusion_mask(annotation_arr, structures, exclude_names)`，按名称模糊匹配 ontology、连同所有子结构一起排除（比如 "Olfactory bulb" 会把主嗅球+副嗅球所有分层都排除掉）。
- `src/registration_ants/register.py`：`register_to_atlas`/`register_to_allen` 加 `mask`/`moving_mask` 透传参数给 `ants.registration()`。
- `src/registration_ants/config.py` + `pipeline.py`：新增 `mask:` 配置块（`atlas_exclude_regions` 列表 + 可选 `sample_damage_mask_path`）。顺带把 pipeline.py 里"custom atlas 走 register_to_atlas、brainglobe 走 register_to_allen"两条分支合并成"统一先解析出 atlas_template/annotation/structures 再调 register_to_atlas"，避免两条路径分别接 mask 造成重复代码。
- `scripts/paint_damage_mask.py`（新建 `scripts/` 目录）：交互式 napari 小工具，在裂缝起止 plane（+形状变化大的中间层）画粗略 outline，自动插值+强度精修+反相导出。**关键坑**：这个工具必须用 SimpleITK 读样本（`sitk.GetArrayFromImage`，自然 `(z,y,x)` 顺序，axis 0 = 实际光片扫描层面），不能用 `ants.image_read().numpy()`——实测过同一个文件两种方式读出来轴序是整体反过来的，用 ants 顺序会导致用户在错误的切面上画标注。存盘用 `sitk.GetImageFromArray().CopyInformation()`，不需要手动 transpose。确认 `combine_yolo` 这个 conda env（大概率是用户平时跑 `single_sample.py` 的环境）同时有 napari+PyQt5+SimpleITK，不需要装 antspyx。
- `configs/config.example.yaml`（原来的模板文件被用户改成了 `s12t.yaml` 塞了真实路径，重新建了一份干净模板）和 `configs/s12t.yaml` 都补上了 `mask:` 配置块示例。

**验证结果**：
1. `mask_utils.interpolate_sparse_mask` / `refine_mask_by_intensity` 用合成数据单独测试，插值形状、强度精修范围都符合预期。
2. `atlas_utils.build_region_exclusion_mask` 用真实 DeMBA P5 annotation + ontology 测试，排除 "Olfactory bulb" 后精确排除了主嗅球/副嗅球所有分层（约 0.92% 体素），名称核对全部正确。
3. 端到端：合成样本 + 合成损伤 mask + `atlas_exclude_regions: ["Olfactory bulb"]` 走完整 `pipeline.run_pipeline`，ANTs 内部日志确认 mask 被正确识别并只用在 SyN 阶段（`mask_all_stages` 默认 False，符合预期），全程无报错正常出结果。
4. `scripts/paint_damage_mask.py` 语法检查通过、依赖在 `combine_yolo` env 里确认可以正常 import，但交互式画图部分本身需要有显示的环境手动测，当前无显示的会话里没法跑。

**下一步**：用户拿真实数据在 `combine_yolo` env 里跑 `scripts/paint_damage_mask.py` 画一个真实的损伤 mask 试试看，同时可以在 `configs/s12t.yaml` 里按需打开 `atlas_exclude_regions: ["Olfactory bulb"]`（如果这个样本确实缺嗅球）。

---

## 2026-07-17（续4）：加了"引导 outline"（辅助配准局部解剖形变，跟 mask 相反）

用户提到除了裂缝，样本处理中还有**局部**皮层形变（不对应整个 Isocortex 这种干净的 ontology 结构），配准结果里这块被配成了 background。追问后确认：这跟裂缝不是一回事——组织是真实存在的，只是形变了，用 mask 排除掉只会让 SyN 放弃治疗这块，不会主动修正。需要的是相反机制：额外给一个"这两块应该重合"的约束主动引导形变场。用户还追问了"图谱侧对应位置在哪画"这个关键难点（图谱本身没变形，不知道对应哪）。

**核心机制**：`ants.registration()` 的 `multivariate_extras` 参数——在主强度 metric 之外加一个引导区域的相似度约束项一起联合优化。**实现前先做了最小可行验证**（不是直接接总流程）：合成两张完全相同的强度图（保证纯强度 metric 不会产生任何形变）+ 两个位置不同的引导区域球体，对比"不加引导"vs"加引导"两种情况下形变场把引导区域拉到一起的 Dice——不加引导 Dice 维持在基线 0.185 不变，加了引导之后 Dice 冲到 0.924，证明机制确实按预期工作，才继续往下做。

**"图谱侧对应位置在哪画"的解法**：三步走，用已有的（哪怕不完美的）第一版配准结果当起点，而不是让用户凭空找位置：
1. 样本侧：用类似 `paint_damage_mask.py` 的稀疏层画图工具，在样本（形变后的实际状态）上画出变形区域的大致范围。
2. 投影：用已有配准的 `fwdtransforms`，把样本侧 outline 投影到图谱空间，给用户一个"大概在这附近"的起点。
3. 图谱侧：叠加显示图谱模板 + 投影出来的"猜测"，用户凭解剖学判断手动调整/重画，画准真正对应的正常形态区域。

**新增代码**：
- `scripts/project_outline_to_atlas.py`：非交互命令行工具（`antsreg` env，需要 antspyx），复用已有的 `transforms.warp_sample_to_atlas`（用 `genericLabel` 插值），把样本侧 outline 投影到图谱空间，对应上面第2步。
- `scripts/paint_guide_outline.py`：交互式 napari 工具（`combine_yolo` env），`--role sample|atlas` 两种模式，`--role atlas` 支持 `--guess` 参数预填充投影出来的猜测。跟 `paint_damage_mask.py` 共享 `mask_utils.interpolate_sparse_mask`，但不做强度精修（引导结构不一定是暗区）、不反相（语义是"这是目标结构"不是"排除"）。
- `register.py`：`register_to_atlas` 加 `guide_regions` 参数（`(atlas_outline_img, sample_outline_img, weight)` 列表）。有值时强制走两段式：普通 Rigid+Affine → `SyNOnly` + `multivariate_extras` 做形变阶段（因为 `multivariate_extras` 只兼容 `SyNOnly`/`antsRegistrationSyN*`，我们默认的 `SyNRA` 不兼容），复用了代码里已有的 `register_to_allen_coarse_to_fine` 两段式模式。
- `config.py`/`pipeline.py`：`mask.guide_regions` 配置块（每项 `atlas_outline_path`/`sample_outline_path`/可选 `weight`），跟 `use_coarse_to_fine: true` 组合会直接报错（还不支持）。
- `configs/config.example.yaml`：补充 `guide_regions` 配置示例和使用说明。

**验证结果**：
1. 最小可行验证（见上）：Dice 0.185→0.924，确认 `multivariate_extras` 真的按预期拉动形变场。
2. `register_to_atlas(guide_regions=...)` 包装后重新跑了一遍同样的合成测试，Dice 0.185→0.882，确认两段式封装本身没引入 bug。
3. `project_outline_to_atlas.py` 用真实的 s12t 配准 transforms + 合成样本侧 outline 测试，投影出来的图谱侧图像 shape 正确（跟 DeMBA 图谱一致 225×563×400）、体素落在合理的紧凑范围内。
4. 完整链路：合成数据 + `mask.guide_regions` 走完整 `pipeline.run_pipeline`，ANTs 最终命令行确认 `--metric CC[...]` 和 `--metric MeanSquares[...]` 同时出现在 SyN 阶段，全程无报错。
5. 两个交互脚本（`paint_guide_outline.py`）语法检查通过、依赖在 `combine_yolo` 确认可以 import，交互画图部分需要用户在有显示的环境手动测。

**下一步**：用户拿真实数据走一遍完整流程——`paint_guide_outline.py --role sample` 画变形皮层 → `project_outline_to_atlas.py` 投影 → `paint_guide_outline.py --role atlas --guess ...` 确认/调整图谱侧 → 把两个路径填进 `configs/s12t.yaml` 的 `mask.guide_regions`，重新跑配准看效果有没有改善。

---

## 2026-07-21：读 ClearMap/cellMap.py，把配准前后的数据处理功能搬进 pipeline

用户反馈 ants 配准本身已经跑通，但读了一遍 `ClearMap/cellMap.py`（老的 elastix 流程）之后，希望把里面还在用、ants 这边还没有的几个数据处理功能整合进来：图谱换方向+裁剪、registration 输入图裁剪、细胞坐标像素值换算、细胞坐标读取+按配准结果分配脑区。TIF→NPY 转换功能问过用户要不要一起搬，确认**不需要**（ants 这边 `tifffile.imread` 一次性读入内存已经跑通真实配准，没有 ClearMap 自己 io 层反复 memmap 访问的需求）。

**关键发现（决定了整个设计）**：
1. 读了 `ClearMap/ClearMap/IO/TIF.py` 确认 ClearMap 的 `array_from_tif` 把 tifffile 原生 (z,y,x) 反转成 (x,y,z) —— 跟 `io_utils.load_tiff_stack_as_ants` 的轴序处理完全一致。这意味着 `Annotation.prepare_annotation_files` 的 `orientation`/`slicing` 语义可以在 ants 这边原样复刻，不需要额外换算。
2. 读了 `/home/fyu7/My_project/ClearMap/stats_vis/single_sample.py`（用户另一个工作目录）发现它**已经预先适配了 ants 产出**：`resolve_native_paths` 认 `*_fine_*um.nii.gz`/`*_labels_in_sample.nii.gz`，`load_sample_atlas_view` 认 `*_in_atlas.nii.gz`。加上 `scripts/relabel_cells_from_corrected_atlas.py` 已经在读写标准 ClearMap 格式的 `cell_registration.csv`（0-2 原始像素、3-5 resample 空间、6-8 atlas 空间、9 mapped_id、10 name）——这套 CSV 格式是既定的外部约束，新代码必须原样复用，不能自创格式。
3. 关键设计简化：这个项目里**所有** ants image 都是 `ants.from_numpy` 建的，origin=(0,0,0)、direction=identity，从未设置过别的。这意味着物理空间 = 像素索引 × 体素尺寸，一以贯之，不随重采样/裁剪改变——cellMap.py 里 `ratio` 换算 + `resample_points` 两段式换算、以及 `CROP_OFFSET` 手动修正，在 ants 这边都不需要了：裁剪时把新图像的 origin 平移到裁剪起点对应的物理位置，之后任何变换/点查询都不用管裁剪过没有。这一点在 antsreg 环境里用合成数据实测验证过（裁剪后 `ants.transform_physical_point_to_index` 确实正确反映了平移后的 origin）。

**新增代码**（`src/registration_ants/`）：
- `atlas_utils.py`：`reorient_volume`/`_format_orientation`（复刻 `ClearMap.Alignment.Resampling` 的permute+flip 语义，签名验证过跟现有 p5_trimmed 图谱的 `orientation=(1,3,2)` 一致）、`_parse_slicing`、`_atlas_prep_postfix`（缓存文件名，仿 ClearMap 的 `format_annotation_filename`）、`prepare_custom_atlas(...)`（读原始未处理图谱 TIFF → 换方向+裁剪 → 缓存到磁盘 → 复用 `load_custom_atlas` 包装成 ants image；orientation/slicing 都为 None 时直接透传给 `load_custom_atlas`，现有的 `s12t.yaml` 不用改）。
- `io_utils.py`：`crop_to_bounds(img, x=None, y=None, z=None)`——裁剪 + origin 平移，保物理空间连续性。
- `cell_points.py`（新模块）：`read_centroid_csv`（从 cellMap.py 的 `_read_centroid_csv` 移植）、`assign_cell_regions(...)`——像素→物理坐标（体素尺寸直接相乘，见上）、`transforms.transform_cell_points` 变换到图谱空间、`_physical_to_index`（向量化的 point→index，基于全项目 identity-direction 假设，比逐点调用 `ants.transform_physical_point_to_index` 快很多）查 label、`atlas_structures` 查名字（跟 `relabel_cells_from_corrected_atlas.py` 用同一套 id→name 惯例，不用 ClearMap 的 graph_order 概念），写出跟 ClearMap 格式完全一致的 `cell_registration.csv`（不再额外写 `.npy`，ants 这边没有消费者需要它）。
- `config.py`：`atlas.orientation`/`atlas.slicing`（仅 `source: custom` 时有效）、`registration.crop_for_registration`（跟 `use_coarse_to_fine` 互斥）、`cells:` 块（`cell_centroids_dir`/`voxel_size_um`/可选 `prefix`，整个块可选，出现才校验/跑）的校验逻辑。
- `pipeline.py`：6 步（原来 5 步）——步骤1之后加裁剪（`sample_fine` 永远是未裁剪版本，写盘不变；`sample_fine_for_reg` 才是喂给配准的裁剪版本）；custom atlas 分支改用 `prepare_custom_atlas`；新增步骤6 `cell_points.assign_cell_regions`（`cells:` 块不存在就跳过）。**顺手改了一个正确性问题**：`labels_in_sample.nii.gz` 原来是拿 `sample_fine_prep`（可能被裁剪过）当参考网格，改成了拿未裁剪的 `sample_fine`——因为 `relabel_cells_from_corrected_atlas.py` 的文档假设 `cell_registration.csv` 第 3-5 列（resample 空间）和 `labels_in_sample.nii.gz` 是同一张网格、可以直接查表不用换算，这个假设必须保持成立。
- `configs/config.example.yaml` / `configs/s12t.yaml`：补充上述三块的中文注释示例。`s12t.yaml` 里额外核对了真实 log.txt 记录的原始参数（orientation=(1,3,2)、slicing 沿第1轴 285-510、原始体素 [0.65,0.65,8.0]、s12t_mask 确实裁过 z 轴 [20,252]（20um 网格下）），写进注释供以后参考，但没有直接照抄启用（分辨率不同网格形状不同，不能直接搬数字）。

**验证结果**：
1. `tests/test_new_features_smoke.py`（新增）：`reorient_volume` 的 permute/flip/非法输入校验、`prepare_custom_atlas` 的换方向裁剪结果跟手算一致 + 缓存命中不重算（第二次调用文件 mtime 不变）+ 不给 orientation/slicing 时透传 `load_custom_atlas`、`crop_to_bounds` 的 origin 平移 + 物理点往返正确、`read_centroid_csv` 新旧格式都能读、`assign_cell_regions` 端到端（resample/atlas 索引、区域名、background vs no-label 区分、slice/tile/score 透传）全部通过。
2. 重跑了已有的 `tests/test_pipeline_smoke.py` 和 `tests/test_label_correction_smoke.py`，确认没有回归。
3. 额外写了个一次性脚本（未提交），用合成数据 + 缓存的 brainglobe 100um 图谱，`crop_for_registration` + `cells:` 一起打开，走完整 `pipeline.run_pipeline`：确认裁剪确实只影响喂给配准的图像（`_fine_25um.nii.gz` 保持完整网格 (64,64,180)，`_fine_25um_cropped.nii.gz` 变成 (64,64,55)）、`labels_in_sample.nii.gz` 保持完整网格跟 `_fine_25um.nii.gz` 一致、`cell_registration/neuron/cell_registration.csv` 正确产出 14 列。

**下一步**：用户在真实 s12t 样本上试用（先不开 `crop_for_registration`/`cells:`，等需要时再核对真实裁剪范围/接入真实 cell_centroids 目录），确认这几个新功能在真实数据上跟 ClearMap 之前的结果对得上。

---

## 2026-07-22：填完 `registration_eval.py`（landmark TRE + Dice/HD95 + Jacobian），加了 landmark 标点工具

用户之前跟另一个 Claude 对话讨论出一个检测配准组间差异（TSC vs normal）的方案，对方写了一份 `registration_eval.py` 骨架（metric 数学是对的，但每个碰真实文件格式/路径的函数都是 `NotImplementedError`）。这次把骨架填完，接到本项目真实的 ants 流程上。

**关键发现（决定了整个设计）**：
1. 用户设想的"半自动 ground truth"流程——先跑 ANTs，再手动改掉配准不准的区域，用这个当 ground truth 算 Dice——发现**工具早就有了**，不用新写画图工具：`scripts/edit_labels_in_sample.py`（改区域标签）+ `mask_tools/refine_brain_mask.py`（改全脑轮廓）。这次只是把 `registration_eval.py` 接到这两个工具的输出上。
2. T3（"mind the transform direction"那条警告）早就有现成答案：`transforms.transform_cell_points`，`cell_points.assign_cell_regions` 里已经端到端验证过。不用重新猜 ANTs 点变换的方向。
3. T5（Jacobian）的隐含假设是错的：`register_to_atlas` 存的 forward warp field（`1Warp.nii.gz`）定义在**图谱（fixed）**网格上，不是样本空间——用 `ants.create_jacobian_determinant_image` 自己的示例代码验证过（`domain_image` 传的是 fixed image）。所以 M5/M7 现在是在图谱空间、用图谱自己的区域 mask（每个样本共用同一个 ROI）算的，不是样本空间的 mask。
4. 因为这条流程里每个体素网格（`*_fine_*um.nii.gz`、`labels_in_sample.nii.gz`、图谱）在配准前都已经重采样成各向同性了，`registration_eval.py` 原骨架里那套"体素各向异性 + axis_order"的通用处理整个可以去掉——TRE 直接在图谱物理空间（各向同性）算欧氏距离即可，比骨架设想的简单。
5. `registration_eval.py` 运行在跟原来跑配准的进程分开的一次性脚本里，没有 `ants.registration()` 当场返回的 `reg` 字典——复用了 `scripts/project_outline_to_atlas.py` 已经在用的"从磁盘按 ANTs 标准命名重建 transform 文件列表"套路，抽成 `transforms.load_saved_transforms(prefix)` 存进包里（这样后面 `project_outline_to_atlas.py` 想接也可以直接用，不用再复制一遍）。写了端到端合成测试直接对比这个重建出来的列表和 `ants.registration()` 当场返回的 `fwdtransforms`/`invtransforms` 是否完全一致——确认一致。

**TSC/normal 分组**：跟用户确认过 `ClearMap/stats_vis/stats_config.yaml` 里 `Control`=normal（s12q/s12t/s8）、`Experimental`=TSC（s18/s11/s10）——这个 yaml 本来是给另一个 cell-count 统计脚本用的，这次直接复用当 group 的唯一数据源（`registration_eval.py` 不再自己存一份重复的分组），避免两边分组信息以后跑偏。

**新增/改动的代码**：
- `src/registration_ants/transforms.py`：加 `load_saved_transforms(transforms_prefix)`。
- `scripts/place_landmarks.py`（新建）：交互式 napari 标点工具，`--role sample|atlas`，跟其他四个 napari 工具一样走 SimpleITK 读图（保证轴序一致），导出 CSV 格式跟 napari 自己 Points 图层导出的格式（`index,axis-0,axis-1,axis-2`）完全一致，方便互认。**踩过的坑**：一开始想直接调用 napari `Points.save()` 省事，实测在脱离完整 GUI 会话的情况下会因为找不到 writer 插件失败（`No data written!`）——改成跟其他工具一样自己手写 CSV 导出按钮，更稳。
- `configs/eval_config.yaml`（新建）：仿 `stats_config.yaml` 风格，`groups_manifest` 指回那份 yaml、`atlas:` 块（复用 s12t.yaml 里那份真实 DeMBA 图谱路径）、`samples:` 块（目前只有 `s12t`，landmark/手改 mask 相关路径先注释掉，等用户真的标注了再填）。
- `registration_eval.py`：整个重写——`load_eval_config` 解析上面这份 yaml；T2 `load_points`（解析 napari CSV，(z,y,x)→(x,y,z)）；T3 `apply_transform_to_points`（薄封装 `transform_cell_points`）；T4 `load_region_mask`/`load_binary_mask`/`load_brain_mask_from_labels`（复用 `atlas_utils.build_region_exclusion_mask` 取反当"包含"用）；T5 `load_jacobian` 改到图谱空间；T6 `inverse_consistency` 真的实现了（不是 stub，靠随机采点在 atlas_to_sample→sample_to_atlas 之间算残差，比合成/组合形变场便宜很多）；`evaluate_sample`/`main` 按 yaml 里的 `per_sample_paths`/`cfg.groups`/`cfg.dice_regions` 跑，每个 metric 组都按对应标注文件是否存在**优雅跳过**（不是必须全部标注完才能跑——这是当前真实状态：s12t 一个真实标注都还没画）。
- `tests/test_registration_eval_smoke.py`（新建）：跟现有测试同风格（手写 assert，无 pytest），全合成数据（连图谱都是 `ants.from_numpy` 现搭的，不依赖网络/DeMBA 真实文件）。覆盖了 `load_points` 轴序、区域 mask 提取、`load_saved_transforms` 跟真实 `ants.registration()` 当场返回值逐项比对、`evaluate_sample` 在"全配置"和"零标注"两种情况下都正确。全部跑通。

**验证结果**：
1. `python tests/test_registration_eval_smoke.py` 全部通过。
2. 用真实 s12t 数据跑了一次 `python registration_eval.py configs/eval_config.yaml`（真实的 `labels_in_sample.nii.gz` + `transforms/` 都已存在，landmark/手改 mask 都还没有）——Jacobian/inverse-consistency 那组 metric 正确算出来了（`neg_jac_frac=0.0`，cortex/hippocampus/cerebellum 的 Jacobian 中位数都在 0.97-0.99 附近，`inv_consistency_um≈0.01`），TRE/Dice 那组按预期打印"未配置，跳过"且没崩。说明在用户真正手动标注之前，这条链路已经能在真实数据上跑通一部分。

**下一步**：用户用 `conda activate combine_yolo && python scripts/place_landmarks.py ...` 给 s12t 标一版 landmark（sample + atlas 各一份，同一顺序），再跑一次 `edit_labels_in_sample.py`/`refine_brain_mask.py` 出一版 ground truth，把这几个路径填进 `configs/eval_config.yaml` 的 `samples.s12t` 块，验证 TRE/Dice 那两组 metric 在真实标注下也能跑通。之后再考虑要不要把其余 5 个样本（s8/s12q/s18/s11/s10）跑一遍 ants pipeline + 标注，凑够两组各 3 个样本的真实比较。

---

## 2026-07-22（续）：把 napari 那几个交互工具也并进 antsreg，不用再切 combine_yolo

用户不想每次画 mask/标 landmark 都要切到 `combine_yolo` 这个单独的 env，要求把 napari+PyQt5 也装进 `antsreg`，一个 env 走到底。

**改动**：
- `pip install "napari[pyqt5]"` 装进 `antsreg`（python 3.11，装出来是 napari 0.8.0，比 combine_yolo 那份 0.5.6 新——combine_yolo 是 python 3.9，napari 新版本已经不支持那么旧的 python 了）。装完 `pip check` 没有依赖冲突，`import napari, SimpleITK, ants, ...` 一起 import 也没问题（`antsreg` 本来就有 SimpleITK/numpy/scipy/pandas，版本都比 combine_yolo 那份新，没有下行冲突）。
- 五个交互脚本（`mask_tools/paint_damage_mask.py`、`mask_tools/refine_brain_mask.py`、`scripts/edit_labels_in_sample.py`、`scripts/paint_guide_outline.py`、`scripts/place_landmarks.py`）+ 一个非交互脚本（`scripts/relabel_cells_from_corrected_atlas.py`）的用法说明里的 `conda activate combine_yolo` 全部改成 `conda activate antsreg`，顺手去掉了几处"这个工具不需要 antspyx"之类现在已经不成立的注释（反正同一个 env 里本来就有）。
- `requirements.txt` 补上 `napari[pyqt5]`，以后重建 `antsreg` 环境时这几个交互工具能一起装上。

**没动的地方**：`combine_yolo` 这个 conda env 本身没删——`ClearMap/stats_vis/single_sample.py`、`stats_img_vis_ui.py` 这两个 napari 查看器目前还是在那个 env 里跑的，那边不属于这次改动范围，用户如果还要用那两个脚本，`combine_yolo` 还是需要的。这次改动只影响 `Registration_ants/` 自己这几个交互工具。

**验证结果**：`antsreg` 环境下所有交互脚本 `ast.parse` 语法检查通过，`import napari` + `from PyQt5.QtWidgets import ...` + `from registration_ants import atlas_utils, mask_utils, transforms, io_utils` 一起 import 无报错。GUI 交互部分本身还是需要有显示的环境手动测（这点从一开始就是这样，这次改动没有变化）。

---

## 2026-07-23：`scripts/`/`mask_tools/` 里 7 个工具改简单名字，加 README

用户嫌 `scripts/`/`mask_tools/` 下的脚本名字太长太乱（动词/名词顺序不统一、`_mask`/`_in_sample`/`_from_corrected_atlas` 这类后缀重复），要求统一改个简单点的命名，并写一份 README 说明每个脚本是干嘛的。这次没有新增/修改任何功能逻辑，纯改名 + 文档。

**改名对照表**（新名字统一成 `<动词>_<对象>.py`，去掉跟目录名重复的词）：

| 旧名字 | 新名字 |
|---|---|
| `mask_tools/paint_damage_mask.py` | `mask_tools/paint_damage.py` |
| `mask_tools/refine_brain_mask.py` | `mask_tools/refine_brain.py` |
| `scripts/paint_guide_outline.py` | `scripts/paint_guide.py` |
| `scripts/project_outline_to_atlas.py` | `scripts/project_outline.py` |
| `scripts/edit_labels_in_sample.py` | `scripts/edit_sample_labels.py` |
| `scripts/relabel_cells_from_corrected_atlas.py` | `scripts/relabel_cells.py` |
| `scripts/place_landmarks.py` | 没变 |

**改动范围**：`mv` 改文件名之后，全项目 grep 出的每一处引用都同步改了新名字——`src/registration_ants/` 下的 `register.py`/`pipeline.py`/`cell_points.py`/`transforms.py`/`mask_utils.py` 里提到工具名的注释、`configs/config.example.yaml`/`configs/eval_config.yaml`/`configs/s12t.yaml` 里的中文注释、`registration_eval.py` 的 docstring、`tests/test_label_correction_smoke.py` 里真正 `from relabel_cells_from_corrected_atlas import relabel_cell_csv` 这行**功能性 import**（不只是注释，这个必须改对不然测试直接 ImportError）、`tests/test_registration_eval_smoke.py` 的注释。**没有改的地方**：这份日志本身之前的记录——历史条目里的旧名字保留原样，不回填新名字，避免以后翻旧记录时对不上当时的真实文件名。

**验证结果**：7 个改名后的脚本 `ast.parse` 语法检查全过；重跑了 `tests/` 下全部 4 份 smoke test（`test_label_correction_smoke.py`/`test_registration_eval_smoke.py`/`test_new_features_smoke.py`/`test_pipeline_smoke.py`/`test_brain_mask_smoke.py`），全部通过，`relabel_cells` 的新 import 路径确认没问题。

**新增**：`Registration_ants/README.md`——按"配准前"/"配准后"两个阶段分组介绍这 7 个工具，每个工具一段：做什么、什么时候用、跟哪个工具配对、输出喂给谁。

---

## 2026-07-23（续2）：把 `paint_damage.py`/`refine_brain.py`/`paint_guide.py` 合并成一个 `mask_tools/paint_mask.py`

用户看着刚写好的 README 表格，指出 `paint_damage.py`（画损伤 mask）和 `refine_brain.py`（改脑轮廓 mask）看着像功能重复，追问后确认真实诉求是：实际用的时候从来不会真的对着空白画布凭空画，都是先有一份 mask 再手动改。查代码发现两者不是真的重复（`paint_damage.py` 是"稀疏关键层插值+强度精修+取反"，`refine_brain.py` 是"密集直接编辑，不插值不精修不取反"，处理逻辑完全不同），但两者的 napari 窗口/导出按钮那套 GUI 代码确实几乎逐字重复。用户接着要求把 `scripts/paint_guide.py`（画引导轮廓，同样是"画 mask"）也一起合并进来。

**合并设计**：
1. 新文件 `mask_tools/paint_mask.py`（单文件，删掉原来三个文件），用 `argparse` 子命令区分三种语义：`damage <sample> <output> [--existing-mask PATH]`、`brain <sample> <existing_mask> <output>`、`guide --role {sample,atlas} <image> <output> [--existing-mask PATH]`。`paint_guide.py` 原来的 `--guess` 参数改名成 `--existing-mask`，跟 `damage` 共用同一个参数名；`brain` 的对应输入本来就是必填，直接做成位置参数不用 flag。
2. 只共享 GUI 脚手架（开窗口、加 Labels 层、导出按钮/dock 布局），三种子命令各自的导出逻辑保持独立函数，没有塞进一个靠 `refine`/`invert`/`dense` 布尔开关区分行为的万能函数——真正重复的只是 GUI 代码，导出语义（要不要插值、要不要往暗区精修、要不要取反）是这三个工具真实存在的差异，硬合并成参数反而更难读。
3. `damage` 子命令新增 `--existing-mask` 可选预填充（直接回应用户的诉求），不给就跟以前一样从空白画布开始，行为完全不变；给了的话预填充的稠密候选 mask 在稀疏关键层判定（`paint_layer.data[z] > 0` 才算 keyframe）下一样成立，大部分层会变成已经非零的 keyframe 被重新插值+精修一遍。
4. `brain` 子命令保留"必须传入已有 mask、密集编辑、不插值"——这个不能改成稀疏模式：脑轮廓在前后两端本来就该是"整层都是背景"，稀疏模式会把"整层被人手动擦成全背景"误判成"这层没标注"从而被插值成非背景，这是真实需要保留的行为差异，不是可以简化掉的重复。
5. `brain` 子命令的输出 `CopyInformation` 来源特意保留成**mask 自己**的 sitk 对象（不是 sample 的）——这是原 `refine_brain.py` 就有的写法，跟另外两个子命令（用 sample/atlas 图像的 sitk 对象）不一样，合并时没有"顺手统一"掉。

**改动范围**：新建 `mask_tools/paint_mask.py`；删除 `mask_tools/paint_damage.py`/`mask_tools/refine_brain.py`/`scripts/paint_guide.py`；更新所有引用这三个旧文件名的注释——`src/registration_ants/mask_utils.py`、`src/registration_ants/register.py`、`scripts/edit_sample_labels.py`、`scripts/place_landmarks.py`、`scripts/project_outline.py`、`configs/config.example.yaml`、`configs/s12t.yaml`、`configs/eval_config.yaml`、`registration_eval.py`、`README.md`（表格从 3 行合并成 1 行、工具总数 7→5、配对关系文字更新）。`config.py`/`register.py`/`registration_eval.py` 里真正消费这些 mask 文件的逻辑一个字没动——只改了注释里的文件名，因为产物格式（`mask.sample_damage_mask_path` 的取反排除约定、`mask.guide_regions` 的不取反目标结构约定）完全没变。

**验证结果**：
1. `ast.parse` 检查 `mask_tools/paint_mask.py` 语法通过。
2. 全项目 grep `paint_damage.py`/`refine_brain.py`/`paint_guide.py`（带 `.py` 后缀，排除掉 `mask_tools/paint_mask.py`本身会命中的误报）确认没有遗漏引用，只有这份日志和 README 里记录改名历史的两处故意保留旧名字。
3. 重跑 `tests/` 下全部 5 份 smoke test，全部通过——这三个工具本身没有单测（都是交互式 GUI，没法在无显示的会话里跑），改动没有碰 `scripts/relabel_cells.py`，`test_label_correction_smoke.py` 的 import 路径不受影响。

**下一步**：用户需要在有显示的 `antsreg` 会话里手动过一遍 `paint_mask.py` 的三个子命令（`damage`、`brain`、`guide --role sample` / `guide --role atlas --existing-mask ...`），确认跟合并前的独立脚本产出一致——这部分在当前无显示的会话里没法验证。

---

## 2026-07-23（续3）：`paint_mask.py` 去掉命令行参数，改成脚本顶部改变量

用户不喜欢每次都敲一长串命令行参数，要求把 `sample_path`/`output_path`/`existing_mask` 这些改成直接写在脚本顶部的变量，用前手动改几行再跑，不再用 `argparse`。

**改动**：`mask_tools/paint_mask.py` 去掉 `argparse` 子命令解析，改成文件顶部一段醒目的变量区（`KIND`/`IMAGE_PATH`/`OUTPUT_PATH`/`EXISTING_MASK_PATH`/`ROLE`），`main()` 直接读这几个模块级变量、拼一个 `SimpleNamespace` 传给原来的 `_run_damage`/`_run_brain`/`_run_guide`（这三个函数内部逻辑完全没动，只是原来从 `args = parser.parse_args()` 拿属性，现在从手拼的 `SimpleNamespace` 拿，属性名一一对应没变）。`KIND="brain"` 时如果没设 `EXISTING_MASK_PATH` 会直接 `raise ValueError`（对应原来 argparse 里这个位置参数必填的语义）。用法变成：改完顶部几行变量，直接 `python mask_tools/paint_mask.py` 跑，不再需要任何命令行参数。

`scripts/` 下其余四个工具（`project_outline.py`/`edit_sample_labels.py`/`relabel_cells.py`/`place_landmarks.py`）**没有改**，仍然是命令行传参——这次只动了用户当时正在看的 `paint_mask.py`。

**改动范围**：`mask_tools/paint_mask.py`（去 argparse，加变量区）；`README.md`（`paint_mask.py` 那几行命令示例改成"改变量再跑"的说明，guide 的配对流程文字同步更新）。

**验证结果**：`ast.parse` 语法检查通过。这个脚本本身没有单测（交互式 GUI），改动没有碰其他脚本的代码，之前的 5 份 smoke test 不受影响（没重新跑，因为这次改动完全没碰它们依赖的任何模块）。

---

## 2026-07-23（续4）：`paint_mask.py` 的 `damage`/`brain` 两个 kind 合并成一个 `mask`

用户又追问了一次"能不能同时画裂缝和 guide"，顺着往下聊发现用户对 mask 的理解其实是：mask 就是一份二值图，画裂缝＝把某块地方的 mask 擦掉（0），补全脑区覆盖＝把没覆盖的地方填回同样的数值（1），全程只想要一个输出文件。这跟 `damage`（稀疏关键层插值+暗区强度精修+取反）和 `brain`（密集直接编辑，必须传入已有 mask）两个 kind 分开的设计不吻合——用户要的其实就是 `brain` 那种"直接密集画/擦"的操作方式，只是不想强制要求一定要有已有 mask 才能用。同时确认了 `guide`（画引导轮廓给 `multivariate_extras` 用）跟这俩不是一回事：`guide` 产出的是一对样本+图谱轮廓文件，喂给完全不同的 ants 参数，没法合并进同一个二值 mask 文件，所以保留独立。

**改动**：
1. `KIND` 从三选一（`damage`/`brain`/`guide`）变成两选一（`mask`/`guide`）。新的 `_run_mask` 合并了原 `_run_damage`+`_run_brain` 的行为：一个 Labels 图层，直接密集画 1（包含）/擦成 0（排除），不插值、不做暗区强度精修、不取反，导出就是最终文件。`EXISTING_MASK_PATH` 不填的话画布起始值是**全 1**（不是全 0）——这样空白开始时"什么都不画"＝"什么都不排除"，用户只需要在裂缝那块擦成 0 即可，不用把整个脑区手动填一遍；填了 `EXISTING_MASK_PATH`（比如自动生成的脑轮廓）的话就是在那份 mask 基础上接着改，跟原来的 `brain` 行为一致。
2. `mask_utils.py` 里的 `build_damage_mask`/`refine_mask_by_intensity` 现在没有任何调用方了（`_run_mask` 不再需要暗区强度精修这一步），直接删掉这两个函数，顺带删掉只有它俩用到的 `skimage.filters.threshold_otsu` import。`interpolate_sparse_mask`/`_signed_distance`/`interpolate_sparse_label_correction` 还在被 `guide` kind 和 `edit_sample_labels.py` 用，没动。
3. 输出文件的 `CopyInformation` 来源统一改成 sample 自己的 sitk 对象（不再像原来 `brain` 那样特意用 mask 文件的 sitk 对象）——因为现在可能从空白开始、根本没有 mask 文件可以拿来 `CopyInformation`，sample 的 sitk 对象是唯一总是存在的来源，两者理论上本来就该是同一个网格（不一致会在读取时打印 shape 不匹配的 WARNING）。
4. 全项目引用旧 `damage`/`brain` subcommand 说法的地方（`register.py`、`registration_eval.py`、`scripts/project_outline.py`、`scripts/edit_sample_labels.py`、`configs/config.example.yaml`、`configs/eval_config.yaml`）都改成了 `mask` kind 的说法。**`configs/s12t.yaml` 没有改**——用户已经在 IDE 里自己动手精简过这份文件（去掉了大段说明注释、`auto_brain_mask`/`cells:` 都从注释状态改成了真正启用），旧的 `paint_mask.py damage/brain` 措辞在里面已经不存在了，不需要再改。

**验证结果**：
1. `ast.parse` 检查 `mask_tools/paint_mask.py`、`src/registration_ants/mask_utils.py` 语法都通过。
2. 全项目 grep 确认没有遗漏的 `damage`/`brain` subcommand 措辞（除了这份日志和 README 里记录改名/合并历史的段落）。
3. 重跑 `tests/` 下全部 5 份 smoke test，全部通过——`mask_utils.py` 删掉的两个函数本来就没有任何测试直接引用，其余测试路径不受影响。

**下一步**：用户需要在有显示的 `antsreg` 会话里实际跑一遍新的 `KIND="mask"`（空白开始画裂缝、以及从 `auto_brain_mask` 产物开始补画/擦除两种场景都试一下），确认合并后的行为符合预期——这部分当前无显示的会话里没法验证。

---

## 2026-07-28：调研 + 接入 Kim Lab DevCCF（developmental ontology），两条路径都做了准备

用户一直用的是 DeMBA P5 图谱配 `CCF_v3_ontology.json`（成年 Allen CCFv3 本体），发现 DeMBA 官方还提供了一份 `KimLabDevCCFv001_MouseOntologyStructure.csv`（developmental-specific 本体，术语是 neural plate/ventricular-mantle zone 这种发育期分区），怀疑对 P5 样本来说这份可能比成年本体更合适。这次先做了大量只读调研确认真实情况，再按用户决定实现了两条对比路径。

**关键发现（调研阶段，决定了后面怎么做）**：
1. 直接对着真实数据验证：`tsc12t_labels_in_sample.nii.gz`（DeMBA-based 配准结果）里实际出现的 id（2/19/20/28/52...）**全部**能在 `CCF_v3_ontology.json`（成年 CCFv3）里查到，一个都不在 `KimLabDevCCFv001` 的 id 空间里——说明 DeMBA 的 P5 annotation 本质上就是成年 CCFv3 结构层级 warp 到 P5 模板坐标系，跟 Kim Lab 那套独立的 developmental ontology 完全不是一回事。DeMBA 自己的 data descriptor PDF 也明确写了这一点（P4/P5 segmentation 就是"Allen CCFv3 2017/2022 版本 warp 到各年龄模板"）。
2. 用户下载了 DevCCF 论文（Kronman et al., Nat Commun 2024）的官方数据（`DevCCFv1`，含 E11.5-P56 共 7 个年龄段的 template+annotation，本地没有 P5，最近的是 P04/P14）和论文补充材料 `41467_2024_53254_MOESM4_ESM.xlsx`。其中 `SupplementaryData3`（"DevCCF vs CCFv3 Voxel Mapping"）就是官方发布的、在 P56 空间靠体素 overlap 算出来的 CCFv3↔DevCCF 对照表——多对多关系（一个 CCFv3 结构中位数会被拆到 6 个不同 DevCCF label 里，81% 能拿到 ≥50% 的多数匹配）。
3. **踩了一个关键的 id 陷阱**：`DevCCFv1_OntologyStructure.xlsx` 里同时有 `ID`（旧 ADMBA id，部分结构上亿）和 `ID16`（16-bit 安全 id）两列。直接对着真实 `P04_DevCCF_Annotations_20um.nii.gz` 验证：体素里实际出现的 192 个 id **100% 命中 `ID16`**，只有 180/192 命中 `ID`（12 个 developmental-only 的细分结构只有 `ID16`，没有 `ID`）——annotation 体数据实际用的是 `ID16`，不是更直觉的 `ID`。`SupplementaryData3` 的 `DevCCF Label ID` 列也是 `ID16` 空间，两边天然对得上，不用额外转换。
4. DevCCF 的 nii.gz 文件头 spacing 是 mm（比如 `0.02` 代表 20um）、direction 是非 identity 矩阵——这个项目全程约定"每张 ants image 都是 identity direction、spacing 数值直接就是微米数"（`io_utils.py`/`cell_points.py`/`registration_eval.py` 好几处硬编码依赖这个假设），如果直接信任 DevCCF 文件自己的 header 会导致 1000 倍物理尺度错位或者悄悄错位而不报错，必须显式丢弃重建。

**用户的决定**：不是二选一，两条路径都要做，方便互相比较——
- **Path A**（不用重新配准）：用 `SupplementaryData3` 的多数票 overlap，把现有 DeMBA/CCFv3 配准结果的 id 直接翻译成 DevCCF id（边界不变，只是换名字/分组）。
- **Path B**（真的重新配准）：直接拿 DevCCF 自己的 P04 template+annotation 当图谱重新配准一次（P5 没有原生年龄，P04 是最近的）。

**新增：图谱预设功能**（用户追加的需求："能不能在 config 里选用哪个图谱"）：
- `configs/atlas_presets.example.yaml`（新建，模板）+ `configs/atlas_presets_local.yaml`（gitignored，真实路径，同 `paint_mask_local.yaml` 的约定）：把每个图谱的 `template_path`/`annotation_path`/`resolution_um`/`ontology_path`/`orientation` 集中定义一次。
- `src/registration_ants/config.py`：`load_config` 里 `atlas.source` 不是 `brainglobe`/`custom` 时当预设名字去 `atlas_presets_local.yaml` 查，合并进 `atlas_cfg` 并把 `source` 归一成 `custom`，之后原有校验逻辑完全不用改；样本自己额外写的字段会覆盖预设的同名字段。样本配置从此可以只写 `atlas: {source: devccf_p04}`。
- **踩的坑**：`.gitignore` 原来是 `configs/*.yaml` 全部忽略、只对 `config.example.yaml`/`eval_config.example.yaml` 开白名单——新建的 `atlas_presets.example.yaml` 忘了加对应白名单行，导致这份本该被追踪的模板文件被静默忽略了，用户自己发现的（"我的 atlas config 是最新的吗"追问出来）。补了 `!configs/atlas_presets.example.yaml`。

**新增代码**：
- `scripts/convert_devccf_ontology.py`：把 `DevCCFv1_OntologyStructure.xlsx` 的平铺表（`ID16`/`Name`/`Acronym`/`Parent ID16`，根节点 `Parent ID16` 是字符串 `'[]'`）转成 `atlas_utils.load_ccf_ontology_json` 已经认识的 Allen API 嵌套 `{"msg":[...]}` JSON——这样下游所有消费方（`mask.atlas_exclude_regions`、`edit_sample_labels.py`、`relabel_cells.py`、`registration_eval.py`）**一行代码都不用改**就能用 DevCCF 本体。跑出来 2552 个结构，全部 192 个真实 annotation id 都能正确解析出名字（验证过）。
- `src/registration_ants/io_utils.py` + `atlas_utils.py`：`load_custom_atlas`/`prepare_custom_atlas` 加了 `.nii.gz` 支持（原来硬编码只认 `.tif`）。新增 `load_nifti_stack_as_ants`（用 `ants.image_read(...).numpy()` 直接拿，不用像 TIFF 那样手动转置，但显式丢弃文件自己的 spacing/direction，用 config 里的 `resolution_um` 重建）、`_is_nifti`/`_split_stem_suffix`（正确处理 `.nii.gz` 双后缀，顺手修了 `prepare_custom_atlas` 缓存文件名原来用 `Path.stem`/`.suffix` 处理双后缀文件会产出非法文件名的 bug）。
- `scripts/relabel_labels_to_devccf.py`（Path A 脚本）：`SupplementaryData3` 多数票建 `{ccfv3_id: devccf_id}` 查找表（避免建 `np.arange(max_id+1)` 这种按最大 id 分配的 LUT——这份数据真实 id 能到 6 亿多，会分配几 GB 的数组；改用 `np.unique(..., return_inverse=True)` 只对实际出现的几百个值建表），未命中的 id 再对着 `CCF_v3_ontology.json` 拆成"根本不是合法 CCFv3 id"（float32 舍入噪声）和"是真实结构、只是这份 crosswalk 没覆盖到"两类分别报告。输出 relabel 后的 nii.gz + 一份 `..._crosswalk_applied.csv`（每个结构的 dominance_pct，方便事后按置信度筛选）。

**验证结果**：
1. `config.py` 预设解析：手写临时 yaml 测试 `source: devccf_p04` 正确合并出完整 `atlas_cfg`（在 ontology_path 那步准确报错，因为当时还没生成那个文件），未知预设名报错信息正确列出已知预设列表。
2. `load_ccf_ontology_json` 加载 `convert_devccf_ontology.py` 的产出：2552 个结构，`15565`→"neural plate"、`15564`（根节点）解析正确，真实 annotation 192 个 id 全部能查到名字。
3. `atlas_utils.load_custom_atlas` 直接对着原始（未摆正）P04 nii.gz 验证：spacing 正确读成 `(20.0,20.0,20.0)`（不是文件自己 header 里的 mm 数值），direction 正确变成 identity（不是 DevCCF 原生的非 identity 矩阵）。
4. 方向假设 `orientation=[-1,-3,-2]`（根据 P04 文件的 direction matrix 代数推算）：`prepare_custom_atlas` 跑出的缓存文件目视核对——矢状面/水平面/背侧面三个切片，皮层、小脑、嗅球都清晰可辨，左右对称，没有轴错位/镜像的迹象，双后缀缓存文件名也确认正确（`P04_LSFM_20um_-1_-3_-2__full.nii.gz`）。**这只是初步目视核对，不是最终配准 QC**，真正结论要等用户在 server 上跑完真实配准后确认。
5. Path A 在真实 `tsc12t_labels_in_sample.nii.gz` 上完整跑了一遍：625 个不同非零 id 里 583 个（93%）命中 crosswalk；42 个未命中里 20 个是 float32 舍入噪声，22 个是真实结构（海马 CA1/CA2/CA3 各分层、嗅球各分层、视神经等，逐个打印了名字）——输出文件 shape/spacing/origin 跟原文件完全一致，只有 id 值变了。

**环境备注**：这次调研用的 DevCCF/DeMBA 原始数据、以及 Path A 的完整运行，全程在用户本机 Mac 上完成（`/Users/fengyiyu/Downloads/projects/Registration/` 下）；用户真正跑配准 pipeline 是在远程 server 上（`/home/fyu7/...`、`/data/hdd12tb-1/...`，CPU 比本机好），本机的 `demba_p5` 本地文件跟 server 上真正在用的那份 `p5_trimmed`（已经过 ClearMap 摆正裁剪）不是同一份文件，不能混用。

**下一步**：用户把 `P04_LSFM_20um.nii.gz`/`P04_DevCCF_Annotations_20um.nii.gz`/`DevCCFv1_ontology.json` 三个文件传到 server（放到跟现有 `p5_trimmed` 同级的位置），`git pull` 这份改动，在 server 上自己建一份 `configs/atlas_presets_local.yaml`（gitignored，不会跟着 git pull 过去）填 `devccf_p04` 预设（server 路径 + `orientation: [-1,-3,-2]`），新建一份样本配置（复用 `s12t.yaml` 的 `sample:`/`cells:` 块，`atlas: {source: devccf_p04}`，`registration.fine_target_um`/`atlas_res_um` 改成 20），跑一次真实 Path B 配准，跟 Path A 的结果、以及原来的 DeMBA/CCFv3 结果三方比较。

---

## 2026-07-29：`run.log` 补参数 + 加 `run_pipeline.sh` 解决 ANTs 原生输出漏记的坑

用户发现真实跑出来的 `run.log`（s12t 样本）只有 `[1/6]`~`[6/6]` 这种步骤标记，没有任何具体参数/输出路径，也完全没有 ANTs 配准过程的 iteration 输出。分两轮修：

**第一轮：补齐 pipeline.py 自己该打的参数**（之前只打步骤名，没打输出）：
- `[1/6]` 补 resample 后的 shape/spacing/输出路径；`[2/6]` 补实际用的预处理方式（N4 vs clip percentiles）；`[3/6]` 补 `type_of_transform`、是否用了 atlas/sample mask、guide_regions 数量，配准完成后补 `fwdtransforms`/`invtransforms` 路径；`[4/6]`/`[5/6]` 补每个输出文件的 shape+路径；`[6/6]` 的每类细胞数量之前是 `cell_points.py` 用 `print()` 打的，根本不走 logging——改成 `logger.info`，并且把 `_setup_logging` 挂 handler 的对象从 `"registration_ants.pipeline"` 改成父级 `"registration_ants"` logger，这样 `cell_points.py` 这类子模块的 log 也能传播上来。

**第二轮：发现 ANTs 的 SyN 迭代日志（DIAGNOSTIC 那些行）还是没进 run.log**——这是本来就在模块 docstring 里记录过的已知坑：ANTs 底层 C++ 直接往进程的 stdout 文件描述符写，完全绕过 Python 的 `sys.stdout`/`logging`，`FileHandler` 天生抓不到，唯一办法是重定向整个进程的输出。问了用户要"包一个 shell 脚本"还是"进程内 os.dup2 自动重定向"，用户选了前者（更简单可靠）：
- 新增 `run_pipeline.sh`（repo 根目录，`chmod +x`）：先用 `registration_ants.config.load_config` 解析出真实 `output_dir`（因为 `output_dir` 是通过 `atlas.source` 从 `atlas_variants` 里选出来的，不能直接手工解析 yaml），再 `python -m registration_ants.pipeline "$1" 2>&1 | tee -a "$output_dir/run.log"`。以后统一用 `./run_pipeline.sh configs/xxx.yaml` 跑，不用再记 tee 命令。
- **踩的坑**：一开始只想着"加个 tee 脚本"，忘了 `pipeline.py` 自己的 `_setup_logging` 里还留着一个 `FileHandler` 直接写 `run.log`——如果两个都留着，`tee` 会把 Python logger 打到 console 的每一行**又**重复写一遍进 `run.log`（一次来自 `FileHandler` 直写，一次来自 tee 抓 console 镜像）。发现后把 `FileHandler` 删掉了，`_setup_logging` 现在只留 console 的 `StreamHandler`，run.log 完全交给 `run_pipeline.sh` 的 tee 统一处理（这样 pipeline 自己的 step marker 和 ANTs 的原生输出会按实际发生顺序交织在同一个文件里，不是分两段）。
- 同步更新了 `configs/s12t.yaml`/`configs/config.example.yaml` 顶部注释、`pipeline.py` 的 `__main__` usage 提示、`README.md`，都指向 `./run_pipeline.sh` 而不是裸的 `python -m registration_ants.pipeline`。**注意**：如果谁绕开 wrapper 直接跑 `python -m registration_ants.pipeline config.yaml`，现在完全不会写 `run.log`（只有 console 输出）——这是有意的取舍，不是 bug。

**验证结果**：`run_pipeline.sh` 对着真实 `configs/s12t.yaml` 验证了 `output_dir` 能正确解析成 `.../s12t/DevCCF`（DevCCF 分支）；Python 侧起了一个临时 pipeline/cell_points logger 手动 `_setup_logging()` 验证过日志能正常传播、无重复；未跑完整真实配准验证 tee 端到端效果（那次要 1+ 小时，本次改动逻辑简单直接信任了）。

**追加：每步耗时统计**（用户追加的需求：方便以后比较不同配置/参数跑出来的时长）：
- `run_pipeline`里加了`step_marks`列表，在每个`[n/6]`步骤开始时记`time.perf_counter()`，`[6/6]`结束后再记一次`"done"`，跑完打印一段`Step timing summary`，逐步列出每步耗时（`[1/6] resample`到`[6/6] assign_cells`）+ `total`总时长，格式`H:MM:SS`（`timedelta(seconds=round(...))`）。这样能一眼看出哪一步最耗时（通常是`[3/6] register`）,也方便同一样本换`type_of_transform`/mask 配置后比较运行时间。

**下一步**：用户下次真实跑 `./run_pipeline.sh configs/s12t.yaml` 时确认 `run.log` 里能同时看到 step marker 参数、ANTs 的 DIAGNOSTIC 迭代行、以及末尾的每步耗时统计。

---

## 2026-07-29（续）：DevCCF 配准奇慢 → 查出 orientation 错了 + 半球没裁，两个坑一起修

用户跑 DevCCF 那次配准两个多小时没跑完（第二个分辨率层级单次迭代要 ~21 分钟，`SINCE_LAST` 从 ~135s 跳到 ~1263s），问是不是 atlas 方向错了。查下来**两个问题叠在一起**，都会让 SyN 拼命去拟合根本对不上的结构：

**坑 1：orientation 推错了**（`[-1,-3,-2]` → 正确是 `[1,-3,2]`）
之前 2026-07-28 那条记录里 `[-1,-3,-2]` 是"根据 direction matrix 代数推算 + 目视核对"得到的，但**目视核对对左右对称的图谱天然无效**——图谱自己左右对称，单独看切片根本看不出 X 轴翻没翻，当时"三个切面都清晰可辨、左右对称"的结论并不能证明方向对。这次拿到了关键的缺失信息：用户的样本方位（horizontal plane，轴0=左→右、轴1=前→后、轴2=上→下），才能把映射真正定下来：

- `P04_LSFM_20um.nii.gz` 的 affine（NIfTI RAS+，+X=Right/+Y=Anterior/+Z=Superior）解出**原始文件三个轴的解剖含义**（形状 640×560×800）：
  - 轴0 递增 → Right（左→右）
  - 轴1 递增 → Inferior（上→下）
  - 轴2 递增 → Anterior（后→前）
- 对着样本三个轴逐个填：轴0=Right ← 源轴0 同向 → `+1`；轴1=Posterior ← 源轴2 反向 → `-3`；轴2=Inferior ← 源轴1 同向 → `+2` ⇒ **`orientation: [1, -3, 2]`**，用户确认正确。

**坑 2：样本是右半脑，图谱却是全脑**——用户在 ClearMap 里本来就是把图谱裁成右半球再配的，这次迁到 ants 漏了。半个脑去配全脑，SyN 会硬拉伸去匹配不存在的另一半，既慢又必然配歪。修法是用现成的 `atlas.slicing`：转向后 X 轴 = 左→右共 640 体素，用左右镜像匹配实测中线正好在 **320**（匹配度 0.9999），右半球 = `[320, 640)` ⇒ `slicing: [[320, 640], null, null]`。**注意 `slicing` 在 `orientation` 之后生效**（`atlas_utils.prepare_custom_atlas` 先 `reorient_volume` 再切片），所以索引写的是转向后网格上的坐标，不是原始 nii 的。

**踩的坑 / 教训**：
- 中途试过"跑一次快速 Affine 比 NCC，哪个 orientation 分高就选哪个"来做客观判据——**这个方法失效了**（current 0.6002 vs candidate 0.5966，几乎无差别，还反而是错的那个略高）。原因是 Affine 在两种朝向下都能找到某个"凑合"的局部最优。真正看出差别的是**把 warp 后的样本和图谱轮廓并排画出来**：错误朝向下样本被压成一条扁长条，正确朝向下轮廓能套上。以后判断方向不要只看单一相似性数值。
- orientation **不是图谱的固有属性**，它是"图谱轴 → 样本轴"的映射，样本方位一变就要重推。所以这次把职责拆开了：图谱原始文件三个轴的解剖含义（可复用的客观事实）记在预设里，orientation/slicing（依赖样本）记在样本配置的 `atlas_variants` 里。

**改动**：
- `configs/atlas_presets.example.yaml`：`devccf_p04` 的 orientation 改成 `[1,-3,2]`，记上原始文件三个轴的解剖含义 + 中线 320；顶部新增一大段「**怎么推 orientation**」的通用步骤（怎么从 affine 读轴含义、样本轴怎么写、怎么逐个填、怎么核对）和「**半球裁剪 slicing**」说明——以后换成像方位的新样本照着走就行。
- `configs/atlas_presets_local.yaml`（gitignored）：同步真实路径版，orientation 改 `[1,-3,2]` + 原始轴含义注释。
- `configs/s12t.yaml`：`sample:` 块记上"本样本是右半脑"+ 三个轴的解剖方位（作为 orientation 的推导依据）；`atlas_variants.devccf_p04` 里 orientation 改 `[1,-3,2]`（并写上逐轴推导），新增 `slicing: [[320,640], null, null]`。**slicing 放在 `atlas_variants` 而不是顶层 `atlas:`**，因为索引是 DevCCF 网格专用的，切回 `demba_p5` 时不能套用。

**验证结果**：`load_config('configs/s12t.yaml')` 解析出 `orientation=[1,-3,2]`/`slicing=[[320,640],None,None]`/`output_dir=.../s12t/DevCCF`/20um；`prepare_custom_atlas` 实跑出 template+annotation 都是 `(320, 800, 560)`（X 正好半边）、spacing `(20,20,20)`、annotation 里 193 个 label。另外确认了 `mask.atlas_exclude_regions: ["Olfactory bulb"]` 在 DevCCF 本体里能匹配到（1 个："olfactory bulb, principal part"，子结构会一并排除），不会静默失效。**没有跑完整配准验证**——最终对齐质量要等用户真实跑完看。

**产物/清理**：`atlas/DevCCF/_orientation_check/` 下导了三个 TIFF 供肉眼核对（`atlas_CURRENT_orient_-1_-3_-2.tif`、`atlas_CANDIDATE_orient_1_-3_2.tif`、`atlas_FINAL_orient_1_-3_2_rightHemi.tif`，各 ~0.3-0.6GB，核对完可以整个目录删掉）。`atlas/DevCCF/` 下旧的 `*_-1_-3_-2__full.nii.gz` 两个缓存文件已经作废（新缓存是 `*_1_-3_2__320-640_full_full.nii.gz`），可以删。

**下一步**：用户重跑 `./run_pipeline.sh configs/s12t.yaml`，预期比之前快很多（图谱体素少一半 + 不再硬拟合缺失的半个脑），跑完检查 `tsc12t_in_atlas.nii.gz` 的实际对齐质量。

---

## 2026-07-29（当天第二轮）：DevCCF 配准跑不动 → 查出 orientation 错 + 漏了半球裁剪

用户反馈 DevCCF 那次配准跑了 2 个多小时还没完（SyN 第二个分辨率层级单次迭代要 ~1263s，第一层级只要 ~135s），怀疑是不是 atlas 方向错了。排查过程和结论：

**先排除的假象**：ANTs 的 DIAGNOSTIC 日志本身是正常的（metricValue 单调变负、convergenceValue 单调下降、前几次迭代显示 `inf` 是滑动窗口没攒够、第二段重新从 1 计数是进入下一个金字塔层级，不是重启）。用户还问过"metricValue 才 -0.004 是不是太小"——ANTs 的 `CC` 是带邻域半径的局部互相关，没有固定值域，绝对大小受邻域半径和强度归一化影响，判断收敛要看趋势不是绝对值。所以慢不是"配准崩了"，是别的原因。

**两个真正的问题**：

1. **样本是右半脑，图谱是全脑，之前完全没裁**。用户明确说了样本只有右半球（ClearMap 里当初也是把图谱裁成半球再配的），而 DevCCF 全脑图谱一直整个喂给 SyN——SyN 只能去硬拉伸匹配根本不存在的另一半，既慢又必然配不准。

2. **`orientation` 之前是错的**。原来配的 `[-1, -3, -2]`（2026-07-28 那轮"根据 direction matrix 代数推算 + 目视核对"的初步值，当时日志里就写了"这只是初步目视核对，不是最终配准 QC"）。这次做了一次快速 Affine-only 对比试验（各 30-40s，不跑耗时的 SyN）：两个候选方向的 NCC 几乎一样（0.6002 vs 0.5966，**这个指标没有区分度，不能用来判方向**），但把 warp 后的样本画出来一看差别很明显——旧方向下样本被压成一个扁长条、跟图谱轮廓完全对不上（affine 陷入局部最优的典型表现），新方向下轮廓能套上。用户最后自己用导出的 TIFF 目视确认了新方向 `[1, -3, 2]` 是对的。

**为什么之前的目视核对没发现**：DevCCF 图谱左右对称，单独看图谱的三视图**看不出 X 轴翻没翻**（镜像后长得一模一样，实测左右镜像匹配度 0.9999）。必须拿图谱去跟样本对照才能定方向——这是这次踩到的关键教训。

**orientation 的正确推法（已写进 `configs/atlas_presets.example.yaml` 顶部，含完整步骤）**：orientation 描述的是"把图谱的轴转成跟**样本**的轴一致"，**同时取决于图谱和样本，不是图谱的固有属性**，样本方位一换就要重推。

- DevCCF P04 原始 nii 的 affine（NIfTI RAS+ 约定，+X=Right/+Y=Anterior/+Z=Superior）解出来的三个数组轴含义（形状 640×560×800）：
  - 轴0 递增 → **Right**（左→右）
  - 轴1 递增 → **Inferior**（上→下）
  - 轴2 递增 → **Anterior**（后→前）
- tsc12t 样本（horizontal plane 成像）的三个轴：轴0=左→右、轴1=前→后、轴2=上→下。
- 逐个目标轴映射（第 d 项 = 样本轴 d 对应哪个源轴，1/2/3 表示图谱轴 0/1/2，反向加负号）：
  - 样本轴0 = Right ← 图谱轴0 就是 Right，同向 → `+1`
  - 样本轴1 = Posterior ← 图谱轴2 是 Anterior，反向 → `-3`
  - 样本轴2 = Inferior ← 图谱轴1 就是 Inferior，同向 → `+2`
  - ⇒ **`orientation: [1, -3, 2]`**

**半球裁剪（`slicing`）**：转向后 X 轴 = 左→右共 640 体素，中线实测在 320（用左右镜像匹配度扫出来的，最佳点 0.9999），所以**右半球 = `[320, 640)`**，左半脑样本就是 `[0, 320]`。关键顺序：`atlas_utils.prepare_custom_atlas` 是**先 reorient 再 slicing**（`atlas_utils.py:198-200`），所以 slicing 索引写的是**转向之后**网格上的坐标，不是原始文件的。

**改动落地位置**（有意分成两层，避免以后切图谱/换样本时错配）：
- `configs/atlas_presets_local.yaml` + `configs/atlas_presets.example.yaml`：记录 **DevCCF 原始文件三个轴的解剖含义**（图谱的固有属性，以后不用再从 affine 重算）+ 完整的「怎么推 orientation」步骤 + 「半球裁剪」说明；`orientation` 修正为 `[1, -3, 2]`。
- `configs/s12t.yaml`：`sample:` 块下用注释记录**本样本三个轴的解剖方位**（orientation 的推导依据）；`atlas_variants.devccf_p04` 下放 `orientation: [1, -3, 2]` + `slicing: [[320, 640], null, null]`。放在 `atlas_variants` 而不是 `atlas:` 顶层，是因为 `slicing` 索引是 DevCCF 网格专属的——切回 `demba_p5`（不同 shape）时不会被错误地套上（`slicing` 本来就已经是 `_ATLAS_VARIANT_FIELDS` 支持的字段，`config.py:57`）。

**验证结果**：`load_config('configs/s12t.yaml')` 解析出 `orientation=[1,-3,2]` / `slicing=[[320,640],None,None]` / `output_dir=.../s12t/DevCCF` / fine=atlas=20um；`prepare_custom_atlas` 实跑产出 template+annotation 都是 `(320, 800, 560)`（X 正好半个）、spacing `(20,20,20)`、annotation 里 193 个不同 label。另外顺手确认了 `mask.atlas_exclude_regions: ["Olfactory bulb"]` 在 DevCCF 本体里能匹配上（命中 "olfactory bulb, principal part"，子结构会一起排除）——`build_region_exclusion_mask` 是子串匹配且匹配不到会**静默不排除**，换本体后值得每次确认一遍。

**顺带产出/需要清理的文件**（都在 `atlas/DevCCF/` 下，都不该进 git）：
- `_orientation_check/atlas_CURRENT_orient_-1_-3_-2.tif`、`_orientation_check/atlas_CANDIDATE_orient_1_-3_2.tif`（各 ~573MB，方向对比用，确认完可删）
- `_orientation_check/atlas_FINAL_orient_1_-3_2_rightHemi.tif`（最终用的图谱，转向+右半球裁剪后，可留着做 QC 对照）
- `P04_*_-1_-3_-2__full.nii.gz`（旧错误方向的缓存，已失效可删）；新缓存是 `P04_*_1_-3_2__320-640_full_full.nii.gz`

**下一步**：用新配置重跑 `./run_pipeline.sh configs/s12t.yaml`。图谱体素少了一半 + 方向对了（affine 不会再陷局部最优），预期比之前那次 2 小时没跑完快很多。跑完要做的 QC：看 `tsc12t_in_atlas.nii.gz` 跟裁剪后的图谱对不对得上，以及 `tsc12t_labels_in_sample.nii.gz` 的脑区边界落在样本上合不合理。

## 2026-07-31：删掉粗-精两级配准（coarse-to-fine）

**动机**：这条路径从阶段4加进来后一直是 `use_coarse_to_fine: false`，实际项目里从没用过；而且它跟后来加的几乎所有功能都不兼容（custom atlas、`crop_for_registration`、`mask.guide_regions`、mask 支持），每加一个功能就要多写一条「跟 coarse 组合会报错」的校验。留着的维护成本大于价值，直接删干净。

**删除范围**：
- `register.py`：删掉 `register_to_allen_coarse_to_fine()` 整个函数，模块 docstring 改成「one-shot SyNRA」。
- `pipeline.py`：`[3/6] register` 步骤里的 `if reg_cfg["use_coarse_to_fine"]: ... else: ...` 分支去掉，原来 else 里的单次配准逻辑（图谱解析 + mask 构建 + `register_to_atlas` 调用）整体去掉一层缩进；日志里的 `strategy=single_shot` 字段也去了（已经没有第二种 strategy 可选）。
- `config.py`：`_DEFAULTS["registration"]` 删 `use_coarse_to_fine`/`coarse_target_um`/`coarse_atlas_res_um` 三个键；`load_config` 里删掉 coarse 相关的三段校验（`raw_tiff_coarse`/`voxel_size_coarse_um` 必填检查、`crop_for_registration` 互斥检查、`mask.guide_regions` 互斥检查）。
- `configs/config.example.yaml`、`configs/s12t.yaml`：删掉 `use_coarse_to_fine` 行。
- `README.md`：`register.py` 那行的描述改成「one-shot SyNRA」。

**没动的东西**（同名但无关，别误删）：`io_utils.resample_to_isotropic` 注释里的 "axes coarser than target_um"、`src/eval/registration_eval.py` 里的 "coarse region"（指 M2/M3 的大脑区）、`tests/test_pipeline_smoke.py` 里的 "coarser than target" 注释。

**验证结果**：全仓 grep `coarse_to_fine|raw_tiff_coarse|voxel_size_coarse|coarse_target_um|coarse_atlas_res` 无残留；三个改过的模块 import 正常，`config._DEFAULTS["registration"]` 现在只剩 `{fine_target_um, atlas_res_um, type_of_transform}`。注意 `tests/test_new_features_smoke.py` 目前会挂在 `assign_cell_regions`（假 `reg` dict 缺 `invtransforms` 键），这是**改动之前就存在**的测试自身的问题（已用 `git stash` 对照确认），跟这次删除无关。

---

## 2026-08-04：半球 atlas 配准几乎不变形的排查 + 两处修复

**用户反馈**：7-31 那次 `tsc12t`/DevCCF 真实配准跑完后（用 `scripts/single_sample.py` 新写的 napari 工具目视核对），atlas 看起来完全没有跟样本做实质性配准——像是直接把原图跟 atlas 硬叠在一起，atlas 大部分落在样本图像四周的空白 buffer 区域里，形状也没有明显形变。

**根因排查**（读实际代码路径 + 上次真实跑的 `run.log`，没有靠猜）：
- 样本图像本身没裁：`registration.crop_for_registration` 在 `configs/s12t.yaml` 里一直是注释掉的，所以整张带前后上下 buffer 的原图直接送去配准。事后用 `brain_mask.generate_brain_mask` 在缓存的 `tsc12t_fine_20um.nii.gz` 上实测：脑组织只占整张图的 **21.6%**，其余接近八成是空白，且 buffer 集中在 x 轴（左右方向）。
- 半球 atlas（`slicing: [[320,640], null, null]`）在正中线硬裁，裁切侧（medial）完全没有留白——正好是样本 buffer 集中的那个轴，两边形状/包围盒差异很大。
- `register.py` 的 `register_to_atlas()` 没传 `initial_transform`，`ants.registration` 因此用它自己的默认初始化："Center of mass alignment"（`run.log` 里能直接看到这行）——这个质心是对**整张图原始强度**算的，不受 `mask`/`moving_mask` 限制。样本 buffer 大、atlas 裁切侧又没留白，两边质心对不上，一旦这步初始对齐偏差够大，后面 Affine/SyN 阶段用来算相似度的 mask 区域重叠就很少，优化器基本就卡在这个错误初始位置附近动不了——正好对应用户看到的"atlas 几乎没变形、大部分落在空白区"。

**改动一：`register.py` 里加 mask-aware 的初始平移预对齐**（`register_to_atlas()` 非 `guide_regions` 分支）：只要传了 `mask`/`moving_mask`（`tsc12t` 已经在用），先跑一次受 mask 约束的 `type_of_transform="Translation"` 粗对齐，把结果当 `initial_transform` 喂给主配准，取代 ANTs 自己那个不受 mask 限制的质心初始化。沿用的是这个文件里 `guide_regions` 分支早就在用的"先跑一次粗配准→拿 `fwdtransforms[0]` 当 `initial_transform`"套路，没有新写一套逻辑。用合成数据验证过：模拟一个"图谱贴边无留白、样本四周大 buffer"的极端案例，单靠这一步 Translation 预对齐就能把 30 体素级的错位找回来，配准后 Dice 0.96。

**改动二：`crop_for_registration` 语义重做**——期间用户指出旧设计有个实际问题：裁剪原来是在重采样成 `fine_target_um` 各向同性网格**之后**做的（`crop_to_bounds(sample_fine, ...)`），所以配置里的坐标是 fine 网格的体素号，用户想自己定这个参数就必须先把流水线跑到出 `sample_fine` 为止才知道该填什么数字，没法在跑之前就定好。改成：
- `pipeline.py` 步骤 `[1/6]` 里把 `raw_tiff` 只读一次（`raw_img`，原始各向异性网格），一份直接重采样出完整的 `sample_fine`（下游 cell 坐标/`labels_in_sample` 用的全图网格，不受影响）；如果设了 `crop_for_registration`，另一份是先在 `raw_img` 上按**原始 TIFF 自己的体素序号**（x=列、y=行、z=第几张切片，跟 `sample.voxel_size_um` 顺序一致）裁剪，再重采样成 `sample_fine_for_reg` 送去配准。旧的"裁 fine 网格"路径整个删掉，不是加一层换算并存两套。
- 好处：现在 `crop_for_registration` 的数值就是直接在 ImageJ/Fiji 打开 `raw_tiff` 读到的体素号，配准前就能自己核对好，不用先跑流水线也不用手动换算到 fine 网格。
- `auto_brain_mask` 那行自动建议（`suggest_crop`）算出来还是 fine 网格坐标，同步加了一段通过物理坐标（`sample_fine_prep.origin/spacing` → 原始体素尺寸）换算回原始 TIFF 坐标系的逻辑，日志里打印的建议值现在直接是原始坐标，可以照抄进 config，不用手动换算。
- `configs/s12t.yaml`/`configs/config.example.yaml` 的注释同步改成新语义；`s12t.yaml` 里原先我（Claude）填的一组 fine-网格坐标数值（因为语义变了会被错误当成原始坐标用，裁剪范围会完全错）已经删掉，整块注释掉，具体数值留给用户自己核对原始 TIFF 后再填——这次没有代填猜测值。

**验证结果**：
- `register.py` 改动：合成小体积数据跑通了"有 mask"（触发新的 Translation 预对齐）和"无 mask"（走原来的默认路径，行为不变）两条分支，无报错；上面提到的 30 体素错位恢复实验，Dice 0.96。
- `crop_for_registration` 改动：合成了一个跟真实样本各向异性比例相近的 TIFF（xy 细、z 粗），验证裁剪现在确实发生在重采样之前、裁剪后物理范围跟预期一致（几体素级的离散化误差，可忽略）；又验证了"fine 坐标→原始坐标"这段换算能把已知的原始范围基本准确地换算回来（跟真值差 0-5 个体素，同样是离散化误差）。
- 两处改动都只跑了合成数据，**没有跑真实 `tsc12t` 数据**——`sample.raw_tiff` 之前指向的路径已经不存在（用户已自行把 `s12t.yaml` 里改成 `raw_data/s12t/registration.tif`），且一次完整 SyN 配准要 4+ 小时，等用户自己核对完 `crop_for_registration` 的原始坐标数值、确认路径没问题后再跑。

**下一步**：用户自己拿 `raw_tiff` 在图像查看器里核对组织实际边界，把原始体素坐标填进 `configs/s12t.yaml` 的 `registration.crop_for_registration`（当前是注释掉的，格式说明已经写在旁边），然后重跑 `./run_pipeline.sh configs/s12t.yaml`。跑完要做的 QC：`run.log` 里确认初始对齐不再是纯 "Center of mass alignment" 直接进 Affine/SyN（应该能看到新加的 Translation 预对齐这一步）、Rigid/Affine/SyN 的 metric 有实际收敛趋势；napari 里把 `tsc12t_in_atlas.nii.gz` 叠到裁剪后的 atlas 模板上、`tsc12t_labels_in_sample.nii.gz` 叠到原始样本上，确认这次 atlas 组织落在样本组织上而不是空白区，且形变不再是"贴图"式的刚性搬运。

---

## 2026-08-05：交互工具 + 评估整体搬到 `../GT_tool_for_registration`，新增两个脚本

**动机**：这个仓库同时装着"跑配准的流水线"和"一堆 napari 交互工具"，后者已经长到 5 个脚本、跟配准本身没有代码耦合。用户要求把画 mask / 标 landmark / 改脑区标签 / 看结果这些**可视化脚本**全部挪出去，ants 项目只留配准。

**搬走的东西**（新仓库 `/home/fyu7/My_project/GT_tool_for_registration`，已 `git init`，扁平结构）：

| 原位置 | 新位置 |
|---|---|
| `mask_tools/paint_mask.py` + 两个 `paint_mask_local*.yaml` | `paint_mask.py` + 同名 yaml（仍与脚本同级，`Path(__file__).parent` 逻辑零改动） |
| `scripts/edit_sample_labels.py` | `edit_sample_labels.py` |
| `scripts/place_landmarks.py` | `place_landmarks.py` |
| `scripts/single_sample.py` | `single_sample.py` |
| `scripts/_form_dialog.py` | `_form_dialog.py` |
| `src/eval/registration_eval.py` + `reg_metrics.csv` | `registration_eval.py` + `reg_metrics.csv` |
| `configs/eval_config.example.yaml` | `configs/eval_config.example.yaml` |
| `tests/test_registration_eval_smoke.py` | `tests/test_registration_eval_smoke.py` |

`mask_tools/` 目录整个删掉；`scripts/` 只剩 4 个非交互脚本（`project_outline.py`/`relabel_cells.py`/`convert_devccf_ontology.py`/`relabel_labels_to_devccf.py`）。这些文件在上一个 commit（`8e28a32 edit`）里用户已经自己从 git 里删掉了，所以这次 `mv` 之后 `git status` 里不会出现 deletion。

**依赖处理**（用户选的方案）：搬走的脚本继续 `from registration_ants import ...`，靠 `antsreg` 里已有的 `pip install -e`（`__editable__.registration_ants-0.1.0.pth`）解析——`paint_mask.py`/`edit_sample_labels.py` 顶部那两行 `sys.path.insert(..., "src")` 直接删掉。用到的 `mask_utils`（纯 numpy/scipy）和 `atlas_utils`（纯 json/numpy，`import ants` 是函数内延迟导入）都不需要 antspyx。**依赖只指向一个方向**：GT_tool 用 registration_ants，反过来没有。

**顺手修好的旧 bug**：`tests/test_registration_eval_smoke.py` 里那行 `import registration_eval as ev` 早就失效了——`registration_eval.py` 之前从仓库根挪到 `src/eval/` 时这行没跟着改（实测 `ModuleNotFoundError`，只有 `from eval import registration_eval` 能成功）。新仓库把 `registration_eval.py` 放在根目录，这行重新成立，测试恢复可跑。

**新增脚本 1：`annotate_gt_sam.py`**（napari + micro_sam 稀疏切片脑区 GT 标注）
- 每个脑区只在配置里**预先锁定**的几层 z 上标（层位置逐脑区不同），其余层保持空；每个脑区单独一个 `{brain_id}_{region}.nii.gz`，值只有 0/1。
- **清单写两份**：合并清单 `annotated_slices_manifest.json`（`{"brain01": {"ventricle": [...]}}`，权威来源）+ 每个 mask 旁边的 `{brain_id}_{region}.annotated_slices.json`。后者是 `edit_sample_labels.py` 早就在用、`registration_eval.py:350 load_region_annotation_hint()` 早就会自动读的 sidecar 格式——**这样新画的 GT 一行评估代码都不用改就能进 `eval_config.yaml` 的 `dice_region_masks`**。sidecar 每次从合并清单重新派生，两边不会跑偏。
- **严禁 z 轴传播是结构性保证，不是"记得别点"**：只用 `micro_sam.sam_annotator.annotator_2d`（它的 `Annotator2d._get_widgets()` 只有 segment/autosegment/commit/clear；传播功能只存在于 `annotator_3d` 的 `segment_nd`/`segment_all_slices` 和 `multi_dimensional_segmentation`，本文件一次都没 import）；喂给 micro_sam 的永远是 `volume[z]` 一张 2D 数组，它根本看不到体数据；embedding 按 `(brain, z)` 分文件缓存，从不跨 z 复用。测试里有一条 grep 断言直接检查这些名字不出现在可执行代码里。
- **多切片怎么复用同一个窗口**（这是最容易写错的地方）：第二张之后**不能**再调 `annotator_2d`——那会往同一个窗口再加一个 `image` 图层和第二个 widget dock。正确做法是走 `AnnotatorState`：`state.image_shape = plane.shape` → `state.initialize_predictor(plane, ..., predictor=state.predictor, decoder=state.decoder)`（传入已加载的 predictor，跳过重新加载 checkpoint，只重算这一层的 embedding）→ 换 `viewer.layers["image"].data` → `state.annotator._update_image(segmentation_result=...)`。另外 `_update_image` 在 `state.skip_recomputing_embeddings` 为 True 时会**提前 return**（该标志由 micro_sam 自己的 embedding widget 设置），所以调用前显式置 False，否则上一层的 `committed_objects` 会留在屏幕上。
- 续标：已完成的层标 `[done]`、被 "Next unfinished" 跳过，不覆盖；要重画就选中该行点 "Redo slice"，已有 mask 通过 `segmentation_result=` 预填回 `committed_objects` 编辑，保存时覆盖并更新清单。
- 导出用 `CopyInformation(reference)` 后**读回来再校验**（调 `align_masks.check_geometry`），不通过就报错而不是默默写出去。
- 一个要提前说明的现实：SAM 点提示对**闭合、强度均匀**的物体最好用（脑室=暗 CSF 很合适）；`cortex_surface`/`cortex_wm` 本质是**界面**不是物体，实际会更依赖在 `committed_objects` 上手改。这是 SAM 的性质，不是脚本坏了。

**新增脚本 2：`align_masks.py`**（mask 头信息校验/强制对齐 + 自检）
- 功能一 `unify_headers()`：把源图像的 spacing/origin/direction 覆盖到一批 mask 上、**另存不原地改**；shape 不一致直接**报错停止**（说明是重采样/拿错文件这类更严重的问题，改头文件只会掩盖），且**先校验完所有 shape 再写第一个文件**，不会留下半成品输出目录；也拒绝会覆盖输入的 `--out-dir`。
- 功能二 `check_geometry()`：size/spacing/origin/direction **逐项**报 pass/fail + 实际差值，不是一个笼统的布尔。
- **表面距离必须以物理单位算**——坑在轴序：`np.argwhere` 出来的索引是 numpy 的 `(z,y,x)`，而 `GetSpacing()` 是 SimpleITK 的 `(x,y,z)`，**正好相反**。所以必须 `spacing_zyx = np.asarray(img.GetSpacing())[::-1]` 再相乘；进 KD-tree 的坐标已经是微米，下游再也不会出现"体素数"。
- **自检设计**（`python align_masks.py --selftest`，全合成，两个 env 都能跑）：
  1. identity：dice=1、距离=0。
  2. known translation：用**单体素厚的平面**沿法线平移——这种形状下对称平均表面距离**精确**等于 `k × spacing[该轴]`（平面上每点的最近点就是正对面那个点），比立方体干净（平移后的立方体尾面会落进对方内部，没有解析解，只能做模糊断言）。**各向异性 spacing `(10,25,40)` + 非立方 shape + 三个轴各测一遍**——只测一个轴的话 spacing 转置了照样能蒙对。x 方向平移 10 体素必须报 100µm，轴序写反会报 400µm。每个各向异性 case 再拿 SimpleITK 自己的 `TransformIndexToPhysicalPoint` 当**独立预言机**交叉核对（它还顺带覆盖了非零 origin）。
  3. shape 不一致：断言抛异常且**一个文件都没写**。
  4. `unify_headers` 改完头信息后体素值逐位不变、原文件没被动过。
  5. 跟 `registration_eval.dice`/`_surface_distances` 数值交叉核对（antspyx 不在时优雅跳过）。
- **`dice`/表面距离是有意的独立最小实现**，不是直接 import `registration_eval`：后者 `from registration_ants import transforms` 会拉进 antspyx，而 `align_masks` 必须能在没有 antspyx 的 `gt_sam` env 里被 `annotate_gt_sam.py` import。第 5 条自检保证两份实现不会跑偏。

**环境**：新建 `gt_sam`（`conda create -y -n gt_sam -c conda-forge -c pytorch python=3.11 micro_sam napari pyqt simpleitk pyyaml`），实测 micro_sam 1.8.8 / napari 0.8.0 / torch 2.10.0 / SimpleITK 2.5.6 / python 3.11。**没有装进 `antsreg`**——micro_sam 会拉 torch 和它自己钉的 napari/numpy，很可能把跑通配准的 numpy 2.3.5 / napari 0.8.0 / antspyx 降级。其余所有工具（含 `registration_eval.py`，它要 antspyx）仍在 `antsreg` 里跑。`nvidia-smi` 目前报 driver/library version mismatch，CPU 路线可用（配置里可以显式写 `device: cpu`）。

**验证结果**：
1. `align_masks.py --selftest`：`antsreg` 和 `gt_sam` **两个 env 都全过**（gt_sam 里第 5 条按预期跳过）。
2. `tests/test_annotate_gt_sam_smoke.py`（新增，两个 env 都过）：配置校验、清单+sidecar 一致、导出几何、续标不丢工作、以及 `--verify` 的**四种失败**都能抓到（清单说标了某层但那层是空的 / mask 有清单没记的层 / 值不在 {0,1} / 标了锁定列表外的层）、禁止传播的 grep 断言。
3. `tests/test_annotate_gt_sam_microsam_e2e.py`（新增）：**真的把 micro_sam 跑起来了**（xvfb + 软件 GL，`QT_QPA_PLATFORM=offscreen` 单独不够，napari 的 vispy canvas 要真 GL）——第一张切片建出全部 6 个图层和 1 个 dock，第二张走 state 复用路径后**没有多出重复的 image 图层和第二个 dock**、`committed_objects` 被正确清空、图像确实换成了另一层；已完成的层被跳过、redo 能把存下的 mask 读回来；`--verify` 通过；embedding 缓存正好一层一个文件。命令见该文件 docstring。
4. `tests/test_registration_eval_smoke.py` 在新仓库跑通（同时验证了上面那个旧 import bug 修好了）。
5. `Registration_ants/tests/` 剩下的 `test_pipeline_smoke.py`/`test_brain_mask_smoke.py`/`test_label_correction_smoke.py` 全过（`test_new_features_smoke.py` 有 7-31 就存在的失败，跟这次无关，没纳入判据）。
6. GT_tool 下 11 个 .py 全部 `ast.parse` + **真实 import** 通过（含 napari/PyQt5/registration_ants 一起 import）。
7. 全仓 grep：`Registration_ants` 里除 `PROGRESS_LOG.md`（历史条目按既有约定保留原文，不回填）外，已无指向搬走文件的路径。
8. **真实数据实测 `align_masks.py`**（见下面"发现"）。

**顺带发现的一件真实数据的事**（不是 bug，但会影响 GT 怎么做）：`DevCCF_ver2_0804/tsc12t_brain_mask.nii.gz` 的网格是 `(176,510,219)` origin `(1144,130,640)`，跟同目录的 `tsc12t_fine_20um.nii.gz`（`(295,517,251)`，origin 0）**对不上**，但跟 `tsc12t_fine_20um_cropped.nii.gz` **完全一致**。这是 8-04 那次改动的正常结果——`auto_brain_mask` 是在裁剪后的 `sample_fine_prep` 上算的。含义：以后要拿 brain mask 跟别的东西比 Dice，得先确认两边在同一个网格上。另一方面 `DevCCF_ver1_0801/tsc12t_labels_in_sample.nii.gz` 跟未裁剪的 `tsc12t_fine_20um.nii.gz` **完全同网格（PASS）**——所以 `annotate_gt_sam.py` 的 `volume:` 应该指向**未裁剪**的 `tsc12t_fine_20um.nii.gz`，这样画出来的 GT 跟 `labels_in_sample.nii.gz` 天然同网格，不用重采样。配置模板里已经按这个填了。

**改动范围（本仓库侧，只改注释/文档，没碰任何执行逻辑）**：`README.md`（"Evaluation" + "Manual correction tools" 两节换成一段指向 GT_tool 的说明，`tests/` 列表去掉搬走的那份）、`src/registration_ants/` 下 `register.py`/`mask_utils.py`/`config.py`/`atlas_utils.py`/`transforms.py` 的 docstring 路径、`scripts/project_outline.py`/`relabel_cells.py`/`convert_devccf_ontology.py`、`configs/atlas_presets.example.yaml`、`tests/test_label_correction_smoke.py` 的注释、`.gitignore`（删掉 `!configs/eval_config.example.yaml` 白名单和 `scripts/.dialog_state/` 两条，都跟着搬走了）、`requirements.txt`（`napari[pyqt5]` **保留**——搬走的工具仍跑在 `antsreg` 里，注释改成说明这一点）。

**下一步**：
1. 用户在有显示的机器上手动过一遍 `annotate_gt_sam.py` 的交互部分（放点 → `s` 分割 → `c` commit → 手改 → Save slice），这部分自动化测不了，验收清单写在 GT_tool 的 README 里。
2. 先在 `configs/gt_annotation.yaml` 里把 4 个脑区各自的 5 个 z 位置**定下来**（现在模板里是占位数字），定之前建议先用 `single_sample.py` 翻一遍 `tsc12t_fine_20um.nii.gz` 挑有代表性的层。
3. 标完一版之后：`python annotate_gt_sam.py ... --verify` → 把 `{brain_id}_{region}.nii.gz` 填进 `configs/eval_config.yaml` 的 `dice_region_masks` → 跑 `registration_eval.py`，看 Dice/HD95 那组 metric 在真实标注下能不能跑通（之前一直因为没有标注被跳过）。

---

## 2026-08-07：接着 8-04 排查——又挖出三个真 bug + Affine-only 快速诊断验证半球思路是对的

**背景**：8-04 那次加了 mask-aware Translation 预对齐 + `crop_for_registration` 改成原始 TIFF 坐标系，但没跑过真实数据验证。这次用户提议"先只跑 Affine（不等 SyN 那几小时）看看效果，Affine 都不行后面也白搭"，跑起来之后接连挖出三个之前没暴露过的真 bug。

**Bug 3：`ants.registration()` 的 `mask_all_stages` 默认 `False`——`SyNRA` 的 Rigid/Affine 两个 stage 实际完全没用 mask**

直接读真实 `run.log` 里 antsRegistration 落地的命令行发现的：`SyNRA` 是 Rigid+Affine+SyN 三个 metric stage 打包成一次调用，只有最后 SyN 那行 `-x` 是真的 mask 指针，Rigid/Affine 两行都是 `-x [NA,NA]`。查了 antspyx 源码确认：`ants.registration()` 只把 mask/moving_mask 传给一个变换的**最后一个** stage，除非显式传 `mask_all_stages=True`（默认 `False`），"早期" stage 一律拿 `[NA,NA]`。

这解释了 8-04 那次加的 Translation 预对齐为什么没用——预对齐本身是单 stage、mask 正常生效，算出的初始位置是对的，但紧接着 `SyNRA` 里完全不受 mask 限制的 Rigid+Affine 又把这个好的初始对齐拽回了错误位置（用的是全图未裁剪范围，跟最开始那个 bug 是同一种机制，只是发生在配准中段而不是初始化）。

**改动**：`register.py` 里所有涉及 mask 的 `ants.registration()` 调用（主配准、`guide_regions` 分支的 Affine/SyNOnly、新加的 Translation 预对齐）统一加 `mask_all_stages=True`。

**Bug 4：`intensity_clip_percentiles` 配置项被静默忽略**

`pipeline.py` 的 `_preprocess()`：`n4_bias_correction: true` 时直接调 `preprocess.preprocess_for_registration(img)`（不带参数），内部 `clip_and_normalize` 用的是硬编码默认值 `(0.5, 99.5)`，config 里配的 `intensity_clip_percentiles: [0.1, 99.9]` 只有 `n4_bias_correction: false` 才会读到——但用户一直开着 N4，这个字段实际从没生效过。`run.log` 也能对上：只打印了 "N4 bias correction" 一行，没有 "intensity clip_and_normalize percentiles=..." 那行。

**改动**：`preprocess.preprocess_for_registration` 加 `lower_pct`/`upper_pct` 参数（默认值不变，向后兼容）；`pipeline.py` 的 `_preprocess` 不管走哪条分支都显式把 `intensity_clip_percentiles` 传进去。

**Bug 5（Affine-only 测试直接测出来的）：`ants.apply_transforms`/`apply_transforms_to_points` 的 `whichtoinvert=None` 自动推断，对单一 `.mat` 的 `invtransforms` 不生效**

第一次 Affine-only 真实跑完，`tsc12t_labels_in_sample.nii.gz` 文件只有 149KB（之前 SyNRA 那次 1200KB），一查是**整个体积全是 0**。根因是 ants 自己的文档写明的："若 transformlist 是 matrix 后面跟 warp field，`whichtoinvert` 默认 `(True, False)`；否则默认 `[False]*len(transformlist)`"。`SyNRA` 的 `invtransforms = [affine.mat, InverseWarp.nii.gz]` 正好命中"matrix+warp"这个特例，一直以来矩阵都被自动求逆、结果是对的；但纯 `Affine`（无形变场）的 `invtransforms` 只有 `[affine.mat]` 一个元素，不命中特例，矩阵没被求逆，方向整个反了，所有点都映射到样本网格外，输出全 0。**这个 bug 只在用单一线性变换类型（Affine/Rigid/Translation，没有 SyN 形变场）时才会触发，`SyNRA` 正式跑不受影响，之前所有真实结果都是对的**——纯粹是这次第一次真的跑 `type_of_transform: Affine` 才暴露出来。

实测验证：同一个矩阵文件，默认 `whichtoinvert=None` 求出 0% 非零；显式传 `whichtoinvert=[True]` 求出 32.4% 非零、190 个不同 label，跟同一样本 SyNRA 那次的 31.7%/176 个量级吻合。

**改动**：`transforms.py` 新增 `_mat_entries_to_invert(transformlist)`（按文件是不是 `.mat` 显式算 `whichtoinvert`，不再依赖 ants 那个只覆盖两元素特例的自动推断），`warp_labels_to_sample` 和 `transform_cell_points`（`sample_to_atlas` 方向）都改成显式传这个值。`fwdtransforms` 方向不用改——单个 `.mat` 在 fwdtransforms 里本来就不需要求逆，默认行为已经是对的。

**顺带修的环境问题**：用户把 `/data/hdd12tb-1/fengyi/COMBINe/clearmap/` 顶层目录整个重命名成了 `Registration/`，`s12t.yaml` 里 `sample.raw_tiff`/`cells.cell_centroids_dir`/两处 `output_dir` 还指向旧路径，run 之前 `load_config` 就会直接报 `FileNotFoundError`，一并改成新路径。另外用户手改 config 时把 `mask.atlas_exclude_regions` 从 `["Olfactory bulb"]` 改成 `[""]`（想表示"不排除任何区域"），但 `build_region_exclusion_mask` 是子串匹配，空字符串是任何字符串的子串，会导致**几乎整个 atlas mask 变成全 0**（约等于把整张图谱都排除出配准相似度计算）——提醒后用户自己改成了正确的空列表 `[]`。

**Affine-only 诊断结果（三处 bug 修完之后）**：
1. **半球图谱**（`slicing: [[320,640],null,null]`，跟 8-04 之前一致）+ Affine：`tsc12t_in_atlas.nii.gz` 跟半球 atlas 叠加，整体位置/大小/形状对上了，明显好于修 bug 之前。用户目视核对："半脑还说得过去"。唯一问题：外侧皮层区域对上了，中线附近深部结构（丘脑等）基本没跟着动——分析是 Affine 只有一套全局线性参数，没法对不同区域分别拉伸，这部分差异性状不均匀，理论上要交给 SyN 的局部形变场处理，不算 Affine 阶段的异常。
2. **完整双侧图谱**（`slicing: [null,null,null]`，用户临时改的对照组）+ Affine：用户目视核对后反馈"全脑atlas效果非常差"。印证了半球裁剪这个思路本身是对的、不该改成拿完整双侧图谱去配（跟用户此前确认的"ClearMap 当年也是配的半脑图谱"一致），不用再纠结要不要放弃半球裁剪。

**用户提供的关键背景**：这份样本做过基因敲除 + iDISCO 清透处理，形变比较明显——长度上跟 atlas 差不多，但宽度大约是 atlas 的一半，且不是均匀缩放（结合上面"皮层对上、中线没动"的现象）。讨论结论：SyN 是微分同胚形变场，理论上能处理这种局部、非均匀的形状差异（这正是 SyN 存在的意义），但它有硬限制——如果某个结构是真的发育异常/缺失（不是被压缩，是真没有），SyN 没法凭空生成对应组织，那部分区域的 label 边界不会有意义，需要 `GT_tool_for_registration/edit_sample_labels.py` 之类的工具手动兜底，不能指望自动配准全解决。

**验证方式**：三个 bug 都各自用真实数据/实际 run.log 核实过（不是纯代码审查猜的），`register_to_atlas` 合成数据烟雾测试仍过；`transform_cell_points`/`warp_sample_to_atlas`（fwdtransforms 方向）未受影响，没改。

**下一步**：`configs/s12t.yaml` 改回 `type_of_transform: SyNRA`、`atlas_variants.devccf_p04.slicing` 改回半球裁剪 `[[320,640],null,null]`、`output_dir` 指到正式目录，跑一次完整 SyNRA（预计仍要小时级），重点看中线区域这次能不能被局部形变拉过去贴合样本。
