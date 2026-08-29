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

---

## 2026-08-08：SyN"只位移不形变"的真正原因——CC 半径被默认值设成 32 + 最细层 0 次迭代

**背景**：8-07 那三个 bug 修完后，用户跑了一次完整 SyNRA（`DevCCF_0807/run.log`，register 步骤 2:55:24），目视核对发现**结果跟 affine-only 那次几乎一样**——atlas 只是整体位移了一下，没有实质形变。用户已经知道自己的样本因为基因敲除 + iDISCO 清透跟正常脑差别大（宽度约为 atlas 一半且非均匀），但这个结果依然不合理。读 `run.log` 落地的 antsRegistration 命令行 + 收敛轨迹，查出两个参数问题，都不是样本的锅。

**问题 1：CC 的邻域半径是 32,应该是 4**

`run.log` 里 SyN 阶段是 `--metric CC[...,1,32]`。这是 antspyx 的参数重载陷阱，读源码确认（`ants/registration/registration.py:111-112` 的 docstring 原文）：

```
syn_sampling : scalar
    the nbins or radius parameter for the syn metric
```

**同一个参数两种含义**——`syn_metric="mattes"` 时是直方图 bin 数（32 是标准值），`syn_metric="CC"` 时是**局部邻域半径（体素）**。而函数默认值写死 `syn_sampling=32`（`registration.py:36`），是按 mattes 定的。`register.py` 里一直只传了 `syn_metric="CC"` 没传 `syn_sampling`，于是 CC 拿到了半径 32。

半径 32 意味着每个体素用 65³ = 274,625 体素的窗口算局部相关，在 20µm 网格上是**边长 1.3mm 的立方体**——而裁剪后的样本才 3.5 × 10.2 × 4.4 mm，这个"局部"窗口边长超过样本宽度的三分之一。后果：局部结构被平均掉，梯度只反映大尺度强度分布（而那部分 affine 已经做完了），所以 SyN 每步更新极小、形变场基本是全局性的；同时慢得离谱（shrink=4 层 47 秒/次迭代，shrink=2 层 420 秒/次，2小时55分基本全花在这）。

同一份 antspyx 源码 `registration.py:670-671` 他们自己的 `SyNCC` 预设注释里写着 `# syn_sampling = 4`——作者自己用 CC 时是配 4 的，只是这个配对关系没做成自动的。4 也是 `antsRegistrationSyN.sh` 的官方默认。

**问题 2：`reg_iterations` 默认 `(40, 20, 0)`，最细层一次都不跑**

命令行 `--convergence [40x20x0,1e-7,8]` 配 `--shrink-factors 4x2x1`：全分辨率那层 **0 次迭代**，形变场最细只在 2 倍降采样（等效 40µm）上估计然后上采样输出。跟半径无关，是独立的第二个问题。

另外两个非零层**都是跑满上限被切断的，不是收敛退出**：第一层 40 次跑完时 convergenceValue 还有 7.6e-4，第二层 20 次跑完 5.5e-4，阈值是 1e-7，差四个数量级，metric 全程单调下降没走平。

（诚实的补充：上次每步走得少，部分原因就是半径 32 把梯度抹平了；半径修好后同样 40 次可能走得远很多。但最细层的 `0` 无论如何都得改，而且 `reg_iterations` 是**上限不是固定开销**——某层收敛值跌破 1e-7 会提前退出，所以调高的主要代价是给足时间预算而不是无条件多跑。用户问过这一条，讨论后决定两个一起改。）

**改动**（两个参数都做成 config 可调，不写死）：
- `register.py`：`register_to_atlas` / `register_to_allen` 新增 `syn_sampling=4` / `reg_iterations=(100,70,50)` 参数，**两个** `ants.registration()` 调用点都传（主配准分支 + `guide_regions` 的 SyNOnly 分支——后者同样是 `syn_metric="CC"` 不传 sampling，有一模一样的 bug）。docstring 里写清 syn_sampling 一参两义、以及 reg_iterations 的长度会决定金字塔层数。
- `config.py`：`_DEFAULTS["registration"]` 加 `syn_sampling: 4` / `reg_iterations: [100, 70, 50]`，并加校验（reg_iterations 必须是非空、非负整数列表——长度错了会静默改变金字塔深度；syn_sampling 必须是正整数）。
- `pipeline.py`：透传两个参数，并把它们打进 `Registration params:` 那行 log（下次看 run.log 能直接确认生效，不用再去读命令行）。
- `configs/config.example.yaml` / `configs/s12t.yaml`：补中文注释说明两个参数是什么、为什么不能用默认值。

**验证结果**：
1. 合成小体积实跑 `register_to_atlas`，抓 antspyx 落地的 antsRegistration 命令行确认：`--metric CC[...,1,4] --transform SyN[0.2,3,0] --convergence [100x70x50,1e-7,8] --smoothing-sigmas 2x1x0 --shrink-factors 4x2x1`——半径变 4、三层金字塔、**最细层是全分辨率且有 50 次迭代**。函数默认值和从 `configs/s12t.yaml` 读出来的值两条路径都验过。
2. `load_config('configs/s12t.yaml')` 解析出 `{type_of_transform: SyNRA, syn_sampling: 4, reg_iterations: [100,70,50]}`。
3. `tests/` 下 `test_pipeline_smoke.py` / `test_brain_mask_smoke.py` / `test_label_correction_smoke.py` 全过（`test_new_features_smoke.py` 的失败是 7-31 就存在的，跟这次无关）。
4. **没有跑真实数据**——真实跑一次是小时级，留给用户。

**顺带发现（不是这次改的）**：`DevCCF_0807` 那次是 `atlas_mask=False`，而 8-06 的 affine 测试是 `atlas_mask=True`——应该是 `mask.atlas_exclude_regions` 从 `[""]` 改成 `[]` 之后图谱侧 mask 就没有了（`[]` 是正确写法，`[""]` 那个才是 bug，见 8-07 记录）。sample mask 还在，影响不大，但要注意这两次运行不是完全同条件。

**下一步**：用户重跑 `./run_pipeline.sh configs/s12t.yaml`。跑起来后的**早期判断点**：看第一层（shrink=4）的 `SINCE_LAST`，如果远低于上次的 47 秒说明半径修对了，再按实测速度外推总时长决定要不要调 `reg_iterations`，不用等三小时才知道。跑完的 QC 重点仍是中线区域的深部结构（丘脑等）这次有没有被局部形变拉过去——8-07 的 affine 诊断显示外侧皮层能对上、中线附近不动，那部分正是要靠 SyN 的局部形变场解决的。

---

## 2026-08-10：找到"atlas 几乎不形变"的真正根因——脑轮廓 mask 抑制了撑开（8-08 的判断是次要因素）

**背景**：8-08 改完 `syn_sampling=4` / `reg_iterations=[100,70,50]` 后，用户跑了 `DevCCF_0809`（register 步骤 1:43:38，比 0807 的 2:55 快很多），但目视核对**结果和调参前几乎一样**，atlas 仍然只是位移、没有实质形变。

**先确认参数确实生效了**（不是又一次静默失效）：`run.log` 里 `Registration params: ... syn_sampling(CC radius)=4, reg_iterations=[100, 70, 50]`，落地命令行 `--metric CC[...,1,4] --convergence [100x70x50,1e-7,8] --smoothing-sigmas 2x1x0 --shrink-factors 4x2x1`，三层金字塔真的都跑了（100 次 / 51 次【收敛退出】/ 50 次），metric 从 -0.09 单调优化到 -0.22。**参数没问题，是别的地方不对。**

**量化"到底动了多少"**（这一步是关键，比目视可靠）：直接读两次的 `1Warp.nii.gz` 比较位移大小——

| | 0807(半径32) | 0809(半径4) |
|---|---|---|
| 位移中位数 | 20µm | 19µm |
| 90 分位 | 103µm | 100µm |
| 最大位移 | 393µm | 392µm |

**两次几乎逐项相同**。而 warp 后样本 vs atlas 的包围盒缺口：上下 2.3mm、前后 1.5mm、左右 0.6mm，**图谱有组织的体素里 35% 底下是空的**。0.39mm 的形变对 2.3mm 的缺口杯水车薪。

**样本实测尺寸**（用 brain mask 量的真实组织范围，对比 DevCCF P04 右半脑）：前后 10.08mm vs 11.66mm（86%）、上下 **4.34mm vs 7.08mm（61%）**、左右 3.32mm vs 4.38mm（76%）、体积 65.9 vs 117.2 mm³（56%）。用户确认**组织是完整的**，压扁是 iDISCO 清透 + 半脑平放成像导致的，不是缺失。（嗅球确实没有，但那是处理过程中脱落的，属于另一回事。）

**根因（合成实验查出来的，不是推理）**：造了个和真实情况同构的幻影——结构完整、沿一轴压扁到 61%、高度用真实同量级的体素数（360）——跑四种组合：

| 配置 | 补上缺口 |
|---|---|
| 3层 + 轮廓mask（= 当时的配置） | **16%**（撑开 +0.38mm） |
| 4层 + 轮廓mask | 35% |
| 3层 + 无mask | **100%** |
| 4层 + 无mask | 100% |

合成的"3层+mask"撑开 **+0.38mm**，真实数据是 **+0.36mm**（4.34→4.7mm）——几乎一样，说明幻影准确复现了失败模式，结论可以外推。

**机制**：`auto_brain_mask` 产出的全脑轮廓被当成 `moving_mask`。ANTs 的 `-x` 是"limit voxels considered by the metric"，一个采样点要计入必须映射后落在 moving mask 内。于是**图谱中样本尚未覆盖的那 35%，在目标函数里根本不存在**——没有任何一项要求样本长大，"缩在中间"和"撑开填满"得分一样好。

**第二轮实验（更严苛的幻影：半脑图谱内侧切平贴边 + 样本压扁 + 人为偏移 + 四周 buffer，复刻 8-04 那个几何）**：

| 配置 | 补缺口 | 质心偏差 |
|---|---|---|
| A. 全程有mask（现状） | **9%** | 0.14mm |
| B. 全程无mask | 101% | 0.00mm |
| C. 线性阶段有mask + SyN无mask | 64% | 0.06mm |
| D. **带mask的Translation预对齐 + 主配准无mask** | **101%** | **0.00mm** |

**C 只有 64% 这一条最有信息量**：说明受害的主要是 **Affine 阶段**而不是 SyN。Affine 自带缩放参数，61%→100% 的撑开本该由它一步完成；戴上 mask 后它不会去放大，全部工作被推给形变场——用形变场干缩放该干的活，事倍功半。这也回头解释了 8-07 那次纯 Affine 诊断为什么始终没撑开（那次也是全程带 mask）。

**改动（采用 D）——把两种 mask 的角色分开**：
- `register.py`：`register_to_atlas` 新增 `prealign_moving_mask` 参数，**只**喂给 Translation 预对齐，不进主配准；预对齐的触发条件相应改成 `mask is not None or prealign_mm is not None`。docstring 里写清了机制、实测数字，以及为什么"轮廓 mask"和"排除 mask"必须区别对待。
- `pipeline.py`：`auto_brain_mask` 的产物不再并进 `sample_mask`，改走 `prealign_moving_mask`；日志新增 `prealign_mask=` 字段。
- **有意保留的**：`sample_damage_mask_path`（手画裂缝）和 `atlas_exclude_regions`（缺失结构）仍走 `moving_mask`/`mask` 进主配准——它们排除的是**具体一小块**（damage mask 是"全 1 里擦掉一块"），不会像轮廓那样把未覆盖的图谱区域整片藏起来。**注意**：如果用户拿 `auto_brain_mask` 产物当 `paint_mask.py` 的 `EXISTING_MASK_PATH` 起点去画 damage mask，产出的又会是个轮廓，会重新触发这个问题。

**验证结果**：
1. 改后的 `register_to_atlas` 在严苛幻影上复跑：轮廓当 `moving_mask` 补缺口 12%/质心偏差 0.11mm → 当 `prealign_moving_mask` 补缺口 **101%**/质心偏差 **0.01mm**。
2. `load_config('configs/s12t.yaml')` 正常；`test_pipeline_smoke.py` / `test_brain_mask_smoke.py` 通过。
3. **没跑真实数据**（一次 1.5~3 小时），留给用户。

**关于 `reg_iterations`**：加一层（`[200,100,70,50]`）在实验里把有 mask 的情况从 16% 提到 35%，有帮助但不解决问题；去掉 mask 后 3 层已经 100%。**建议下次真实跑先只改 mask 这一处、保持 `[100,70,50]`**，这样变量干净——好了就确知是 mask 的锅，还差再加层数。

**顺带查实的事（用户本轮决定暂不处理）**：`atlas_exclude_regions: ["Olfactory bulb"]` 在 DevCCF 本体里命中 6 个 label（5 个嗅球分层规矩待在最前端 118~230，第 6 个 "OB olfactory fiber layer" 拖到 468——嗅束沿腹侧后行，解剖学上真实），共 3.69% 体素。**更重要的发现**：嗅球在图谱里占 118~230（2.2mm），而实测前端缺口只有 0.74mm，说明**配准已经把样本往前拽去填嗅球的空位**——warp 后样本前端边缘落在 142，正盖在嗅球地盘上约 1.8mm，即额叶皮层正在被贴上嗅球标签。形变问题解决后需要回来处理这个。

**待修（本轮没动）**：`guide_regions` 分支的 Affine 调用仍然没有 Translation 预对齐、也没做这次的 mask 角色分离；真要用 guide_regions 之前得先补。

**下一步**：用户把 `output_dir` 换个新目录（保住 0809 做对比），重跑 `./run_pipeline.sh configs/s12t.yaml`。跑完的 QC：`run.log` 里确认 `prealign_mask=True` 且 `sample_mask=False`；量 `1Warp.nii.gz` 的位移分位数（这次应该远超 0.4mm）；量 warp 后样本 vs atlas 的包围盒缺口（上下那 2.3mm 应该基本消失）。

**追加（同日）：启用 `atlas_exclude_regions: ["Olfactory bulb"]` + 验证图谱侧排除 mask 的行为**

用户决定把嗅球加进排除列表。改 `configs/s12t.yaml` 时发现文件里当时是 `atlas_exclude_regions: [""]` —— 又是 8-07 记录过的那个坑（空字符串是任何名字的子串，会把**整个图谱**排除掉、配准完全没有相似度信号）。一并改成 `["Olfactory bulb"]`，并在配置注释里显式标注这个坑（"不排除任何区域" 要写 `[]`）。

**验证（合成幻影：图谱前端多一个"嗅球"、样本没有且压扁到 61%）**：

| | 补缺口 | 样本前缘落点（图谱本体前缘 0.62mm） |
|---|---|---|
| 不排除嗅球 | 91% | 0.34mm |
| 排除嗅球（atlas mask） | **95%** | 0.24mm |

1. **主要风险已排除**：图谱侧的排除 mask **不会**像轮廓 mask 那样抑制撑开（91%→95%，反而略好）。机制上说得通——它排除的是一小块具体区域，不会把"未覆盖的图谱territory"整片藏起来。**配置可以放心用。**
2. 顺带印证：这个幻影用 `[100,70,50]` 三层就撑开了 91~95%，支持"去掉轮廓 mask 后 `reg_iterations` 保持不动"的判断。
3. **但排除嗅球并没有阻止样本被推进嗅球地盘**（前缘 0.34→0.24mm，反而更靠前）。**mask 的语义是"这块不计分"，不是"这块不许进"**——被排除的区域没有数据项，SyN 的形变场只是从邻近（正在扩张的）区域平滑外推过去，没有力量把组织推回来。排除真正买到的是"额叶皮层的强度不再被拿去和嗅球强度做匹配"，改善的是内部对应关系，不是外轮廓边界。
4. **这个测试没能回答"标签错配改善了多少"**：幻影的嗅球只有 0.62mm（真实是 2.2mm），且"前缘位置"测的是外轮廓、测不出内部对应关系。留给真实跑完看。

**真实跑完新增一条 QC**：打开 `labels_in_sample.nii.gz` 看额叶最前端还挂不挂着嗅球标签。如果还有，说明仅"排除出计分"不够，需要更强手段（比如用 `atlas.slicing` 把图谱在嗅球处直接裁掉，而不只是 mask 掉）。

**追加（同日）：0810 正式跑之前的配置定版 + 预检**

改完代码后做了一次跑前预检（一次真实跑 1.5~3 小时，可检查的东西不该留到运行时才炸）。查出唯一的拦路项：`output_dir` 还指着 `DevCCF_0809`（已有 8 个文件），直接跑会覆盖掉上一次的结果——而那正是这次要对比的基线（位移中位数 19µm / 上下缺口 2.3mm）；`run.log` 又是 tee 追加写的，新旧日志还会混在同一个文件里。用户已把 `output_dir` 改成 `DevCCF_0810`，0809 保留作基线。

**这次跑的配置定版**（`configs/s12t.yaml`，其余项预检全绿：raw_tiff / 三个图谱文件都在，cells 已配置所以步骤 6 会跑）：

| 项 | 值 | 备注 |
|---|---|---|
| `type_of_transform` | `SyNRA` | |
| `syn_sampling` | 4 | CC 邻域半径，8-08 改的 |
| `reg_iterations` | `[100, 70, 50]` | **有意保持三层不变**——去掉轮廓 mask 后三层已够（幻影 91~95%），这样变量干净 |
| `mask.auto_brain_mask` | `true` | 配置不用改，代码侧已自动改走 `prealign_moving_mask` 通道 |
| `mask.atlas_exclude_regions` | `["Olfactory bulb"]` | 本轮新启用 |
| `mask.sample_damage_mask_path` | 未配置 | |
| `crop_for_registration` | x[440,1790] y[50,3977] z[20,157] | 原始 TIFF 体素序号 |
| atlas | DevCCF P04 20µm, orientation `[1,-3,2]`, slicing `[[320,640],null,null]` | 右半球 |

**本轮同时改了两处**（mask 角色分离 + 嗅球排除）。前者是决定性的、幻影验证充分（12%→101%）；后者影响较小且方向明确。**如果结果仍不理想，主要怀疑对象是形变量本身，不是嗅球那一项。**

**跑完的 QC 清单**：
1. `run.log` 里确认 `prealign_mask=True, sample_mask=False, atlas_mask=True`（这是本次改动是否生效的直接证据）
2. 量 `1Warp.nii.gz` 的位移分位数 —— 0809 是中位数 19µm / 90分位 100µm / 最大 392µm，这次应有数量级变化
3. 量 warp 后样本 vs 图谱的包围盒 —— 上下那 2.3mm 缺口应基本消失
4. `labels_in_sample.nii.gz` 里额叶最前端还挂不挂着嗅球标签（见上一条追加里 "mask 不是一堵墙" 的说明）

**顺带发现（没动）**：`configs/s12t.yaml` 里 `atlas_variants.devccf_p04` 上方那段注释还停留在 8-06 跑 Affine-only 诊断的时期（"确认 Affine 效果之后把 type_of_transform 改回 SyNRA、output_dir 改回 DevCCF_ver2_0804 再正式跑"），跟现在的状态已经对不上了，容易误导以后翻配置的人。没有替用户改，留作提醒。

---

## 2026-08-10（续）：0810 真实结果 QC —— mask 修复救回了 Affine，SyN 仍不动；接入 landmark 初始形变场

**0810 QC 结果**（对比 0809，两次唯一差别是 mask 角色分离 + 嗅球排除）：

改动确认生效：`run.log` 里 `atlas_mask=True, sample_mask=False, prealign_mask=True`，嗅球排除 482734 体素，命令行 `CC[...,1,4] --convergence [100x70x50,...]` 全对。register 步骤 2:13:05。

| 指标 | 0809 | 0810 |
|---|---|---|
| 图谱悬空占比 | 34.9% | **25.0%** ✓ |
| 左右缺口(内侧) | +0.66mm | **+0.14mm** ✓ |
| 前后缺口 | 0.74 / 0.72mm | 0.46 / 0.76mm |
| 上下缺口(腹侧) | +1.66mm | +1.56mm ✗ |
| **SyN 位移** 中位/90分位/最大 | 19/100/392µm | **20/95/380µm** ✗ |

**结论：全局对齐（Affine）明显改善，SyN 的局部形变量一字未动。** 用户目视也确认"中线挪过来了"，但图 3 那种扭曲区域、海马边缘仍然没对齐。

这跟幻影实验的差异说明了问题的性质：幻影里两边强度纹理完全对应，去掉 mask 后 SyN 能补完剩余缺口；真实数据在扭曲区/海马边缘**强度信息本身就建立不起对应关系**。参数只能让优化器"看得见、走得快"，不能替它回答"这块该对到哪"。**参数路线的收益已经吃完**（CC 半径、金字塔、mask 角色、嗅球排除四项全部生效且验证过），符合之前定的判据：升级到 landmark。

（用户猜测中线附近可能有半球切割损失。数据部分支持：内侧缺口已降到 0.14mm，说明整体切割损失不大；但局部扭曲区完全可能是局部缺损+扭曲叠加，全局数字看不出来——而这正是 landmark 能编码的信息。）

**分工**：交互式标点工具归 `/home/fyu7/My_project/Registration_toolkit`（用户在那边用 Claude Code 写 `fit_initial_transform.py`，prompt 已给，含本轮实测出的方向语义/双向场/坑）；本仓库负责 pipeline 侧接入。

**新增代码（本仓库）**：
- `register.py`：`register_to_atlas` 加 `initial_transform` / `initial_inverse` 参数。给了 `initial_transform` 就**跳过** Translation 预对齐（`ants.registration` 只接受一个 initial_transform，且 landmark 场已经同时确定了位姿和粗形变）。
- `register.py` 新增 `_repair_invtransforms_for_initial()`，修**两个都是静默的**问题：
  1. **ANTs 无法给 `-r` 传入的形变场求逆**，初始形变在 `invtransforms` 里直接缺席 → 所有 atlas→sample 产物系统性偏移。实测：缺这一环 atlas→sample Dice 0.86，补上（用同一批点交换角色再拟合的反向场，放在列表**最前面**，因为 ANTs 的变换列表是从后往前作用的）0.99。
  2. **antspyx 自己的列表构建对这种文件布局是错的**：它 glob 后只丢弃**一个**正向 warp（`idx != findfwd[0]`），而有初始变换时磁盘上有两个（`0Warp` 初始场 + `2Warp` SyN 结果），于是 **SyN 的正向 warp 被留在了 invtransforms 里**。复现 antspyx 的 glob 逻辑验证过：返回的是 `[1GenericAffine.mat, 2InverseWarp.nii.gz, 2Warp.nii.gz]`。`fwdtransforms` 不受影响（重放对比 `warpedmovout` 最大差 0.0，已验证正确）。
  - 同时把反向场复制成 `{prefix}0InverseWarp.nii.gz`（**在 registration 返回之后才复制**——调用期间多一个 `*InverseWarp` 文件会改变 antspyx 自己的 glob 结果）。
- `transforms.load_saved_transforms()`：认两种文件布局（有无初始变换时 ANTs 的 stage 编号整体后移一位），否则 `registration_eval.py` 会静默拿错变换。
- `config.py`：`registration.initial_transform` 块校验，**`inverse_path` 强制必填**（缺了不是报错而是静默算错，所以挡在配置阶段）。
- `pipeline.py`：透传 + 日志新增 `initial_transform=` 字段。
- `configs/config.example.yaml`：补配置示例和说明（默认注释掉）。

**验证结果**（合成幻影，样本沿一轴压到 0.65，27 个点对按解析映射生成）：
1. `fwdtransforms = [2Warp, 1GenericAffine, 0Warp]`、`invtransforms = [0InverseWarp, 1GenericAffine, 2InverseWarp]`（修复后）。
2. 正向 sample→atlas Dice **0.993**（未配准 0.785）；反向 atlas→sample Dice **0.990**。
3. `load_saved_transforms` 重建的两个列表与 live 返回**完全一致**，用它重放反向 Dice 同为 0.990。
4. `test_pipeline_smoke` / `test_brain_mask_smoke` / `test_label_correction_smoke` 全过；`load_config('configs/s12t.yaml')` 正常且 `initial_transform=None`（未启用的样本完全不受影响）。

**已实测记录的关键事实**（给上游脚本用）：`ants.fit_transform_to_paired_points(moving_pts, fixed_pts)` 返回的是**重采样方向**（apply 到 fixed 空间点得到 moving 空间对应点；已知形变幻影上 apply(fixed点) 距 moving 对应点 5.8µm、距原点 106µm），可直接当 `initial_transform`，**不需要反转**。

**下一步**：用户在 Registration_toolkit 写完 `fit_initial_transform.py` → 用 `place_landmarks.py` 标两份 CSV（样本侧 + 图谱侧，**同一批解剖位置同一顺序**；图谱侧必须标在转向裁剪后的缓存图谱 `P04_LSFM_20um_1_-3_2__320-640_full_full.nii.gz` 上，不能用原始 DevCCF 文件）→ 跑脚本出两个场 → config 填 `registration.initial_transform` 两行 → 重跑。标点重点覆盖：扭曲区、海马轮廓、腹侧（1.56mm 缺口所在）、疑似中线缺损区；另留 5~8 个点**不参与驱动**，只给 `registration_eval.py` 算 TRE（拿驱动点评估自己是循环论证）。

---

## 2026-08-11：转向区域引导（guide_regions）——修 guide 分支 + 一个失败的基准实验 + 三条被纠正的错误说法

**方向变化**：用户否掉了 landmark 驱动路线（两侧手标点、靠行号配对，工作量大），改成**区域引导**：在样本上圈 3~5 个脑区，图谱侧**不用画**——图谱自带完整标注，按名字取 label 即可。这正是 `guide_regions`（`multivariate_extras`）的机制，antspyx 也有现成的 `ants.label_image_registration()`（内部就是算两侧区域质心拟合初始变换 + 每个区域当 MSQ 额外项）。`place_landmarks.py` 不作废，但用途退回**评估**（TRE），不再做驱动。

**调研：DevCCF 本体里没有用户想画的那两个区**（决定了可选区域清单）。实测 192 个真实出现的 label：
- **没有 hippocampus 标签**，只有碎片：dentate gyrus 0.73mm³、subiculum 1.15mm³、presubiculum 0.31mm³、fimbria 0.23mm³、perihippocampal cortex 0.77mm³。CA1/CA2/CA3 完全没有。
- **cortex 没有单一标签**，是 36 个分层（`layer N of FCx/PCx/OCx/...`），合计 30.1mm³，可以并起来用。
- 可用的大块：alar plate of m1 (7.1)、cerebellar hemisphere (4.2)、alar plate of p2/alar thalamus (3.8)、corpus callosum (2.8)、cerebellar vermis (1.7)。
- 备选路线（未采纳）：切回 `demba_p5`，它配的是成年 CCFv3 本体，有干净的 `Hippocampal formation`/`Isocortex`。代价是整条配准路线分叉。

**改动：修 `guide_regions` 分支**（它停留在 8-04 之后所有修复之前的状态，用户一打开就会绕过全部修复）：
- 之前**没有 mask 预对齐** → 用 ANTs 默认质心初始化，正是 8-04 查出会把半球图谱配歪的那个。
- 之前 Affine 直接吃 `moving_mask` → 若那是脑轮廓，就是 8-10 查出的"Affine 不肯放大、只补 9% 缺口"的坑。
- 现在：Translation 预对齐（用 `prealign_moving_mask`）→ Affine（不吃轮廓 mask）→ SyNOnly + 引导项，与主分支一致；并支持 `initial_transform` 覆盖预对齐。docstring 里"（outside the guide_regions branch）"那句限定去掉了。

**一个失败的基准实验（记下来是为了别再犯）**：想量化"3~5 个粗糙手圈区域能帮多少"，跑了两版幻影，**两版都无效**：
1. 第一版：给整个体积加了噪声后用 `>0.1` 阈值量尺寸，背景噪声被算成组织 → 两边都报满长度、"需撑开 0.00mm"、后续百分比全 nan。
2. 第二版（噪声只进强度图、尺寸用无噪声 body mask 量）：**基线太容易**——无引导就已补上 98% 缺口、全脑 Dice 0.972，说明该幻影里强度信息完全够用，**和真实数据的困境正好相反**。在这个前提下粗糙引导只会把结果往回拽（0.972→0.940→0.933）。而且 roughen 的腐蚀把小区域弄没了（样本侧 dors 0.00mm³、cc 只剩图谱侧的 20%），喂了垃圾进去。

**从失败实验里仍然站得住的两点**：
- 引导确实能在特定区域产生大幅改善：cbl 0.68→0.96、dors 0.06→0.46。与当初接入时那个**为"强度提供零信息"设计的**验证一致（两张完全相同的强度图 + 位置不同的球体，Dice 0.185→0.924）——那个实验才对得上真实场景。
- **系统性尺寸偏差是真正的杀手，对小结构致命**：cc 样本侧只有图谱侧 20%，引导直接起反作用（Dice 0.46→0.15）。随机抖动会互相抵消，**始终画大/画小不会**。

**给用户的实操建议**（从机制推的，不是从那组无效数字推的）：优先大结构（1mm³ 以下的别选）；画之前叠着 `labels_in_sample` 确认自己对边界的定义和图谱标签语义一致（这比画得精细重要得多）；只在真正对不上的地方加引导，强度已经配得好的地方别加。**决定不再跑第三版幻影**——构造得出"强度失效"，但那是我设计的失效方式，不是真实样本的失效方式；真实数据本身就是最好的测试。

**三条我在给 toolkit 的 prompt 里写错、被对方实测纠正的说法（这里记下正确版本，避免以后翻记录时把错的当事实）**：
1. **sidecar 格式我援引错了**。仓库真实约定是 `edit_sample_labels.py` 写、`registration_eval.py:366 load_region_annotation_hint()` 读的 `<mask>.annotated_slices.json`，内容是 `{"hand_drawn_slices": [...]}`，**针对单个二值 mask**，结构里没有 per-label 分解也没有 label→区域名映射（我错误地说它已支持，还错误地把 `annotate_gt_sam.py` 一并援引为同一约定）。对方的解法是写两份：`.regions.json`（新格式，带 label→名字和逐 label 的层号）+ `.annotated_slices.json`（所有 label painted planes 的并集），这样产物能原样落进现有 eval 路径。
2. **合并插值的失效模式是"湮灭"不是"边界渗透"**。我说的"多 label 混在一起插值会在边界互相污染"是想当然。实际：符号距离场在两个**不相交**截面之间插值，`{½·d_A + ½·d_B < 0}` 在两块距离超过各自尺度时是**空集** —— 夹在异区域关键层之间的层**整段消失**。合成用例实测 700 体素静默丢失，逐 label 插值后为 0。**结论（必须逐 label 插值）不变，理由换成这个。**
3. **重叠只可能来自插值，不可能来自关键层**。单个 napari Labels 图层每个体素只有一个整数，关键层上物理上无法重叠。我原来建议的测试构造（"故意让两个 label 重叠"）不可实现；唯一能触发的路径是把一个 label 的关键层**夹在**另一个的关键层之间，让插值出的实体穿过彼此（实测 125 个争夺体素，精确可算）。

**另一个决定**：`display_scale_zyx`（各向异性显示，原始 TIFF 是 2.6/2.6/32µm，12:1，不设的话 napari 里 z 压扁 12 倍）**两条 kind 路径都要加**，放进共享的 `_launch_viewer`。我原先说"不要动 `_run_mask`"指的是**导出语义**（密集编辑/不插值/不取反 —— 那些是 mask 与 guide 之间真实存在的差异，不该合并掉）；显示比例是**图像的属性不是 kind 的属性**，且只影响显示、不改变导出的体素索引。

**下一步**：用户按 prompt 在 Registration_toolkit 改好 `paint_mask.py` 的多脑区支持 → 在原始 `registration.tif` 上给 3~5 个脑区各画 5 层左右 → Registration_ants 这边还需要实现「图谱侧按脑区名自动生成区域」（现在 `mask.guide_regions` 要求两侧都给文件路径，得改成图谱侧只写名字）→ 跑一次带引导的配准。

**追加（同日）：实现「图谱侧按名字自动生成引导区域」——只画样本侧**

用户决定 label→脑区名的权威映射放在 ANTs 侧 config、不放画图工具里。这不只是偏好问题，**结构上也只能这样**：画图工具的 `region_labels` 是 `{label: 一个名字}`，而真实需求是一对多（DevCCF 里 cortex 是 36 个分层、cerebellum 是 hemisphere+vermis 两个）。画图工具那份降级成人类可读的备忘（napari 窗口显示、落进 `.regions.json`），下游不读。

**新增/改动**：
- `atlas_utils.py`：抽出 `_structure_ids_matching(structures, names)`（子串匹配 + 经 structure_id_path 带上所有子结构），`build_region_exclusion_mask` 改为复用它（行为不变，已回归验证仍是 482734 体素）；新增 `build_region_inclusion_mask(annotation_arr, structures, include_names)`，返回 `(mask, matched)`，`matched` 是 `{结构名: 体素数}`。**返回 matched 是刻意的**——子串匹配不中会给出全 False 的 mask 且不报错，下游看起来就是"这个引导什么也没做"。
- `config.py`：`mask.guide_regions` 支持**两种形式**，按类型区分：dict = 新的多区域形式（`regions_mask`/`voxel_size_um`/`atlas_names`/`weight`），list = 旧的一区一对文件形式（保留，用于没有对应本体名字、必须手画图谱侧的区域）。`voxel_size_um` 必填（在原始 TIFF 上画的 mask 头里没有 spacing，读回来是 1,1,1）；`atlas_names` 的 key 归一成 int、value 归一成 list（允许写单个字符串）；`weight` 可以是单值或 `{label: 权重}`。
- `pipeline.py`：新增 `_build_guide_regions_from_labels()`。读多 label 文件（用 `io_utils.load_nifti_stack_as_ants` 显式重建 spacing、丢弃文件头）→ 若网格与 `sample_fine_prep` 不同则 `resample_image_to_target(interp_type="genericLabel")` 重采样（物理空间一致：origin 0、spacing 即微米、裁剪只平移 origin，所以是纯粹的 regrid）→ 逐 label 抠图谱侧 → 组装三元组。日志逐 label 打印样本/图谱体素数和命中的结构名。
- `configs/config.example.yaml`：补完整配置示例 + 关键层间隔的实测数据。

**三条刻意做成报错而非静默的检查**（今天已经因为静默失效吃过两次亏）：
1. `atlas_names` 里某个 label 一个结构都没匹配到 → 报错（否则引导项永远无法被满足）
2. `regions_mask` 里画了、但 `atlas_names` 没配的 label → 报错（画了却用不上，肯定是配漏了）
3. 配了 `atlas_names`、但 mask 里没有该 label → 报错

**验证结果**：
1. `build_region_inclusion_mask` 对真实 DevCCF：`["layer 1 of","layer 5 of"]` 命中 16 个结构 17.50mm³，名字全部核对正确；不存在的名字给出空 mask + 空 matched。`build_region_exclusion_mask` 回归检查仍是 482734 体素（重构没改行为）。
2. `_build_guide_regions_from_labels` 端到端：合成的"原始 TIFF 网格多 label 文件"（各向异性 spacing）+ 真实图谱，正确产出 2 个三元组，**图谱侧在图谱网格 (320,800,560)、样本侧被重采样到配准网格**，断言通过。
3. 三条错误路径逐条断言报错信息正确。
4. `test_pipeline_smoke` / `test_brain_mask_smoke` / `test_label_correction_smoke` 全过；`load_config('configs/s12t.yaml')` 正常且 `guide_regions=None`（未启用不受影响）。
5. **没跑真实数据**——等用户画完区域。

**关键层间隔的实测数据**（用真实 DevCCF 结构按用户实际画的切面方向采样后插值，与真形状比 Dice；原始 TIFF z 间距 32µm，"每 20 层" = 640µm）：

| 结构 | 体积 | 每5层 | 每10层 | 每20层 | 每40层 |
|---|---|---|---|---|---|
| cortex（36分层并集） | 27.5mm³ | 0.966 | 0.902 | **0.816** | 0.629 |
| cerebellum | 6.0mm³ | 0.988 | 0.972 | **0.901** | 0.541 |
| corpus callosum | 2.8mm³ | 0.933 | 0.762 | **0.396** | 0.145 |
| dentate gyrus | 0.73mm³ | 0.938 | 0.778 | **0.360** | 0.104 |

结论：形状连续的大结构每 20 层够用（cortex 10 层、cerebellum 5 层的工作量）；细长/弯曲的结构每 20 层直接废掉（0.36~0.40 已经是形状错了，考虑到"系统性偏差比随机抖动危险"，这种引导比不加更糟），要画就每 10 层。**首尾层必须画**——`interpolate_sparse_mask` 只在关键层之间插值，`[min,max]` 之外一律留空。

**下一步**：用户用改好的 `paint_mask.py`（多 label 已落地，commit `22494cb`）在原始 `registration.tif` 上画 3~5 个区 → 填 `mask.guide_regions` → 跑。建议先只画 cortex 一个区试手、导出后拖回 napari 核对插值层形状，走通再一次画完。

---

## 2026-08-18：图谱中线"永远笔直"的真正原因 —— SyN 形变场的零边界条件（实测确诊）

**现象**（用户报告）：配准后图谱半脑的中线依然笔直，完全不贴合样本那条偏斜的中线；图谱其他位置**有**正常形变。此外图谱整体偏大、没有按画的 mask 范围收缩。

**诊断（先量后改，没有靠猜）**。拿 `DevCCF_0818` 那次跑的变换直接量：

1. **六个面全部精确为零**。`tsc12t_1Warp.nii.gz` 和 `tsc12t_1InverseWarp.nii.gz` 两个场，六个面的 `max|位移|` 都是 **0.000000**，而内部中位数是 110µm（组织内）。这是 ANTs/ITK 在 fixed（图谱）网格边界上对形变场施加的 **Dirichlet 零边界条件**。
2. **中线正好落在那个被钉死的面上**。`slicing: [[320, 536], ...]` 的下界 320 就是解剖中线，裁完之后 axis0 的 lo 面上有 **105259 个组织体素**，间隙 0。其余五个面各有 10 体素间隙。
3. **位移沿 x 轴从 0 线性爬升**：x=0 是 0.0000，x=1 是 11.9µm，x=2 是 21.8µm……到 x≈25~30 才接近内部中位数。爬升段就是正则化从被钉死的边界往内平滑的结果。

**结论**：这**不是**引导不够、metric 不行、迭代不足。约束加在**形变场本身**上，不在目标函数里，所以换 metric、加迭代、加 `guide_regions` 引导项、把 weight 调到天上去，**全都不可能移动那个面**。之前讨论的"给中线加 guide outline"如果不先解决这个，做了也是白做。

**幻影验证**（`hemisphere` 方块，内侧切面平直 vs 样本侧倾斜 10 体素，SyNOnly + CC(4) + [100,70,50]）：

| pad | 切面 \|d\| max | 切面 \|d\| mean | Dice(atlas→sample) 前→后 | warp 后切面倾斜量 vs 真值 |
|---|---|---|---|---|
| 0 | **0.000** | **0.000** | 0.989 → 0.990 | **0** vs 4 |
| 5 | — | — | 0.982 → **1.000** | **7** vs 7 |
| 10 | 83.8 | 41.3 | 0.982 → **1.000** | 7 vs 7 |
| 20 | 85.7 | 41.2 | 0.982 → **1.000** | 7 vs 7 |
| 40 | 84.5 | 41.1 | 0.982 → **1.000** | 7 vs 7 |

pad=0 时切面位移恒为 0、倾斜量永远是 0（内部却有 64.6µm 形变）；**只要有 padding 就完全解决**，5 就够，再大没有额外收益。

**修法：补零背景，不是放宽 slicing**。新增 `atlas.background_margin_voxels`（走 `atlas_variants`），语义是"补到组织与六个面之间至少隔这么多空体素"，**不是**"再加这么多"——已经够宽的面不动，重复跑不会越补越大。

**为什么不能改成"slicing 放宽 + 把对侧组织 mask 掉"**（我最初给用户的建议，被这次测量推翻）：内侧面那个**组织/背景的强度台阶**正是 metric 唯一能咬住的边；放进真实对侧组织就把台阶抹掉了，mask 只是"不计分"、并不会重建那个边。补零是把被钉死的面挪开、同时把台阶原样留下 —— 严格更优。

**改动**：
- `atlas_utils.py`：新增 `background_pad_width()`（测量逻辑 + 完整原因写在 docstring 里）；`_atlas_prep_postfix` 加 margin 段，**margin 为空时输出不变**，所以此前写好的缓存文件名照旧有效；抽出 `_read_atlas_array_xyz`/`_write_atlas_array_xyz`；`prepare_custom_atlas` 改为**两个文件一起处理**（padding 把它们耦合了：pad 宽度从 annotation 量出来，必须原样应用到 template，否则两者不再共享网格），缓存改成"两个都在才复用"。
- `pipeline.py`：新增 `_log_atlas_face_clearance()`，每次跑都打印六个面的实际间隙；**只要有面是 0 就 WARNING**。这个失效模式静默且巨大（配准照样收敛、metric 照样下降，那个面只是悄悄不动），日志里不写出来根本看不见。
- `config.py`：`atlas.background_margin_voxels` 校验（非负整数、拒绝 bool/float/字符串），并入 `_ATLAS_VARIANT_FIELDS`；非 custom 图谱用它会报错。
- `tests/test_new_features_smoke.py`：新增 `test_atlas_background_margin`（只补不足的面、组织值不变、幂等、缓存 key 向后兼容）。

**真实图谱验证**：DevCCF P04 + 现有 orientation/slicing + margin=20 → 形状 `(216,582,353)` → `(246,602,373)`，六个面间隙全部 =20，192 个 label 一个不少。体积 +24%。

**测试**：`test_pipeline_smoke` / `test_brain_mask_smoke` / `test_label_correction_smoke` 全过；`test_new_features_smoke` 新增项通过，**该文件第 5 项 `test_assign_cell_regions` 失败是既有问题**（`KeyError: 'invtransforms'`，测试里构造的合成 `reg` dict 缺这个键），改动前后完全一致，本次没动。

**下一步 / 跑完的 QC**：
1. `run.log` 里确认六个面的 clearance 都是 20、没有 FLUSH 警告。
2. 量新 `1Warp.nii.gz` 在中线那一层的位移 —— 0818 是恒为 0，这次必须显著非零。
3. 目视 `labels_in_sample.nii.gz` 的中线是否跟着样本偏斜了。
4. **注意**：本轮只解决了"中线被钉死"。用户报的另外两条（图谱整体偏大、没按 mask 范围收缩）是**全局尺度**问题，归 Affine 阶段管，padding 不会修好它 —— 而 `register.py` 的 guide 分支里 Affine 是**完全不吃 guide_regions 的**（引导项只进 SyNOnly），手画区域携带的尺度信息在 Affine 阶段被整个丢掉了。如果这次跑完中线对了但整体尺度还是不对，下一个要查的就是这里（`ants.label_image_registration` 的做法是用两侧区域质心先拟合初始变换）。

---

## 2026-08-21：0820_pad20 结果 QC —— 中线钉死已解决，但图谱只盖住样本 57%；引导 mask 查出硬伤 + 新增 `scripts/qc_guide_mask.py`

**用户报告**：中线解剖结构仍呈直线；很多非 pallium 结构盖在 pallium 上；label7（融合 diencephalon + peduncular hypothalamus + hypothalamic prosomere 2，为了做出完整中线结构）没起作用；怀疑 pad 也没用。并提出三个机制问题（形变有没有限度、相邻脑区是否互相影响、mask 是否每个平面都要覆盖所有脑区）+ 要一个插值后 mask 的可视化工具。

**结论先行：pad 是有效的，用户看到的"直线"已经是另一个问题了。** 三件事按重要性排：图谱整体太小 > 引导 mask 有硬伤 > 中线（已修）。

### 1. 中线钉死：已解决（实测）

拿 `DevCCF_0820_pad20/transforms/tsc12t_1Warp.nii.gz` 在图谱内侧面（padding 之后组织起始于 axis0=20）量组织体素的位移：

| | 0818（pad 前） | 0820（margin=20） |
|---|---|---|
| 内侧面组织体素 \|d\| mean | **0.0** | **154.7 µm** |
| 同上 median / max | 0.0 / 0.0 | 150.7 / 553.6 µm |
| x 分量 p5~p95 | 0 | −224 ~ +236 µm（±11 体素） |

六个面本身仍恒为 0（Dirichlet 边界照旧），但那六个面现在全是 padding 背景，`run.log` 里 clearance 六面均为 20、无 FLUSH 警告。**"中线不能动"这条约束已经不存在了。**

### 2. 真正的问题：Affine 把图谱缩得太小，43.4% 的样本组织根本没有标签

在样本空间量 `tsc12t_labels_in_sample.nii.gz` vs `tsc12t_brain_mask.nii.gz`（brain mask 在裁剪网格上，按 origin 差算出偏移 (57,6,32) 贴回全网格）：

| 指标 | 值 |
|---|---|
| 样本脑体积 | 65.9 mm³ |
| 其中**有**标签 | 37.3 mm³ |
| 其中**无**标签 | **28.6 mm³ = 43.4%** |
| 标签落在脑外 | 0.1 mm³ = 0.3% |
| 内侧缺口（label 内侧缘 − 组织内侧缘） | mean **27.6 体素 = 552 µm**，中位 29，p90 45 |
| 有 >100µm 内侧缺口的 (y,z) 列 | **90.1%** |

**图谱几乎完整地缩在样本内部**（只有 0.3% 溢出）。全局尺度：图谱组织 104.7 mm³ / 样本 65.9 mm³ → 真实体积比 **0.629**（线性 0.857；三轴范围比 0.81 / 0.90 / 0.65，z 方向差 2.3mm）。而实际用的 Affine `det = 0.390`（奇异值 1.003 / 0.736 / 0.529）—— **体积上比应有值又多缩了 0.62 倍（线性 0.85）**。

用户看到的"中线是直线"因此是这个的表现，不是钉死：图谱内侧缘停在离样本中线 0.5mm 的组织内部，那里没有强度台阶可咬，SyN 无处发力，于是保持 Affine 给的平面形状（label 内侧缘的 std 只有 7.3 体素，而组织内侧缘是 15.4）。同理"非 pallium 结构盖在 pallium 上"——整个图谱被压小并内缩，区域身份整体错位。

**SyN 补不动这个缺口**：全场位移中位 10.6 µm / p90 190 µm / **max 629 µm**。需要的内侧修正平均就是 552 µm、90% 的列都要，而 629 µm 是它在**整个体积任何一处**能达到的上限。这不是加迭代能解决的量级差——量级本身归 Affine 管。

### 3. 引导 mask 的硬伤（`atlas/mask/s12t_guide6.nii.gz`）

**失效模式 A：游离关键层。** 画的时候一次误点就在那一层留下几个体素的"关键层"。`interpolate_sparse_mask` 在**相邻**关键层之间插符号距离场，所以一个 4 体素的关键层夹在两个 50 万体素的关键层中间，不是"影响很小"，而是**把它两侧的整个区间都毁掉**——两个 SDF 相差超过各自尺度时 `{(1−t)d₀ + t d₁ < 0}` 是空集，中间层整段消失（8-11 记录过这个机制，这次是第一次在真实数据上抓到）。

**失效模式 B：与其他区域不一致的尺寸。** 引导项是 MeanSquares，编码的是"这两块是同一个东西"。

实测（`scripts/qc_guide_mask.py` 输出，painted/atlas 体积；`rel` = 该比值 ÷ 各区共享的中位数 49%）：

| lab | 区域 | painted mm³ | atlas mm³ | ratio | rel | 问题 |
|---|---|---|---|---|---|---|
| 1 | pallium | 22.47 | 43.04 | 52% | 106% | 游离关键层 z=148(259vox)、156(16vox)；3 层空 |
| 3 | subpallium | 4.98 | 10.35 | 48% | 98% | 游离关键层 z=124(**4vox**)、149(32vox)；3 层空 |
| 4 | midbrain | **0.37** | 7.94 | **5%** | **9%** | 游离关键层 z=41(4),46(1),50(1),99(371),110(1)；**32 层空**（70 层里） |
| 5 | hindbrain | 7.61 | 20.80 | 37% | 74% | 偏小 |
| 6 | olfactory bulb | 2.11 | 3.86 | 55% | 110% | ok |
| 7 | diencephalon | 6.59 | 13.02 | 51% | 102% | ok |

**关键的读法修正**：不要看 ratio 的绝对值。样本整体就只有图谱的 63%，每个区都继承这个因子，Affine 本来就该吸收掉它——**六个区一致地偏小不是六个错误**。要看的是**偏离共享因子的那一个**。按这个读法，1/3/6/7 画得是一致的（98~110%），5 偏小（74%，边缘），**4 是灾难（9%）**：它在告诉 SyN 把整个图谱中脑（7.94 mm³）挤进一个 0.37 mm³ 的碎片里，而 SyN 是微分同胚的，这个拉扯必然连带周围所有区域——正是"非 pallium 结构盖到 pallium 上"的一个直接来源。

**关于 label 7 为什么没做出中线**：引导项是**区域重叠**项不是边界项——填充区内部两边都是 1，梯度为 0，只有边界带在出力。所以合并多个脑区成一个大 label，约束的是**这个合并块的外轮廓**；而实际画出来的 label 7 内侧缘落在组织内部（见 `atlas/mask/s12t_guide6_qc/label7.png`），并不是中线本身。要让引导约束中线，label 的边界必须**就是**中线。

### 4. 回答三个机制问题

- **形变有限度吗**：没有幅度硬上限（`SyN[0.2, 3, 0]`，total field variance=0 = 贪心累积），但有两个实际限制：(a) update field 每次迭代按 σ=3 体素高斯平滑 → **形变场里做不出比 ~3 体素更细的空间细节**，最粗层（shrink 4）那是 240 µm 的物理平滑；(b) 迭代预算 × 步长决定实际能走多远，本次实测上限就是 629 µm。
- **相邻脑区互相影响吗**：**是，而且是强制的**。σ=3 的平滑把 ~9 体素（180 µm）半径内的位移耦合在一起；更根本的是 SyN 保持微分同胚（雅可比恒正、不许折叠撕裂），所以任何一块的形变都必须和邻块自洽。这就是 label 4 能污染 pallium 的机制。
- **每个平面都要覆盖所有脑区吗**：**不需要**，覆盖率不是要点。要点是每个 label 的 3D 范围要和它配对的图谱结构描述**同一个体积**——系统性画小比不画更糟（本轮 label 4 是实例）。真正必须注意的两条是：首尾关键层必须画（`interpolate_sparse_mask` 在 `[min,max]` 之外一律留空），以及关键层间不能有游离层。

### 5. 新增 `scripts/qc_guide_mask.py`

跑配准前审 mask 用，不需要跑配准：

```
python scripts/qc_guide_mask.py configs/s12t.yaml            # 报表 + 每个 label 一张 PNG
python scripts/qc_guide_mask.py configs/s12t.yaml --napari   # 再打开 napari（带 display_scale）
python scripts/qc_guide_mask.py configs/s12t.yaml --no-atlas # 不碰图谱，只查插值
```

做四件事：(1) 逐 label 列出关键层面积并标出游离层；(2) 找出画的跨度内被插值弄空的层；(3) 用 pipeline 同一条路径（`atlas_ids` > `atlas_names` > sidecar `region_ids`，含 `atlas_exclude_ids`）算图谱侧体积，报 ratio 和相对共享因子的 `rel`；(4) 出 PNG 拼图——**手画层绿框、插值层红框、空层品红框**，青色轮廓叠在原始 tiff 上，游离层和空层强制入选（不管等距采样有没有采到）。PNG 默认写到 `<mask 同目录>/<stem>_qc/`（在 gitignored 的 `atlas/` 下）。

napari 那条路径用 `tifffile.memmap` 惰性读 2.8GB 栈，contrast 只从中间一层采（否则 napari 会扫全栈），`scale=(32, 2.6, 2.6)` 免得 z 被压扁 12 倍。

### 6. 下一步（按付出/收益排）

1. **修 mask**（便宜、确定）：删掉游离关键层（1: z=148,156；3: z=124,149；4: z=41,46,50,99,110）；label 4 要么重画到真实中脑范围要么进 `ignore_labels`——**照现在这样它一定是负贡献**；label 5 复查一下 74%。改完重跑 `qc_guide_mask.py` 直到没有 PROBLEMS。
2. **修全局尺度**（这是真正的瓶颈）。Affine 给 0.390，真值 0.629，43% 的样本没有标签。当前 guide 分支的 Affine **完全不吃 guide_regions**（8-18 就标记过这个待查项），手画区域携带的尺度信息在 Affine 阶段被整个丢掉。用 6 个区的质心最小二乘拟合 12-DOF Affine 试算过：全用 det=0.329，去掉 label 4 后 det=**0.669**（真值 0.629）——方向对，但 6 个点拟合 12 个自由度太脆（去掉一个点 det 就从 0.33 跳到 0.67），**不能直接当结论**；等 mask 修好、区域数够了再评估。另一条更稳的候选：把整脑轮廓对（`brain_mask` vs 图谱组织 mask）当成第 7 个 `guide_regions` MSQ 项（`ants.label_image_registration` 的"whole brain label"做法）——注意它和 8-10 查出的"轮廓当 moving_mask 会抑制撑开"是**两种不同机制**，那次是"不计分"，这次是"区域匹配项"，不能拿那次的结论否掉这次。风险是样本 brain mask 含有越过中线的蚓部/脑干，会把图谱往对侧拽。
3. 中线本身**这一轮不用再动**——约束已经解除，等尺度对了再看它还差多少。

**没跑真实数据**：本轮全部是对已有产物的测量 + 新工具，没有重跑配准。

### 追加（同日，晚）：更正上面两条 + 查清 Affine 才是瓶颈 + 新增 `scripts/affine_probe.py`

用户指出 **0814（图谱未裁切那次）pallium 表现好得多**。查证属实，并且顺着查下去推翻了上面"下一步"里的第 2 条。

**先更正上面写错的两处**：

1. **"Affine 的目标是 det 0.629"是错的。** 0.629 是样本/图谱的组织体积比，但那个比值来自**样本缺了一块图谱有的地盘**，不是"样本更小"。把两边的组织面积剖面按"平移+缩放"拟合（只在样本自己的范围内）：y 轴最佳 scale **0.99**（rmse 0.043）、z 轴最佳 scale **1.20**（rmse 0.063），都不是 0.63。z 剖面在**纯平移不缩放**下和图谱吻合到样本长度的 80%，然后样本直接停在体积的最后一个平面（组织 z 范围 30..249，体积共 251 层）——**是截断，不是压缩**，图谱在样本覆盖范围以下还有 2.02mm。x 轴这个方法给不出结论（图谱 x 剖面从 98% 峰值起步，那是中线平切面；样本从 0 缓升，是斜的/圆的内侧面，两个形状不同源，rmse 0.321）。
2. **"图谱整体太小 / Affine 缩得太小"这个因果说反了一半。** 不是 Affine 算出了偏小的答案，而是**Affine 在这份数据上根本不稳定**：同一样本同一预处理，只改图谱周围留多少背景，det 就在 0.39~1.00 之间跳。

**关键测量：把线性部分做极分解，分开旋转和拉伸**（只看 det 会漏掉"大旋转 + 各向异性压扁"这类解——它们 det 可以接近正常，却把每个结构都放错位置）：

| | det | 旋转 | 拉伸特征值 |
|---|---|---|---|
| 0814 实跑 | 0.982 | **0.07°** | [0.978, 1.000, 1.005] |
| 0818 实跑 | 0.416 | 12.25° | [0.544, 0.766, 0.998] |
| 0820 实跑 | 0.390 | **21.11°** | [0.529, 0.736, 1.003] |

**0814 的 Affine 是彻底的 no-op**——旋转 0.07°、拉伸全 1，结果就是 Translation 预对齐本身。而紧裁图谱的两次产生了 12~21° 旋转配上 0.53/0.74 的压扁，这两者是一体的：要沿非轴对齐方向压扁就必须配旋转。**那 21° 不是真实位姿修正，是错误压扁的一部分。**

**隔离实验**（`scripts/affine_probe.py`，同一样本同一预处理，只换图谱裁切，每档 15~45 秒）：

| atlas.slicing | 图谱组织 | det | 旋转 | 拉伸 | **无标签** | 溢出组织外 |
|---|---|---|---|---|---|---|
| `[[320,536],[108,690],[129,482]]`（当前） | 104.7mm³ | 0.390 | 21.1° | [0.53,0.74,1.00] | **40.6%** | 2.7% |
| `[[320,536],[108,690],[151,371]]` | 94.5mm³ | 1.000 | **0.01°** | [1.00,1.00,1.00] | **14.8%** | 35.8% |
| `[[320,640],null,null]`（0814 式） | 104.7mm³ | 0.980 | 0.07° | [0.98,1.00,1.00] | 21.0% | 34.0% |
| `z[151,371]+y[143,647]` | 94.1mm³ | 0.525 | — | [1.00,0.83,0.63] | 29.1% | 5.5% |

（样本脑 65.9mm³；全部 Affine-only 不含 SyN。目视叠图与数字一致：当前配置的标签停在组织中间一条竖线上、内侧 1/3 空着；z 裁切后标签覆盖满组织且皮层分层弧线贴着皮层走，多余部分溢到内侧**背景**里——背景没有细胞，无害。）

**结论**：`z[151,371]` 在覆盖率和目视上都最好（40.6% → 14.8%），但它之所以好是因为**裁切替 Affine 干了活**，Affine 本身仍是 no-op。对这个样本这是对的答案（剖面分析独立地也说 y/z 该按 scale≈1 对齐），但**不是稳健的**——换一个真的需要旋转/缩放的样本，会拿到同样的 no-op。

**Affine 在任何一种取景下都没有正常工作，这才是真正的瓶颈**，不该再靠调裁切参数绕过去。上面 8-11 和 8-18 两次标记过的待办项——`register.py` 的 guide 分支里 Affine 完全不吃 `guide_regions`——现在从"可以顺便做"升级成主线：手画区域携带的正是 Affine 需要的解剖对应关系（`ants.label_image_registration` 的做法是用两侧区域质心先拟合初始变换）。**前提是先修好 mask**：label 4 现在是图谱结构的 5%，拿它算质心只会得到垃圾。

**本轮踩到的坑（仓库自己记过，我还是踩了）**：`ants.apply_transforms` 对 **Affine-only** 的 `invtransforms`（一个裸 `.mat`）必须显式传 `whichtoinvert=[True]`；默认的自动推断只特判 `[matrix, warp]` 两元素那种形状，其余一律 all-False，把矩阵**正向**贴上去。症状就是 `transforms._mat_entries_to_invert` docstring 里写的"100% 空"。我第一版探针漏了这个，据此错误地撤回了"裁 z"的建议，又据错误数字改了一轮结论。**探针类脚本复用 `transforms.py` 里现成的封装，不要自己拼 `apply_transforms`。**

**新增 `scripts/affine_probe.py`**：只跑 Translation+Affine，报 det / 极分解出的旋转和拉伸 / 无标签% / 溢出%，可选 `--png` 出叠图；`--slicing` 可给多个变体一次对比。把一次 1.5~3 小时的试错压到 30 秒。识别并显式提示 "Affine no-op"（旋转和拉伸同时是单位）这种情况，因为它在 det 上看起来完全正常。

**下一步（覆盖上面那份清单的第 2 条）**：
1. 先只改 `slicing[2]` 为 `[151, 371]`（`background_margin_voxels: 20` 保留，新的腹侧切面同样需要），跑 `affine_probe.py` 确认，再跑全流程。**一次一个变量**——mask 的问题下一轮再修。
2. 修 mask（游离关键层 + label 4），然后才谈把 guide 区域接进 Affine。

### 追加（同日，深夜）：查清 Affine 为什么不动 —— 强度互信息不足，且找到唯一能真正优化的驱动方式

用户追问"Affine 为什么没工作 / 现在在工作吗"。查到底了。

**1. 机制没坏（控制实验）。** 拿图谱和它自己经过已知仿射的副本配：1.20× 拉伸、10° 旋转、0.85/0.90/0.85 缩放**三个全部精确恢复**（注意返回的是真值的**逆**——`apply_transforms(moving=at, transformlist=[T])` 造出的合成图，registration 求的是 T⁻¹；我第一次判成 FAILED 是比较约定写反了）。控制组 level 1 跑满 48 次迭代、metric 从 −0.9805 走到 −1.7150。

**2. 真实数据上没有梯度可跟（迭代轨迹）。**

| | 控制组 | 真实数据 |
|---|---|---|
| level 1 | 48 iters，−0.9805 → **−1.7150**（改善 0.73） | 12 iters，−0.170210 → −0.170349（改善 **1.4e-4**） |
| level 2 | 31 iters | 9 iters，**变差** |
| level 3 | 31 iters | 10 iters，改善 6e-7 |
| level 4 | 10 iters | 10 iters，**变差** |

**Mattes 互信息量级差 10 倍**（控制组 −0.98~−1.71，真实 −0.12~−0.17）。ANTs 收敛阈值 1e-6，真实数据每次迭代的变化就在这个量级，所以 9~12 次迭代后**正当地**判定收敛退出。不是 bug，不是参数没调好。

**3. 扰动测试（判据：能不能把故意推歪的初值拉回来）。** 起点 z×1.20 → 结果 1.199；起点转 10° → 结果 10.05°；起点 z×0.80 → 漂到 [0.628,0.817,1.00]+11° 更差处。**它把输入原样吐回，没有回拉能力。**

**4. 换 metric / 加 mask / 提采样率全无效**（六个变体 dz 全是 +0.20）：mattes / GC / meansquares、图谱组织 mask + 脑 mask、采样率 0.2→0.8。这些改的是"怎么读信号"，信号本身不存在。

**5. 质心驱动不行**（我一度推荐、当场被否）：6 个区质心最小二乘拟合 12-DOF —— det 0.385 / 旋转 **40.7°** / 无标签 49.1%；去掉 label 4 后 det 0.751 / 旋转 **49.7°** / 26.7%。6 个点定 12 个自由度严重欠定，在点张不满的方向上完全不受约束。**`ants.label_image_registration` 的质心初始化不能照搬。**

**6. 形状驱动可以 —— 这是整个排查里唯一真正在优化的 Affine。** 把区域的**整个二值形状**（高斯平滑 σ=2 给出梯度带）当图像配，meansquares：

| 驱动信号 | 从单位阵 | 无标签 | 从 z×1.20 | 无标签 | 拉回来？ |
|---|---|---|---|---|---|
| 强度（现状） | [1,1,1] 0.0° | 14.7% | [1,1,**1.2**] | 13.0% | ✗ |
| 整脑轮廓 | [1,1,1] 0.0° | 15.0% | [1,1,**1.2**] | 13.0% | ✗ |
| **guide 区域并集（5 个好 label）** | [0.662,0.838,1.013] 2.9° | 27.3% | [0.658,0.838,1.023] 4.3° | 27.5% | **✓** |

两个完全不同的起点收敛到同一个答案 —— **有真正的吸引域**。整脑轮廓不行（形状太简单，约束不住），必须是**多个内部区域**。

**但它的覆盖率反而更差（27.3% vs 14.7%），原因闭环回到 mask**：它忠实地把画的区域配到图谱结构上，而画的区域系统性只有图谱结构的 **49%**。拟合 stretch 立方根 0.825，mask 画小比例立方根 0.788，对得上。

**由此更正本日早些时候写下的一条**：早先说"六个区一致地偏小不是六个错误，Affine 本来就该吸收掉这个因子"。**那只在 Affine 被强度驱动、完全无视 mask 时成立。** 一旦 mask 驱动 Affine，共享因子不再被吸收而是被**照单执行**。画 mask 的要求因此提高：不只要各区之间一致，**每个区都要画到图谱结构真实的解剖范围**。`scripts/qc_guide_mask.py` 里 `RELATIVE_RATIO_OK` 上方的注释已按此改写。

**修正后的下一步（取代之前所有版本）**：
1. **修 mask，画到真实解剖范围**（不再是"一致就行"）。删游离关键层（1: z=148,156；3: z=124,149；4: z=41,46,50,99,110）；label 4 重画或 `ignore_labels`；label 5 的 74% 复查。跑 `qc_guide_mask.py` 到无 PROBLEMS。**这一步现在是阻塞项，不是可选项** —— 形状驱动的 Affine 直接吃它。
2. **把 Affine 改成形状驱动**（`register.py` guide 分支）：用 guide 区域并集的平滑二值图跑 Affine，而不是强度图。这是唯一被实测证明有吸引域的方案。注意**不能用质心**（第 5 条），也**不能用整脑轮廓**（第 6 条表格第 2 行）。
3. `slicing[2]` 改 `[151,371]` 仍然值得做，但它只是让 Translation 落点更好，属于顺带。

**方法论教训**：这一轮判断反复了三次（建议裁 z → 撤回 → 恢复），根因都是**测量本身出错**：漏 `whichtoinvert`（仓库自己 docstring 里写着的坑）、控制实验比较约定写反。**任何"结论反转"之前先复核测量代码，不要先改结论。**

---

## 2026-08-21（续,傍晚)：形状驱动 Affine 接入 `register.py` 并实跑 —— 覆盖率改善,中线位置对齐良好（但这一步的"直"不代表什么)

### 改动

`src/registration_ants/register.py` 的 guide 分支：
1. 新增 `_guide_union_image()`——把 `guide_regions` 里每一侧（图谱/样本)的画出区域做并集、高斯平滑（σ=2 体素),给 Affine 当匹配目标。
2. guide 分支的 Affine 现在用这个形状图跑（`aff_metric="meansquares"`),不再用原始强度图——本日早些时候已经测出强度在这份数据上没有可用梯度（见上文"Affine 为什么不动"整节),形状驱动是唯一实测有吸引域的方案。
3. `type_of_transform` 在 guide 分支里原来完全不生效（不管填什么,永远跑完 Rigid+Affine+SyN)。现在 `type_of_transform.lower() == "affine"` 会在 Affine 跑完后直接返回,不再进入 SyN——配合 `../Registration_toolkit/affine_probe.py` 的诊断工作流用。
4. 顺带修了一个自己引入的 bug：早停到 Affine 的这条路径最初没有把 `outprefix` 传给那次 `ants.registration()` 调用,导致 `.mat` 只写到系统临时目录、不落进这次运行的 `transforms/`。已修（对继续跑 SyN 的路径无影响,那条路径靠后续 SyN 调用的 `outprefix` 重新落盘)。

Mask 侧：用户已经在 `paint_mask.py` 里删除 label 1/3/4 的游离关键层,重新导出为 `s12t_guide7.nii.gz`,并把 `configs/s12t.yaml` 的 `mask.guide_regions.ignore_labels` 从 `[2]` 改成 `[2, 4]`（label 4/midbrain 关掉——见下方"label4 为什么关掉"）。

### 跑了什么

`configs/s12t.yaml`：`type_of_transform: affine`（诊断用,早停),`atlas.slicing` 仍是 `[[320,536],[108,690],[129,482]]`（没改,腹侧裁切那条建议本轮没动),`output_dir` 指向新目录 `DevCCF_0821_pad20`,`guide_regions` 用 `s12t_guide7.nii.gz` + `ignore_labels: [2,4]`（5 个生效区：1 pallium、3 subpallium、5 hindbrain、6 olfactory bulb、7 diencephalon)。

全流程（含重采样、预处理、guide 区域构建、Affine、warp 回样本空间、cell 归属)**4 分钟跑完**——因为提前在 Affine 就停了,没有跑 SyN。

### 结果（Affine-only,同一张样本脑 65.9 mm³)

| | 无标签 | 溢出组织外 |
|---|---|---|
| 强度驱动 Affine（同一图谱裁切,历史值) | 40.6% | 2.7% |
| **形状驱动 Affine（这次)** | **26.0%** | **12.3%** |

目视（`DevCCF_0821_pad20/qc/affine_only_{horizontal,coronal}.png`)：标签轮廓从之前"贴一条竖直线、内侧一半空着"变成基本铺满整个组织,外层边界跟着软脑膜表面的弯曲走。中线位置和样本实际中线重合度很高。

**溢出从 2.7% 涨到 12.3% 是预期代价**：图谱被拉伸/旋转去贴合画出的区域,边缘部分跑到组织外的背景里（不影响细胞归属,背景没有细胞)。

### 中线为什么现在是直的,以及这为什么不能算"解决了"

**这一步没有测试到中线该测的东西。** Affine 是单一的全局线性变换（12 自由度：旋转+缩放+切变+平移),仿射变换保直线——任何直线经过仿射变换之后还是直线,和切割位点在哪完全无关,是这一步变换本身的数学性质。所以"中线现在是直的"是必然结果,不是这次跑出来的信息；"中线位置和样本对得上"才是这次实际测到的、值得记的改善。

真正会让中线跟着样本局部弯曲的是 **SyN**,而本轮完全没有跑到 SyN——`type_of_transform: affine` 就是为了在这里停下诊断。**中线问题是否真正解决,要等 SyN 跑完之后再看**：如果那时候中线还是直的,才是需要报警的信号；现在是直的不提供任何一方向的证据。

本次 `.mat` 由于上述 outprefix bug 没有落盘（只在内存里参与了这次的 warp,输出的 `.nii.gz` 数值可信,但变换文件本身不可复用),想要留档需要重新跑一次（bug 已修,重跑约 4 分钟)。

### 下一步

1. 用户决定：这个 Affine 结果是否已经"good enough"到可以接上 SyN。如果是,把 `type_of_transform` 改回 `SyNRA`（或任意非 `"affine"` 的值,guide 分支只用这一个值判断是否早停),预期 1.5~3 小时,启动前会再确认。
2. `atlas.slicing` 第三轴收紧到 `[151,371]`（此前测过,能去掉样本没有的腹侧 2mm 图谱地盘)这条建议本轮没有应用,仍然待做——可以和这次的形状驱动 Affine 叠加,预期覆盖率还能进一步改善,值得在正式跑 SyN 之前再探一次。
3. label 5（hindbrain,`rel` 74%)之前标注"边缘情况,值得看一眼",还没有专门看过。

---

## 2026-08-25：在配准结果上改标签 —— `paint_mask.py` 新增 `mode: labels`，并把 partition 定义成"手工挑的分组树"而不是本体层级

全部改动在 `../Registration_toolkit`，本仓库一行未动。

### 需求

现有两条路各缺一半：`paint_mask.py`（`mode: guide`）能在原图上从零画、能给每个画笔号指定图谱脑区，但不知道配准结果长什么样；`tools/edit_sample_labels.py` 从 `labels_in_sample.nii.gz` 起手、能按本体层级切换粒度，但导出的是像素级 delta，喂不回配准。要的是两者合体：**在配准输出上画，只画个别层，导出时整层进 keyframe、层间插值，并且能按脑区逐级细化**。

### 关键结论：partition 不是本体深度

最初的方案是"导出时锁定一个 ontology level，把所有 keyframe 折叠到该级"。**对 CCFv3 不成立**，拿 `atlas/DeMBA/DeMBA_P5_annotation.tif` 实测：

| collapse 到 | 全 volume label 数（= guide 项数） | 单层最多 |
|---|---|---|
| level 2 | 24（其中只有 4 个真结构：root / Basic cell groups / fiber tracts / ventricular systems） | 14 |
| level 3 | 35 | 18 |
| level 4 | 59（已经细到 abducens nerve） | 24 |
| level 6 | 184 | 53 |

而且 annotation 里有 **20 个 id 根本不在 `CCF_v3_ontology.json` 里**（182305696、312782560…），`collapse_labels_to_level` 对它们原样穿过——level 2 那 24 个里 20 个是这种孤儿。CCFv3 的深度在"分区"这件事上没有一档是可用的：深度 2 只有细胞群/纤维束/脑室系统，深度 4 已经是脑神经。（DevCCF 不同，它 level 4 只有 15 个结构，最初的建议是照 DevCCF 的数字给的——这是本轮第一次判断失误的根因：**换了本体就得重新量，不能沿用另一套本体的层级手感**。）

反过来看用户已经在用的 `atlas/mask/s12t_DeMBAguide7.regions.json`，那 7 组本来就**不是**任何一个深度：label 2 是 6 条纤维束并起来，label 5 是 Cerebellum + Hindbrain + cerebellum related fiber tracts，label 6 是 MOB + AOB nerve layer。这是按"guide 能有效拉动的尺度"手工挑的分区，而它的格式 `{画笔号: [ccf_ids]}` 恰好就是导出格式本身。

所以把"level"重定义成 **一份 partition：一组分组，每组带若干 CCF 根 id，允许嵌套**。规则是**最深匹配**——一个体素归属于其 `structure_id_path` 上最深的那个组根。展开某一组＝给它的子结构各建新组，父组留下当残余；不用任何额外记账，太小的子结构没被建组就自动还归父组。

### `shared/label_partition.py`（新增，约 490 行含 6 项 selftest）

`Partition` / `Group`，核心 API：`from_regions_json` / `collapse` / `expandable` / `expand` / `merge_back`（递归）/ `atlas_exclude_ids` / `empty_atlas_side`。

几个实现上必须记的点：

- **`collapse` 走 `np.unique` + `searchsorted`，不建稠密 LUT**。CCFv3 的 id 最大到 6.1e8（DeMBA annotation 里有 614454272），`np.arange(max_id+1)` 要 2.4 GB 去描述几百个实际出现的 id。本仓库的 `atlas_utils.collapse_labels_to_level` 目前还是稠密 LUT 写法，对 CCFv3 能跑但很浪费，**这是一处已知可优化点，本轮没动**。
- **`atlas_exclude_ids` 做了极小化**：exclude 项本身也按子树展开，所以列 `695` 就等于列了 `315/698/1089/822/1080`。不极小化的话残余 Cerebral cortex 会列 8 个 id 说同一件事。
- **`empty_atlas_side` 必须用"每个 id 自己的体素数"，不能用 `atlas_reference.voxels_per_ontology_node` 的子树累计**。后者把每个体素记到全部祖先头上，父组的子组全被拆出去之后仍然报告它们的体素、永远不为空。第一版就是写错成子树累计，被 selftest 抓到。

### 实测：展开 Cerebral cortex 该展到哪一级

| 展开到 | annotation 里实际出现 | 其中 >1 mm³ | 尾巴 |
|---|---|---|---|
| 深度 5 | 2 | 2 | Cortical plate 95.8 / Cortical subplate 3.5 |
| 深度 6 | 9 | 3 | Isocortex 58.5、Olfactory areas 18.9、**Hippocampal formation 18.4**，其余 6 个杏仁核/claustrum 在 0.25–1.02 |
| 深度 7 | 36 | 10 | 后 26 个从 1.0 掉到 0.14 |
| 深度 8 | 90 | ~20 | 尾巴是 `Perirhinal area, layer 6b` 0.00 mm³ |

按 `configs/config.example.yaml` 里已经记着的实测结论（<1 mm³ 的 guide 区手圈误差占比太大，会主动把形变往错方向拽），深度 8 那 90 项里 70 个一个都不该成项。所以是**按需逐节点展开**，`expand()` 内建 `min_region_mm3`（默认 1.0）过滤，被过滤的子结构留在残余父组里。

用户的目标（改 hippocampus 及再下一级）在 CCFv3 里是：`688 → 695 → 1089(HPF, 深度6) → 1080/822(深度7) → 375 Ammon's horn 6.4 / 726 Dentate gyrus 2.8(深度8)`。从 guide7 起手只展开 label 1 这一支，到 CA/DG 一级共 14 组、约 12 个生效 guide 项（现状是 7 组、`ignore_labels: [2,4]` 之后**实际只有 5 项生效**）。

**代价**：Dentate gyrus 2.75 mm³ 且细长弯曲，正是日志里记过"每 20 层 Dice 只有 0.36、要每 10 层"的那个结构。展开到 CA/DG 意味着 hippocampus 所在 z 段的 keyframe 要从每 20 层加密到每 10 层。建议先只展到 Hippocampal region 9.3 / Retrohippocampal region 8.9（块状得多，每 20 层够）看效果。

### 一个已经在手动做的事，现在自动了

`configs/s12t.yaml:93` 的 `atlas_exclude_ids: {1: [507, 151]}` 就是一次手工的"非均匀展开 + 减去"——MOB(507) 在 CCFv3 里是 `688 → 695 → 698 → MOB` 的后代，但由 label 6 单独引导，不减掉的话同一批图谱体素被两个 guide pair 拉向不同目标。工具现在按 partition 树自动推导这一段并打进可粘贴的 config 片段。

**一处对不上，值得注意**：从 guide7 的 sidecar 推出来是 `{1: [507]}`，config 里手写的是 `{1: [507, 151]}`。因为 sidecar 里 label 6 记的是 `[507, 1016]`，不含 AOB(151)——151 是手工额外加的。工具只保证"partition 记了什么就减什么"；AOB 目前不属于任何组、在样本侧归入 label 1 的残余，若仍要把它排除在图谱侧，得把它加进 label 6 或保留手写项。

### 第二次判断失误：画布网格搞反了（用户指出）

第一版 `mode: labels` 要求 `image_path` 是重采样后的 `*_fine_25um.nii.gz`，让人在 25 µm 各向同性网格上画。**这是错的**，而且本仓库 `pipeline.py:88` 的注释早就写着为什么：painted volume 必须活在原始 tiff 网格上，"that is the only grid where the structures are still resolvable by eye -- the isotropic resample throws away ~8x of the in-plane detail"。此外 25 µm 网格的 z 层是插值出来的，不是真正扫过的成像平面。

改成：**画布仍是原始 tif，把配准结果重网格升采样到原图网格叠上去**。依据是本仓库的不变量——所有图像共享物理原点 0、identity direction、spacing 单位为微米（`io_utils.load_tiff_stack_as_ants` / `resample_to_isotropic` 从不传非零 origin，`crop_to_bounds` 裁剪时专门平移 origin），所以两网格之间每轴一次乘法即可，不需要任何变换。

实现上两个决定：
1. **先折叠再重网格**。`collapse` 跑在小的各向同性体积上（~20M 体素，uint32→uint8），只有 uint8 结果 gather 到原图网格（s12t 是 157×3974×2273 = 1.4e9 体素）。反过来做的话到达原图网格的是 uint32，4 倍内存。
2. **用 numpy gather 而不是 `ants.resample_image_to_target`**。后者的 float32 往返每份约 5.7 GB；gather 是一次输出大小的分配，且离散 id 没有插值可以搞错。越界钳到边缘不回绕——`crop_for_registration` 之外 `labels_in_sample` 已被清零而原图是完整的，回绕会把对侧脑贴过来。

**顺带修掉自己引入的一个 bug**：第一版写了 `voxel_size_um_from_spacing()`，把 NIfTI header spacing 乘 1000。这对**本仓库自己写出的文件是错的**——`labels_in_sample.nii.gz` 的 spacing 直接就是微米（读回来 25.0），乘完变成 25000 µm。外部下载的才是毫米（DevCCF 读回来 0.02）。改成 `labels_voxel_size_um()`，按数量级区分并打印实际采用哪种，`labels_voxel_size_um` 配置项可强制覆盖，header 是 (1,1,1) 时直接报错不猜。

导出网格＝原图网格，所以打印的 `voxel_size_um` 就是 `[2.6, 2.6, 32.0]`，和 `mode: guide` 完全一致，直接替换 `configs/s12t.yaml` 的 `regions_mask` 即可。

### 两个输出

同一批整层 keyframe 喂 `mask_utils.interpolate_sparse_label_correction` 两次，只换 baseline：

- baseline = 全 0 → **稀疏 guide**（`output_path`）：只有改过的层加层间插值，首尾 keyframe 之外留空，喂 `mask.guide_regions` 重新配准。
- baseline = partition 折叠后的完整体积 → **稠密版**（`atlas_output_path`）：每层有值，下次打开继续画。

用 `interpolate_sparse_label_correction` 而不是 guide 模式的 `interpolate_labels_separately`：这里所有区域共享同一批 keyframe（因为存的是整层），guide 模式那个"各 label keyframe 交错会互相吞掉"的问题不会出现；这里需要的反而是相邻区域为层间体素做逐区域 SDF 竞争，正是前者的行为。

**keyframe 判定不挂 paint 事件**，而是 `paint != baseline` 逐层求或——mode: labels 的 baseline 本身就是折叠结果，diff 是精确的，而且 fill / 多边形擦除 / 撤销 / 批量 relabel 全都自动覆盖。展开 partition 时 `recollapse_keeping_edits()` 按"与旧 baseline 不同即为手改"保留手改像素、其余按新 partition 重新细化——实测在真实数据上：963 个手改体素原样保留、278972 个未碰过的体素细化到 CA/DG 粒度，3 个 keyframe 层仍被识别。

**续做不复利**：`.keyframes.json` 记 `hand_drawn_slices` + **真正 baseline 的路径** + 两个网格的尺寸和体素大小（两个 header 里都没有）。重开时只读回那几层，其余从 `labels_path` 重新折叠重网格。不这么做的话本轮插值猜测会变成下轮"真值"，`paint_mask.load_guide_resume` 的 docstring 量化过：5 层的活第二轮回来变 11 层。

### 实测代价

在 251 GB / 48 核的机器上，按 s12t 的真实平面尺寸（3974×2273）：

- 一次 signed-distance 场：**0.48 s**
- `labels_export`：**2.4 s 每（keyframe 间隔 × 区域）**
- 折叠 + 重网格 157 层：约 3 s

按现有 36 个 keyframe 配十来个区域推算，**一次导出 15–20 分钟**。可接受（一次性），但应该知道。一个未做的优化：把每个区域的 EDT 限制在自己的包围盒内算（远处区域 SDF 恒为大正数不可能赢），能砍掉大部分——要改 `mask_utils` 并单独验证。

### 显示：脑区默认填充

`single_sample.py:936` 原本强制 `labels_layer.contour = 1`（只画边界）。改成默认填充，轮廓保留为控制面板里一个复选框——原注释说得对，轮廓压在高分辨率原图上是判断配准精度的主要手段，只是不该是默认。`_apply_region_contour()` 在两个视图加载函数末尾都调一次（两者都清空重建图层，新层是 napari 默认值）。高亮层 `>> Highlight Atlas <<` **故意不跟随**：它是搜索结果，轮廓化等于把刚搜到的东西藏起来。

`paint_mask.py` 本来就是填充的（napari 0.8 的 Labels 默认 `contour = 0`，从没被覆盖过），加了同名同默认的 Display 面板保持两个工具一致。

### 显示：面板宽度可调

两件不同的事：

- `single_sample.py` 两个面板写了 `setMaximumWidth(320)` —— **硬上限，怎么拖都不可能更宽**。
- `paint_mask.py` 没有上限，但会被**最小宽度**钉住：Qt 从子控件 `minimumSizeHint` 推最小宽度，QLabel 的 hint 是最长那行的宽度，一路传到 QMainWindow，一句长说明就让分隔条推不动。

`tools/atlas_view.py` 里已经有解法（`_shrinkable` = `setMinimumWidth(1)`，带说明）。提到 `shared/ontology_tree_ui.py` 成 `shrinkable()`，另加 `set_dock_width()` —— 用 `QMainWindow.resizeDocks` 给**起始宽度**而不是上限，设完两边照样能拖。应用到 paint_mask 全部 7 个面板（含本体树/列表控件本身）和 single_sample 两个面板。`tools/atlas_view.py` 未改（用户有未提交改动在里面），其私有 `_shrinkable` 仍可用，可择机去重。

### `tests/test_gui_smoke.py`（新增，5 项，3.2 秒）

此前所有测试都是无头的纯 numpy，覆盖不到面板接线。这次发现本机有 `xvfb-run`，napari 可以无头起窗口，于是补上：guide 模式（含 ontology 面板）、labels 模式（画布必须在原图网格、Expand/Merge、导出 5 个文件、稀疏/稠密边界）、续做（只恢复 keyframe）、填充/轮廓开关、面板宽度可调。

三个写测试时才发现的坑：

1. **ssh 转发的 X11 跑不了。** 本机 `DISPLAY=localhost:10.0`（渲染器字符串是 "2.1 Metal"，一路转发回客户端的 Mac），GL 只到 1.4，napari 的 shader 编译不过、进程 core dump 在 vispy 里。所以 Linux 上**即使 `$DISPLAY` 已设也优先用本地 xvfb**——测试从不真去看窗口，不可见但能用的 GL context 胜过可见但坏掉的。`GUI_SMOKE_USE_DISPLAY=1` 可强制。两者都没有则跳过，不算失败。
2. **合成 annotation 写成 uint32 的 3D TIFF，SimpleITK 读回来是乱码**（8111 个 id），而其中碰巧有几个是对的，于是 partition 照样展开、测试照样通过。改成 `.nii.gz` 后干净的 5 个叶节点 id，并加了 `_assert_annotation_loaded()` 在所有测试之前断言"写进去什么就读回什么"。真实图谱两种格式都没问题，只有玩具图谱会踩到。
3. 第一版 guide 模式测试传 `atlas=None`，**于是 `_add_ontology_picker` 从没被调用、用户点名的那个面板压根没被建出来**。改成带合成图谱跑。

面板宽度那一项做了负向验证：把 cap 加回 ontology 面板 / 给 Partition 面板钉最小宽度 / 在 single_sample 恢复 cap，三种都能被抓到并指名道姓。single_sample 的面板建在 `MainController` 里需要真实样本目录，用源码级检查兜底——检查用 `tokenize` 剥掉注释和字符串再比对，因为解释"不要用 setMaximumWidth"的注释本身就含这个词。

### 下一步

1. 用真实 s12t 数据跑一次 `mode: labels`：从 `s12t_DeMBAguide7.regions.json` 起手，展开 label 1 到 Isocortex / Olfactory areas / HPF 一级，先不展到 CA/DG。
2. 拿导出的稀疏 guide 替换 `configs/s12t.yaml` 的 `regions_mask`（`voxel_size_um` 不变，仍是 `[2.6, 2.6, 32.0]`），把自动生成的 `atlas_exclude_ids` 贴进去，重跑一次配准比较。
3. AOB(151) 的归属要定：加进 label 6，还是保留 config 里的手写 exclude。
4. 若 hippocampus 的 Dice 不到位再展开到 CA/DG，并把该 z 段的 keyframe 加密到每 10 层。
5. 待办（本轮记下未做）：`atlas_utils.collapse_labels_to_level` 的稠密 LUT 换成 searchsorted；`labels_export` 的 per-region EDT 限定包围盒；`tools/atlas_view.py` 的 `_shrinkable` 去重。

---

## 2026-08-26：DeMBA P5 annotation 是 float32 —— 127 个 CCFv3 大 id 被舍入，换回官方 uint32 版并堵掉四处 cast

起因很小：用户在 atlas view 里看完实际结构，要求把 `s12t_DeMBAguide7.regions.json` 的 label 2 从 6 条纤维束改成 `776`(corpus callosum) + `484682528`，并顺口问了一句"这个 id 数字太大了，会影响读取吗"。会。而且这一条追下去，从数据文件一直坏到加载器。

### 现象与根因

`atlas/DeMBA/DeMBA_P5_annotation.tif` 的 dtype 是 **float32**。float32 只有 24 位整数精度，超过 2²⁴ = 16,777,216 的整数会被吸附到最近的可表示值（在 4.8e8 量级上步长是 32）。CCFv3 有 **127 个 id 超过这条线**，最大 614,454,277。

用户之所以要加 `484682528`，是因为他在 viewer 里点胼胝体中间那一段，查到的 id 就是它，本体一翻译报成 "commissural branch of stria terminalis"。实际上那是四个结构被压进同一个 float32 值：

| uint32 id | 结构 | 官方文件里的真实体素 |
|---|---|---|
| 484682516 | ccb 胼胝体体部 | 198,512 |
| 484682520 | or 视辐射 | 8,848 |
| 484682524 | ar 听辐射 | 4,360 |
| 484682528 | stc 终纹连合支 | 2,537 |
| | **合计** | **214,257** ← 坏文件里那一桶的体素数，分毫不差 |

后果：`descendant_ids_of(776)` 展开出 `484682516`，但 `np.isin` 在 float32 数组上匹配到 **0 个体素**，胼胝体体部整块消失，不报错不警告。用户说的"水平角度看胼胝体是不连续的"就是这个——实测各水平层的 2D 连通块：

| 水平层(axis1) | 修复前 块数/体素 | 修复后 块数/体素 |
|---|---|---|
| 130 | 3 / 2,270 | **1** / 3,577 |
| 140 | 2 / 1,236 | **1** / 3,687 |
| 170 | 6 / 4,158 | 2 / 6,705 |
| 180 | 3 / 5,747 | **1** / 6,888 |
| 190 | 3 / 5,013 | **1** / 6,019 |

修复前碎成 2–6 块，修复后是完整的 C 形弓。**用户凭解剖知识画的连续结构是对的，图谱是错的**——之前等于拿一个中间断掉的目标去引导一个连续的样本结构。

### 全量损伤评估

annotation 里超过 2²⁴ 的值共落成 24 个 float32 值：**4 个可唯一还原**（FRP6b / ORBm6b / HATA / Su3，32,884 体素），**20 个不可拆**（1,032,402 体素）。合计占全部有标签体素的 **3.63%**。最严重的几桶：

| float32 值 | 合并了 |
|---|---|
| 607344832 | PN + IPR/IPC/IPA/IPL/IPI/IPDM/IPDL/IPRL（9 合 1，脚间核整个塌掉） |
| 312782560 | VISa 全层 + VISli（8 合 1，跨区混） |
| 484682496 | ProS 各亚区 + APr + scwm（8 合 1） |
| 549009216 | LSS/RPF/InCo/MA3/P5/Acs5/PC5/I5（8 合 1） |
| 484682528 | ccb + or + ar + stc（4 合 1） |

同名分层合并的那几个（SSp-un、LGd、MM）影响不大，麻烦的是 VIS 跨区那几组。

### 数据修复

坏的不是本项目造的。旧 `DeMBA_P5_annotation.tif` 跟 `ClearMap/Resources/Atlas/p5_trimmed/DeMBA_P5_annotation_trimmed.tif` **md5 完全相同**（`17b7d73de163cb78ba89d54a70f9b793`），是直接拷过来的；那条链上游全是 float32。同目录的 `p4/DeMBA_P4_segmentation_2022.tif` 却是 uint32、77 个大 id 一个不少——说明是做 P5 那个 tif 时转的，不是官方数据的问题。

DeMBA 的 P4 data descriptor 第 2 页明确警告过同一件事（"elastix introduces some rounding to very large values... e.g. 589508447 for the Hippocampo-amygdalar transition area"），他们的解法是**warp 前把 id 换成连续小整数、存盘前换回来**；文件清单里也区分了 `script_with_metadata/P4_resultSegmentation_2022.nii.gz`（"IDs are not consistent with the Allen ontology"）和顶层 `DeMBA_P4_segmentation_2022.nii.gz`（"IDs corresponding to Allen ontology"）。

从 EBRAINS 重新取（DOI `10.25493/V3AH-HK7`，v2，CC-BY-4.0）。**页面默认只列顶层文件，体积文件在子目录里**，用 data-proxy 的公开 API 才看得到：

```
interpolated_segmentations/AllenCCFv3_segmentations/20um/2022/DeMBA_P5_segmentation_2022_20um.nii.gz  (11.3 MB, uint32)
interpolated_volumes/DeMBA_templates/DeMBA_P5_brain.nii.gz
compiled_volumes/AllenCCFv3-2022_segmentations/DeMBA_AllenCCFv3-2022_P4_to_P20.nii.gz  (4D 版，262 MB，同样可用)
```

重建（两个纯索引操作，无重采样）：

```python
v   = np.asanyarray(nib.load("DeMBA_P5_segmentation_2022_20um.nii.gz").dataobj)  # uint32 (570,400,705)
arr = np.ascontiguousarray(np.transpose(v, (2,1,0))[39:602])                     # uint32 (563,400,570)
```

- `transpose(2,1,0)`：拿 `DeMBA_P5_brain.nii.gz` 跟 `p5_origin/DeMBA_P5_brain.tif` 比对，转置后逐体素完全相同
- `[39:602]`：拿旧文件中间层去未裁剪体积里搜，唯一匹配在 index 320，反推 offset = 39，再整段 `array_equal` 确认。reference 用同一组参数也是 `array_equal` True
- **无损证明**：`np.array_equal(new.astype(np.float32), old)` 为 True——旧文件就是新文件的 float32 像，边界一个体素没动

新文件 uint32 (563,400,570)，684 个 label，md5 `3eccd2b7e1913b9a2ad0f0022572bb1b`。同时把过期的 `_1_3_2__285-510_full_35-356__pad20.tif` 缓存挪走（`prepare_custom_atlas` 的缓存，下次跑自动重建）。

### 代码修复：四处 cast

**磁盘修好只是必要条件，不充分**——本仓库的加载器每次读盘都在内存里把损坏重演一遍。

| 位置 | 问题 |
|---|---|
| `io_utils.load_tiff_stack_as_ants` | 无条件 `.astype(np.float32)` |
| `io_utils.load_nifti_stack_as_ants` | 同上，而且 **`ants.image_read` 默认 `pixeltype='float'`，在外面改 dtype 来不及**，必须传 `pixeltype='unsigned int'` |
| `atlas_utils.get_allen_atlas` / `_read_atlas_array_xyz` / `_write_atlas_array_xyz` | BrainGlobe 标注、缓存读、缓存写三处都 cast |
| `transforms.warp_labels_to_sample` | 把 annotation 直接送进 `ants.apply_transforms` |

前三处加 `preserve_labels` 开关，annotation 传 True、template 维持 float32（强度图，配准需要浮点）。

**第四处是 `test_pipeline_smoke.py` 抓出来的**：前三处改完后它挂在 `genericLabel interpolation introduced new label ids`。原因是 **ANTs/ITK 内部就是 float32**，annotation 只要经过 `apply_transforms`/`resample_image` 就会重新舍入——之前两边都坏所以对得上，图谱那边一准反而露馅。改成 DeMBA 那个办法：warp 前 `np.searchsorted` 把 id 重编号成 0..N-1，warp 后映射回去（几百个小整数在 float32 里完全精确）。合成数据验证：输入 `[0, 776, 484682516, 614454277]` → 输出完全一致，各 id 体素数不变。

前因后果写成 `io_utils._LABEL_DTYPE_NOTE`，四处改动都指向它，并写明两类消费者的规矩：**只读 `.numpy()` 建 mask 的**（`_build_guide_regions_from_labels`、`build_region_exclusion_mask`，送进 ANTs 的是二值 mask 不是 annotation）加载时别 cast 就够；**要 warp annotation 的**必须先重编号。

### 验证与副作用

走真实加载路径（不是 `tifffile` 直读）：

```
annotation: uint32, pixeltype 'unsigned int'
label 2 [776] -> 600,842 voxels   （修复前 402,330）
   198512 ccb / 117755 fa / 100990 fp / 67084 ccg / 55733 ec / 55197 ccs / 5571 ee
```

sidecar 的 label 2 最终定为 `[776]` / `["corpus callosum"]`。**`484682528` 去掉了**——当初加它是为了捞回匹配不到的 ccb，修复后 `776` 自己就能正确展开，留着只会塞进 2,537 体素不相干的终纹连合支（修复前那一桶还会连带混进视辐射+听辐射 13,208 体素，约 6.5% 污染）。

测试：4 个 smoke test 过 3 个。`test_new_features_smoke.py` 挂在 `KeyError: 'invtransforms'`，`git stash` 验过**改动前就是这样**，与本轮无关。

**一个方法论教训**：中途一度宣布"已经修好了"，但那次验证用的是 `tifffile.imread` 直读文件，绕过了 pipeline 自己的加载器；走真实路径才发现 label 2 仍是 402,330。**改完数据必须用生产代码路径复验，不能用旁路脚本。**

### 纠正 08-25 条目里的一处说法

08-25 记的"annotation 里有 **20 个 id 根本不在 `CCF_v3_ontology.json` 里**（182305696、312782560…），`collapse_labels_to_level` 对它们原样穿过——level 2 那 24 个里 20 个是这种孤儿"——**那不是孤儿 id，正是本轮这些被 float32 舍入出来的假值**。换 uint32 之后 684 个 label 全部能在本体里查到，`np.isin` 匹配为 0 的情况不再存在。当时基于"孤儿 id"对 partition 层级做的计数（level 2 = 24 项等）应当重新量一遍。

### 顺带记下

- `kim_annotation_P4_to_P20.nii.gz`（DeMBA 同一数据集，uint32，4D 第 4 维是年龄）**index 1 = P5**——用 P5 brain mask 逐期算 Dice 确认：P4 0.841 / **P5 0.941** / P6 0.896 / P7 0.815。它是原生 P5 的 DevCCF 标注且跟本图谱同网格，比 sidecar `converted_from` 现在引的 `P04_DevCCF_Annotations_20um.nii.gz`（P4、另一套网格）合适得多。
- `DeMBA_P5_lsfm.nii.gz` / `DeMBA_P5_mri.nii.gz` 是 25 µm 单模态平均图，(456,320,564)，跟图谱不是一套网格，用不上。
- `atlas_utils.collapse_labels_to_level` 的稠密 LUT（08-25 已记的待办）现在更值得改：max id 由舍入后的 614454272 变回真值 614454277，`np.arange(max_id+1)` 仍是 2.4 GB。
- 文件大小：uint32 和 float32 都是 4 字节/体素，改整数一个字节不多。真正影响体积的是压缩——这份 tif 加 zlib 是 514 MB → 4.9 MB（读写更快、`array_equal` 一致），本轮**按用户要求没有压缩**，维持无压缩 tif。
- `atlas/` 整个目录在 `.gitignore` 里，`atlas/mask/` 下手画的 mask 和 sidecar 既不可复现也不在版本控制中，**需要单独备份**。

### README

根 `README.md` 新增 `## Atlas data` 一节，从读者角度写：用的是 DeMBA P5 + Allen CCFv3-2022（20 µm，DOI 10.25493/V3AH-HK7），三个文件名（`DeMBA_P5_segmentation_2022_20um.nii.gz` / `DeMBA_P5_brain.nii.gz` / `CCF_v3_ontology.json`，并提醒 2022 vs 2017 是不同 CCFv3 版本）、体积文件在哪两个子目录、转换代码，以及"annotation 必须读回 uint32"一句（细节指向 `_LABEL_DTYPE_NOTE`）。

### 下一步

1. sidecar 的 `converted_from` 换成 `kim_annotation_P4_to_P20.nii.gz` 的 P5 层。
2. 重新量 08-25 那份 partition 层级统计（孤儿 id 已消失）。
3. `collapse_labels_to_level` 换 searchsorted。

---

## 2026-08-28：0827 三组结果对比（baseline / rescale / weighted）—— label 4 的 id 是对错问题不是权重问题；形变不平滑的根源是 SyN guide 项用了二值 mask，已改为高斯平滑

起因：用户看完 0827 全部结果，发现海马下缘被 root 覆盖、皮层一小块被推到 background，怀疑"自己手绘 mask + interpolate 不够准，不能正确指导配准"。把 `DeMBA_0827`（基线）/ `DeMBA_0827_rescale`（label 4 改用 8 个中脑核团 id）/ `DeMBA_0827_weighted`（label 4 保持 [313] 但 weight 0.3）三组结果和手绘 mask 系统比了一遍。结论：**mask 的插值不是瓶颈，rescale 的做法正确，weighted 的做法有副作用；真正的平滑度问题在代码里（SyN guide 项的二值边界），本轮已修。**

### rescale vs weighted

每个 guide label：warp 到样本空间的图谱区域 vs 手绘该 label 的 Dice（括号为体积比 warp/手绘）：

| lbl | 区域 | 0827 基线 | rescale | weighted |
|---|---|---|---|---|
| 1 | Cerebral cortex | 0.931 | 0.927 | 0.928 |
| 3 | Cerebral nuclei | 0.928 | 0.927 | **0.937** |
| 4 | Midbrain | 0.124 (13.3x) | **0.896 (1.01x)** | 0.201 (7.9x) |
| 5 | Cerebellum/HB | 0.681 | 0.707 | **0.811** |
| 6 | Main olfactory bulb | **0.973** | **0.973** | 0.815 (0.77x) |
| 7 | Interbrain | 0.928 | 0.919 | 0.892 |

形变质量（1Warp 的 Jacobian）与"手绘组织落到 background/root"的泄漏率：

| | %J≤0 | mean \|∇log J\| | p1(J) | 泄漏 |
|---|---|---|---|---|
| 0827 | 0.0026% | 0.0296 | 0.311 | 4.31% |
| rescale | 0.0054% | 0.0300 | 0.287 | 4.35% |
| weighted | **0.0178%** | **0.0385** | **0.185** | **5.25%** |

根因：手绘 label 4 根本不是 CCFv3 `Midbrain(313)` 的全部——313 展开 1,529,791 体素，画的只有 45,647，**33 倍失配**。基线把整个中脑往那一小块上拽（体积比 13.3x）。rescale 换成实际画到的 8 个核团（MRN 128 / APN 215 / RN 214 / NB 580 / NOT 628 / PPT 1061 / OP 706 / DT 75，共 229,310 体素）后体积比 1.01、Dice 0.896——id 对了，问题消失。weighted 是"错误配对被小声执行"：折叠体素 3.3x，且副作用溢出——**嗅球 Dice 0.973→0.815（体积掉到 0.77x）**，丘脑内侧沿中线出现一条 12 万体素的 background 带。weighted 在 label 5 上的收益（0.681→0.811）是中脑被放松的连带效果，不值这个代价。

**决策：采用 rescale 配置（config 已是），weight 全 1.0；若要 label 5 的收益，单独给 5 调。**

### 手绘 mask 自检：插值不是瓶颈

留一法（拿前后两个手绘层的 SDF 插值去预测中间那个手绘层；间隔翻倍，是悲观估计）：

| lbl | 平均 Dice | 平均表面误差 (2.6 µm px) |
|---|---|---|
| 1 CTX | 0.845 | 22.7 (~59 µm) |
| 3 CNU | 0.846 | 22.8 |
| 4 MB | 0.616 | 40.0 |
| 5 CB/HB | 0.927 | 8.9 |
| 6 MOB | 0.932 | 7.8 |
| 7 IB | 0.895 | 17.4 |
| 2 cc | **0.001** | 270 |

label 2（胼胝体）在水平面上是逐层剧变的薄弓，SDF 插值完全失效——`ignore_labels: [2]` 是正确的，别加回来。

关键检验：用与手绘完全独立的 auto brain mask 作参照，按"该层离最近手绘关键帧的距离"分组量配准轮廓误差（20 µm 体素，rescale/weighted）：0 层距 6.76/6.10，0.6–2 层 6.86/6.08，2–4 层 6.39/5.86，4–7 层 6.48/5.74，>7 层 6.85/5.21。**完全平坦——在关键帧那层（插值误差为零）配准并不更准。多画层数不会改善配准，插值不是瓶颈。**

### 海马下缘的 root 带：图谱自带，不是配准错误

HPF 与丘脑之间的 root(997) 条带：图谱本身 5,386 体素、平均厚 2.21 体素；warp 后（rescale）3,135 体素、厚 2.22 体素，按 HPF 体积归一化只放大 1.14x（weighted 1.28x）。**CCFv3 在海马-丘脑之间（侧脑室/伞/脉络丛一带）本来就没有叶子标签**，改 mask 消不掉。要消：warp 后对 997 做最近邻叶子填充，或下游把 997 当"未分配"排除。

### 形变不平滑的真正根源：SyN guide 项的二值边界（已修）

粗糙度按"离最近 guide 区域边界的距离"分箱（rescale）：0–2 体素处 mean |∇log J| = 0.1473，40+ 体素处 0.0253——**贴边界处粗糙 6 倍，且全部折叠体素（J≤0）100% 落在 guide 边界 10 体素以内**。机制：`register.py` 送进 SyN `multivariate_extras` 的是原始 0/1 mask，MeanSquares 在其内外梯度为零、全部作用力集中成边界上一圈 δ，SyN 只能沿这条线剪切——而手绘边界自身就有 ±2–3 体素不确定度。Affine 阶段的 `_guide_union_image` 早就为同一理由做了 σ=2 平滑，SyN 这一路漏掉了。

修复：新增 `register._smoothed_outline`（σ=2 高斯，与 `_guide_union_image` 同参数），extras 改为对每个 outline 平滑后再交给 ANTs。合成验证：48³ 球体 + 偏置 guide 区域，完整 guide 分支（Translation 预对齐 → shape Affine → SyNOnly + smoothed extras）跑通，warp 后 guide 区域 Dice 0.968（旧二值版当年的验证值是 0.92）。**"皮层被推到 background"的那几块（最大 2.2 万体素）正是边界剪切把组织挤出图谱轮廓所致，预期此修复直接缓解。**

### sidecar 回写

`s12t_DeMBAguide7.regions.json` 的 `region_ids[4]` 从 `[313]` 改为上述 8 个核团 id，`regions[4]` 同步为 8 个结构名，`converted_from.manual_overrides[4]` 记录了两次改动的历史与依据。config 与 sidecar 现在一致，`_build_guide_regions_from_labels` 的不一致 WARNING 不再触发。

### 关键帧密度：画多少层才够

同一份留一法数据回答"下个样本每个区要画几层"。**决定密度的是形状变化速率，不是面积大小**——"最宽处"通常在中段，恰恰变化最慢，一层就够：

- **平滑中段 20-30 层间隔足够**。皮层 z=66→86（隔 25）、z=86→110（隔 33 重建 z=99）留一 Dice 仍有 0.83-0.90，而留一是双倍间隔的悲观估计。MOB 画了 15 层是浪费。
- **必须加密的是起止端**。皮层 z=26 Dice 0.694（-42%）、z=142 0.639、z=151 0.727（-43%）——SDF 线性插值让面积匀速缩小，真实结构在端部快速收口。且 `interpolate_sparse_mask` 在首末关键帧之外一律留空，端点画短了直接截断结构。策略：起止层画在结构真正出现/消失处，再往内 3-5 层各补一层。
- **形状剧变处**。label 4 是反例：间隔只有 8-11 层，Dice 仍只有 0.35-0.80——不是画少了，是截面在层间平移+变形，线性 SDF 跟不上。这种地方 3-5 层一画。
- 相邻 label 尽量画在同一批层上（guide7 已经是：66/86/99/110 各 label 共用），共享边界在插值层上才不会互相错开。

经验值：**起止各一层 + 端部内侧各一层 + 中段每个形状拐点一层，平滑段 20-30 层一层**。皮层这种大结构 6-8 层够，不需要现在的 18 层。

### 合并 label 省掉胼胝体

下个样本若不想单独画胼胝体，可以把皮层+胼胝体圈成一个实心 label。图谱侧**不能只写 `[688, 776]`**：两者之间还夹着 supra-callosal white matter(484682512, 7.3 万体素)、cingulum bundle(940, 5.3 万体素) 等薄层，只取 688∪776 会留 12.9 万体素内缝，而手绘是实心的，两边形状系统性不一致。补齐为 `[688, 776, 484682512, 940, 466, 884]`（+alveus, +amygdalar capsule）后剩余内缝仅 317 体素，实测是实心的。机制上无损失：guide 项只用轮廓，内部平坦无梯度，而胼胝体本来就是 `ignore_labels`。`atlas_exclude_ids: {该 label: [507, 151]}` 照旧。注意 484682512 是超 2²⁴ 的大 id，依赖 08-26 的 uint32 修复。

### 工具：scripts/guide_mask_selftest.py

把本轮的留一法分析固化成正式工具（之前是一次性脚本）。读任意 `<mask>.regions.json` sidecar，对每个 label 的每个内部关键帧做留一重建，报 Dice / 平均表面距离（可选 `--voxel-size-um` 换算成微米），并单列 Dice < 阈值的关键帧和各 label 的端点。只依赖 numpy/scipy/nibabel，不需要 ANTs。

    python scripts/guide_mask_selftest.py atlas/mask/s12t_DeMBAguide7.nii.gz --voxel-size-um 2.6 2.6 32.0

docstring 里写清了两个读法上的坑：结果是**双倍间隔的悲观估计**；以及**配准精度不与它成正比**（上面那张平坦的关键帧距离表），低分意味着"插值在编造这个形状"，不等于配准就差了。

拿旧的 `s12t_guide6.nii.gz` 当反例跑，顺手发现并加了一项检查：**误笔关键帧**。guide6 里有 8 个平面只画了 1-259 像素（label 4 的 plane 46 只有 1 个像素，label 3 的 plane 124 只有 4 个），显然是手滑而不是标注。这种平面不只自己得 0 分——它夹在 SDF 混合中间，会把**两侧邻居**的重建一起拉垮，于是一个错误读起来像三个（label 3 的 119 和 126 双双塌到 0.041/0.002，元凶是中间那个 4 像素的 124）。工具现在按"该 label 关键帧面积中位数的 2%"判定，单列一节并在失败行上标注 `<- stray mark` / `<- neighbour is a stray mark; fix that first`。用中位数而非均值，免得一个误笔把基准拉低到藏住下一个。当前在用的 guide7 是 0 个误笔——这条检查同时验证了 guide6→guide7 的重画确实修掉了这批问题。

### 下一步

1. 用 rescale 配置 + smoothed extras 重跑一次，验证边界处 |∇log J| 与折叠体素下降、皮层 background 泄漏块缩小。
2. 考虑拆开 label 5（CB+HB+纤维束一团，Dice 仅 0.71）——比加密关键帧有用。
3. 997 的下游处理定一个策略（最近邻填充 or 排除）。

## 2026-08-29：半脑样本局部越中线的处理 —— guide mask 新增 `damage_labels`

**背景/决定**：
- 6 个半脑样本切割时中线不完全一致：有的中段 z 局部略越过中线（多出一条对侧组织），有的略少于中线。定下的规则：**图谱永远严格按解剖中线裁切（所有样本同一参考），样本侧把越过中线的组织从 metric 里排除**——否则样本内侧的"组织/背景"强度台阶在物理切面而不是解剖中线上，metric 会把图谱中线拉去对齐切面，内侧结构被系统性拉宽、对侧细胞被错归进内侧核团（与图谱侧"不放宽 slicing"是同一逻辑的镜像）。
- 越中线的样本经此处理后与"正好切在中线"等价，不引入 bias；**切少于中线的样本才是真正的 bias 来源**（组织物理缺失，无法补回），且只影响紧贴中线一薄层的结构。计划的应对：把各样本 brain mask warp 到图谱空间算每脑区覆盖率，覆盖率掉下来的中线脑区用密度（细胞数/实际覆盖体积）代替绝对计数或从统计中排除；并确认越线/欠线没有和实验分组重合。
- 排除方式：`crop_for_registration` 是轴对齐框，对"只有中段 z 局部越线"切不准；单独画一份 damage mask 文件工作量大。改为**在画 guide mask 的同一次会话里多画一个 label**，流水线把它解释成"样本上存在、但图谱里没有对应物的组织"。

**做了什么**：
- `pipeline._build_guide_regions_from_labels` 支持 `mask.guide_regions.damage_labels`：列出的画笔 label 不进 guide pair，而是并进 moving_mask（与 `sample_damage_mask_path` 同语义、两者可叠加），返回值改为 `(triples, damage_hole)`。paint_mask.py 的 guide 导出本来就按 label 独立做关键帧插值，所以薄片只需画几层。
- `config.py` 校验：`ignore_labels`/`damage_labels` 规范为 int 列表；同一 label 同时出现在两者、或同时在 damage_labels 和 atlas_ids/atlas_names 里都直接报错。未配对 label 的报错信息里加了 damage_labels 这个出口。
- `config.example.yaml` 和 README 的 guide-mask 章节补了用法说明。
- 新增烟雾测试 `test_guide_damage_labels`（hole 精确、不进 guide pair、未配置/未画仍报错）；顺手修了 `test_assign_cell_regions` 的陈旧 fake reg（`transform_cell_points` 的 sample_to_atlas 改用 `invtransforms` 后测试没跟上，补上 `"invtransforms": []`）。`python tests/test_new_features_smoke.py` 全绿。

**下一步**：
- 对越线样本：在 guide mask 会话里把对侧薄片画成新 label（几个关键帧即可），config 里列进 `damage_labels`，重跑配准。
- 配准完成后做覆盖率检查，确定受影响的中线脑区名单，决定密度校正还是排除。

## 2026-08-29（续）：paint_mask.py 的 ontology 树加 "damage / no atlas counterpart" 伪节点

**背景**：damage_labels 落地后，画的时候那个 label 在 picker 里没有可选的脑区——不报错但导出一路 warning（"painted but has no region_labels entry"）、面板一直显示未分配，还得手动往 pipeline config 里抄 damage_labels。

**做了什么**（Registration_toolkit/paint_mask.py）：
- ontology 树顶部插入伪节点 "damage / no atlas counterpart"（sentinel `DAMAGE_ID = -1`，负数保证永不撞真实 ontology id）。像普通脑区一样 Assign to label；同一 label 不能既是 damage 又配脑区（GUI、config 解析、_seed_assignment 三处都拦）。
- 导出时 damage label 不进 region_ids/guide pair，写进 sidecar 新键 `damage_labels`；打印的 YAML snippet 带上 `damage_labels: [...]`；导出 warning/逐 label 报告用 DAMAGE_NAME 显示但【不】写进 sidecar 的 regions 键（避免 resume 时被当脑区名去解析）。resume 从 sidecar 读回 damage 标记重新 seed。paint config 也支持手写 `damage_labels:`（configs/paint_mask.example.yaml 有说明）。
- Registration_ants 侧：`atlas_utils.load_regions_sidecar_damage_labels` + pipeline 自动 union sidecar 的 damage 标记（同 region_ids 的"读 sidecar、不手抄"逻辑，pipeline config 可以完全不写）；显式配了 atlas_ids/atlas_names/ignore_labels 的 label 优先于 sidecar 的 damage 标记（打 WARNING）。
- 测试：paint_mask `--selftest` 扩了 sidecar/seed/assignment_rows/config-normalizer 四处并全绿；Registration_ants 烟雾测试加了 sidecar fallback 断言，全绿。

**下一步**：在 s12t 之外那个越中线样本的 guide 会话里实际用一次：树顶选 damage、画对侧薄片几个关键帧、导出、直接重跑配准（config 不用改）。
