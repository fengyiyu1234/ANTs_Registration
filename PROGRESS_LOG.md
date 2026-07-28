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
