# registration_ants

ANTs-based registration of LSFM brain samples to the Allen CCF atlas, plus tools to evaluate and manually correct the result.

## Core pipeline (`src/registration_ants/`)

- `run_pipeline.sh` — recommended entry point: `./run_pipeline.sh configs/mouse01.yaml`. Wraps `python -m registration_ants.pipeline` and captures both the pipeline's own log lines and ANTs' native stdout output into `<output_dir>/run.log`.
- `pipeline.py` — runs the full registration pipeline end-to-end from a YAML config.
- `io_utils.py` — converts raw anisotropic TIFF stacks into isotropic ANTs/NIfTI images.
- `preprocess.py` — bias correction and intensity normalization before registration.
- `brain_mask.py` — Otsu-threshold tissue/brain mask generation.
- `mask_utils.py` — registration masks: atlas-side region exclusion and sparse-mask interpolation.
- `atlas_utils.py` — fetches/loads the atlas template and annotation (BrainGlobe or custom).
- `register.py` — runs the ANTs registration itself (one-shot SyNRA) to the CCF.
- `transforms.py` — applies computed transforms to images and cell-coordinate points.
- `cell_points.py` — reads ClearMap cell-centroid CSVs.
- `config.py` — loads and validates the pipeline YAML config.

## Annotation, visualization and evaluation → `../GT_tool_for_registration`

Everything that opens a napari window (painting masks, placing landmarks,
hand-correcting region labels, QC-viewing a registered sample) plus
`registration_eval.py` now lives in a separate repo,
[`../GT_tool_for_registration`](../GT_tool_for_registration). Those tools still
run in this same `antsreg` env — they import `registration_ants` through its
editable install — so nothing here needs to change to use them.

This repo keeps the non-interactive post-processing that is part of the
pipeline itself:

- `scripts/project_outline.py` — warps a hand-painted sample-space guide outline into atlas space.
- `scripts/relabel_cells.py` — rewrites per-cell region assignments using corrected labels.
- `scripts/convert_devccf_ontology.py` — converts the DevCCF ontology spreadsheet into the nested JSON the atlas loader expects.
- `scripts/relabel_labels_to_devccf.py` — translates CCFv3 label ids into DevCCF ids via the published voxel-overlap crosswalk.

## Atlas data (`atlas/`)

Gitignored (hundreds of MB), so a fresh checkout has to download it. The
pipeline registers to **DeMBA P5 with the Allen CCFv3-2022 segmentation**, 20 µm
isotropic, from the DeMBA dataset on EBRAINS (v2, CC-BY-4.0, DOI
[10.25493/V3AH-HK7](https://doi.org/10.25493/V3AH-HK7)).

Three files from that dataset:

- `DeMBA_P5_segmentation_2022_20um.nii.gz` — the annotation (one CCFv3
  structure id per voxel). Note the **2022**:
  `DeMBA_P5_segmentation_2017_20um.nii.gz` sits next to it and is a different
  CCFv3 release, with different ids.
- `DeMBA_P5_brain.nii.gz` — the matching template.
- `CCF_v3_ontology.json` — the Allen ontology naming those ids.

The dataset page lists only top-level files; the two volumes live in the
`interpolated_segmentations/` and `interpolated_volumes/` subdirectories.

Both volumes then get the same conversion — NIfTI `(x,y,z)` to this codebase's
TIFF `(z,y,x)` order, and an AP crop to the 563-plane grid the painted guide
masks are drawn on:

```python
import nibabel as nib, numpy as np, tifffile
for src, dst in [("DeMBA_P5_segmentation_2022_20um.nii.gz", "DeMBA_P5_annotation.tif"),
                 ("DeMBA_P5_brain.nii.gz",                  "DeMBA_P5_reference.tif")]:
    v = np.asanyarray(nib.load(src).dataobj)                       # (570, 400, 705)
    tifffile.imwrite(dst, np.ascontiguousarray(np.transpose(v, (2, 1, 0))[39:602]))
```

Keep the annotation's integer dtype — casting it to float32 silently merges
CCFv3 structures whose ids exceed 2²⁴, so the converted annotation must read
back as `uint32`. `io_utils._LABEL_DTYPE_NOTE` has the details and the rules
any new code touching the annotation has to follow.

Put the three files wherever you like and point
`configs/atlas_presets_local.yaml` at them (gitignored — see
`configs/atlas_presets.example.yaml` for the format). That config also supplies
the 20 µm voxel size: neither TIFF carries spacing in its header.

## Config / project files

- `configs/config.example.yaml` — template pipeline config (copy for real runs; real configs are gitignored).
- `pyproject.toml` — package metadata (`registration_ants`, source in `src/`).
- `requirements.txt` — dependencies for the `antsreg` conda env.

## Tests (`tests/`)

- `test_pipeline_smoke.py` — end-to-end smoke test of the registration pipeline on synthetic data.
- `test_brain_mask_smoke.py` — smoke test for brain mask generation.
- `test_label_correction_smoke.py` — smoke test for the post-registration label-correction workflow.
- `test_new_features_smoke.py` — smoke tests for miscellaneous newer pipeline features.

Full details for any script/module are in its own docstring.
