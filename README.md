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
