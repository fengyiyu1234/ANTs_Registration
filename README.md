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
- `reposition.py` — closes the gaps left by tissue that split open, as per-plane
  in-plane rigid moves applied before registration (`sample.reposition_plan`).
  `grab_plane` takes a piece from one click as its own connected component, so
  nothing has to be outlined by hand; the planes between the ones clicked are
  filled by `densify_fragments` when the plan is applied.
  Plans are drawn in `paint_mask.py`'s Reposition panel; one config key moves the
  stack, the guide outlines, the damage mask and the cell centroids together.
  `scripts/apply_reposition.py` writes the repositioned copies out, and `--invert`
  takes a result back onto the original geometry for QC.
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

- `scripts/guide_mask_selftest.py` — checks a hand-painted guide mask before you register with it: how well its slice interpolation holds up, and whether any plane is a stray brush mark. See [Drawing a guide mask](#drawing-a-guide-mask--what-actually-helps).
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

## Drawing a guide mask — what actually helps

A guide mask is a few hand-painted outlines that tell the registration
"this blob in the sample is that structure in the atlas". It is the single
biggest lever you have on the result, and it is easy to spend hours drawing
things that change nothing. What follows is measured on the s12t sample, not
folklore — the numbers are in PROGRESS_LOG.md (2026-08-28).

**Draw the boundaries the image cannot show.** The whole point of a guide is
to supply information the intensity metric does not have. Measured against
the noise floor of the sample image, the cortex/striatum boundary has *zero*
contrast (1.0x) — the registration is completely blind there, and your
outline is carrying all of it. The brain's outer surface, by contrast, is
11.7x. Both are worth drawing; the first is worth much more.

**Do not draw boundaries you cannot see.** Cortical area boundaries (MOs vs
SSp and friends) sit at 1.4x the noise floor — invisible at this resolution,
because they are defined by cytoarchitecture. If you draw them you are
drawing your own guess, the registration reproduces it faithfully, and the
result *looks* validated when it is not. That is worse than an honest error.
This is about visibility, not size: every cortical area in this sample is at
least 8 voxels thick, so size was never the problem.

**Fewer planes than you think.** Between the planes you draw, the pipeline
interpolates (blending signed-distance fields). On smooth stretches that
works over gaps of 20-30 planes. Drawing more does not make the registration
more accurate — measured, the registered outline was no closer to the truth
on hand-drawn planes than on interpolated ones. Spend the time on:

  - **The two ends.** Planes outside your first and last keyframe are left
    empty, so an end placed short truncates the structure outright. And the
    interpolation shrinks a structure at a constant rate, while real ones
    close up fast — so draw the true first/last plane, then one more a few
    planes inside each end.
  - **Places where the shape changes fast**, wherever those are. Not "the
    widest plane" — the widest part is usually in the middle, where the
    shape is changing slowest, and one plane covers it.

  Rule of thumb: ends + one just inside each end + one per shape inflection,
  and 20-30 planes apart through smooth stretches. A structure like cortex
  needs 6-8 planes, not 18.

**Draw neighbouring labels on the same planes.** Shared boundaries only stay
aligned on the interpolated planes in between if both sides were drawn
together.

**Skip thin sheets seen edge-on.** A structure that appears as a thin arc
that moves between planes (corpus callosum in a horizontal stack is the
classic) cannot be interpolated at all — measured Dice ~0.00. Either draw
every plane, or leave it out via `ignore_labels` and fold it into a
neighbour: one label can map to several atlas structures, so "cortex +
corpus callosum" as one filled outline is legitimate. If you do that, make
sure the atlas side is filled too — cortex and corpus callosum have thin
sheets sandwiched between them (supra-callosal white matter, cingulum), and
leaving them out puts a gap in the atlas side that your solid outline does
not have.

**Tissue with no atlas counterpart** — e.g. a hemisphere sample that locally
crosses the midline, leaving a sliver of contralateral tissue a hemisphere
atlas cannot match: paint it as its own label in the same guide-mask session
and assign it to the "damage / no atlas counterpart" entry at the top of the
ontology tree (or list the label under `mask.guide_regions.damage_labels` by
hand — the exported sidecar records the GUI assignment, and the pipeline
reads it automatically, so no config entry is needed). It is punched out of
the metric via moving_mask (same semantics as `mask.sample_damage_mask_path`)
instead of becoming a guide pair, and gets the same per-label keyframe
interpolation as every other label, so a few planes suffice. Unlike
`ignore_labels` ("pretend it was never painted"), this actively excludes
those voxels from the objective.

**Check the mask before you spend hours registering with it:**

```
python scripts/guide_mask_selftest.py atlas/mask/<your_mask>.nii.gz --voxel-size-um 2.6 2.6 32.0
```

It re-derives each hand-drawn plane from its neighbours and reports how close
it got, so you can see which labels the interpolation is guessing at, and it
catches stray brush marks — a plane with a handful of stray pixels scores
badly *and* drags both its neighbours down with it.

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
