"""Interactive tool: directly hand-correct the Allen/CCF region boundaries
that ANTs already warped into sample space (`<name>_labels_in_sample.nii.gz`),
on a sparse set of z-planes, then auto-interpolate the corrections into a
dense corrected label volume -- WITHOUT re-running registration.

This is a different tool from mask_tools/paint_mask.py's `mask`/`guide` kinds:
those two operate *before* registration (paint a guide, then re-run SyN so
the deformation goes and matches it). This one operates *after* -- you
already have labels_in_sample.nii.gz from a completed registration, you
directly repaint whichever region(s) came out wrong on a handful of
representative slices (using real CCF region ids, picked from the ontology),
and the correction is interpolated across the volume in sample space. No
re-registration involved -- see scripts/relabel_cells.py
for the next step, which uses the corrected volume to fix cell-to-region
assignments in cell_registration.csv.

Usage (needs a display; the antsreg env has napari+PyQt5+SimpleITK alongside
antspyx -- one env for the whole pipeline):
    conda activate antsreg
    python scripts/edit_sample_labels.py \\
        <sample_fine.nii.gz> <labels_in_sample.nii.gz> <out_corrected.nii.gz> \\
        --ontology-json <ontology.json>
    # or, for a BrainGlobe-sourced atlas instead of a custom one:
    #     --atlas-source brainglobe --atlas-res-um 25

    --level-min / --level-max (default 4 / 6): restrict the region picker to
    structures at that CCF ontology tree depth (root = level 1), so you're
    choosing from major structures instead of scrolling every fine leaf area.

Workflow:
    1. A napari window opens with the sample image and a Labels layer
       PRE-FILLED with the current labels_in_sample -- you're editing the
       existing (imperfect) registration result in place, not painting from
       scratch. Sliced along axis 0, the actual imaging z-planes (same
       SimpleITK convention as the other two paint scripts -- see their
       docstrings for why this matters).
    2. Pick a region from the dock's region list (filtered to
       --level-min/--level-max) -- this sets the paint brush to that
       region's real CCF id. Moving the mouse over the image shows which
       region is currently under the cursor, so you can tell what you're
       about to overwrite.
    3. Repaint only the area that's wrong, on a handful of representative
       z-planes (where the mismatch starts, ends, and where its shape
       changes a lot) -- leave everything else on those planes alone, and
       leave every other plane untouched entirely. This is a delta edit on
       top of the existing labels, not a full resegmentation.
    4. Click "Export Corrected Labels". This auto-detects which planes/pixels
       you actually touched (by diffing against the original), interpolates
       the correction (not the whole plane) between edited planes using
       shape-aware per-region signed-distance blending, and writes a full
       corrected label volume on the exact same grid as the input --
       ready for scripts/relabel_cells.py.
"""
import argparse
import sys
from pathlib import Path

import napari
import numpy as np
import SimpleITK as sitk
from PyQt5.QtWidgets import QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import atlas_utils, mask_utils  # noqa: E402


def _load_structures(args):
    if args.ontology_json:
        return atlas_utils.load_ccf_ontology_json(args.ontology_json)
    if args.atlas_source == "brainglobe":
        _, _, structures = atlas_utils.get_allen_atlas(args.atlas_res_um)
        return structures
    raise ValueError("Pass either --ontology-json or --atlas-source brainglobe --atlas-res-um <res>")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_path")
    parser.add_argument("labels_path")
    parser.add_argument("output_path")
    parser.add_argument("--ontology-json", default=None)
    parser.add_argument("--atlas-source", choices=["brainglobe"], default=None)
    parser.add_argument("--atlas-res-um", type=float, default=25)
    parser.add_argument("--level-min", type=int, default=4)
    parser.add_argument("--level-max", type=int, default=6)
    args = parser.parse_args()

    structures = _load_structures(args)
    pickable = atlas_utils.structures_at_levels(structures, args.level_min, args.level_max)
    id_to_name = {sid: info["name"] for sid, info in structures.items()}
    sorted_pickable = sorted(pickable.items(), key=lambda kv: kv[1]["name"])

    # Same axis-order reasoning as mask_tools/paint_mask.py:
    # SimpleITK's Array<->Image round trip gives natural (z,y,x), axis 0 = the
    # actual imaging planes -- NOT ants.image_read().numpy()'s reversed order.
    sample_sitk = sitk.ReadImage(args.sample_path)
    sample_arr = sitk.GetArrayFromImage(sample_sitk)

    labels_sitk = sitk.ReadImage(args.labels_path)
    original_labels = sitk.GetArrayFromImage(labels_sitk).astype(np.uint32)
    if original_labels.shape != sample_arr.shape:
        print(f"WARNING: labels shape {original_labels.shape} != sample shape {sample_arr.shape}")

    viewer = napari.Viewer(title="Edit labels in sample space")
    viewer.add_image(sample_arr, name="sample", colormap="gray")
    labels_layer = viewer.add_labels(original_labels.copy(), name="labels (edit here)", opacity=0.5)

    status_label = QLabel(
        "Pick a region below, repaint only the wrong area on a few planes,\n"
        "then click Export. Hover the image to see the region under the cursor."
    )
    status_label.setWordWrap(True)

    hover_label = QLabel("Region under cursor: -")

    search_box = QLineEdit()
    search_box.setPlaceholderText("Filter regions by name...")
    region_list = QListWidget()

    def refresh_list(filter_text=""):
        region_list.clear()
        filter_text = filter_text.lower()
        for sid, info in sorted_pickable:
            if filter_text and filter_text not in info["name"].lower():
                continue
            region_list.addItem(f"{info['name']} ({sid})")
        region_list._ids = [sid for sid, info in sorted_pickable
                             if not filter_text or filter_text in info["name"].lower()]

    refresh_list()
    search_box.textChanged.connect(refresh_list)

    def on_region_selected():
        row = region_list.currentRow()
        if row < 0:
            return
        labels_layer.selected_label = region_list._ids[row]

    region_list.currentRowChanged.connect(lambda _row: on_region_selected())

    def on_mouse_move(_layer, event):
        if labels_layer.data.ndim != 3:
            return
        pos = labels_layer.world_to_data(event.position)
        z, y, x = (int(round(p)) for p in pos)
        shape = labels_layer.data.shape
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            return
        region_id = int(labels_layer.data[z, y, x])
        name = id_to_name.get(region_id, f"unknown id {region_id}")
        hover_label.setText(f"Region under cursor: {name} ({region_id})")

    labels_layer.mouse_move_callbacks.append(on_mouse_move)

    def export():
        edited = labels_layer.data
        keyframe_edits = {}
        for z in range(edited.shape[0]):
            edit_mask = edited[z] != original_labels[z]
            if np.any(edit_mask):
                keyframe_edits[z] = (edit_mask, edited[z])

        if not keyframe_edits:
            status_label.setText("No planes edited yet -- nothing to export.")
            return

        status_label.setText(f"Exporting... ({len(keyframe_edits)} edited planes)")
        corrected = mask_utils.interpolate_sparse_label_correction(keyframe_edits, original_labels)

        out_sitk = sitk.GetImageFromArray(corrected.astype(np.uint32))
        out_sitk.CopyInformation(labels_sitk)
        sitk.WriteImage(out_sitk, args.output_path)

        n_changed = int(np.sum(corrected != original_labels))
        msg = (f"Wrote {args.output_path}\n"
               f"Edited planes: {sorted(keyframe_edits.keys())}\n"
               f"Voxels changed from original: {n_changed}")
        status_label.setText(msg)
        print(msg)

    export_btn = QPushButton("Export Corrected Labels")
    export_btn.clicked.connect(export)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(status_label)
    layout.addWidget(hover_label)
    layout.addWidget(QLabel(f"Regions (level {args.level_min}-{args.level_max}):"))
    layout.addWidget(search_box)
    layout.addWidget(region_list)
    layout.addWidget(export_btn)
    viewer.window.add_dock_widget(dock, area="right", name="Label Correction Export")

    napari.run()


if __name__ == "__main__":
    main()
