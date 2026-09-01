"""Smoke tests for shared/reposition.py: the in-plane rigid transform, its
keyframe interpolation, and the three things a plan gets applied to (image,
labels, cell centroids).

The load-bearing one is test_image_and_points_agree. The image is moved by a
pulled affine resample and cells are moved by a pushed point map -- two
separate pieces of algebra with opposite senses and an (x,y) <-> (y,x) swap
between them. If their conventions ever drift apart nothing raises: the stack
and the cells simply end up describing different anatomy, which is exactly
the failure a cell-counting pipeline cannot detect downstream. So that test
moves a blob and asserts the resampled image's own centre of mass lands where
the point map independently says it should.

Same manual-assert style as the other smoke tests here (no pytest),
all-synthetic data, no /data/... dependency. Run: `python tests/test_reposition_smoke.py`
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from registration_ants import reposition as rp  # noqa: E402


PAINT_UM = (5.2, 5.2, 64.0)          # s18's painted grid
CELLS_UM = (0.65, 0.65, 8.0)         # the grid cells are detected on


def make_stack(shape_zyx=(12, 60, 80), box=(20, 32, 30, 46), planes=(4, 5, 6, 7)):
    """A stack whose only content is one bright box on `planes`, plus a label
    volume marking it as fragment 1. y0,y1,x0,x1 = box."""
    image = np.full(shape_zyx, 100, dtype=np.uint16)
    labels = np.zeros(shape_zyx, dtype=np.uint8)
    y0, y1, x0, x1 = box
    for z in planes:
        image[z, y0:y1, x0:x1] = 4000
        labels[z, y0:y1, x0:x1] = 1
    return image, labels


def centre_of_mass_xy_um(plane, voxel_um, background):
    """The (x, y) micron centroid of whatever is above background on a plane."""
    weight = np.clip(plane.astype(float) - background, 0, None)
    iy, ix = np.nonzero(weight)
    w = weight[iy, ix]
    return np.array([(ix * w).sum() / w.sum() * voxel_um[0],
                     (iy * w).sum() / w.sum() * voxel_um[1]])


def test_rotation_convention():
    print("1. transform_points_um: pure translation, and 90 deg about a centre...")
    tf = rp.make_keyframe(z=0, tx_um=10.0, ty_um=-4.0)
    got = rp.transform_points_um([[100.0, 200.0]], tf)[0]
    assert np.allclose(got, [110.0, 196.0]), got

    # +90 deg CCW on (x, y) about (0,0) sends (1,0) -> (0,1).
    tf = rp.make_keyframe(z=0, theta_deg=90.0, center_um=(0.0, 0.0))
    got = rp.transform_points_um([[1.0, 0.0]], tf)[0]
    assert np.allclose(got, [0.0, 1.0], atol=1e-9), got

    # A point AT the rotation centre is the fixed point -- the property the
    # whole hinge argument rests on (put the centre on the hinge and the tear
    # a rigid move opens is zero).
    tf = rp.make_keyframe(z=0, theta_deg=37.0, center_um=(50.0, 60.0))
    got = rp.transform_points_um([[50.0, 60.0]], tf)[0]
    assert np.allclose(got, [50.0, 60.0]), got
    print("   OK")


def test_image_and_points_agree():
    print("2. apply_to_image lands where transform_points_um says (translation + rotation)...")
    for label_txt, tf_kwargs in [
        ("translate", dict(tx_um=26.0, ty_um=-15.6)),
        ("rotate", dict(theta_deg=20.0, center_um=(200.0, 130.0))),
        ("both", dict(tx_um=10.4, ty_um=5.2, theta_deg=-12.0, center_um=(180.0, 120.0))),
    ]:
        image, labels = make_stack()
        plan = rp.make_plan(image.shape, PAINT_UM, [
            rp.make_fragment(1, [rp.make_keyframe(z=z, **tf_kwargs) for z in (4, 7)])])
        moved = rp.apply_to_image(image, labels, plan)

        for z in (4, 7):
            before = centre_of_mass_xy_um(image[z], PAINT_UM, 100)
            after = centre_of_mass_xy_um(moved[z], PAINT_UM, 100)
            predicted = rp.transform_points_um([before], rp.plane_transform(
                plan["fragments"][0], z))[0]
            err = np.abs(after - predicted).max()
            assert err < 0.6, f"{label_txt} z={z}: image landed at {after}, points say {predicted}"
        # The source must actually be vacated, not duplicated.
        assert (moved[4] > 1000).sum() > 0
        print(f"   {label_txt}: image/point centroids agree to <0.6 um")
    print("   OK")


def test_dz_moves_between_planes():
    print("3. dz_planes moves content onto another plane, and cells with it...")
    image, labels = make_stack(planes=(4, 5))
    plan = rp.make_plan(image.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=z, dz_planes=2) for z in (4, 5)])])
    moved = rp.apply_to_image(image, labels, plan)
    assert (moved[4] > 1000).sum() == 0, "source plane 4 should be empty"
    assert (moved[6] > 1000).sum() > 0, "content should have landed on plane 6"
    assert (moved[7] > 1000).sum() > 0

    # Painted voxel (z=4, y=25, x=38) is inside the box; in cell pixels that is
    # x=38*5.2/0.65=304, y=200, z=4*64/8=32. +2 painted planes is +16 cell planes.
    cx, cy, cz, mv = rp.apply_to_cells([304.0], [200.0], [32.0], labels, plan, CELLS_UM)
    assert mv[0] == 1, "cell on the fragment should have been moved"
    assert np.isclose(cz[0], 48.0), cz
    print("   OK")


def test_keyframe_interpolation():
    print("4. keyframes: interpolate between, identity outside, and the False reading...")
    frag = rp.make_fragment(1, [rp.make_keyframe(z=10, tx_um=0.0, theta_deg=0.0),
                                rp.make_keyframe(z=20, tx_um=100.0, theta_deg=8.0)])
    mid = rp.plane_transform(frag, 15, interpolate=True)
    assert np.isclose(mid["tx_um"], 50.0) and np.isclose(mid["theta_deg"], 4.0), mid
    assert rp.plane_transform(frag, 9, interpolate=True) is None, "below the span must not move"
    assert rp.plane_transform(frag, 21, interpolate=True) is None, "above the span must not move"

    assert rp.plane_transform(frag, 15, interpolate=False) is None
    assert rp.plane_transform(frag, 10, interpolate=False) is not None
    assert rp.fragment_source_planes(frag, 30, interpolate=True) == list(range(10, 21))
    assert rp.fragment_source_planes(frag, 30, interpolate=False) == [10, 20]

    # dz is rounded, never blended -- half a section is not a place to put content.
    frag2 = rp.make_fragment(1, [rp.make_keyframe(z=0, dz_planes=0),
                                 rp.make_keyframe(z=4, dz_planes=1)])
    assert [rp.plane_transform(frag2, z)["dz_planes"] for z in range(5)] == [0, 0, 0, 1, 1]
    print("   OK")


def test_cells_only_move_on_their_fragment():
    print("5. apply_to_cells moves cells on the fragment and nothing else...")
    image, labels = make_stack(box=(20, 32, 30, 46), planes=(4, 5, 6, 7))
    plan = rp.make_plan(image.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=z, tx_um=52.0) for z in (4, 7)])])

    # Painted voxel (z=5, y=25, x=38) is inside the box; in cell pixels that is
    # x=38*5.2/0.65=304, y=200, z=5*64/8=40. The second cell is well outside.
    cx = np.array([304.0, 40.0])
    cy = np.array([200.0, 40.0])
    cz = np.array([40.0, 40.0])
    nx, ny, nz, moved = rp.apply_to_cells(cx, cy, cz, labels, plan, CELLS_UM)

    assert moved[0] == 1 and moved[1] == 0, moved
    assert np.isclose(nx[0], 304.0 + 52.0 / 0.65), nx      # +52 um = +80 cell pixels
    assert np.isclose(ny[0], 200.0) and np.isclose(nz[0], 40.0)
    assert np.isclose(nx[1], 40.0) and np.isclose(ny[1], 40.0), "off-fragment cell moved"
    print("   OK")


def test_invert_plan_round_trips():
    print("6. invert_plan takes points and images back where they started...")
    tf_kwargs = dict(tx_um=31.2, ty_um=-10.4, theta_deg=15.0, center_um=(190.0, 125.0))
    image, labels = make_stack()
    plan = rp.make_plan(image.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=z, **tf_kwargs) for z in (4, 7)])])

    pts = np.array([[150.0, 110.0], [220.0, 160.0]])
    fwd = rp.transform_points_um(pts, rp.plane_transform(plan["fragments"][0], 5))
    back = rp.transform_points_um(fwd, rp.plane_transform(
        rp.invert_plan(plan)["fragments"][0], 5))
    assert np.abs(back - pts).max() < 1e-9, back

    # And the same on the image, through the moved label volume the inverse
    # has to be applied from.
    moved_img = rp.apply_to_image(image, labels, plan)
    moved_lab = rp.apply_to_labels(labels, plan)
    restored = rp.apply_to_image(moved_img, moved_lab, rp.invert_plan(plan))
    before = centre_of_mass_xy_um(image[5], PAINT_UM, 100)
    after = centre_of_mass_xy_um(restored[5], PAINT_UM, 100)
    assert np.abs(after - before).max() < 0.6, (before, after)
    print("   OK")


def test_two_fragments_erase_before_paste():
    print("7. one fragment's source overlapping another's target survives...")
    # Fragment 2 is parked exactly where fragment 1 is about to land. If the
    # erase and paste passes were interleaved, moving 1 first and then erasing
    # 2's source would delete what 1 just pasted.
    image = np.full((6, 40, 60), 100, dtype=np.uint16)
    labels = np.zeros((6, 40, 60), dtype=np.uint8)
    image[2, 10:20, 5:15] = 4000
    labels[2, 10:20, 5:15] = 1
    image[2, 10:20, 25:35] = 3000
    labels[2, 10:20, 25:35] = 2

    plan = rp.make_plan(image.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=2, tx_um=20 * PAINT_UM[0])]),   # 1 -> where 2 was
        rp.make_fragment(2, [rp.make_keyframe(z=2, tx_um=20 * PAINT_UM[0])]),   # 2 -> further right
    ])
    moved = rp.apply_to_image(image, labels, plan)
    assert moved[2, 15, 30] > 3500, "fragment 1 should now occupy fragment 2's old spot"
    assert moved[2, 15, 50] > 2500, "fragment 2 should have landed further right"
    assert moved[2, 15, 10] < 500, "both source regions should be vacated"
    print("   OK")


def test_boundary_report_flags_only_real_steps():
    print("8. boundary_report warns on a live edge, stays quiet on a tapered one...")
    # Fragment ends abruptly: plane 7 still carries the full box, plane 8 has none.
    image, labels = make_stack(planes=(4, 5, 6, 7))
    plan = rp.make_plan(image.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=z, tx_um=104.0) for z in (4, 7)], "abrupt")])
    rows = rp.boundary_report(plan, labels)
    # The default voxel threshold is sized for real flaps; this synthetic box is
    # 192 voxels a plane, so the test states its own separation (192 vs the
    # tapered case's 4) rather than depending on a tuned production default.
    warnings = rp.boundary_warnings(rows, voxel_threshold=100)
    assert len(warnings) == 2, warnings              # both edges end abruptly here
    assert all(w.startswith("label 1 (abrupt)") for w in warnings), warnings
    assert any("z=7" in w for w in warnings) and any("z=4" in w for w in warnings), warnings

    # Same move, but the fragment thins to a sliver at its edges -- no step,
    # because there is almost nothing there to be stepped.
    image2 = np.full((12, 60, 80), 100, dtype=np.uint16)
    labels2 = np.zeros((12, 60, 80), dtype=np.uint8)
    for z, half in ((4, 1), (5, 6), (6, 6), (7, 1)):
        labels2[z, 26 - half:26 + half, 38 - half:38 + half] = 1
        image2[z, 26 - half:26 + half, 38 - half:38 + half] = 4000
    plan2 = rp.make_plan(image2.shape, PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=z, tx_um=104.0) for z in (4, 7)], "tapered")])
    assert rp.boundary_warnings(rp.boundary_report(plan2, labels2), voxel_threshold=100) == []
    print("   OK")


def test_plan_roundtrip_and_validation(tmp_dir):
    print("9. plan write/read, and the hand-edit mistakes read_plan catches...")
    plan = rp.make_plan((12, 60, 80), PAINT_UM, [
        rp.make_fragment(1, [rp.make_keyframe(z=4, tx_um=1.0), rp.make_keyframe(z=7, tx_um=2.0)],
                         "cortex flap A")], interpolate=True, feather_um=10.0)
    path = rp.write_plan(tmp_dir / "plan.json", plan)
    assert rp.read_plan(path) == plan

    for broken, expect in [
        ({"format": "nope"}, "format"),
        ({"fragments": [rp.make_fragment(1, []), rp.make_fragment(1, [])]}, "more than one"),
        ({"fragments": [rp.make_fragment(0, [])]}, "background"),
        ({"fragments": [rp.make_fragment(1, [rp.make_keyframe(z=4), rp.make_keyframe(z=4)])]},
         "only have one transform"),
    ]:
        bad = dict(plan)
        bad.update(broken)
        rp.write_plan(tmp_dir / "bad.json", bad)
        try:
            rp.read_plan(tmp_dir / "bad.json")
        except ValueError as exc:
            assert expect in str(exc), f"wrong error for {broken}: {exc}"
        else:
            raise AssertionError(f"read_plan accepted {broken}")
    print("   OK")



def test_pipeline_integration(tmp_dir):
    print("10. pipeline wiring: one config key moves the stack, the outlines and the cells...")
    import ants
    import pandas as pd
    import SimpleITK as sitk
    from registration_ants import cell_points, pipeline

    # A plan and its fragment outlines on a (10, 10, 10) painted grid.
    fragments = np.zeros((10, 10, 10), dtype=np.uint8)
    fragments[8, 4:8, 1:5] = 1        # (z, y, x)
    fragments_path = tmp_dir / "frag.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(fragments), str(fragments_path))
    painted_um = (10.0, 10.0, 10.0)
    plan = rp.make_plan((10, 10, 10), painted_um,
                        [rp.make_fragment(1, [rp.make_keyframe(z=8, tx_um=50.0, ty_um=20.0)], "flap")],
                        labels_path=str(fragments_path))
    plan_path = rp.write_plan(tmp_dir / "s.reposition.json", plan)

    sample_cfg = {"reposition_plan": str(plan_path), "voxel_size_um": list(painted_um)}
    loaded, frag_arr = pipeline._load_reposition(sample_cfg)
    assert loaded == plan and np.array_equal(frag_arr, fragments)
    assert pipeline._load_reposition({}) == (None, None)

    # A plan whose outlines are missing or the wrong shape must stop the run:
    # it says how far to move tissue but not which tissue.
    orphan = dict(plan, labels_path=str(tmp_dir / "gone.nii.gz"))
    rp.write_plan(tmp_dir / "orphan.json", orphan)
    # A plan drawn on a different grid -- or, the (1,1,1) case, on none at all.
    for drawn, expect in ((( 20.0, 20.0, 10.0), "um/voxel"), ((1.0, 1.0, 1.0), "voxel counts")):
        rp.write_plan(tmp_dir / "grid.json", dict(plan, voxel_size_um=list(drawn)))
        try:
            pipeline._load_reposition({"reposition_plan": str(tmp_dir / "grid.json"),
                                       "voxel_size_um": list(painted_um)})
        except ValueError as exc:
            assert expect in str(exc), exc
        else:
            raise AssertionError(f"a plan drawn on {drawn} was applied to {painted_um}")

    try:
        pipeline._load_reposition({"reposition_plan": str(tmp_dir / "orphan.json"),
                                   "voxel_size_um": list(painted_um)})
    except FileNotFoundError as exc:
        assert "labels_path" in str(exc), exc
    else:
        raise AssertionError("a plan with no fragment volume was accepted")

    # _reposition_volume works in ANTs' (x, y, z) order and refuses any other grid.
    img_xyz = np.zeros((10, 10, 10), dtype=np.float32)
    img_xyz[1:5, 4:8, 8] = 1000.0                      # the fragment, in (x, y, z)
    moved = pipeline._reposition_volume(
        ants.from_numpy(img_xyz, spacing=painted_um), plan, fragments, "image", "test")
    assert moved.numpy()[1:5, 4:8, 8].max() < 1.0, "the source region was not vacated"
    # +50 um and +20 um at 10 um/voxel: five voxels in x, two in y.
    assert moved.numpy()[6:10, 6:10, 8].min() > 900.0, "the fragment did not land where planned"
    try:
        pipeline._reposition_volume(
            ants.from_numpy(np.zeros((8, 8, 8), dtype=np.float32)), plan, fragments, "image", "wrong")
    except ValueError as exc:
        assert "only be applied on the grid it was drawn on" in str(exc), exc
    else:
        raise AssertionError("a plan was applied to a volume of a different shape")

    # And the cells: moved with their fragment, but columns 0-2 keep the raw
    # position the detector actually reported.
    atlas_annotation = ants.from_numpy(np.full((10, 10, 10), 5, dtype=np.float32),
                                       spacing=(25.0, 25.0, 25.0))
    reg = {"fwdtransforms": [], "invtransforms": [], "atlas_annotation": atlas_annotation}
    sample_fine = ants.from_numpy(np.zeros((20, 20, 20), dtype=np.float32), spacing=(50.0,) * 3)
    cells_dir = tmp_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    # raw (6,10,16) * 5 um = (30,50,80) um = painted voxel (x3, y5, z8), on the
    # fragment; raw (2,2,2) = 10 um = painted (1,1,1), off it.
    pd.DataFrame({"cx": [6.0, 2.0], "cy": [10.0, 2.0], "z": [16.0, 2.0],
                  "score": [0.5, 0.5], "slice_name": ["a", "b"], "tile_name": ["t", "t"]}
                 ).to_csv(cells_dir / "ob_c.csv", index=False)

    cell_points.assign_cell_regions(
        str(cells_dir), str(tmp_dir / "out"), (5.0, 5.0, 5.0), sample_fine, reg,
        atlas_structures={5: {"name": "R5"}}, prefix="ob_",
        reposition_plan=plan, reposition_fragments=fragments)
    got = pd.read_csv(tmp_dir / "out" / "cell_registration" / "c" / "cell_registration.csv",
                      header=None)
    assert np.allclose(got.iloc[0, 0:3].values.astype(float), [6.0, 10.0, 16.0]), \
        "columns 0-2 must stay the raw detected position, not the moved one"
    # (30,50,80) um moved to (80,70,80); the resample grid is 50 um, so the
    # index is (1.6, 1.4, 1.6) where an unrepositioned run would give (0.6, 1.0, 1.6).
    assert np.allclose(got.iloc[0, 3:6].values.astype(float), [1.6, 1.4, 1.6]), got.iloc[0, 3:6]
    assert np.allclose(got.iloc[1, 3:6].values.astype(float), [0.2, 0.2, 0.2]), \
        "the off-fragment cell moved"
    print("   OK")



def test_grab_plane_takes_the_piece_and_nothing_else():
    print("11. grab_plane: one click takes the piece on that plane, not the brain...")
    tissue = np.zeros((12, 40, 60), dtype=bool)
    tissue[:, 14:38, 5:55] = True                 # the brain
    tissue[:, 4:12, 12:34] = True                 # a piece, across a one-voxel crack

    got, note = rp.grab_plane(tissue, (5, 8, 20))
    assert got[5, 4:12, 12:34].all(), "the piece was not taken"
    assert not got[5, 14:38, 5:55].any(), "the grab leaked across the crack into the brain"
    assert not got[4].any() and not got[6].any(), "grab_plane reached another plane"
    assert "plane 5" in note, note

    # A crack too tight to threshold apart: the pieces touch, so one component
    # spans both. A few strokes in `exclude` separate them again -- the cut
    # brush, and the only thing that gets a grab past a join.
    welded = tissue.copy()
    welded[:, 12:14, 12:34] = True
    leaked, _ = rp.grab_plane(welded, (5, 8, 20))
    assert leaked[5, 20:38, 10:50].any(), "the phantom does not weld; the test is void"
    cut = np.zeros(tissue.shape, dtype=np.uint8)
    cut[:, 12:14, 12:34] = 255
    fixed, _ = rp.grab_plane(
        rp.threshold_planes(welded.astype(np.uint8), 0, exclude=cut), (5, 8, 20))
    assert not fixed[5, 20:38, 10:50].any(), "the cut did not separate the pieces"
    assert fixed[5, 4:12, 12:34].all(), "the cut ate into the piece"

    # Clicking off tissue is a mistake worth naming, not an empty result.
    try:
        rp.grab_plane(tissue, (5, 0, 0))
    except ValueError as exc:
        assert "not on tissue" in str(exc), exc
    else:
        raise AssertionError("a seed on background was accepted")
    print("   OK")


def test_pipeline_integration(tmp_dir):
    print("10. pipeline wiring: one config key moves the stack, the outlines and the cells...")
    import ants
    import pandas as pd
    import SimpleITK as sitk
    from registration_ants import cell_points, pipeline

    # A plan and its fragment outlines on a (10, 10, 10) painted grid.
    fragments = np.zeros((10, 10, 10), dtype=np.uint8)
    fragments[8, 4:8, 1:5] = 1        # (z, y, x)
    fragments_path = tmp_dir / "frag.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(fragments), str(fragments_path))
    painted_um = (10.0, 10.0, 10.0)
    plan = rp.make_plan((10, 10, 10), painted_um,
                        [rp.make_fragment(1, [rp.make_keyframe(z=8, tx_um=50.0, ty_um=20.0)], "flap")],
                        labels_path=str(fragments_path))
    plan_path = rp.write_plan(tmp_dir / "s.reposition.json", plan)

    sample_cfg = {"reposition_plan": str(plan_path), "voxel_size_um": list(painted_um)}
    loaded, frag_arr = pipeline._load_reposition(sample_cfg)
    assert loaded == plan and np.array_equal(frag_arr, fragments)
    assert pipeline._load_reposition({}) == (None, None)

    # A plan whose outlines are missing or the wrong shape must stop the run:
    # it says how far to move tissue but not which tissue.
    orphan = dict(plan, labels_path=str(tmp_dir / "gone.nii.gz"))
    rp.write_plan(tmp_dir / "orphan.json", orphan)
    # A plan drawn on a different grid -- or, the (1,1,1) case, on none at all.
    for drawn, expect in ((( 20.0, 20.0, 10.0), "um/voxel"), ((1.0, 1.0, 1.0), "voxel counts")):
        rp.write_plan(tmp_dir / "grid.json", dict(plan, voxel_size_um=list(drawn)))
        try:
            pipeline._load_reposition({"reposition_plan": str(tmp_dir / "grid.json"),
                                       "voxel_size_um": list(painted_um)})
        except ValueError as exc:
            assert expect in str(exc), exc
        else:
            raise AssertionError(f"a plan drawn on {drawn} was applied to {painted_um}")

    try:
        pipeline._load_reposition({"reposition_plan": str(tmp_dir / "orphan.json"),
                                   "voxel_size_um": list(painted_um)})
    except FileNotFoundError as exc:
        assert "labels_path" in str(exc), exc
    else:
        raise AssertionError("a plan with no fragment volume was accepted")

    # _reposition_volume works in ANTs' (x, y, z) order and refuses any other grid.
    img_xyz = np.zeros((10, 10, 10), dtype=np.float32)
    img_xyz[1:5, 4:8, 8] = 1000.0                      # the fragment, in (x, y, z)
    moved = pipeline._reposition_volume(
        ants.from_numpy(img_xyz, spacing=painted_um), plan, fragments, "image", "test")
    assert moved.numpy()[1:5, 4:8, 8].max() < 1.0, "the source region was not vacated"
    # +50 um and +20 um at 10 um/voxel: five voxels in x, two in y.
    assert moved.numpy()[6:10, 6:10, 8].min() > 900.0, "the fragment did not land where planned"
    try:
        pipeline._reposition_volume(
            ants.from_numpy(np.zeros((8, 8, 8), dtype=np.float32)), plan, fragments, "image", "wrong")
    except ValueError as exc:
        assert "only be applied on the grid it was drawn on" in str(exc), exc
    else:
        raise AssertionError("a plan was applied to a volume of a different shape")

    # And the cells: moved with their fragment, but columns 0-2 keep the raw
    # position the detector actually reported.
    atlas_annotation = ants.from_numpy(np.full((10, 10, 10), 5, dtype=np.float32),
                                       spacing=(25.0, 25.0, 25.0))
    reg = {"fwdtransforms": [], "invtransforms": [], "atlas_annotation": atlas_annotation}
    sample_fine = ants.from_numpy(np.zeros((20, 20, 20), dtype=np.float32), spacing=(50.0,) * 3)
    cells_dir = tmp_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    # raw (6,10,16) * 5 um = (30,50,80) um = painted voxel (x3, y5, z8), on the
    # fragment; raw (2,2,2) = 10 um = painted (1,1,1), off it.
    pd.DataFrame({"cx": [6.0, 2.0], "cy": [10.0, 2.0], "z": [16.0, 2.0],
                  "score": [0.5, 0.5], "slice_name": ["a", "b"], "tile_name": ["t", "t"]}
                 ).to_csv(cells_dir / "ob_c.csv", index=False)

    cell_points.assign_cell_regions(
        str(cells_dir), str(tmp_dir / "out"), (5.0, 5.0, 5.0), sample_fine, reg,
        atlas_structures={5: {"name": "R5"}}, prefix="ob_",
        reposition_plan=plan, reposition_fragments=fragments)
    got = pd.read_csv(tmp_dir / "out" / "cell_registration" / "c" / "cell_registration.csv",
                      header=None)
    assert np.allclose(got.iloc[0, 0:3].values.astype(float), [6.0, 10.0, 16.0]), \
        "columns 0-2 must stay the raw detected position, not the moved one"
    # (30,50,80) um moved to (80,70,80); the resample grid is 50 um, so the
    # index is (1.6, 1.4, 1.6) where an unrepositioned run would give (0.6, 1.0, 1.6).
    assert np.allclose(got.iloc[0, 3:6].values.astype(float), [1.6, 1.4, 1.6]), got.iloc[0, 3:6]
    assert np.allclose(got.iloc[1, 3:6].values.astype(float), [0.2, 0.2, 0.2]), \
        "the off-fragment cell moved"
    print("   OK")



def test_outline_polygon_and_rigid_fit_round_trip():
    print("14. outline_polygon + fit_from_points: drag the silhouette, get the pose back...")
    fragments = np.zeros((6, 60, 80), dtype=np.uint8)
    fragments[2, 20:40, 30:52] = 1
    fragments[2, 20:26, 52:60] = 1        # asymmetric, so a rotation is identifiable

    poly = rp.outline_polygon(fragments, 1, 2)
    assert poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3, poly.shape
    assert not np.allclose(poly[0], poly[-1]), "the closing duplicate vertex should be dropped"
    ys, xs = poly[:, 0], poly[:, 1]
    assert 19 <= ys.min() and ys.max() <= 40 and 29 <= xs.min() and xs.max() <= 60, poly

    # The contract: whatever rigid move the vertices underwent comes back as
    # the numbers that produced it, whatever centre it is expressed about.
    src = np.column_stack([xs * PAINT_UM[0], ys * PAINT_UM[1]])
    applied = rp.make_keyframe(2, tx_um=41.0, ty_um=-17.0, theta_deg=8.0, center_um=(120.0, 90.0))
    moved = rp.transform_points_um(src, applied)
    for centre in (None, (120.0, 90.0), (0.0, 0.0)):
        tx, ty, theta, c, scale = rp.fit_from_points(src, moved, center_um=centre)
        assert abs(theta - 8.0) < 1e-6, theta
        assert abs(scale - 1.0) < 1e-9, scale
        back = rp.transform_points_um(src, rp.make_keyframe(2, tx, ty, theta, 0, c))
        assert np.abs(back - moved).max() < 1e-6, np.abs(back - moved).max()

    # A resize is measured, never folded into the pose: napari's selection box
    # resizes if a corner handle is dragged, and a plan that quietly absorbed
    # it would move the tissue by the wrong amount.
    stretched = (moved - moved.mean(axis=0)) * 1.2 + moved.mean(axis=0)
    tx, ty, theta, c, scale = rp.fit_from_points(src, stretched)
    assert abs(scale - 1.2) < 1e-6, scale
    assert abs(theta - 8.0) < 1e-6, "a resize must not be read as a rotation"

    # Vertices added or removed: refused, because index correspondence is gone.
    try:
        rp.fit_from_points(src, moved[:-1])
    except ValueError as exc:
        assert "vertices were added or removed" in str(exc), exc
    else:
        raise AssertionError("a mismatched vertex count was accepted")

    # A plane the fragment was never grabbed on has no outline to copy.
    try:
        rp.outline_polygon(fragments, 1, 3)
    except ValueError as exc:
        assert "nothing on plane 3" in str(exc), exc
    else:
        raise AssertionError("an empty plane produced an outline")
    print("   OK")


def test_densify_fills_between_grabbed_planes_only():
    print("13. densify: sparse grabs fill in, labels stay apart, dense input is untouched...")
    sparse = np.zeros((12, 40, 60), dtype=np.uint8)
    sparse[2, 10:20, 10:30] = 1          # fragment 1, grabbed on 2 and 6
    sparse[6, 10:20, 10:30] = 1
    sparse[8, 25:35, 10:30] = 2          # fragment 2, grabbed on 8 and 10
    sparse[10, 25:35, 10:30] = 2

    dense = rp.densify_fragments(sparse)
    assert sorted(int(z) for z in np.unique(np.nonzero(dense == 1)[0])) == [2, 3, 4, 5, 6]
    assert sorted(int(z) for z in np.unique(np.nonzero(dense == 2)[0])) == [8, 9, 10]
    assert not (dense == 1)[7].any(), "filling ran past the last grabbed plane"
    assert not (dense == 2)[7].any(), "filling ran before the first grabbed plane"
    assert not ((dense == 1) & (dense == 2)).any()

    # Per plane, without building the volume -- what the GUI preview uses.
    for z in range(12):
        assert np.array_equal(rp.densify_plane(sparse, z), dense[z]), z

    # Already dense in, the same volume out: one code path serves a sparse
    # export and an outline someone traced the long way.
    assert np.array_equal(rp.densify_fragments(dense), dense)
    print("   OK")


def main():
    tmp_dir = Path("/tmp/claude-1004/-home-fyu7-My-project-Registration-ants/"
                   "aac47064-271f-44d0-8e1d-aadcd5b55308/scratchpad/reposition_smoketest")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_rotation_convention()
    test_image_and_points_agree()
    test_dz_moves_between_planes()
    test_keyframe_interpolation()
    test_cells_only_move_on_their_fragment()
    test_invert_plan_round_trips()
    test_two_fragments_erase_before_paste()
    test_boundary_report_flags_only_real_steps()
    test_plan_roundtrip_and_validation(tmp_dir)
    test_grab_plane_takes_the_piece_and_nothing_else()
    test_outline_polygon_and_rigid_fit_round_trip()
    test_densify_fills_between_grabbed_planes_only()
    test_pipeline_integration(tmp_dir)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
