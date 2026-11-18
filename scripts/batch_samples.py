"""Sample table + path helpers for the TSC / WT batch registration
(scripts/orientation_review.py, scripts/make_batch_configs.py).

Every sample is a half brain stitched by brain_detector's
src/core/stitch_raw_tiles.py at 2.6 x 2.6 x 8 um. Registration uses the soma
channel image (stitched_imgs/soma: RFP, or GFP for t4/t10/t70); cell
centroids come from <pATHRESULT>/7_colocalization/cell_centroids.

Frames: cells are in the stitching_reference_channel frame (Olig2, or Sox9
for the old TSC samples) -- Stage 3 places tile t at ABS(t) - min ABS of the
reference XML, and z at raw slice - (max ABS_D - ABS_D(t)). The soma image
uses the soma channel's own xml_merging.xml the same way, with the min/max
taken over that XML. soma_cell_offset() returns the median per-tile
difference, i.e. what to add to a cell coordinate (0.65 um px, 8 um slices)
to land on the same point in the soma image. The per-tile spread around that
median is ~5-20 px (a few um) -- left uncorrected.
"""
import glob
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DETECTOR_CONFIG_DIR = Path("Y:/Fengyi/brain_detector/config")
STITCHED_DIR = Path("X:/Fengyi/stitched_imgs")
OUTPUT_ROOT = Path("X:/Fengyi/registration_ants")
REVIEW_DIR = OUTPUT_ROOT / "orientation_review"
REVIEW_XLSX = REVIEW_DIR / "orientation_review.xlsx"
REPO = Path(__file__).resolve().parents[1]
CONFIG_OUT_DIR = REPO / "configs" / "batch"

IMAGE_VOXEL_UM = [2.6, 2.6, 8.0]
CELL_VOXEL_UM = [0.65, 0.65, 8.0]

# name = stitched-image output_name; old = the old registration config (configs/<old>.yaml)
SAMPLES = [
    # TSC FF
    dict(name="TSC_FF6", group="TSC_FF", config="TSC2/config_TSC_FF6.json"),
    dict(name="TSC_FF8", group="TSC_FF", config="TSC2/config_TSC_FF8.json"),
    dict(name="TSC_FF9", group="TSC_FF", config="TSC2/config_TSC_FF9.json"),
    dict(name="TSC_FF11", group="TSC_FF", config="TSC2/config_TSC_FF11.json"),
    dict(name="TSC_FF19", group="TSC_FF", config="TSC2/config_TSC_FF19.json"),
    dict(name="TSC8", group="TSC_FF", config="TSC/config_TSC8.json", old="s8"),
    dict(name="TSC12q", group="TSC_FF", config="TSC/config_tsc12q.json", old="s12q"),
    dict(name="TSC12t", group="TSC_FF", config="TSC/config_tsc12t.json", old="s12t"),
    # TSC FW
    dict(name="TSC_FW2", group="TSC_FW", config="TSC2/config_TSC_FW2.json"),
    dict(name="TSC_FW6", group="TSC_FW", config="TSC2/config_TSC_FW6.json"),
    dict(name="TSC_FW7", group="TSC_FW", config="TSC2/config_TSC_FW7.json"),
    dict(name="TSC_FW8", group="TSC_FW", config="TSC2/config_TSC_FW8.json"),
    dict(name="TSC_FW11", group="TSC_FW", config="TSC2/config_TSC_FW11.json"),
    dict(name="TSC18", group="TSC_FW", config="TSC/config_tsc18.json", old="s18"),
    dict(name="TSC10", group="TSC_FW", config="TSC/config_TSC10.json", old="s10"),
    dict(name="TSC11", group="TSC_FW", config="TSC/config_tsc11.json", old="s11"),
    # WT controls (shared with the EGFR analysis)
    dict(name="t70", group="WT", config="EGFR/2/config_EGFR_t70.json"),
    dict(name="t4", group="WT", config="EGFR/2/config_EGFR_t4.json"),
    dict(name="t10", group="WT", config="EGFR/2/config_EGFR_t10.json"),
    dict(name="ET10", group="WT", config="EGFR/3/config_EGFR_ET10.json"),
    dict(name="ETail", group="WT", config="EGFR/3/config_EGFR_ETail.json"),
    dict(name="ET1", group="WT", config="EGFR/4/config_EGFR_ET1.json"),
    dict(name="ERH", group="WT", config="EGFR/4/config_EGFR_ERH.json"),
]

# Which end of the stack z=0 is. Old TSC samples: decided in their old configs
# (s10 is the only ventral-first one); everything else defaults to dorsal.
OLD_Z_FIRST = {"s8": "dorsal", "s10": "ventral", "s11": "dorsal",
               "s12q": "dorsal", "s12t": "dorsal", "s18": "dorsal"}

# DeMBA P5 orientation/slicing per z_first, copied from the old configs
# (s8/s11/s12q/s12t/s18 for dorsal, s10 for ventral). Ventral-first with the
# midline still on the image's left is a left hemisphere seen from below:
# flipping atlas x and z together is a proper rotation, not a mirror.
DEMBA_P5 = {
    "dorsal": dict(orientation=[1, 3, 2], slicing=[[285, 510], None, [35, 356]]),
    "ventral": dict(orientation=[-1, 3, -2], slicing=[[285, 514], None, [44, 365]]),
}


def to_local(path):
    return (str(path).replace("/rsstu/users/a/agrinba/DeepDesign", "Y:")
            .replace("/rsstu/users/t/tekaeser/Chlamydia", "X:"))


def _strip_json_comments(text):
    out, i, in_str = [], 0, False
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\":
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(ch)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def detector_config(sample):
    text = (DETECTOR_CONFIG_DIR / sample["config"]).read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(text))


def cell_centroids_dir(cfg):
    return Path(to_local(cfg["paths"]["pATHRESULT"])) / "7_colocalization" / "cell_centroids"


def reference_xml(cfg):
    marker = cfg["stitching_reference_channel"]
    route = next(r for r in cfg["channels_routing"] if r["id"] == marker)
    return marker, Path(to_local(cfg["paths"][route["dir_key"]])) / "xml_merging.xml"


def soma_image(name):
    """(tif, manifest dict) of the finished soma image, or (None, None)."""
    for marker in ("RFP", "GFP"):
        manifest = STITCHED_DIR / "soma" / f"{name}_{marker}_xy4x.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            return manifest.with_name(m["file"]), m
    return None, None


def xml_positions(path):
    root = ET.parse(path).getroot()
    return {s.get("DIR_NAME"): np.array([int(s.get(k)) for k in ("ABS_H", "ABS_V", "ABS_D")])
            for s in root.find("STACKS")}


def soma_cell_offset(manifest, ref_xml):
    """Median [dx, dy, dz] (cell px / slices) taking cell coords to soma-image coords,
    plus the per-tile 90th-percentile |deviation| from it in x/y."""
    soma_xml = Path(to_local(next(iter(manifest["channels"].values()))["xml"]))
    ps, pr = xml_positions(soma_xml), xml_positions(ref_xml)
    common = sorted(set(ps) & set(pr))
    if not common:
        raise ValueError(f"no shared tiles between {soma_xml} and {ref_xml}")
    diff = np.array([ps[k] - pr[k] for k in common])
    ref = np.array(list(pr.values()))
    med = np.median(diff, axis=0)
    dx = med[0] - (manifest["global_min_abs_h"] - ref[:, 0].min())
    dy = med[1] - (manifest["global_min_abs_v"] - ref[:, 1].min())
    dz = med[2] - (manifest["global_max_abs_d"] - ref[:, 2].max())
    spread = np.percentile(np.abs(diff[:, :2] - med[:2]), 90, axis=0)
    return [float(dx), float(dy), float(dz)], [float(v) for v in spread], soma_xml


def default_z_first(sample):
    return OLD_Z_FIRST.get(sample.get("old"), "dorsal")


def read_review_annotations(path=REVIEW_XLSX):
    """{sample: 'dorsal'|'ventral'} from the user's column of the review sheet."""
    if not Path(path).exists():
        return {}
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)["orientation"]
    header = [c.value for c in ws[1]]
    i_name, i_user = header.index("sample"), header.index("z0_user")
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        value = (row[i_user] or "").strip().lower() if row[i_user] else ""
        if value in ("dorsal", "ventral"):
            out[row[i_name]] = value
    return out
