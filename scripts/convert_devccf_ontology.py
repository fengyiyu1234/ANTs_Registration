"""Convert DevCCF's flat ontology table (DevCCFv1_OntologyStructure.xlsx) into the
Allen-API-style nested {"msg": [...]} JSON tree that atlas_utils.load_ccf_ontology_json
already parses, so every existing ontology consumer in this repo
(mask.atlas_exclude_regions, ../GT_tool_for_registration/edit_sample_labels.py, scripts/relabel_cells.py,
../GT_tool_for_registration/registration_eval.py) works against DevCCF completely unmodified.

Keyed by ID16, not the plain ID/ADMBA ID column -- confirmed empirically that every
DevCCF annotation volume's actual voxel values are ID16, not ID: some structures only
have an ID16 (no plain ID at all), and DevCCF's annotation volumes can't physically
store the plain ID column's largest values (legacy ADMBA ids run past 10^8) in the 16-bit
id space those volumes are stored in.

Usage:
    python scripts/convert_devccf_ontology.py \\
        /path/to/DevCCFv1_OntologyStructure.xlsx \\
        /path/to/DevCCFv1_ontology.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_ontology_tree(xlsx_path, sheet_name="DevCCFv1_Ontology"):
    """Parse the flat ID16/Parent ID16 table into one nested root node (Allen-API
    'msg' tree shape: id/name/acronym/children). Raises if the sheet doesn't have
    exactly one root (Parent ID16 == '[]') or has a child row pointing at a Parent
    ID16 that isn't itself a row in the sheet -- fail loudly rather than silently
    dropping structures.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)

    nodes = {}
    children_by_parent = {}
    root_id = None
    for _, row in df.iterrows():
        node_id = int(row["ID16"])
        if node_id in nodes:
            raise ValueError(f"duplicate ID16: {node_id}")
        nodes[node_id] = {
            "id": node_id,
            "name": str(row["Name"]),
            "acronym": str(row["Acronym"]),
            "children": [],
        }

        parent = row["Parent ID16"]
        if isinstance(parent, str) and parent.strip() == "[]":
            if root_id is not None:
                raise ValueError(f"multiple root rows (Parent ID16 == '[]'): {root_id} and {node_id}")
            root_id = node_id
        else:
            children_by_parent.setdefault(int(parent), []).append(node_id)

    if root_id is None:
        raise ValueError("no root row found (expected exactly one row with Parent ID16 == '[]')")

    orphans = sorted(pid for pid in children_by_parent if pid not in nodes)
    if orphans:
        raise ValueError(
            f"{len(orphans)} child row(s) reference a Parent ID16 not present as its own "
            f"row: {orphans[:10]}{'...' if len(orphans) > 10 else ''}"
        )

    for parent_id, child_ids in children_by_parent.items():
        nodes[parent_id]["children"] = [nodes[cid] for cid in child_ids]

    return nodes[root_id]


def _count_nodes(node):
    return 1 + sum(_count_nodes(c) for c in node["children"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xlsx_path")
    parser.add_argument("out_json_path")
    args = parser.parse_args()

    root = build_ontology_tree(args.xlsx_path)
    print(f"Built ontology tree: root={root['name']!r} (id={root['id']}), {_count_nodes(root)} structures total")

    out_path = Path(args.out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"msg": [root]}, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
