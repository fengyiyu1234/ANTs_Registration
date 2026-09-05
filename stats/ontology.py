"""CCF ontology tree, ClearMap-free.

Replaces `ClearMap.Alignment.Annotation` (the `ano` module the old
stats_group_compare.py was built on) with a self-contained tree built
straight from the Allen-API-style ontology JSON this project already ships
(`atlas/DeMBA/CCF_v3_ontology.json`), via
`registration_ants.atlas_utils.load_ccf_ontology_json`.

Two things it has to provide that the rest of the stats code leans on:

1. A dense `order` index 0..n-1 so counts/volumes can live in plain arrays
   and DataFrames keyed by one small integer instead of by the raw CCF
   structure id (which reaches 6.1e8 -- a dense id-indexed LUT would be
   2.4 GB, the same trap noted for atlas_utils.collapse_labels_to_level).
   Ids are mapped to orders with np.searchsorted instead.

2. Hierarchical rollup: count(region) = cells directly in it + all its
   descendants. `order` is assigned in tree pre-order (sorted by
   structure_id_path), which guarantees parent_order < child_order for every
   node, so a single reverse sweep accumulates the whole tree in O(n).

The `order` index here is NOT ClearMap's `order` -- it comes from a different
tree walk over a different file. It is stable for a given ontology JSON, which
is all any of the outputs need, but numbers from before the ClearMap cut are
not comparable by `order`; join on `id` instead.
"""
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from registration_ants.atlas_utils import load_ccf_ontology_json  # noqa: E402


class Ontology:
    """Order-indexed view of a CCF structure tree.

    Attributes, all order-indexed arrays of length n:
      ids           int64 CCF structure id
      names         object, structure name
      acronyms      object
      levels        int, tree depth (root = 0)
      parent_order  int, order of the parent (-1 for the root)
    plus `children` (list of lists of orders) and `root_order`.
    """

    def __init__(self, structures):
        if not structures:
            raise ValueError("empty ontology")
        # Pre-order: sorting by structure_id_path puts every node after its
        # own ancestors (a path is a prefix of all its descendants' paths).
        items = sorted(structures.values(), key=lambda s: tuple(s["structure_id_path"]))

        self.ids = np.array([s["id"] for s in items], dtype=np.int64)
        self.names = np.array([s["name"] for s in items], dtype=object)
        self.acronyms = np.array([s.get("acronym", "") for s in items], dtype=object)
        self.levels = np.array([len(s["structure_id_path"]) - 1 for s in items], dtype=int)
        self.n = len(items)

        order_by_id = {int(i): o for o, i in enumerate(self.ids)}
        self.parent_order = np.full(self.n, -1, dtype=int)
        for o, s in enumerate(items):
            path = s["structure_id_path"]
            if len(path) > 1:
                # A parent missing from the file would silently detach a whole
                # subtree from the rollup, so fail loudly instead.
                parent = order_by_id.get(int(path[-2]))
                if parent is None:
                    raise ValueError(
                        f"structure {s['id']} ({s['name']}) lists parent {path[-2]}, "
                        "which is not in the ontology"
                    )
                self.parent_order[o] = parent

        self.children = [[] for _ in range(self.n)]
        for o in range(self.n):
            p = self.parent_order[o]
            if p >= 0:
                self.children[p].append(o)

        roots = np.where(self.parent_order < 0)[0]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root, found {len(roots)}")
        self.root_order = int(roots[0])

        # searchsorted lookup table (ids are not sorted in pre-order)
        self._sorted_ids = np.sort(self.ids)
        self._sorted_to_order = np.argsort(self.ids, kind="stable")

    @classmethod
    def from_json(cls, path):
        return cls(load_ccf_ontology_json(path))

    def order_of_ids(self, ids):
        """Map raw CCF structure ids -> order index. Unknown ids give -1."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=int)
        pos = np.searchsorted(self._sorted_ids, ids)
        pos_clipped = np.clip(pos, 0, len(self._sorted_ids) - 1)
        hit = self._sorted_ids[pos_clipped] == ids
        out = np.where(hit, self._sorted_to_order[pos_clipped], -1)
        return out.astype(int)

    def rollup(self, direct):
        """direct[order] -> rolled[order] = direct + sum over all descendants.

        Relies on parent_order[o] < o for every node (guaranteed by the
        pre-order construction above), so one reverse sweep suffices.
        """
        direct = np.asarray(direct, dtype=float)
        if direct.shape != (self.n,):
            raise ValueError(f"expected shape ({self.n},), got {direct.shape}")
        rolled = direct.copy()
        for o in range(self.n - 1, 0, -1):
            p = self.parent_order[o]
            if p >= 0:
                rolled[p] += rolled[o]
        return rolled

    def bincount_ids(self, ids):
        """Raw structure ids -> direct (non-hierarchical) counts per order.
        Ids not in the ontology are dropped; the count of those is returned
        alongside so callers can report it rather than lose it silently."""
        orders = self.order_of_ids(ids)
        unknown = int((orders < 0).sum())
        counts = np.bincount(orders[orders >= 0], minlength=self.n).astype(float)
        return counts, unknown

    def metadata_frame(self):
        import pandas as pd

        return pd.DataFrame({
            "order": np.arange(self.n),
            "id": self.ids,
            "acronym": self.acronyms,
            "name": self.names,
            "level": self.levels,
            "parent_id": np.where(self.parent_order >= 0, self.ids[self.parent_order], -1),
        })

    def descendants_of(self, ids):
        """All orders in the subtree(s) rooted at the given structure ids
        (the ids themselves included). Used by the region whitelist."""
        return self.descendant_orders(o for o in self.order_of_ids(ids) if o >= 0)

    def descendant_orders(self, orders):
        """Same, but starting from orders rather than ids. Used by the
        gatekeeping chain, which walks down from whichever regions were
        significant at the previous tested level."""
        out = set()
        stack = list(orders)
        while stack:
            o = int(stack.pop())
            if o in out:
                continue
            out.add(o)
            stack.extend(self.children[o])
        return sorted(out)

    def ancestor_at_level(self, level):
        """-> array over orders: the ancestor of each node at depth `level`,
        or -1 for nodes shallower than `level` (they have no such ancestor).

        A node at exactly `level` maps to itself. This is what collapses a
        voxel-level annotation to a chosen ontology depth: a voxel labelled
        CA1 (level 8) reads out its level-5 ancestor HPF, so a level-5 map
        paints CA1's voxels with HPF's value.

        One forward pass suffices because pre-order guarantees the parent is
        visited before the child.
        """
        out = np.full(self.n, -1, dtype=int)
        for o in range(self.n):
            if self.levels[o] == level:
                out[o] = o
            elif self.levels[o] > level:
                p = self.parent_order[o]
                out[o] = out[p] if p >= 0 else -1
        return out

    def subtree_mask(self, orders):
        """Boolean array over all orders: True inside the subtree(s) rooted at
        `orders`."""
        m = np.zeros(self.n, dtype=bool)
        idx = self.descendant_orders(orders)
        if idx:
            m[idx] = True
        return m
