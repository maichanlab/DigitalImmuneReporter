"""
Patches CellViT-plus-plus's `OverlapCellCleaner._remove_overlap` for Shapely 2.x.

CellViT-plus-plus was written against Shapely 1.x, where (a) `shapely.geometry.Polygon` allows
attaching arbitrary attributes (`poly.uid = idx`) and (b) `shapely.strtree.STRtree.query()`
returns the matched geometry objects themselves. Shapely 2.x's `Polygon` uses `__slots__` (no
more ad-hoc attributes — `poly.uid = idx` raises `AttributeError`) and `STRtree.query()` returns
integer indices into the tree's input list instead of geometry objects — both hard, unavoidable
API changes, not just a conservative version pin.

This codebase pins `Shapely>=2.1` for Trident's own Step 1 tissue segmentation (see
setup_env.sh's comments), so rather than downgrading Shapely (which would break Trident),
this monkey-patches a Shapely-2.x-compatible reimplementation of the exact same
overlap-removal algorithm (uid tracked via a parallel list instead of a geometry attribute;
`STRtree.query()`'s indices resolved back through that list) in at runtime — same
"patch before use" approach as `torch_load_patch.py` / `slidevips_mpp_patch.py` /
`pathopatch_pydantic_patch.py`. Call `apply()` before constructing any `OverlapCellCleaner`
(i.e. before `CellViTInferenceMemory.process_wsi()`).
"""
from collections import deque

import numpy as np

_applied = False


def _remove_overlap_shapely2(self, cleaned_edge_cells):
    from shapely import strtree
    from shapely.geometry import MultiPolygon, Polygon

    merged_cells = cleaned_edge_cells

    for iteration in range(20):
        poly_list = []
        uids = []
        for idx, cell_info in merged_cells.iterrows():
            poly = Polygon(cell_info["contour"])
            if not poly.is_valid:
                self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                fixed = poly.buffer(0)
                if isinstance(fixed, MultiPolygon):
                    if len(fixed.geoms) > 1:
                        poly_idx = int(np.argmax([p.area for p in fixed.geoms]))
                        poly = Polygon(fixed.geoms[poly_idx])
                    else:
                        poly = Polygon(fixed.geoms[0])
                else:
                    poly = Polygon(fixed)
            poly_list.append(poly)
            uids.append(idx)

        tree = strtree.STRtree(poly_list)

        merged_idx = deque()
        iterated_cells = set()
        overlaps = 0

        for query_i, query_poly in enumerate(poly_list):
            query_uid = uids[query_i]
            if query_uid in iterated_cells:
                continue

            intersected_indices = tree.query(query_poly)  # includes self-intersection
            if len(intersected_indices) > 1:
                submergers = []  # (polygon, uid) pairs overlapping with query
                for inter_i in intersected_indices:
                    inter_uid = uids[inter_i]
                    if inter_uid == query_uid or inter_uid in iterated_cells:
                        continue
                    inter_poly = poly_list[inter_i]
                    if (
                        query_poly.intersection(inter_poly).area / query_poly.area > 0.01
                        or query_poly.intersection(inter_poly).area / inter_poly.area > 0.01
                    ):
                        overlaps += 1
                        submergers.append((inter_poly, inter_uid))
                        iterated_cells.add(inter_uid)

                if len(submergers) == 0:
                    merged_idx.append(query_uid)
                else:  # merging strategy: keep the biggest overlapping cell
                    selected = max(submergers, key=lambda pair: pair[0].area)
                    merged_idx.append(selected[1])
            else:
                merged_idx.append(query_uid)
            iterated_cells.add(query_uid)

        self.logger.info(f"Iteration {iteration}: Found overlap of # cells: {overlaps}")
        if overlaps == 0:
            self.logger.info("Found all overlapping cells")
            break
        elif iteration == 20:
            self.logger.info(
                f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, "
                f"we stop iterations now. Please raise an issue in git or increase number of iterations."
            )
        merged_cells = cleaned_edge_cells.loc[cleaned_edge_cells.index.isin(merged_idx)].sort_index()

    return merged_cells.sort_index()


def apply() -> None:
    global _applied
    if _applied:
        return
    from cellvit.inference.overlap_cell_cleaner import OverlapCellCleaner

    OverlapCellCleaner._remove_overlap = _remove_overlap_shapely2
    _applied = True
