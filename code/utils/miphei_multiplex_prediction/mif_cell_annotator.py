"""
"Glue" logic that turns CellViT-plus-plus's binary cell detections + MIPHEI-ViT's predicted
multiplex (mIF) channels into hierarchy-based cell types.

Ported/adapted from AI4HE-Spatial's `experiments/crc_benchmark_miphei/utils.py` (per-cell mean
marker intensity extraction) and `sp_annotator/tools/CellTypeAnotator.py` (hierarchy resolution),
so this module has no runtime dependency on the AI4HE-Spatial repo itself (numpy/pandas/torch/
opencv/tifffile only) — matching the same "ported, not imported" convention already used by
`spatial_feature_computer.py`, `cell_density.py`, and `gcross.py` elsewhere in this codebase.
"""
import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch

logger = logging.getLogger(__name__)


def convert_contours_to_labeled_mask(
    contours: List[np.ndarray],
    width: int,
    height: int,
    start_label: int = 1,
) -> np.ndarray:
    """
    Create a labeled mask where each contour is filled with a unique integer (its 1-based
    index in `contours`, i.e. its CellID). Background pixels are 0.
    """
    mask = np.zeros((height, width), dtype=np.int32)
    for i, contour in enumerate(contours, start=start_label):
        contour_formatted = np.asarray(contour).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [contour_formatted], color=i)
    return mask


def _read_marker_channel_region(tiff_file: tifffile.TiffFile, channel_index: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Read a (h, w) region of one channel from the MIPHEI-ViT OME-TIFF (one channel per page)."""
    return tiff_file.pages[channel_index].asarray()[y:y + h, x:x + w]


def extract_mean_marker_intensities(
    ome_tiff_path: str,
    labeled_cell_mask: np.ndarray,
    marker_channels: List[str],
    chunk_size: int = 40000,
    overlap: int = 200,
) -> pd.DataFrame:
    """
    Per-cell mean predicted intensity (0-255 uint8 scale) for every channel in `marker_channels`,
    read from the MIPHEI-ViT OME-TIFF in overlapping chunks to stay memory-safe on WSI-scale
    masks (one channel = one TIFF page, in `marker_channels` order).

    A cell whose contour straddles a chunk boundary is measured in every chunk it appears in;
    the reading with the largest pixel coverage (closest to the whole cell) is kept.

    Returns:
        pd.DataFrame indexed by CellID (1-based, matching `labeled_cell_mask` labels), one
        column per marker in `marker_channels`.
    """
    height, width = labeled_cell_mask.shape
    n_cells = int(labeled_cell_mask.max())
    n_channels = len(marker_channels)
    stride = chunk_size - overlap

    means = np.zeros((n_cells, n_channels), dtype=np.float64)
    best_pixel_coverage = np.zeros(n_cells, dtype=np.int64)

    with tifffile.TiffFile(ome_tiff_path) as tiff_file:
        for y in range(0, height, stride):
            y_end = min(y + chunk_size, height)
            for x in range(0, width, stride):
                x_end = min(x + chunk_size, width)

                mask_chunk = labeled_cell_mask[y:y_end, x:x_end]
                mask_flat = mask_chunk.ravel()
                valid = mask_flat > 0
                if not valid.any():
                    continue

                logger.info(f"Extracting marker intensities for chunk ({y}:{y_end}, {x}:{x_end})...")
                chunk_w, chunk_h = x_end - x, y_end - y
                channel_chunks = np.stack([
                    _read_marker_channel_region(tiff_file, c, x, y, chunk_w, chunk_h)
                    for c in range(n_channels)
                ])  # (n_channels, chunk_h, chunk_w)

                # Vectorized per-cell aggregation (one bincount pass per channel over the whole
                # chunk) instead of looping per cell — the latter is O(n_cells * n_pixels) and
                # becomes impractically slow (hours) once a chunk holds thousands of cells.
                labels_valid = mask_flat[valid]
                pixel_counts = np.bincount(labels_valid, minlength=n_cells + 1).astype(np.int64)  # index 0..n_cells

                # Only cells whose coverage in *this* chunk beats their best-so-far get updated
                # (handles cells split across overlapping chunk boundaries).
                update_mask = pixel_counts[1:] > best_pixel_coverage
                if not update_mask.any():
                    continue

                for c in range(n_channels):
                    channel_flat = channel_chunks[c].ravel()[valid].astype(np.float64)
                    sums = np.bincount(labels_valid, weights=channel_flat, minlength=n_cells + 1)
                    chunk_means = np.zeros(n_cells + 1, dtype=np.float64)
                    nonzero = pixel_counts > 0
                    chunk_means[nonzero] = sums[nonzero] / pixel_counts[nonzero]
                    means[update_mask, c] = chunk_means[1:][update_mask]
                best_pixel_coverage[update_mask] = pixel_counts[1:][update_mask]

    cell_means = pd.DataFrame(means, columns=marker_channels)
    cell_means.index = pd.RangeIndex(start=1, stop=n_cells + 1, name="CellID")
    return cell_means


def binarize_markers(
    cell_means: pd.DataFrame,
    calibration_classifier_path: str,
    marker_channels: List[str],
    nucleus_marker: str,
) -> pd.DataFrame:
    """
    Binarize per-cell mean marker intensities into positive/negative calls via the calibration
    classifier (a `torch.nn.Linear(len(marker_channels) - 1, len(marker_channels) - 1)` trained
    on real-vs-MIPHEI-predicted mIF pairs to correct MIPHEI-ViT's imperfect virtual staining).
    The nucleus marker (e.g. Hoechst) is excluded from the classifier and instead set to 1 for
    every row, since CellViT already restricted detections to nuclei.

    `marker_channels` must be in the exact order the classifier was trained on (the checkpoint's
    `config.yaml: data.targ_channel_names`); `cell_means` must have one column per entry.

    Returns:
        pd.DataFrame indexed like `cell_means`, with one `"{marker}_binarized"` column (0/1)
        per entry in `marker_channels`.
    """
    non_nucleus_markers = [m for m in marker_channels if m != nucleus_marker]

    classifier = torch.nn.Linear(len(non_nucleus_markers), len(non_nucleus_markers))
    classifier.load_state_dict(torch.load(calibration_classifier_path, map_location="cpu"))
    classifier.eval()

    inputs = torch.from_numpy(cell_means[non_nucleus_markers].to_numpy()).float()
    with torch.no_grad():
        predictions = (torch.sigmoid(classifier(inputs)) > 0.5).numpy()

    binarized = pd.DataFrame(
        predictions.astype(int),
        columns=[f"{m}_binarized" for m in non_nucleus_markers],
        index=cell_means.index,
    )
    binarized[f"{nucleus_marker}_binarized"] = 1
    return binarized


def hierarchy_level_categories(hierarchy: list) -> Dict[str, List[str]]:
    """
    Derive `{"level_1": [names of hierarchy nodes at depth 1], "level_2": [...], ...}` from a
    hierarchy definition, for use as `SpatialFeatureComputer`'s `level_categories`/
    `immune_level_categories`. Each level only lists names that are genuinely new at that
    depth (i.e. real "name" fields of hierarchy nodes) — not the backfilled parent labels that
    `annotate_cell_hierarchy` fills in for cells with no matching subtype — so density/G-cross/
    CT-PT features computed per level don't double-count the same cells under two names.
    """
    level_categories: Dict[str, List[str]] = {}

    def walk(nodes: list, depth: int):
        if not nodes:
            return
        level_key = f"level_{depth}"
        level_categories.setdefault(level_key, [])
        for node in nodes:
            level_categories[level_key].append(node["name"])
            if "subtypes" in node:
                walk(node["subtypes"], depth + 1)

    walk(hierarchy, 1)
    return level_categories


def validate_cell_hierarchy(hierarchy: list) -> bool:
    """
    Validate a cell-type hierarchy list. Ported verbatim from
    `sp_annotator/tools/CellTypeAnotator.py::validate_cell_hierarchy`.

    Rules: hierarchy is a list of dicts, each with "name" (str), "markers" (non-empty list of
    marker strings ending in '+'/'-', or AND-groups of such strings), and optional "subtypes"
    (recursively validated).
    """
    def check_node(node, path):
        if not isinstance(node, dict):
            raise ValueError(f"{path}: must be a dict, got {type(node).__name__}")
        if "name" not in node or not isinstance(node["name"], str):
            raise ValueError(f"{path}: missing or invalid 'name' (must be str)")
        if "markers" not in node:
            raise ValueError(f"{path}: missing 'markers' key")
        if not isinstance(node["markers"], list):
            raise ValueError(f"{path}: 'markers' must be a list")
        if len(node["markers"]) == 0:
            raise ValueError(f"{path}: 'markers' list cannot be empty")
        for m in node["markers"]:
            if not isinstance(m, str) and not isinstance(m, list):
                raise ValueError(f"{path}: marker {m!r} is not a string or a list")
            if isinstance(m, str) and not (m.endswith('+') or m.endswith('-')):
                raise ValueError(f"{path}: marker {m!r} must end with '+' or '-'")
            if isinstance(m, list):
                for n in m:
                    if not (n.endswith('+') or n.endswith('-')):
                        raise ValueError(f"{path}: marker {n!r} must end with '+' or '-'")
        if "subtypes" in node:
            if not isinstance(node["subtypes"], list):
                raise ValueError(f"{path}: 'subtypes' must be a list if present")
            for idx, child in enumerate(node["subtypes"]):
                check_node(child, path + f" -> {node['name']}[{idx}]")

    if not isinstance(hierarchy, list):
        raise ValueError("Top-level hierarchy must be a list of dicts")
    for i, top in enumerate(hierarchy):
        check_node(top, f"hierarchy[{i}]")
    return True


def annotate_cell_hierarchy(df: pd.DataFrame, hierarchy: list, nuclear_marker: Optional[str] = None) -> pd.DataFrame:
    """
    Annotate each cell with a hierarchical cell type. Ported from
    `sp_annotator/tools/CellTypeAnotator.py::CellTypeAnotator.annotate_cell_type`.

    Args:
        df: rows are cells, columns are "<Marker>_binarized" 0/1 values.
        hierarchy: nested list of {"name", "markers", "subtypes"} dicts (see
            `validate_cell_hierarchy`). Marker logic: `["A+","B+"]` -> OR,
            `[["A+","B+"]]` -> AND, `[["A+","B+"],"C+"]` -> (A AND B) OR C.
        nuclear_marker: if given, cells with `"{nuclear_marker}_binarized" != 1` are dropped
            before annotation (not applicable here since CellViT already restricts detections
            to nuclei and `binarize_markers` always sets this to 1, but kept for parity).

    Returns:
        `df` (row-filtered if `nuclear_marker` given) concatenated with per-cell `level_1`,
        `level_2`, ..., `ambiguity_reason`, and `full_path` columns.
    """
    validate_cell_hierarchy(hierarchy)

    if nuclear_marker is not None:
        nuclear_marker_col = f"{nuclear_marker}_binarized"
        if nuclear_marker_col not in df.columns:
            raise KeyError(f"Column '{nuclear_marker_col}' not found in DataFrame")
        df = df[df[nuclear_marker_col] == 1]

    def marker_to_col(marker: str) -> str:
        return f"{marker.rstrip('+-')}_binarized"

    def marker_condition_mask(df, marker: str) -> pd.Series:
        """Boolean mask for one marker string's own condition: `"<Marker>+"` -> binarized
        column == 1, `"<Marker>-"` -> binarized column == 0. `validate_cell_hierarchy` already
        enforces every marker ends in '+'/'-'."""
        is_positive = df[marker_to_col(marker)] == 1
        return is_positive if marker.endswith("+") else ~is_positive

    def get_hierarchy_depth(nodes: List[Dict], depth=1) -> int:
        max_depth = depth
        for node in nodes:
            if "subtypes" in node:
                max_depth = max(max_depth, get_hierarchy_depth(node["subtypes"], depth + 1))
        return max_depth

    max_depth = get_hierarchy_depth(hierarchy)
    logger.info(f"Cell-type hierarchy maximum depth: {max_depth}")

    labels = pd.DataFrame({f"level_{i + 1}": ["Other"] * len(df) for i in range(max_depth)}, index=df.index)
    labels["ambiguity_reason"] = ""

    def evaluate_marker_logic(df, markers):
        group_masks = []
        for m in markers:
            if isinstance(m, list):
                mask = pd.concat([marker_condition_mask(df, x) for x in m], axis=1).all(axis=1)
            else:
                mask = marker_condition_mask(df, m)
            group_masks.append(mask)
        if not group_masks:
            return pd.Series(False, index=df.index)
        return pd.concat(group_masks, axis=1).any(axis=1)

    def eval_level(nodes, parent_mask=None, prefix="", depth=1):
        level_masks = {}
        presence_masks = {}
        program_marker_cols = {}

        for node in nodes:
            name = node["name"]
            markers = node["markers"]

            logic_mask = evaluate_marker_logic(df, markers)
            if depth > 1 and parent_mask is not None:
                logic_mask &= parent_mask

            flat_markers = []
            for m in markers:
                if isinstance(m, list):
                    flat_markers.extend(m)
                else:
                    flat_markers.append(m)
            presence_cols = [marker_to_col(x) for x in flat_markers]
            # Sign-aware, like logic_mask above - a "-" marker's own condition being met (i.e.
            # the underlying channel reading NEGATIVE) is what counts as this node showing
            # evidence, not raw column positivity (which would make e.g. CD163+ M2 and CD163-
            # M1 look simultaneously "present" for every cell, since it's the same raw column).
            presence_mask = pd.concat([marker_condition_mask(df, x) for x in flat_markers], axis=1).any(axis=1)

            full_name = name if prefix == "" else f"{prefix}::{name}"
            program_marker_cols[full_name] = presence_cols
            level_masks[full_name] = logic_mask
            presence_masks[full_name] = presence_mask

        if not level_masks:
            return

        ambiguous = pd.Series(False, index=df.index)
        conflict_reason = pd.Series("", index=df.index, dtype="object")

        if len(nodes) > 1:
            presence_df = pd.DataFrame(presence_masks)
            active_counts = presence_df.sum(axis=1)
            ambiguous = active_counts > 1

            for idx in ambiguous[ambiguous].index:
                active_programs = presence_df.columns[presence_df.loc[idx]].tolist()
                if len(active_programs) <= 1:
                    continue
                program_strings = []
                for prog in active_programs:
                    active_markers = [
                        col.replace("_binarized", "")
                        for col in program_marker_cols[prog]
                        if df.loc[idx, col] == 1
                    ]
                    prog_name = prog.split("::")[-1]
                    program_strings.append(f"{prog_name}({','.join(active_markers)})" if active_markers else prog_name)
                conflict_reason.loc[idx] = " vs ".join(program_strings)

        labels.loc[ambiguous, "ambiguity_reason"] = conflict_reason.loc[ambiguous]

        # "Ambiguous" can only be assigned at the top level.
        if depth == 1:
            labels.loc[ambiguous, f"level_{depth}"] = "Ambiguous"

        for full_name, mask in level_masks.items():
            assignable = mask & (~ambiguous)
            labels.loc[assignable, f"level_{depth}"] = full_name.split("::")[-1]

        # Recurse into subtypes only for non-ambiguous cells matching their parent.
        if depth < max_depth:
            for node in nodes:
                if "subtypes" not in node:
                    continue
                parent_name = node["name"] if prefix == "" else f"{prefix}::{node['name']}"
                child_mask = level_masks[parent_name] & (~ambiguous)
                if child_mask.any():
                    eval_level(node["subtypes"], parent_mask=child_mask, prefix=parent_name, depth=depth + 1)

    eval_level(hierarchy)

    # Propagate labels downward: a deeper level that's "Other" (no subtype matched) or whose
    # parent was "Ambiguous" inherits the parent's label instead.
    for i in range(1, max_depth):
        parent_col, child_col = f"level_{i}", f"level_{i + 1}"
        labels[child_col] = labels[child_col].mask(
            (labels[child_col] == "Other") | (labels[parent_col] == "Ambiguous"),
            labels[parent_col],
        )

    def build_full_path(row):
        parts = []
        for i in range(max_depth):
            lv = row[f"level_{i + 1}"]
            if i == 0:
                parts.append(lv)
            elif lv != parts[-1]:
                parts.append(lv)
            else:
                break
        return "::".join(parts)

    labels["full_path"] = labels.apply(build_full_path, axis=1)

    return pd.concat([df.copy(), labels], axis=1)
