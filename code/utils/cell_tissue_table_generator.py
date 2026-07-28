from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd


def generate_tissue_area_table(
    tissue_mask: np.ndarray,
    tissue_id2label: Dict[int, str],
    mpp: float
) -> pd.DataFrame:
    """
    Generate a table containing the area (in square micrometers) of each tissue type.

    Args:
        tissue_mask (np.ndarray): 2D array of the same size as the image, where each pixel contains
                                  an integer tissue type ID.
        tissue_id2label (Dict[int, str]): Mapping from tissue type IDs to human-readable labels.
        mpp (float): Microns per pixel (image resolution), used to convert area from pixel count
                     to square micrometers.

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - 'tissue_label': Corresponding label.
                      - 'pixel_count': Number of pixels assigned to the tissue.
                      - 'area_sq_microns': Area in square micrometers.
    """
    unique_ids, counts = np.unique(tissue_mask, return_counts=True)

    records = []
    for tissue_id, count in zip(unique_ids, counts):
        label = tissue_id2label.get(tissue_id, f"Unknown({tissue_id})")
        area = (count * mpp * mpp)  # pixel area = mpp^2
        records.append({
            "tissue_label": label,
            "pixel_count": count,
            "area_sq_microns": area
        })

    return pd.DataFrame(records)


def generate_cell_table(
    cells: List[Dict[str, Any]],
    tissue_mask: np.ndarray,
    tissue_id2label: Dict[int, str],
    cell_id2label: Dict[int, str],
    mpp: float,
) -> pd.DataFrame:
    """
    Generate a table associating each cell with the tissue type that contains it.

    Args:
        cells: List of dictionaries containing cell information. Each dictionary must contain:
            - 'centroid': (x, y)
            - 'contour': (N, 2) array
            - 'label' or 'type'
            - optional 'score' or 'type_prob'
        tissue_mask: 2D array (H, W) with tissue type IDs.
        tissue_id2label: Mapping from tissue ID to label.
        cell_id2label: Mapping from cell type ID to label.
        mpp: Microns per pixel.

    Returns:
        pd.DataFrame with cell and tissue annotations.
    """

    height, width = tissue_mask.shape
    records = []

    for cell in cells:
        centroid = cell["centroid"]
        contour = cell["contour"]

        cell_label = cell.get("label", cell.get("type"))
        cell_score = cell.get("score", cell.get("type_prob"))

        contour = np.round(np.array(contour)).astype(int)
        x_coords = contour[..., 0]
        y_coords = contour[..., 1]

        xmin = np.min(x_coords)
        xmax = np.max(x_coords)
        ymin = np.min(y_coords)
        ymax = np.max(y_coords)

        if xmax <= xmin or ymax <= ymin or xmax > width or ymax > height:
            continue

        # Binary cell mask
        cell_mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)

        contour_offset = contour.copy()
        contour_offset[..., 0] -= xmin
        contour_offset[..., 1] -= ymin

        cv2.drawContours(
            cell_mask,
            [contour_offset.reshape(-1, 1, 2)],
            contourIdx=-1,
            color=1,
            thickness=-1,
        )

        try:
            overlapping_tissues = tissue_mask[ymin:ymax, xmin:xmax][cell_mask.astype(bool)]
        except IndexError:
            continue

        if overlapping_tissues.size == 0:
            tissue_id = -1
            tissue_label = "Unknown"
        else:
            tissue_id = int(np.bincount(overlapping_tissues).argmax())
            tissue_label = tissue_id2label.get(tissue_id, f"Unknown({tissue_id})")

        cell_label_str = cell_id2label.get(cell_label, f"Unknown({cell_label})")

        x_pixel = centroid[0]
        y_pixel = centroid[1]

        records.append(
            {
                "x": x_pixel * mpp,
                "y": y_pixel * mpp,
                "cell_label": cell_label_str,
                "tissue_label": tissue_label,
                "x_pixel": x_pixel,
                "y_pixel": y_pixel,
                "score": cell_score,
            }
        )

    return pd.DataFrame(records)
