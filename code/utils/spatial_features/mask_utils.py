import warnings

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw


def _hex_to_rgb(color):
    """Convert hex color to RGB tuple."""
    if isinstance(color, str):
        color = color.lstrip("#")
        if len(color) != 6:
            raise ValueError(f"Invalid hex color: {color}")
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    return color


def convert_label_mask_to_rgb(
    mask: np.ndarray,
    label_colors: dict,
    ignore_unseen_labels: bool = False
) -> np.ndarray:
    """
    Converts a single-channel integer-labeled mask to a 3-channel RGB color mask.
    Supports both RGB tuples and HEX color codes.
    """

    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Normalize colors (convert HEX -> RGB if needed)
    normalized_colors = {}
    for label, color in label_colors.items():
        color = _hex_to_rgb(color)

        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"Invalid color value for label {label}: must be RGB tuple or HEX string.")

        normalized_colors[label] = tuple(color)

    if ignore_unseen_labels:
        warnings.warn("Labels not defined in `label_colors` will be ignored")
        unique_labels = list(normalized_colors.keys())
    else:
        unique_labels = np.unique(mask)
        unseen_labels = list(set(unique_labels) - set(normalized_colors.keys()))
        if unseen_labels:
            raise ValueError(
                f"The following labels in the mask are not defined in `label_colors`: {unseen_labels}"
            )

    for label in unique_labels:
        rgb_mask[mask == label] = normalized_colors[label]

    return rgb_mask


def convert_geojson_to_mask(geojson_file_path: str, width: int, height: int) -> np.ndarray:
    """
    Converts a GeoJSON annotation file into a binary mask.

    Reads polygon geometries (e.g., tissue/background contours) from a GeoJSON file and
    rasterizes them into a binary mask of the specified image dimensions. Pixels inside
    any polygon are assigned a value of 1.

    Args:
        geojson_file_path: Path to the GeoJSON file containing polygon or multipolygon geometries.
        width: Width of the target mask (in pixels).
        height: Height of the target mask (in pixels).

    Returns:
        np.ndarray: A 2D array of shape (height, width) with 1 where a polygon is present, 0 elsewhere.
    """
    gdf = gpd.read_file(geojson_file_path)

    mask = Image.new("1", (width, height), 0)

    for geom in gdf.geometry:
        if geom.is_valid:
            draw = ImageDraw.Draw(mask)

            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                draw.polygon(coords, outline=1, fill=1)
            elif geom.geom_type == "MultiPolygon":
                for polygon in geom.geoms:
                    coords = list(polygon.exterior.coords)
                    draw.polygon(coords, outline=1, fill=1)

    return np.array(mask, dtype=np.uint8)
