from pathlib import Path

import cv2
import numpy as np
from openslide import OpenSlide
from shapely.geometry import Polygon


class WSIPatcher:
    """Tile-based iterator over a whole-slide image.

    Optionally restricted to tissue regions defined by a Trident contours
    GeoJSON, so tiles that fall outside tissue are skipped entirely.
    """

    def __init__(self, wsi_path, tile_size, overlap=0, contours_geojson_path=None):
        self.wsi_path = wsi_path
        self.tile_size = tile_size
        self.stride = tile_size - overlap
        self.wsi_slide = OpenSlide(wsi_path)
        self.width, self.height = self.wsi_slide.dimensions
        self.tile_coords = [
            (x, y)
            for x in range(0, self.width, self.stride)
            for y in range(0, self.height, self.stride)
        ]
        self.number_of_tiles_per_column = len(range(0, self.height, self.stride))
        print("Number of tiles: ", len(self.tile_coords))

        if not contours_geojson_path:
            return

        try:
            import geopandas as gpd

            gdf = gpd.read_file(contours_geojson_path)
            all_tissue_contours = []
            for geom in gdf.geometry:
                if not geom.is_valid:
                    continue
                if geom.geom_type == "Polygon":
                    all_tissue_contours.append(list(geom.exterior.coords))
                elif geom.geom_type == "MultiPolygon":
                    for polygon in geom.geoms:
                        all_tissue_contours.append(list(polygon.exterior.coords))

            all_tissue_contours = sorted(all_tissue_contours, key=len, reverse=True)
            self.tile_coords = self.filter_tiles_by_contours(
                self.tile_coords,
                tile_size,
                all_tissue_contours,
            )
            print("Number of tiles after ignoring non-tissue regions: ", len(self.tile_coords))
        except Exception as e:
            print(f"Error filtering tiles: {e}")

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        x1, y1 = self.tile_coords[idx]
        y2 = min(y1 + self.tile_size, self.height)
        x2 = min(x1 + self.tile_size, self.width)

        tile = np.array(self.wsi_slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1)))[:, :, :3]
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

        pad_bottom = self.tile_size - (y2 - y1)
        pad_right = self.tile_size - (x2 - x1)
        tile = cv2.copyMakeBorder(
            tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0,
        )

        return tile, np.array([x1, y1, x2, y2, pad_right, pad_bottom])

    def filter_tiles_by_contours(self, tile_coords, tile_size, contours, min_overlap=0.01):
        valid_tiles = []
        tile_area = tile_size * tile_size
        contour_polys = [Polygon(cnt) for cnt in contours if len(cnt) >= 3]

        for (x, y) in tile_coords:
            tile_poly = Polygon(
                [(x, y), (x + tile_size, y), (x + tile_size, y + tile_size), (x, y + tile_size)]
            )
            for poly in contour_polys:
                if tile_poly.intersects(poly):
                    overlap = tile_poly.intersection(poly).area / tile_area
                    if overlap >= min_overlap:
                        valid_tiles.append((x, y))
                        break

        return valid_tiles


__all__ = ["WSIPatcher"]
