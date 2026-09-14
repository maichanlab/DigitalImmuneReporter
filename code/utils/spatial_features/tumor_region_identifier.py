import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import (
    binary_closing, label as nd_label, binary_dilation,
    binary_fill_holes, distance_transform_edt, binary_opening
)

logger = logging.getLogger(__name__)


class TumorRegionIdentifier:
    """
    Processes single-cell spatial data to identify core tumor (CT) and peritumoral (PT) regions
    based on tumor cell labels, spatial proximity, and morphological operations. Provides grid-based
    mapping of cells and visualization of tumor regions.
    """
    def __init__(
        self,
        cell_table: pd.DataFrame,
        tumor_cell_binary_col: str,

        grid_resolution: float = 150,
        peritumoral_margin_um: int = 300,
        k_neighbors: int = 25,
        tissue_gap_closing_um: float = None,
        tissue_density_radius_um: float = 50.0,
        tissue_min_neighbors: int = 3,
        min_margin_tissue_fraction: float = 3.0,
        verbose: bool = True,
    ):
        """
        Initialize the TumorRegionIdentifier object.

        Parameters
        ----------
        cell_table : pd.DataFrame
            DataFrame containing detected cell information, including
            - 'x', 'y': x, y coordinates in micrometer unit
            - tumor_cell_binary_col
        tumor_cell_binary_col : str
            Column name in cell_table indicating whether a cell is initially labeled as tumor (binary).
        grid_resolution : float, default 150
            Size of each grid cell in micrometers for spatial mapping.
        peritumoral_margin_um : int, default 300
            Width of the peritumoral (PT) region in micrometers.
        k_neighbors : int, default 25
            Number of neighbors used in k-NN tumor mask estimation.
        tissue_gap_closing_um : float, optional
            Real-distance size of the morphological closing used to turn "grid cells with
            >=1 detected cell" into a smooth tissue mask, bridging small gaps from locally
            sparse cell density (e.g. acellular stroma) that aren't actually background. The
            PT margin is only grown into grid cells this mask marks as tissue. Defaults to
            `peritumoral_margin_um` — gaps are bridged at the same spatial scale as the margin.
        tissue_density_radius_um : float, default 50.0
            Radius (um) used to count each cell's neighbors when deciding whether it reflects
            real tissue or is likely a spurious/false detection. Cells are excluded from the
            tissue mask before it is built from grid cell counts, so an isolated false-positive
            detection sitting in true background can't seed a "tissue" grid cell there.
        tissue_min_neighbors : int, default 3
            Minimum number of other cells a cell must have within `tissue_density_radius_um`
            to count toward the tissue mask. Cells below this are treated as noise, not tissue.
        min_margin_tissue_fraction : float, default 3.0
            Where the tissue-constrained margin buffer runs up against background before
            reaching the full margin distance, PT membership additionally requires the locally
            available tissue depth (tumor edge to background) to be at least
            `min_margin_tissue_fraction * peritumoral_margin_um`. Cells with less available
            tissue are folded into CT instead — a thin sliver of tissue pinched between the
            tumor and the tissue edge isn't a meaningful peritumoral zone, and labeling it PT
            would fragment the band into slivers hugging the tissue-background interface. 1.0
            requires the full margin depth; lower values are more lenient.
        verbose : bool, default True
            To print detailed steps
        """
        if {'x', 'y', tumor_cell_binary_col} - set(cell_table.columns):
            raise ValueError(f"cell_table must contain 'x', 'y', '{tumor_cell_binary_col}' columns.")

        self.cell_table = cell_table
        self.tumor_cell_binary_col = tumor_cell_binary_col

        # Computational parameters
        self.grid_resolution = grid_resolution
        self.peritumoral_margin_um = peritumoral_margin_um
        self.k_neighbors = k_neighbors
        self.tissue_gap_closing_um = (
            tissue_gap_closing_um if tissue_gap_closing_um is not None else peritumoral_margin_um
        )
        self.tissue_density_radius_um = tissue_density_radius_um
        self.tissue_min_neighbors = tissue_min_neighbors
        self.min_margin_tissue_fraction = min_margin_tissue_fraction

        # Cell coordinates system
        self.all_coords = self.cell_table[['x', 'y']].values
        self.x_coords = self.cell_table['x'].values
        self.y_coords = self.cell_table['y'].values
        self.x_min = self.x_coords.min()
        self.y_min = self.y_coords.min()

        # Grid system
        self.grid_x_coords = np.arange(self.x_coords.min(), self.x_coords.max(), self.grid_resolution)
        self.grid_y_coords = np.arange(self.y_coords.min(), self.y_coords.max(), self.grid_resolution)
        self.cell_grid_x_idx = ((self.x_coords - self.x_min) / self.grid_resolution).astype(int)
        self.cell_grid_y_idx = ((self.y_coords - self.y_min) / self.grid_resolution).astype(int)
        self.cell_within_grid_extent = (
            (self.cell_grid_x_idx >= 0) & (self.cell_grid_x_idx < len(self.grid_x_coords)) &
            (self.cell_grid_y_idx >= 0) & (self.cell_grid_y_idx < len(self.grid_y_coords))
        )
        self.valid_cell_grid_x_idx = self.cell_grid_x_idx[self.cell_within_grid_extent]
        self.valid_cell_grid_y_idx = self.cell_grid_y_idx[self.cell_within_grid_extent]
        # Placeholders
        self.grid_tissue = None                       # STEP 0
        self.grid_in_tumor = None                     # STEP 1
        self.grid_in_tumor_smoothed = None             # STEP 2
        self.grid_tumor_core = None                    # STEP 3
        self.grid_tumor_filled = None                  # STEP 4
        self.grid_tumor_expanded = None                # STEP 4 (core expanded outward by peritumoral_margin_um)
        self.grid_tumor_expanded_boundary = None       # STEP 4 (external boundary of the expanded core)
        self.grid_dist_from_filled_um = None           # STEP 4 (outward distance from the original tumor core)
        self.grid_margin_reclassified_to_ct = None     # STEP 5 (thin-tissue margin cells folded into CT)
        self.grid_pt = None                            # STEP 5
        self.grid_ct = None                            # STEP 5

    def map_grid_to_cells(self, grid_array: np.ndarray):
        """
        Map a boolean or numeric 2D grid array to cell-level binary mask.
        """
        cell_mask = np.zeros(len(self.cell_table), dtype=grid_array.dtype)
        cell_mask[self.cell_within_grid_extent] = grid_array[
            self.valid_cell_grid_y_idx, self.valid_cell_grid_x_idx
        ]
        return cell_mask

    def run(self):
        """
        Execute the full CT/PT detection pipeline.

        Steps executed:
        0. Tissue mask estimation (any detected cell, not just tumor)
        1. k-NN-based tumor mask estimation
        2. Morphological smoothing of tumor region
        3. Adaptive selection of tumor components
        4. Filling tumor core holes, expanding the core outward by the peritumoral margin
           (grown only into tissue, per STEP 0), and finding the expanded external boundary
        5. Definition of Core Tumor (CT) and Peritumoral (PT) regions via an inward-only
           distance transform from the expanded boundary

        Returns
        -------
        pd.DataFrame
            Updated cell_table with additional columns:
            - 'is_dense_cell', 'in_tumor', 'in_tumor_smoothed', 'in_tumor_core',
              'in_tumor_core_filled', 'in_CT', 'in_PT', 'distance_to_boundary'
        """
        logger.info("=" * 70)
        logger.info("EXTERNAL BOUNDARY CT/PT DETECTION")
        logger.info("=" * 70)

        self._step0_build_tissue_mask()
        self._step1_knn_tumor_mask()
        self._step2_morphological_smoothing()
        self._step3_adaptive_component_selection()
        self._step4_fill_tumor_holes_and_find_boundary()
        self._step5_define_ct_pt()

        logger.info("=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)

        return self.cell_table

    def _step0_build_tissue_mask(self):
        """
        Build a smoothed grid-level mask of where tissue actually is (any detected cell,
        tumor or not), as opposed to background/off-tissue area (e.g. glass slide, tissue
        tears, folded-away regions). Does not depend on tumor labels at all.

        This exists so STEP 4 can grow the peritumoral margin only into real tissue, not into
        background — otherwise a tumor sitting near the edge of the sampled/imaged area would
        get a margin band extending into empty space, which has no biological meaning.

        - Density filter: a cell only counts toward the tissue grid if it has at least
          `tissue_min_neighbors` other cells within `tissue_density_radius_um` (via `KDTree`),
          so an isolated false-positive detection in true background can't seed a "tissue"
          grid cell (the closing step below only fills gaps between already-tissue cells, it
          doesn't remove isolated speckles on its own).
        - Raw tissue presence: grid cells containing at least one density-filtered cell of
          any type.
        - Closed with a kernel sized by `tissue_gap_closing_um` to bridge small gaps from
          locally sparse cell density (e.g. acellular/hyalinized stroma) that aren't actually
          background, then hole-filled.
        """
        logger.info("-" * 70)
        logger.info("STEP 0: Tissue Mask (all cells, any type)")
        logger.info("-" * 70)

        tree = KDTree(self.all_coords)
        neighbor_counts = tree.query_ball_point(
            self.all_coords, r=self.tissue_density_radius_um, return_length=True
        ) - 1  # exclude the point itself
        is_dense_cell = neighbor_counts >= self.tissue_min_neighbors
        self.cell_table['is_dense_cell'] = is_dense_cell
        logger.info(
            f"Density filter: {is_dense_cell.sum():,}/{len(is_dense_cell):,} cells kept "
            f"({is_dense_cell.sum() / len(is_dense_cell) * 100:.2f}%), "
            f"{(~is_dense_cell).sum():,} treated as likely false-positive noise"
        )

        is_dense_and_valid = is_dense_cell[self.cell_within_grid_extent]
        grid_any_cell = np.zeros((len(self.grid_y_coords), len(self.grid_x_coords)), dtype=np.uint32)
        np.add.at(
            grid_any_cell,
            (self.valid_cell_grid_y_idx[is_dense_and_valid], self.valid_cell_grid_x_idx[is_dense_and_valid]),
            1,
        )
        grid_tissue_raw = grid_any_cell > 0

        closing_grids = max(1, int(np.ceil(self.tissue_gap_closing_um / self.grid_resolution)))
        k = 2 * closing_grids + 1
        grid_tissue_closed = binary_closing(grid_tissue_raw, structure=np.ones((k, k)))
        self.grid_tissue = binary_fill_holes(grid_tissue_closed)
        logger.info(
            f"Tissue mask: {grid_tissue_raw.sum()} -> {self.grid_tissue.sum()} grid cells "
            f"({k}x{k} closing, {self.tissue_gap_closing_um} um gap-bridging, + hole fill)"
        )

    def _step1_knn_tumor_mask(self):
        """
        Identify a coarse tumor mask using k-nearest neighbors (k-NN) based on initial tumor labels.
        Each cell is classified as tumor if more than 50% of its k nearest neighbors are labeled tumor.
        """
        logger.info("-" * 70)
        logger.info(f"STEP 1: k-NN Tumor Mask (k={self.k_neighbors})")
        logger.info("-" * 70)

        logger.info(f"Building k-NN model with k={self.k_neighbors}...")
        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            algorithm='ball_tree',
            n_jobs=-1
        ).fit(self.all_coords)

        all_indices = nbrs.kneighbors(self.all_coords, return_distance=False)

        self.cell_table['in_tumor'] = (
            self.cell_table[self.tumor_cell_binary_col].values[all_indices].mean(axis=1) > 0.5
        )
        logger.info(f"Tumor region: {self.cell_table['in_tumor'].sum():,} cells ({self.cell_table['in_tumor'].sum()/len(self.cell_table)*100:.2f}%)")

    def _step2_morphological_smoothing(self):
        """
        Apply morphological closing to smooth the coarse tumor mask, with an adaptive kernel
        size based on component dominance.
        """
        logger.info("-" * 70)
        logger.info("STEP 2: Morphological Smoothing")
        logger.info("-" * 70)

        self.grid_in_tumor = np.zeros((len(self.grid_y_coords), len(self.grid_x_coords)), dtype=np.uint8)
        in_tumor_cells = self.cell_table[self.cell_table['in_tumor']]
        in_tumor_cell_grid_x_idx = ((in_tumor_cells['x'] - self.x_min) / self.grid_resolution).astype(int)
        in_tumor_cell_grid_y_idx = ((in_tumor_cells['y'] - self.y_min) / self.grid_resolution).astype(int)
        valid = (in_tumor_cell_grid_x_idx >= 0) & (in_tumor_cell_grid_x_idx < len(self.grid_x_coords)) & (in_tumor_cell_grid_y_idx >= 0) & (in_tumor_cell_grid_y_idx < len(self.grid_y_coords))
        np.add.at(self.grid_in_tumor, (in_tumor_cell_grid_y_idx[valid], in_tumor_cell_grid_x_idx[valid]), 1)
        self.grid_in_tumor_binary = self.grid_in_tumor > 0

        labeled, num_components = nd_label(self.grid_in_tumor_binary)

        if num_components > 1:
            component_sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
            largest = max(component_sizes)
            second_largest = sorted(component_sizes, reverse=True)[1] if len(component_sizes) > 1 else 0
            dominance_ratio = largest / second_largest if second_largest > 0 else float('inf')

            if dominance_ratio > 3:
                k = 3
            elif dominance_ratio > 1.5:
                k = 4
            else:
                k = 5
            self.grid_in_tumor_smoothed = binary_closing(self.grid_in_tumor_binary, structure=np.ones((k, k)))
            logger.info(f"Found {num_components} components, dominance ratio: {dominance_ratio:.2f}")
            logger.info(f"Applied {k}x{k} smoothing")
        else:
            self.grid_in_tumor_smoothed = binary_closing(self.grid_in_tumor_binary, structure=np.ones((5, 5)))
            logger.info("Single component detected, applied 5x5 smoothing")

        self.cell_table['in_tumor_smoothed'] = self.map_grid_to_cells(self.grid_in_tumor_smoothed)
        logger.info(f"Smoothed tumor cells: {self.cell_table['in_tumor_smoothed'].sum():,}")
        logger.info(f"Grid resolution: {self.grid_resolution:.2f} um per grid cell")

    def _step3_adaptive_component_selection(self):
        """
        Select tumor core components based on size and dominance.
        """
        logger.info("-" * 70)
        logger.info("STEP 3: Adaptive Component Selection")
        logger.info("-" * 70)

        self.labeled_tumor_regions, self.num_regions = nd_label(self.grid_in_tumor_smoothed)
        logger.info(f"Found {self.num_regions} separate tumor regions")

        if self.num_regions > 1:
            component_sizes = [(self.labeled_tumor_regions == i).sum() for i in range(1, self.num_regions + 1)]
            component_sizes_sorted = sorted(component_sizes, reverse=True)

            largest = component_sizes_sorted[0]
            second_largest = component_sizes_sorted[1] if len(component_sizes_sorted) > 1 else 0
            total_tumor_mass = sum(component_sizes)
            largest_fraction = largest / total_tumor_mass

            if second_largest > 0:
                dominance = largest / second_largest
            else:
                dominance = float('inf')

            logger.info(f"  Dominance ratio: {dominance:.2f}")
            logger.info(f"  Largest fraction: {largest_fraction:.2f}")

            if dominance > 5.0 and largest_fraction > 0.75:
                self.mode = "single-component"
                largest_idx = np.argmax(component_sizes) + 1
                self.grid_tumor_core = (self.labeled_tumor_regions == largest_idx)
                logger.info("Single-component mode (filtering fragments)")
            else:
                self.mode = "multi-component"
                size_threshold = largest * 0.50
                tumor_adaptive = np.zeros_like(self.grid_in_tumor_smoothed, dtype=bool)
                kept = 0
                for i in range(1, self.num_regions + 1):
                    if component_sizes[i-1] >= size_threshold:
                        tumor_adaptive |= (self.labeled_tumor_regions == i)
                        kept += 1
                self.grid_tumor_core = tumor_adaptive
                logger.info(f"Multi-component mode (kept {kept}/{self.num_regions} components)")

            self.dominance_ratio = dominance
        else:
            self.mode = "single-component"
            self.grid_tumor_core = (self.labeled_tumor_regions == 1)

        self.cell_table['in_tumor_core'] = self.map_grid_to_cells(self.grid_tumor_core)

    def _find_external_boundary(self, grid_mask: np.ndarray) -> np.ndarray:
        """
        Find the 1-grid-cell-wide ring of `grid_mask` that borders the background region
        connected to the grid's outer edge (as opposed to the edge of a background hole fully
        enclosed by `grid_mask`, which there shouldn't be once holes have been filled).
        """
        grid_non_mask = ~grid_mask
        labeled_non_mask, _ = nd_label(grid_non_mask)
        edge_labels = set()
        edge_labels.update(np.unique(labeled_non_mask[0, :]))
        edge_labels.update(np.unique(labeled_non_mask[-1, :]))
        edge_labels.update(np.unique(labeled_non_mask[:, 0]))
        edge_labels.update(np.unique(labeled_non_mask[:, -1]))
        edge_labels.discard(0)

        external_non_mask = np.isin(labeled_non_mask, list(edge_labels))
        external_adjacent = binary_dilation(external_non_mask, structure=np.ones((3, 3)))
        return grid_mask & external_adjacent

    def _step4_fill_tumor_holes_and_find_boundary(self):
        """
        Fill internal holes in tumor core components, expand the core outward by the
        peritumoral margin, and identify the external boundary of the expanded region.

        - Fills all internal holes for each tumor component (`grid_tumor_filled`).
        - Expands `grid_tumor_filled` outward by `peritumoral_margin_um`, as a Euclidean
          distance buffer (not a fixed-size structuring element), intersected with
          `grid_tissue` (STEP 0) so the buffer only grows into real tissue, never into
          background — giving `grid_tumor_expanded`.
        - Finds the external boundary of the expanded region (`grid_tumor_expanded_boundary`)
          — diagnostic only; STEP 5 computes CT/PT from `grid_tumor_expanded` directly via a
          distance transform, not from this ring.
        """
        logger.info("-" * 70)
        logger.info("STEP 4: Hole Filling, Margin Expansion, and Boundary Detection")
        logger.info("-" * 70)

        labeled_tumor, num_comps = nd_label(self.grid_tumor_core)
        self.grid_tumor_filled = np.zeros_like(self.grid_tumor_core, dtype=bool)

        for comp_id in range(1, num_comps + 1):
            comp_mask = labeled_tumor == comp_id
            comp_filled = binary_fill_holes(comp_mask)
            self.grid_tumor_filled |= comp_filled

        self.cell_table['in_tumor_core_filled'] = self.map_grid_to_cells(self.grid_tumor_filled)
        logger.info(f"Tumor core filled: {self.grid_tumor_filled.sum()} grid cells")
        logger.info(f"Cells in tumor core: {self.cell_table['in_tumor_core_filled'].sum()} cells")
        if self.grid_tumor_filled.sum() == 0:
            logger.warning("Tumor core is empty after hole-filling — CT/PT assignment in step 5 will be empty.")

        # Expand the tumor core outward by the peritumoral margin, growing only into tissue
        # (STEP 0) so the margin never crosses into background.
        self.grid_dist_from_filled_um = distance_transform_edt(~self.grid_tumor_filled) * self.grid_resolution
        grid_margin_candidate_untissued = self.grid_dist_from_filled_um <= self.peritumoral_margin_um
        grid_margin_candidate = grid_margin_candidate_untissued & self.grid_tissue
        self.grid_tumor_expanded = self.grid_tumor_filled | grid_margin_candidate
        self.grid_tumor_expanded = binary_fill_holes(self.grid_tumor_expanded)
        logger.info(
            f"Tumor core expanded by {self.peritumoral_margin_um} um margin: "
            f"{self.grid_tumor_filled.sum()} -> {self.grid_tumor_expanded.sum()} grid cells "
            f"(+{self.grid_tumor_expanded.sum() - self.grid_tumor_filled.sum()})"
        )
        if self.grid_tumor_expanded.sum() == self.grid_tumor_filled.sum():
            if grid_margin_candidate_untissued.sum() > 0:
                logger.warning(
                    f"Margin expansion added 0 grid cells even though {grid_margin_candidate_untissued.sum()} "
                    f"grid cells were within {self.peritumoral_margin_um} um of the tumor core: none of them "
                    "are marked as tissue (STEP 0). Either the tumor is entirely surrounded by background, "
                    f"or tissue_gap_closing_um ({self.tissue_gap_closing_um} um) needs to be larger to bridge "
                    "real tissue currently treated as background. The PT band in STEP 5 will be empty."
                )
            else:
                logger.warning(
                    f"Margin expansion added 0 grid cells: grid_resolution ({self.grid_resolution:.2f} um) is "
                    f"coarser than peritumoral_margin_um ({self.peritumoral_margin_um} um), so no grid cell's "
                    "center falls within the margin of the tumor core. The PT band in STEP 5 will be empty. "
                    "Use a grid_resolution well below peritumoral_margin_um to resolve a PT band."
                )

        self.grid_tumor_expanded_boundary = self._find_external_boundary(self.grid_tumor_expanded)
        if self.grid_tumor_expanded_boundary.sum() == 0:
            logger.warning(
                "No expanded-region external boundary found — grid_tumor_expanded may be empty "
                "or fully touching the grid edge."
            )

    def _step5_define_ct_pt(self):
        """
        Define Core Tumor (CT) and Peritumoral (PT) regions.

        - Computes an inward-only distance transform from the external boundary of
          `grid_tumor_expanded` (STEP 4): for every grid cell inside that region, its distance
          to the nearest cell outside it.
        - Candidate margin band: within `peritumoral_margin_um` of the expanded boundary
          (measured inward), excluding the original tumor core, then closed (3x3) to clear
          jagged edges. This makes PT a purely outward band from the tumor's own edge out to
          the expanded boundary, rather than a band straddling the tumor edge on both sides.
          No binary_opening here (unlike the previous method) — this one-sided band is only
          about half as thick as the old two-sided band at the same margin, and an
          erosion-then-dilation pass would shatter it into disconnected fragments.
        - Thin-tissue reclassification: a candidate margin cell can sit within tissue while
          having too little of it left to support a full margin (the tumor is close to where
          the tissue itself runs out). The locally available tissue depth is estimated as the
          cell's outward distance from the tumor plus its distance to the nearest background
          cell; where that's below `min_margin_tissue_fraction * peritumoral_margin_um`, the
          cell is folded into CT instead of PT.
        - CT is everything in the expanded (tumor+margin) region that isn't PT.
        """
        logger.info("-" * 70)
        logger.info("STEP 5: Defining CT and PT Regions")
        logger.info("-" * 70)

        if self.grid_resolution > self.peritumoral_margin_um:
            logger.warning(
                f"grid_resolution ({self.grid_resolution:.2f} um) exceeds peritumoral_margin_um "
                f"({self.peritumoral_margin_um} um): distance-transform values are quantized to "
                "multiples of grid_resolution, so no grid cell can satisfy the PT threshold. "
                "in_PT will be empty (STEP 4 should already have logged that margin expansion "
                "added 0 grid cells)."
            )

        # Inward-only distance: for cells inside grid_tumor_expanded, distance to the nearest
        # cell outside it (0 outside the expanded region) — distance measured inward from the
        # expanded region's external boundary.
        grid_dist_inward_um = distance_transform_edt(self.grid_tumor_expanded) * self.grid_resolution
        self.grid_dist_um = grid_dist_inward_um

        grid_margin_band_raw = (
            self.grid_tumor_expanded
            & (grid_dist_inward_um <= self.peritumoral_margin_um)
            & (~self.grid_tumor_filled)
        )
        grid_margin_band = binary_closing(grid_margin_band_raw, structure=np.ones((3, 3)))

        # Thin-tissue reclassification. Pad grid_tissue with True before the background-distance
        # transform so the edge of the *sampled* field of view (possibly just where the analysis
        # was cropped, not a real tissue-background boundary) is never mistaken for background.
        pad = max(1, int(np.ceil(self.peritumoral_margin_um / self.grid_resolution))) + 1
        grid_tissue_padded = np.pad(self.grid_tissue, pad, mode='constant', constant_values=True)
        grid_dist_to_background_padded = distance_transform_edt(grid_tissue_padded) * self.grid_resolution
        grid_dist_to_background_um = grid_dist_to_background_padded[pad:-pad, pad:-pad]

        grid_tissue_strip_width_um = self.grid_dist_from_filled_um + grid_dist_to_background_um
        min_required_um = self.min_margin_tissue_fraction * self.peritumoral_margin_um
        grid_margin_reclassified_to_ct = grid_margin_band & (grid_tissue_strip_width_um < min_required_um)
        self.grid_margin_reclassified_to_ct = grid_margin_reclassified_to_ct
        logger.info(
            f"Margin band: {grid_margin_band.sum()} grid cells; {grid_margin_reclassified_to_ct.sum()} "
            f"reclassified to CT (available tissue < {min_required_um:.2f} um, "
            f"{self.min_margin_tissue_fraction * 100:.0f}% of peritumoral_margin_um), "
            f"{grid_margin_band.sum() - grid_margin_reclassified_to_ct.sum()} remain candidate PT"
        )

        grid_pt = grid_margin_band & (~grid_margin_reclassified_to_ct)
        if grid_pt.sum() == 0:
            logger.warning(
                "grid_pt is empty — no PT cells will be found. This happens when the margin "
                "expansion in STEP 4 added 0 grid cells, or when every candidate margin cell "
                "was reclassified to CT for having too little tissue depth."
            )

        self.grid_pt = grid_pt
        self.grid_ct = self.grid_tumor_expanded | grid_margin_reclassified_to_ct

        self.cell_table['in_PT'] = self.map_grid_to_cells(grid_pt)
        self.cell_table['in_CT'] = self.map_grid_to_cells(self.grid_ct) & (~self.cell_table['in_PT'])
        self.cell_table['distance_to_boundary'] = self.map_grid_to_cells(grid_dist_inward_um)

        self.ct_count = self.cell_table['in_CT'].sum()
        self.pt_count = self.cell_table['in_PT'].sum()

        logger.info("Region assignments:")
        logger.info(f"  CT: {self.ct_count:,} cells")
        logger.info(f"  PT: {self.pt_count:,} cells")

    def plot_ct_pt_regions(self, figsize=(24, 16), dpi=100, save_path=None):
        """
        Plot spatial distribution of tumor regions: all cells, hole-filled tumor core, CT, PT.
        """
        if self.cell_table is None or 'in_CT' not in self.cell_table.columns:
            logger.error("Must call run() first before plotting")
            return None, None

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()

        axes[0].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightblue', s=0.3, alpha=0.5, edgecolors='none')
        axes[0].set_title(f'(All Cells)\n{len(self.cell_table):,} cells', fontsize=14, fontweight='bold')

        tumor_mask = self.cell_table[self.cell_table['in_tumor_smoothed']]
        axes[1].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[1].scatter(tumor_mask['x'],
                        tumor_mask['y'],
                        c='green', s=0.5, alpha=0.7)
        axes[1].set_title(f'Tumor Region\n{len(tumor_mask):,} cells',
                          fontsize=14, fontweight='bold')

        tumor_for_boundary_cells = self.cell_table[self.cell_table['in_tumor_core_filled']]
        axes[2].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[2].scatter(tumor_for_boundary_cells['x'],
                        tumor_for_boundary_cells['y'],
                        c='orange', s=0.5, alpha=0.7)
        axes[2].set_title(f'Main Tumor Component(s)\n{len(tumor_for_boundary_cells):,} cells',
                          fontsize=14, fontweight='bold')

        ct_cells = self.cell_table[self.cell_table['in_CT']]
        axes[3].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[3].scatter(ct_cells['x'], ct_cells['y'],
                        c='red', s=0.5, alpha=0.7)
        axes[3].set_title(f'Core Tumor (CT)\n{len(ct_cells):,} cells',
                          fontsize=14, fontweight='bold')

        pt_cells = self.cell_table[self.cell_table['in_PT']]
        axes[4].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[4].scatter(pt_cells['x'], pt_cells['y'],
                        c='blue', s=0.5, alpha=0.7)
        axes[4].set_title(f'Peritumoral (PT, +/-{self.peritumoral_margin_um:.0f}um)\n{len(pt_cells):,} cells',
                          fontsize=14, fontweight='bold')

        for ax in axes:
            ax.set_xlabel('X coordinate (um)')
            ax.set_ylabel('Y coordinate (um)')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_facecolor('white')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
            plt.close(fig)

        return fig, axes
