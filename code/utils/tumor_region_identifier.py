import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

        grid_resolution: float = 32.5,
        peritumoral_margin_um: int = 100,
        k_neighbors: int = 25,
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
        grid_resolution : float, default 32.5
            Size of each grid cell in micrometers for spatial mapping.
        peritumoral_margin_um : int, default 100
            Width of the peritumoral (PT) region in micrometers.
        k_neighbors : int, default 25
            Number of neighbors used in k-NN tumor mask estimation.
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
        self.grid_in_tumor = None                   # STEP 1
        self.grid_in_tumor_smoothed = None          # STEP 2
        self.grid_tumor_core = None                 # STEP 3
        self.grid_tumor_filled = None               # STEP 4
        self.grid_tumor_external_boundary = None    # STEP 4
        self.grid_pt = None                          # STEP 5

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
        1. k-NN-based tumor mask estimation
        2. Morphological smoothing of tumor region
        3. Adaptive selection of tumor components
        4. Filling tumor core holes and finding external boundary
        5. Definition of Core Tumor (CT) and Peritumoral (PT) regions

        Returns
        -------
        pd.DataFrame
            Updated cell_table with additional columns:
            - 'in_tumor', 'in_tumor_smoothed', 'in_tumor_core', 'in_tumor_core_filled',
              'in_tumor_core_boundary', 'in_CT', 'in_PT', 'distance_to_boundary'
        """
        logger.info("=" * 70)
        logger.info("EXTERNAL BOUNDARY CT/PT DETECTION")
        logger.info("=" * 70)

        self._step1_knn_tumor_mask()
        self._step2_morphological_smoothing()
        self._step3_adaptive_component_selection()
        self._step4_fill_tumor_holes_and_find_boundary()
        self._step5_define_ct_pt()

        logger.info("=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)

        return self.cell_table

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

    def _step4_fill_tumor_holes_and_find_boundary(self):
        """
        Fill internal holes in tumor core components and identify the external tumor boundary.
        """
        logger.info("-" * 70)
        logger.info("STEP 4: Connectivity-based Boundary Detection")
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

        grid_non_tumor = ~self.grid_tumor_filled
        labeled_non_tumor, num_non_tumor = nd_label(grid_non_tumor)
        edge_labels = set()
        edge_labels.update(np.unique(labeled_non_tumor[0, :]))
        edge_labels.update(np.unique(labeled_non_tumor[-1, :]))
        edge_labels.update(np.unique(labeled_non_tumor[:, 0]))
        edge_labels.update(np.unique(labeled_non_tumor[:, -1]))
        edge_labels.discard(0)

        external_non_tumor = np.isin(labeled_non_tumor, list(edge_labels))
        external_adjacent = binary_dilation(external_non_tumor, structure=np.ones((3, 3)))
        self.grid_tumor_external_boundary = self.grid_tumor_filled & external_adjacent
        self.cell_table['in_tumor_core_boundary'] = self.map_grid_to_cells(self.grid_tumor_external_boundary)

    def _step5_define_ct_pt(self):
        """
        Define Core Tumor (CT) and Peritumoral (PT) regions from the distance transform of
        the tumor external boundary.
        """
        logger.info("-" * 70)
        logger.info("STEP 5: Defining CT and PT Regions")
        logger.info("-" * 70)

        grid_dist = distance_transform_edt(~(self.grid_tumor_external_boundary))
        grid_dist_um = grid_dist * self.grid_resolution

        pt_margin_grids = max(1, int(np.ceil(self.peritumoral_margin_um / self.grid_resolution)))
        pt_structuring_element = np.ones(
            (2 * pt_margin_grids + 1, 2 * pt_margin_grids + 1),
            dtype=bool
        )
        grid_pt = (grid_dist_um <= self.peritumoral_margin_um) & binary_dilation(self.grid_tumor_filled, structure=pt_structuring_element)
        grid_pt = binary_opening(binary_closing(grid_pt, structure=np.ones((3, 3))), structure=np.ones((3, 3)))

        self.cell_table['in_PT'] = self.map_grid_to_cells(grid_pt)
        self.cell_table['in_CT'] = (self.cell_table['in_tumor_core_filled']) & (~self.cell_table['in_PT'])
        self.cell_table['distance_to_boundary'] = self.map_grid_to_cells(grid_dist_um)

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
