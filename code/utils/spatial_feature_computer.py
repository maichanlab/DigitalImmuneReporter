"""
Computes spatial TIME (tumor immune microenvironment) features from the AI
predictions in infer.py's output, and generates a CSV of feature values plus
a PDF summary report.

Adapted from AI4HE-Spatial's
experiments/crc_wk/compute_features_from_model_output/ModelOutputFeatureComputer.py,
with the WSI reader swapped for plain OpenSlide (the slide is already
normalized to an OpenSlide-readable format by infer.py's slide_io step) and
the "under investigation" lymphocyte-cluster analysis left out.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import openslide
import pandas as pd
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from utils.mask_utils import convert_geojson_to_mask, convert_label_mask_to_rgb
from utils.cell_tissue_table_generator import generate_cell_table, generate_tissue_area_table
from utils.cell_density import count_cell_density, count_cell_in_tissue
from utils.gcross import area_under_g_cross_curve
from utils.tumor_region_identifier import TumorRegionIdentifier

logger = logging.getLogger(__name__)

DEFAULT_TISSUE_ID2LABEL = {0: "background", 1: "tumor", 2: "stroma", 3: "necrosis", 4: "other"}
DEFAULT_CELL_ID2LABEL = {1: "neutrophil", 2: "tumor", 3: "lymphocyte", 4: "eosinophil", 5: "plasmacell", 6: "other"}


class SpatialFeatureComputer:
    def __init__(
        self,
        slide_path: str,
        cell_json_path: str,
        tissue_mask_path: str,
        output_directory: str,
        mpp: Optional[float] = None,
        malignant_mask_path: Optional[str] = None,  # for visualization only, no processing involved
        tissue_contour_geojson_path: Optional[str] = None,
        tissue_id2label: Dict[int, str] = DEFAULT_TISSUE_ID2LABEL,
        cell_id2label: Dict[int, str] = DEFAULT_CELL_ID2LABEL,
    ):
        if not Path(slide_path).exists():
            raise FileExistsError("H&E image not found")
        if not Path(cell_json_path).exists():
            raise FileExistsError("Cell prediction not found")
        if not Path(tissue_mask_path).exists():
            raise FileExistsError("Tissue prediction not found")

        self.slide = openslide.OpenSlide(str(slide_path))
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if mpp is not None:
            self.mpp = mpp
        else:
            mpp_prop = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X) or self.slide.properties.get("aperio.MPP")
            if mpp_prop is None:
                raise ValueError("MPP not found on the slide; pass --mpp explicitly.")
            self.mpp = float(mpp_prop)
            logger.info(f"MPP value using OpenSlide: {self.mpp}")

        self.tissue_id2label = tissue_id2label
        self.cell_id2label = cell_id2label

        # Retrieve tissue data
        logger.info(f"Loading tissue data from {tissue_mask_path}")
        tissue_mask_path = Path(tissue_mask_path)
        suffix = tissue_mask_path.suffix.lower()
        if suffix == ".png":
            self.tissue_mask = np.array(Image.open(tissue_mask_path), dtype=np.uint8)
        elif suffix in {".tif", ".tiff"}:
            self.tissue_mask = tifffile.imread(tissue_mask_path).astype(np.uint8)
        elif suffix == ".npy":
            self.tissue_mask = np.load(tissue_mask_path).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported tissue mask format: {suffix}")

        tissue_mask_height, tissue_mask_width = self.tissue_mask.shape
        if self.slide.dimensions != (tissue_mask_width, tissue_mask_height):
            raise ValueError("Tissue mask shape mismatch with H&E")

        if tissue_contour_geojson_path:
            logger.info(f"Removing background outside tissue contours from {tissue_contour_geojson_path}")
            binary_tissue_mask = convert_geojson_to_mask(tissue_contour_geojson_path, width=tissue_mask_width, height=tissue_mask_height)
            self.tissue_mask = self.tissue_mask * binary_tissue_mask

        self.malignant_mask = None
        if malignant_mask_path and Path(malignant_mask_path).exists():
            self.malignant_mask = np.load(malignant_mask_path).astype(np.uint8)
            logger.info(f"Loaded malignant-region mask from {malignant_mask_path} (used for report visualization only)")

        # Retrieve cell data
        logger.info(f"Loading cell data from {cell_json_path}")
        with open(cell_json_path) as json_file:
            data = json.load(json_file)
            self.cell_data = data["cells"]
        logger.info(f"Loaded {len(self.cell_data)} cells")

    def compute_features(self) -> Dict:
        output = {}

        cell_table = self.cell_table[
            self.cell_table["tissue_label"].isin(["tumor", "stroma", "necrosis", "other"])
        ].copy()

        cell_table["is_tumor_cell"] = (
            (cell_table["cell_label"] == "tumor")
            & (cell_table["tissue_label"].isin(["tumor", "stroma"]))
        )

        # ----- AREA -----
        if "area_sq_milimeters" not in self.tissue_area_table.columns:
            self.tissue_area_table["area_sq_milimeters"] = (
                self.tissue_area_table["area_sq_microns"] / (1000 * 1000)
            )

        area_table = self.tissue_area_table[
            self.tissue_area_table["tissue_label"].isin(["tumor", "stroma", "necrosis", "other"])
        ]

        tissue_area = {
            row["tissue_label"]: row["area_sq_milimeters"]
            for _, row in area_table.iterrows()
        }

        for t in ["tumor", "stroma", "necrosis", "other"]:
            tissue_area.setdefault(t, 0)

        # ----- CELL COUNT -----
        cell_count = count_cell_in_tissue(cell_table)

        for cell_type in [
            "tumor",
            "other",
            "lymphocyte",
            "eosinophil",
            "plasmacell",
            "neutrophil",
        ]:
            for tissue_type in ["tumor", "stroma", "necrosis", "other"]:

                if len(
                    cell_count[
                        (cell_count["cell_label"] == cell_type)
                        & (cell_count["tissue_label"] == tissue_type)
                    ]
                ) == 0:

                    cell_count = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "cell_label": [cell_type],
                                    "tissue_label": [tissue_type],
                                    "count": [0],
                                }
                            ),
                            cell_count,
                        ]
                    )

        # ----- TUMOR + STROMA -----
        tissue_area["tumor_stroma"] = (
            tissue_area.get("tumor", 0) + tissue_area.get("stroma", 0)
        )

        tumor_stroma_counts = (
            cell_count[cell_count["tissue_label"].isin(["tumor", "stroma"])]
            .groupby("cell_label", as_index=False)["count"]
            .sum()
        )

        tumor_stroma_counts["tissue_label"] = "tumor_stroma"
        cell_count = pd.concat([cell_count, tumor_stroma_counts], ignore_index=True)

        # ----- DENSITY -----
        self.density_table = count_cell_density(cell_count, tissue_area)

        output.update(
            {row["density_label"]: row["density"] for _, row in self.density_table.iterrows()}
        )

        # ----- G-CROSS -----
        radii = np.linspace(0, 50, 100)

        class_i = "tumor"
        points_i = cell_table[cell_table["is_tumor_cell"]][["x", "y"]].to_numpy()

        for class_j in ["neutrophil", "lymphocyte", "eosinophil", "plasmacell"]:

            points_j = cell_table[cell_table["cell_label"] == class_j][["x", "y"]].to_numpy()

            if len(points_i) == 0 or len(points_j) == 0:
                output[f"G_{class_i}:{class_j}_auc_0_20"] = 0
                continue

            output[f"G_{class_i}:{class_j}_auc_0_20"] = area_under_g_cross_curve(
                points_i, points_j, radii, slice=(0, 20)
            )

        # ----- CT / PT -----
        if len(cell_table) >= 20000:
            analyzer = TumorRegionIdentifier(cell_table, "is_tumor_cell")
            self.cell_table = analyzer.run()

            total_CT = len(cell_table[cell_table["in_CT"]])
            total_PT = len(cell_table[cell_table["in_PT"]])

            if total_CT and total_PT:

                for cell_type in [
                    "lymphocyte",
                    "eosinophil",
                    "plasmacell",
                    "neutrophil",
                ]:

                    output[f"{cell_type}_CT"] = (
                        len(
                            cell_table[
                                (cell_table["in_CT"])
                                & (cell_table["cell_label"] == cell_type)
                            ]
                        )
                        / total_CT
                    )

                    output[f"{cell_type}_PT"] = (
                        len(
                            cell_table[
                                (cell_table["in_PT"])
                                & (cell_table["cell_label"] == cell_type)
                            ]
                        )
                        / total_PT
                    )

                    if len(
                            cell_table[
                                (cell_table["in_PT"])
                                & (cell_table["cell_label"] == cell_type)
                            ]
                        ):
                        output[f"{cell_type}_CT/PT_ratio"] = (
                            len(
                                cell_table[
                                    (cell_table["in_CT"])
                                    & (cell_table["cell_label"] == cell_type)
                                ]
                            ) /
                            len(
                                cell_table[
                                    (cell_table["in_PT"])
                                    & (cell_table["cell_label"] == cell_type)
                                ]
                            )
                        )

        # ----- TSP -----
        try:
            output["tumor_stroma_percentage"] = (
                tissue_area.get("stroma", 0)
                / (tissue_area.get("stroma", 0) + tissue_area.get("tumor", 0))
                * 100
            )
        except ZeroDivisionError:
            output["tumor_stroma_percentage"] = 0

        # ----- CELL COUNT -----
        cell_count_overall = (
            cell_count.groupby("cell_label").agg({"count": "sum"}).reset_index()
        )

        cell_count_overall["abundance"] = (
            cell_count_overall["count"] / cell_count_overall["count"].sum()
        )

        for _, row in cell_count_overall.iterrows():
            output[f"{row['cell_label']}_count"] = row["count"]
            output[f"{row['cell_label']}_abundance"] = row["abundance"]

        # ----- NLR -----
        neutrophil = output.get("neutrophil_count", 0)
        lymphocyte = output.get("lymphocyte_count", 0)

        if lymphocyte == 0:
            output["neutrophil_to_lymphocyte_ratio"] = neutrophil
        else:
            output["neutrophil_to_lymphocyte_ratio"] = neutrophil / lymphocyte

        self.features = output
        return output

    def tissue_mask_thumbnail(self):
        h, w = self.tissue_mask.shape[:2]
        scale = 2000 / max(h, w)
        tissue_mask_thumbnail = cv2.resize(self.tissue_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tissue_mask_thumbail_rgb = convert_label_mask_to_rgb(
            tissue_mask_thumbnail,
            label_colors={
                1: (0, 0, 255),
                2: (0, 255, 0),
                3: (255, 0, 0),
                4: (255, 255, 0),
            },
            ignore_unseen_labels=True,
        )
        return Image.fromarray(tissue_mask_thumbail_rgb)

    def malignant_region_mask_thumbnail(self):
        h, w = self.malignant_mask.shape[:2]
        scale = 2000 / max(h, w)
        malignant_mask_thumbnail = cv2.resize(self.malignant_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        malignant_mask_thumbnail_rgb = convert_label_mask_to_rgb(
            malignant_mask_thumbnail,
            label_colors={
                1: (0, 255, 255),
                2: (255, 0, 255),
            },
            ignore_unseen_labels=True,
        )
        return Image.fromarray(malignant_mask_thumbnail_rgb)

    def ct_pt_process_plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Tumor cells
        tumor_mask = self.cell_table[self.cell_table['is_tumor_cell']]
        axes[0].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[0].scatter(tumor_mask['x'],
                        tumor_mask['y'],
                        c='lime', s=0.3, alpha=0.5)
        axes[0].set_title('Tumor cell')
        axes[0].set_xlim(left=0)
        axes[0].set_ylim(bottom=0)
        axes[0].invert_yaxis()
        # Packed tumor cells
        tumor_mask = self.cell_table[self.cell_table['in_tumor_smoothed']]
        axes[1].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.3, edgecolors='none')
        axes[1].scatter(tumor_mask['x'],
                        tumor_mask['y'],
                        c='green', s=0.3, alpha=0.5)
        axes[1].set_title('Packed tumor region')
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(bottom=0)
        axes[1].invert_yaxis()
        # CT/PT cells
        ct_cells = self.cell_table[self.cell_table['in_CT']]
        pt_cells = self.cell_table[self.cell_table['in_PT']]
        axes[2].scatter(self.cell_table['x'], self.cell_table['y'],
                        c='lightgray', s=0.3, alpha=0.5)
        axes[2].scatter(ct_cells['x'], ct_cells['y'],
                        c='yellow', s=0.3, alpha=0.5)
        axes[2].scatter(pt_cells['x'], pt_cells['y'],
                        c='orange', s=0.3, alpha=0.5)
        axes[2].set_title('CT/PT')
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(bottom=0)
        axes[2].invert_yaxis()
        fig.savefig(self.output_directory / "ct_pt_process.jpg")
        plt.close(fig)
        return self.output_directory / "ct_pt_process.jpg"

    def generate_pdf(self, output_path: str):
        PAGE_WIDTH = 6.5 * inch   # content width (Letter minus margins)
        PAGE_HEIGHT = 9 * inch    # content height (Letter minus margins)
        MAX_IMAGE_WIDTH = 3 * inch
        MAX_IMAGE_HEIGHT = 3 * inch

        def make_scaled_image(image_path, max_width=MAX_IMAGE_WIDTH, max_height=MAX_IMAGE_HEIGHT):
            img = Image.open(image_path)
            orig_width, orig_height = img.size

            width_ratio = max_width / orig_width
            height_ratio = max_height / orig_height
            scale = min(width_ratio, height_ratio, 1.0)  # don't upscale

            final_width = orig_width * scale
            final_height = orig_height * scale

            return ReportLabImage(image_path, width=final_width, height=final_height)

        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Digital Immune Microenvironment Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        # ----- H&E & Tissue Mask -----
        he_thumbnail_path = self.output_directory / "he_thumbnail.jpg"
        self.slide.get_thumbnail((2000, 2000)).convert("RGB").save(he_thumbnail_path)

        tissue_mask_thumbnail_path = self.output_directory / "tissue_mask_thumbnail.jpg"
        self.tissue_mask_thumbnail().save(tissue_mask_thumbnail_path)

        thumbnail_cells = [
            Table([
                ["H&E\n"],
                [make_scaled_image(he_thumbnail_path)],
            ]),
            Table([
                ["Tissue mask\nB: Tumor, G: Stroma, R: Necrosis, Y: Other"],
                [make_scaled_image(tissue_mask_thumbnail_path)],
            ]),
        ]

        if self.malignant_mask is not None:
            malignant_mask_thumbnail_path = self.output_directory / "malignant_region_mask_thumbnail.jpg"
            self.malignant_region_mask_thumbnail().save(malignant_mask_thumbnail_path)
            thumbnail_cells.append(
                Table([
                    ["Malignant region mask\nMagenta: Malignant, Cyan: Non-malignant"],
                    [make_scaled_image(malignant_mask_thumbnail_path)],
                ])
            )

        he_tissueMask = Table(
            [thumbnail_cells[i:i+2] for i in range(0, len(thumbnail_cells), 2)],
            colWidths=[MAX_IMAGE_WIDTH + inch, MAX_IMAGE_WIDTH + inch]
        )
        he_tissueMask.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))
        elements.append(he_tissueMask)
        elements.append(Spacer(1, 20))

        # ----- TISSUE AREA & CELL COUNT ----
        area_df = self.tissue_area_table[self.tissue_area_table["tissue_label"] != "background"]
        area_rows = [["Tissue Region", "Area (sq mm)"]]
        for _, row in area_df.iterrows():
            area_rows.append([
                row["tissue_label"],
                round(row["area_sq_milimeters"], 4)
            ])

        count_rows = [["Cell Type", "Count", "Abundance"]]
        for cell in ["tumor", "lymphocyte", "neutrophil", "plasmacell", "eosinophil"]:
            count_rows.append(
                [
                    cell,
                    self.features.get(f"{cell}_count", 0),
                    round(self.features.get(f"{cell}_abundance", 0), 3),
                ]
            )

        cellCount_areaTable = Table(
            [[
                [
                    Paragraph("Cell Counts", styles["Heading2"]),
                    Table(count_rows)
                ],
                [
                    Paragraph("Tissue Area (sq mm)", styles["Heading2"]),
                    Table(area_rows)
                ],

            ]],
            colWidths=[250, 250]
        )
        cellCount_areaTable.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP")
            ])
        )
        elements.append(cellCount_areaTable)
        elements.append(Spacer(1, 20))

        # ----- DENSITY -----
        elements.append(Paragraph("Cell Density Metrics (cells/mm2)", styles["Heading2"]))

        pivot_df = self.density_table.pivot(
            index="cell_label",
            columns="tissue_label",
            values="density"
        ).fillna("-")

        density_rows = [["Cell Type"] + list(pivot_df.columns)]

        for cell_type, row in pivot_df.iterrows():
            density_rows.append(
                [cell_type] + [round(v, 4) if isinstance(v, (int, float)) else v for v in row]
            )

        elements.append(Table(density_rows))
        elements.append(Spacer(1, 20))

        # ----- G-CROSS -----
        elements.append(Paragraph("Tumor-Immune Spatial Proximity (G-cross AUC)", styles["Heading2"]))

        g_rows = [["Proximity", "AUC"]]

        for k, v in self.features.items():
            if "G_tumor" in k:
                g_rows.append([k, round(v, 4)])

        elements.append(Table(g_rows))
        elements.append(Spacer(1, 20))

        # ----- OTHER METRICS -----
        elements.append(Paragraph("Other Metrics", styles["Heading2"]))

        other_rows = [
            ["Tumor Stroma Percentage", round(self.features.get("tumor_stroma_percentage", 0), 3)],
            ["Neutrophil/Lymphocyte Ratio", round(self.features.get("neutrophil_to_lymphocyte_ratio", 0), 3)],
        ]

        elements.append(Table([["Metric", "Value"]] + other_rows))
        elements.append(PageBreak())

        # ----- CT PT IMAGE -----
        ct_pt_process_plot_path = self.ct_pt_process_plot()

        ct_pt_process_plot = Table([
            [Paragraph("CT/PT Segmentation Process", styles["Heading2"])],
            [make_scaled_image(ct_pt_process_plot_path, max_width=PAGE_WIDTH, max_height=PAGE_HEIGHT)],
        ])
        elements.append(ct_pt_process_plot)
        elements.append(Spacer(1, 20))

        # ----- RELATIVE ABUNDANCE (CT vs PT) -----
        elements.append(Paragraph("Relative Abundance by Region (CT vs PT)", styles["Heading3"]))

        cell_types = ["lymphocyte", "eosinophil", "plasmacell", "neutrophil"]

        relative_abundance_rows = [["Cell Type", "CT", "PT"]]

        for cell in cell_types:
            ct = self.features.get(f"{cell}_CT", "-")
            pt = self.features.get(f"{cell}_PT", "-")

            relative_abundance_rows.append([
                cell,
                round(ct, 4) if isinstance(ct, (int, float)) else ct,
                round(pt, 4) if isinstance(pt, (int, float)) else pt
            ])

        relative_abundance_table = Table(relative_abundance_rows)

        elements.append(relative_abundance_table)
        elements.append(Spacer(1, 20))

        pdf = SimpleDocTemplate(output_path)
        pdf.build(elements)

    def run(self):
        logger.info("Generating tissue area table...")
        area_table_csv_path = self.output_directory / "area_table.csv"
        self.tissue_area_table = generate_tissue_area_table(
            self.tissue_mask,
            tissue_id2label=self.tissue_id2label,
            mpp=self.mpp,
        )
        self.tissue_area_table.to_csv(area_table_csv_path, index=False)
        logger.info(f"Tissue area table saved: {area_table_csv_path}")

        logger.info("Generating cell table (assigning each cell to a tissue region)...")
        cell_table_csv_path = self.output_directory / "cell_table.csv"
        self.cell_table = generate_cell_table(
            cells=self.cell_data,
            cell_id2label=self.cell_id2label,
            tissue_mask=self.tissue_mask,
            tissue_id2label=self.tissue_id2label,
            mpp=self.mpp,
        )
        logger.info(f"Cell table built: {len(self.cell_table)} cells matched to a tissue region")

        logger.info("Computing spatial TIME features (densities, G-cross, CT/PT, tumor-stroma %, NLR)...")
        self.features = self.compute_features()
        self.cell_table.to_csv(cell_table_csv_path, index=False)
        logger.info(f"Cell table saved: {cell_table_csv_path}")
        features_csv_path = self.output_directory / "features.csv"
        pd.DataFrame(list(self.features.items()), columns=["feature", "value"]).to_csv(features_csv_path, index=False)
        logger.info(f"Computed {len(self.features)} features. Saved: {features_csv_path}")

        report_path = self.output_directory / "digital_immune_report.pdf"
        self.generate_pdf(str(report_path))
        logger.info(f"Report saved: {report_path}")
        logger.info(f"All outputs saved to {self.output_directory}")
