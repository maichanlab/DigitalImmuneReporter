"""
Run the CRC cell/tissue-type AI prediction and spatial-feature pipeline on a
single whole-slide image, adapted from the AI4HE-Spatial-Web `process.py`
SageMaker processing job (ai4he-crc-cell-tissuetype-processing-v3) and
AI4HE-Spatial's ModelOutputFeatureComputer.py.

5 steps:
    1. preprocess_with_trident    - tissue/background segmentation + CONCH patch features
    2. predict_malignant_region   - malignant/non-malignant patch classification (logistic regression)
    3. predict_tissue_compartment - tumor/stroma/necrosis/other segmentation
    4. predict_cell_type          - cell-level instance segmentation and typing
    5. compute_spatial_features   - spatial TIME feature computation + CSV/PDF report

dMMR/MSI prediction and DZI mask/report generation from the original AI
prediction pipeline are intentionally not included, nor is the "under
investigation" lymphocyte-cluster analysis from ModelOutputFeatureComputer.py.

All model weights are loaded from ../model_weights (never downloaded).

Outputs are written to <output_dir>/, one file per prediction:
    malignant_region_mask.npy            - Step 2 (malignant/non-malignant mask)
    tissue_compartment_mask_raw.tif      - Step 3, before combining with the malignant region (compressed)
    tissue_compartment_mask_combined.npy - Step 3, restricted to the malignant region
    cell_type_predictions.json           - Step 4
    features.csv                         - Step 5, all computed spatial TIME feature values
    digital_immune_report.pdf            - Step 5, summary report
    area_table.csv, cell_table.csv       - Step 5, supporting intermediate tables

output_dir defaults to output/<slide_name> (slide_name taken from --slide_path),
so the containing folder already identifies the slide.

Usage:
    python infer.py --slide_path /path/to/slide.svs
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile
import torch
from sklearn.linear_model import LogisticRegression

from utils.slide_io import prepare_openslide_wsi
from utils.trident_runner import run_trident_preprocessing
from utils.mmdet_utils import model_fn as load_cell_model, infer_single_wsi as infer_cell_wsi
from utils.mmseg_utils import model_fn as load_tissue_model, infer_single_wsi as infer_tissue_wsi
from utils.stabilize_majority_vote import stabilize_majority_vote
from utils.spatial_feature_computer import SpatialFeatureComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS_DIR = REPO_ROOT / "model_weights"

CONCH_ENCODER_PATH = MODEL_WEIGHTS_DIR / "malignant_region_identification" / "pytorch_model.bin"
MALIGNANT_LOGREG_PATH = MODEL_WEIGHTS_DIR / "malignant_region_identification" / "logreg_conch_model_20260204.json"
TISSUE_MODEL_CKPT = MODEL_WEIGHTS_DIR / "tissue_compartment_segmentation" / "iter_40000.pth"
TISSUE_MODEL_CONFIG = MODEL_WEIGHTS_DIR / "tissue_compartment_segmentation" / "segformer_b3_40k_2xb4_tcgacrc_tissue_augment.py"
CELL_MODEL_CKPT = MODEL_WEIGHTS_DIR / "cell_type_prediction" / "epoch_36.pth"
CELL_MODEL_CONFIG = MODEL_WEIGHTS_DIR / "cell_type_prediction" / "mask2former_swin-s-3x_dataset_tcga_lizard_class_weight_log_count.py"

CELL_ID2LABEL = {1: "Neutrophil", 2: "Tumor cell", 3: "Lymphocyte", 4: "Eosinophil", 5: "Plasmacell", 6: "Other"}
TISSUE_ID2LABEL = {1: "Tumor", 2: "Stroma", 3: "Necrosis", 4: "Other"}


def preprocess_with_trident(slide_path, output_dir, mpp, gpu=0):
    """Step 1: tissue/background segmentation + CONCH patch feature extraction via Trident."""
    logging.info("Step 1: Trident preprocessing (tissue segmentation + CONCH features)...")
    geojson_path, conch_features_path = run_trident_preprocessing(
        slide_path=slide_path,
        job_dir=output_dir,
        mpp=mpp,
        gpu=gpu,
        patch_encoder="conch_v1",
        patch_encoder_ckpt_path=str(CONCH_ENCODER_PATH),
        patch_size=512,
        mag=20,
    )
    logging.info(f"Trident preprocessing done. Tissue contours: {geojson_path}, CONCH features: {conch_features_path}")
    return geojson_path, conch_features_path


def predict_malignant_region(conch_features_path, slide_width, slide_height):
    """Step 2: classify tissue patches as malignant/non-malignant using CONCH features + logistic regression."""
    logging.info("Step 2: Malignant region prediction (CONCH features + logistic regression)...")
    with open(MALIGNANT_LOGREG_PATH) as f:
        params = json.load(f)
    logreg_model = LogisticRegression(**params["params"])
    logreg_model.classes_ = np.array(params["classes"])
    logreg_model.coef_ = np.array(params["coef"])
    logreg_model.intercept_ = np.array(params["intercept"])

    with h5py.File(conch_features_path, "r") as f:
        coords = f["coords"][:]
        features = f["features"][:]
        patch_size = dict(f["coords"].attrs)["patch_size_level0"]

    y_pred = logreg_model.predict(features)
    y_pred = stabilize_majority_vote(coords, y_pred)

    # "normal"/"cancer" here are the trained classifier's fixed output
    # vocabulary (model_weights/.../logreg_conch_model_20260204.json), not
    # ours to rename; they map to non-malignant/malignant below.
    malignant_mask = np.zeros((slide_height, slide_width), dtype=np.uint8)
    for (x, y), label in zip(coords, y_pred):
        if label == "normal":
            malignant_mask[y:y + patch_size, x:x + patch_size] = 1  # non-malignant
        elif label == "cancer":
            malignant_mask[y:y + patch_size, x:x + patch_size] = 2  # malignant
        else:
            logging.warning(f"Unknown malignant-region label: {label}")

    logging.info("Malignant region prediction done.")
    return malignant_mask


def predict_tissue_compartment(slide_path, geojson_path, malignant_mask, mpp, device, output_dir):
    """Step 3: segment tissue compartments (tumor/stroma/necrosis/other), restricted to the malignant region.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (raw tissue-compartment mask, mask combined
        with the malignant region so everything outside it is labeled "Other").
    """
    logging.info("Step 3: Tissue compartment segmentation...")
    model = load_tissue_model(
        config_file_path=str(TISSUE_MODEL_CONFIG),
        checkpoint_file_path=str(TISSUE_MODEL_CKPT),
        device=device,
    )
    tissue_mask = infer_tissue_wsi(
        slide_path,
        model,
        tile_size=int(1024 * 0.25 / mpp),
        output_dir=output_dir,
        contours_geojson_path=geojson_path,
    )

    # infer_tissue_wsi() also writes an uncompressed raw-mask PNG as a side
    # effect; we save our own compressed TIFF copy of the same raw mask
    # instead (see main()), so drop the now-redundant PNG.
    auto_png_path = os.path.join(output_dir, f"{Path(slide_path).stem}.png")
    if os.path.exists(auto_png_path):
        os.remove(auto_png_path)

    combined_tissue_mask = np.full_like(tissue_mask, 4, dtype=np.uint8)  # default: "Other"
    malignant_region = malignant_mask == 2
    combined_tissue_mask[malignant_region] = tissue_mask[malignant_region]

    logging.info("Tissue compartment segmentation done.")
    return tissue_mask, combined_tissue_mask


def predict_cell_type(slide_path, geojson_path, mpp, device, output_dir):
    """Step 4: cell-level instance segmentation and typing."""
    logging.info("Step 4: Cell type prediction...")
    model = load_cell_model(
        config_file_path=str(CELL_MODEL_CONFIG),
        checkpoint_file_path=str(CELL_MODEL_CKPT),
        device=device,
    )

    cell_output_path = None
    for batch_size, chunk_size in zip([16, 8, 4, 2], [2048, 1024, 512, 256]):
        try:
            logging.info(f"Attempting cell segmentation with batch_size={batch_size}")
            cell_output_path = infer_cell_wsi(
                slide_path,
                model,
                tile_size=int(512 * 0.25 / mpp),
                output_dir=output_dir,
                batch_size=batch_size,
                chunk_size=chunk_size,
                contours_geojson_path=geojson_path,
            )
            logging.info(f"Cell segmentation succeeded with batch_size={batch_size}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(f"OOM at batch_size={batch_size}, reducing batch size...")
                torch.cuda.empty_cache()
            else:
                raise

    # infer_cell_wsi() names the output after the slide; rename to a fixed
    # name since the containing output_dir already identifies the slide.
    cell_json_path = os.path.join(output_dir, "cell_type_predictions.json")
    if os.path.abspath(cell_output_path) != os.path.abspath(cell_json_path):
        os.replace(cell_output_path, cell_json_path)

    with open(cell_json_path) as f:
        cells = json.load(f)["cells"]

    centroids = [c["centroid"] for c in cells]
    contours = [np.array(c["contour"], dtype=np.int32) for c in cells]
    labels = [c["label"] for c in cells]
    scores = [c["score"] for c in cells]

    logging.info(f"Cell type prediction done. {len(centroids)} cells detected. Saved: {cell_json_path}")
    return cell_json_path, centroids, contours, labels, scores


def compute_spatial_features(slide_path, cell_json_path, tissue_mask_path, malignant_mask_path, geojson_path, mpp, output_dir):
    """Step 5: compute spatial TIME (tumor immune microenvironment) features and generate the CSV + PDF report."""
    logging.info("Step 5: Spatial TIME feature computation + report generation...")
    computer = SpatialFeatureComputer(
        slide_path=slide_path,
        cell_json_path=cell_json_path,
        tissue_mask_path=tissue_mask_path,
        output_directory=output_dir,
        mpp=mpp,
        malignant_mask_path=malignant_mask_path,
        tissue_contour_geojson_path=geojson_path,
    )
    computer.run()
    logging.info("Spatial TIME feature computation done.")


def main():
    parser = argparse.ArgumentParser(description="Run the CRC cell/tissue-type AI prediction and spatial-feature pipeline on a single WSI.")
    parser.add_argument("--slide_path", required=True, help="Path to the input slide (svs, ome.tiff/tiff, czi, or a plain image).")
    parser.add_argument("--output_dir", default=None, help="Directory to store all pipeline outputs. Defaults to 'output/<slide_name>'.")
    parser.add_argument("--mpp", type=float, default=None, help="Microns-per-pixel override. Defaults to the slide's own MPP, or 0.25.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run Trident preprocessing on.")
    args = parser.parse_args()

    slide_name = Path(args.slide_path).stem
    output_dir = args.output_dir if args.output_dir is not None else os.path.join("output", slide_name)
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else None

    import openslide

    slide_path, final_mpp = prepare_openslide_wsi(args.slide_path, output_dir, args.mpp)
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    slide.close()
    logging.info(f"Slide ready: {slide_path} ({width}x{height}, mpp={final_mpp})")

    # Step 1
    geojson_path, conch_features_path = preprocess_with_trident(slide_path, output_dir, final_mpp, args.gpu)

    # Step 2
    malignant_mask = predict_malignant_region(conch_features_path, width, height)
    malignant_mask_path = os.path.join(output_dir, "malignant_region_mask.npy")
    np.save(malignant_mask_path, malignant_mask)
    logging.info(f"Malignant region mask saved: {malignant_mask_path}")

    # Step 3
    tissue_mask, combined_tissue_mask = predict_tissue_compartment(slide_path, geojson_path, malignant_mask, final_mpp, device, output_dir)
    tissue_mask_raw_path = os.path.join(output_dir, "tissue_compartment_mask_raw.tif")
    tifffile.imwrite(tissue_mask_raw_path, tissue_mask, compression="zlib")
    logging.info(f"Original (pre-combination) tissue compartment mask saved: {tissue_mask_raw_path}")
    tissue_mask_combined_path = os.path.join(output_dir, "tissue_compartment_mask_combined.npy")
    np.save(tissue_mask_combined_path, combined_tissue_mask)
    logging.info(f"Combined tissue compartment mask saved: {tissue_mask_combined_path}")

    # Step 4
    cell_json_path, centroids, contours, labels, scores = predict_cell_type(slide_path, geojson_path, final_mpp, device, output_dir)

    # Step 5
    compute_spatial_features(
        slide_path=slide_path,
        cell_json_path=cell_json_path,
        tissue_mask_path=tissue_mask_combined_path,
        malignant_mask_path=malignant_mask_path,
        geojson_path=geojson_path,
        mpp=final_mpp,
        output_dir=output_dir,
    )

    logging.info("Pipeline complete.")
    logging.info(f"Tissue compartment labels: {TISSUE_ID2LABEL}")
    logging.info(f"Cell type labels: {CELL_ID2LABEL}")
    logging.info(f"Detected cells: {len(centroids)}")
    logging.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
