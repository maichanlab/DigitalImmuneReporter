"""
Run the CRC cell/tissue-type AI prediction and spatial-feature pipeline on a
single whole-slide image, adapted from the AI4HE-Spatial-Web `process.py`
SageMaker processing job (ai4he-crc-cell-tissuetype-processing-v3) and
AI4HE-Spatial's ModelOutputFeatureComputer.py.

5 steps:
    1. preprocess_with_trident    - tissue/background segmentation + CONCH patch features
    2. predict_malignant_region   - malignant/non-malignant patch classification (logistic regression)
                                     optional: skip with --no-use_malignant_region (default: run it)
    3. predict_tissue_compartment - tumor/stroma/necrosis/other segmentation; restricted to the
                                     malignant region from Step 2, or used as-is if Step 2 was skipped
    4. predict_cell_type          - cell-level instance segmentation and typing
    5. compute_spatial_features   - spatial TIME feature computation + CSV/PDF report

dMMR/MSI prediction and DZI mask/report generation from the original AI
prediction pipeline are intentionally not included, nor is the "under
investigation" lymphocyte-cluster analysis from ModelOutputFeatureComputer.py.

All model weights are loaded from ../model_weights (never downloaded).

Outputs are written to <output_dir>/, one file per prediction:
    pipeline.log                          - full log of this run (console output is mirrored here)
    malignant_region_mask.tif            - Step 2 (malignant/non-malignant mask), only if not skipped (compressed)
    tissue_compartment_mask_raw.tif      - Step 3, before combining with the malignant region (compressed)
    tissue_compartment_mask_combined.tif - Step 3, restricted to the malignant region (identical to the
                                            raw mask if Step 2 was skipped) (compressed)
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
from utils.logging_utils import setup_logging, TimingTracker

logger = logging.getLogger(__name__)

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
    logger.info("Step 1: Trident preprocessing (tissue segmentation + CONCH features)...")
    logger.info(f"Trident config: patch_encoder=conch_v1, patch_size=512, mag=20, segmenter=hest, gpu={gpu}, mpp={mpp}")
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
    logger.info(f"Trident preprocessing done. Tissue contours: {geojson_path}, CONCH features: {conch_features_path}")
    return geojson_path, conch_features_path


def predict_malignant_region(conch_features_path, slide_width, slide_height):
    """Step 2: classify tissue patches as malignant/non-malignant using CONCH features + logistic regression."""
    logger.info("Step 2: Malignant region prediction (CONCH features + logistic regression)...")
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
    logger.info(f"Classifying {len(coords)} patches (patch_size={patch_size})...")

    y_pred = logreg_model.predict(features)
    unique, counts = np.unique(y_pred, return_counts=True)
    logger.info(f"Raw classifier output before spatial smoothing: {dict(zip(unique.tolist(), counts.tolist()))}")

    y_pred = stabilize_majority_vote(coords, y_pred)
    unique, counts = np.unique(y_pred, return_counts=True)
    logger.info(f"After spatial smoothing (stabilize_majority_vote): {dict(zip(unique.tolist(), counts.tolist()))}")

    # "normal"/"cancer" here are the trained classifier's fixed output
    # vocabulary (model_weights/.../logreg_conch_model_20260204.json), not
    # ours to rename; they map to non-malignant/malignant below.
    malignant_mask = np.zeros((slide_height, slide_width), dtype=np.uint8)
    unknown_labels = set()
    for (x, y), label in zip(coords, y_pred):
        if label == "normal":
            malignant_mask[y:y + patch_size, x:x + patch_size] = 1  # non-malignant
        elif label == "cancer":
            malignant_mask[y:y + patch_size, x:x + patch_size] = 2  # malignant
        else:
            unknown_labels.add(label)
    if unknown_labels:
        logger.warning(f"Unknown malignant-region label(s) encountered and ignored: {sorted(unknown_labels)}")

    total_px = malignant_mask.size
    non_malignant_pct = (malignant_mask == 1).sum() / total_px * 100
    malignant_pct = (malignant_mask == 2).sum() / total_px * 100
    logger.info(
        f"Malignant region prediction done. Area breakdown: "
        f"malignant={malignant_pct:.1f}%, non-malignant={non_malignant_pct:.1f}%, "
        f"background/unclassified={100 - malignant_pct - non_malignant_pct:.1f}%"
    )
    return malignant_mask


def predict_tissue_compartment(slide_path, geojson_path, malignant_mask, mpp, device, output_dir):
    """Step 3: segment tissue compartments (tumor/stroma/necrosis/other).

    If `malignant_mask` is provided, the result is restricted to the malignant
    region (everything outside it is labeled "Other"). If it's None (malignant
    region identification was skipped), the raw tissue-compartment mask is
    used as-is.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (raw tissue-compartment mask, combined mask).
    """
    logger.info("Step 3: Tissue compartment segmentation...")
    logger.info(f"Loading tissue-compartment model from {TISSUE_MODEL_CKPT}")
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

    # infer_tissue_wsi() also writes a compressed-TIFF raw mask as a side
    # effect, under the slide's own stem; we save our own copy of the same
    # raw mask under a fixed name instead (see main()), so drop the
    # now-redundant auto-named one.
    auto_tif_path = os.path.join(output_dir, f"{Path(slide_path).stem}.tif")
    if os.path.exists(auto_tif_path):
        os.remove(auto_tif_path)

    if malignant_mask is None:
        logger.info("No malignant-region mask provided; using the raw tissue-compartment prediction as-is.")
        combined_tissue_mask = tissue_mask.copy()
    else:
        combined_tissue_mask = np.full_like(tissue_mask, 4, dtype=np.uint8)  # default: "Other"
        malignant_region = malignant_mask == 2
        combined_tissue_mask[malignant_region] = tissue_mask[malignant_region]
        logger.info(f"Restricted tissue-compartment mask to the malignant region ({malignant_region.sum() / malignant_region.size * 100:.1f}% of the slide).")

    total_px = combined_tissue_mask.size
    label_breakdown = {
        TISSUE_ID2LABEL.get(label_id, f"id={label_id}"): f"{(combined_tissue_mask == label_id).sum() / total_px * 100:.1f}%"
        for label_id in np.unique(combined_tissue_mask)
        if label_id in TISSUE_ID2LABEL
    }
    logger.info(f"Tissue compartment segmentation done. Area breakdown: {label_breakdown}")
    return tissue_mask, combined_tissue_mask


def predict_cell_type(slide_path, geojson_path, mpp, device, output_dir):
    """Step 4: cell-level instance segmentation and typing."""
    logger.info("Step 4: Cell type prediction...")
    logger.info(f"Loading cell-type model from {CELL_MODEL_CKPT}")
    model = load_cell_model(
        config_file_path=str(CELL_MODEL_CONFIG),
        checkpoint_file_path=str(CELL_MODEL_CKPT),
        device=device,
    )

    cell_output_path = None
    for batch_size, chunk_size in zip([16, 8, 4, 2], [2048, 1024, 512, 256]):
        try:
            logger.info(f"Attempting cell segmentation with batch_size={batch_size}, chunk_size={chunk_size}")
            cell_output_path = infer_cell_wsi(
                slide_path,
                model,
                tile_size=int(512 * 0.25 / mpp),
                output_dir=output_dir,
                batch_size=batch_size,
                chunk_size=chunk_size,
                contours_geojson_path=geojson_path,
            )
            logger.info(f"Cell segmentation succeeded with batch_size={batch_size}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch_size={batch_size}, retrying with a smaller batch size...")
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

    unique, counts = np.unique(labels, return_counts=True) if labels else ([], [])
    label_breakdown = {CELL_ID2LABEL.get(int(label_id), f"id={label_id}"): int(count) for label_id, count in zip(unique, counts)}
    logger.info(f"Cell type prediction done. {len(centroids)} cells detected: {label_breakdown}")
    logger.info(f"Cell predictions saved: {cell_json_path}")
    return cell_json_path, centroids, contours, labels, scores


def compute_spatial_features(slide_path, cell_json_path, tissue_mask_path, malignant_mask_path, geojson_path, mpp, output_dir):
    """Step 5: compute spatial TIME (tumor immune microenvironment) features and generate the CSV + PDF report."""
    logger.info("Step 5: Spatial TIME feature computation + report generation...")
    logger.info(f"Using tissue mask: {tissue_mask_path}" + (f", malignant mask: {malignant_mask_path}" if malignant_mask_path else " (no malignant mask; skipping that visualization)"))
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
    logger.info(
        f"Spatial TIME feature computation done. Wrote features.csv, digital_immune_report.pdf, "
        f"area_table.csv, cell_table.csv to {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the CRC cell/tissue-type AI prediction and spatial-feature pipeline on a single WSI.")
    parser.add_argument("--slide_path", required=True, help="Path to the input slide (svs, ome.tiff/tiff, czi, or a plain image).")
    parser.add_argument("--output_dir", default=None, help="Directory to store all pipeline outputs. Defaults to 'output/<slide_name>'.")
    parser.add_argument("--mpp", type=float, default=None, help="Microns-per-pixel override. Defaults to the slide's own MPP, or 0.25.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run Trident preprocessing on.")
    parser.add_argument(
        "--use_malignant_region", action=argparse.BooleanOptionalAction, default=True,
        help="Run malignant/non-malignant region identification (Step 2) and restrict the tissue-compartment "
             "mask to it. Pass --no-use_malignant_region to skip Step 2 and use the raw tissue-compartment "
             "mask directly.",
    )
    args = parser.parse_args()

    slide_name = Path(args.slide_path).stem
    output_dir = args.output_dir if args.output_dir is not None else os.path.join("output", slide_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = setup_logging(output_dir)
    logger.info(f"Logging to console and {log_path}")
    logger.info(f"Arguments: {vars(args)}")
    timing_tracker = TimingTracker(logger)

    device = "cuda" if torch.cuda.is_available() else None
    logger.info(f"CUDA available: {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    import openslide

    timing_tracker.start_step("0. Slide preparation")
    slide_path, final_mpp = prepare_openslide_wsi(args.slide_path, output_dir, args.mpp)
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    slide.close()
    logger.info(f"Slide ready: {slide_path} ({width}x{height}, mpp={final_mpp})")
    timing_tracker.end_step("0. Slide preparation")

    # Step 1
    timing_tracker.start_step("1. Trident preprocessing")
    geojson_path, conch_features_path = preprocess_with_trident(slide_path, output_dir, final_mpp, args.gpu)
    timing_tracker.end_step("1. Trident preprocessing")

    # Step 2
    timing_tracker.start_step("2. Malignant region prediction")
    malignant_mask = None
    malignant_mask_path = None
    if args.use_malignant_region:
        malignant_mask = predict_malignant_region(conch_features_path, width, height)
        malignant_mask_path = os.path.join(output_dir, "malignant_region_mask.tif")
        tifffile.imwrite(malignant_mask_path, malignant_mask, compression="zlib")
        logger.info(f"Malignant region mask saved: {malignant_mask_path}")
    else:
        logger.info("Step 2: Skipping malignant region prediction (--no-use_malignant_region); using the tissue compartment mask directly.")
    timing_tracker.end_step("2. Malignant region prediction")

    # Step 3
    timing_tracker.start_step("3. Tissue compartment segmentation")
    tissue_mask, combined_tissue_mask = predict_tissue_compartment(slide_path, geojson_path, malignant_mask, final_mpp, device, output_dir)
    tissue_mask_raw_path = os.path.join(output_dir, "tissue_compartment_mask_raw.tif")
    tifffile.imwrite(tissue_mask_raw_path, tissue_mask, compression="zlib")
    logger.info(f"Original (pre-combination) tissue compartment mask saved: {tissue_mask_raw_path}")
    tissue_mask_combined_path = os.path.join(output_dir, "tissue_compartment_mask_combined.tif")
    tifffile.imwrite(tissue_mask_combined_path, combined_tissue_mask, compression="zlib")
    logger.info(f"Combined tissue compartment mask saved: {tissue_mask_combined_path}")
    timing_tracker.end_step("3. Tissue compartment segmentation")

    # Step 4
    timing_tracker.start_step("4. Cell type prediction")
    cell_json_path, centroids, contours, labels, scores = predict_cell_type(slide_path, geojson_path, final_mpp, device, output_dir)
    timing_tracker.end_step("4. Cell type prediction")

    # Step 5
    timing_tracker.start_step("5. Spatial feature computation")
    compute_spatial_features(
        slide_path=slide_path,
        cell_json_path=cell_json_path,
        tissue_mask_path=tissue_mask_combined_path,
        malignant_mask_path=malignant_mask_path,
        geojson_path=geojson_path,
        mpp=final_mpp,
        output_dir=output_dir,
    )
    timing_tracker.end_step("5. Spatial feature computation")

    logger.info("Pipeline complete.")
    logger.info(f"Tissue compartment labels: {TISSUE_ID2LABEL}")
    logger.info(f"Cell type labels: {CELL_ID2LABEL}")
    logger.info(f"Detected cells: {len(centroids)}")
    logger.info(f"All outputs saved to: {output_dir}")
    timing_tracker.log_summary()


if __name__ == "__main__":
    main()
