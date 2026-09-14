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
import os
import sys


def _get_gpu_arg_early(argv):
    """Extract --gpu from argv using only os/sys (nothing that touches CUDA), so
    CUDA_VISIBLE_DEVICES can be set below before torch/mmcv/mmdet - which link against the
    CUDA driver at import time - are imported. Setting the env var any later (e.g. inside
    main(), after those imports have already happened) does not reliably restrict which
    physical GPU gets used: the driver appears to resolve GPU topology at library-load time,
    not at first actual CUDA call, so device_count() and cuda:0 end up "masked" to 1 visible
    device but not necessarily the physical one --gpu asked for. Mirrors argparse's own
    --gpu/--gpu=N syntax and default of 0.
    """
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return "0"


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _get_gpu_arg_early(sys.argv[1:])

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
import torch
from sklearn.linear_model import LogisticRegression

from utils.slide_io.slide_io import prepare_openslide_wsi
from utils.preprocessing.trident_runner import run_trident_preprocessing
from utils.mmlab_prediction.mmdet_utils import model_fn as load_cell_model, infer_single_wsi as infer_cell_wsi
from utils.mmlab_prediction.mmseg_utils import model_fn as load_tissue_model, infer_single_wsi as infer_tissue_wsi
from utils.preprocessing.stabilize_majority_vote import stabilize_majority_vote
from utils.spatial_features.spatial_feature_computer import SpatialFeatureComputer
from utils.logging_utils import setup_logging, TimingTracker
from utils.miphei_multiplex_prediction import miphei_utils, cellvit_utils
from utils.miphei_multiplex_prediction.mif_cell_annotator import (
    convert_contours_to_labeled_mask,
    extract_mean_marker_intensities,
    binarize_markers,
    annotate_cell_hierarchy,
    hierarchy_level_categories,
)

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

# Step 4b (miphei_multiplex cell typing): external repos, cloned by setup_env.sh into
# code/external/ (see there) and registered on sys.path at call time (neither ships a
# setup.py/pyproject.toml, so they aren't pip-installed like trident/CONCH).
EXTERNAL_REPOS_DIR = Path(__file__).resolve().parent / "external"
MIPHEI_VIT_REPO_ROOT = str(EXTERNAL_REPOS_DIR / "MIPHEI-ViT")
CELLVIT_REPO_ROOT = str(EXTERNAL_REPOS_DIR / "CellViT-plus-plus")

MIPHEI_CHECKPOINT_DIR = MODEL_WEIGHTS_DIR / "miphei_multiplex_prediction"
MIPHEI_CALIBRATION_CLASSIFIER_PATH = MIPHEI_CHECKPOINT_DIR / "logreg.pth"
CELLVIT_CHECKPOINT_PATH = MODEL_WEIGHTS_DIR / "cellvit_binary_cell_detection" / "CellViT-SAM-H-x40-AMP.pth"

# Fixed channel order the MIPHEI-ViT checkpoint predicts in and its calibration classifier was
# trained on (model_weights/miphei_multiplex_prediction/config.yaml: data.targ_channel_names) —
# must not be reordered.
MIPHEI_MARKER_CHANNELS_FULL = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA",
]
MIPHEI_NUCLEUS_MARKER = "Hoechst"

MIPHEI_MULTIPLEX_HIERARCHY = [
    {"name": "Tumour", "markers": ["Pan-CK+"]},
    {
        "name": "T Cell",
        "markers": ["CD3e+"],
        "subtypes": [
            {"name": "CytotoxicT", "markers": ["CD8a+"]},
            {"name": "Helper T", "markers": ["CD4+"]},
        ],
    },
    {"name": "B Cell", "markers": ["CD20+"]},
    {"name": "Macrophage", "markers": ["CD68+"]},
]


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


def predict_miphei_multiplex_cell_type(slide_path, slide_width, slide_height, mpp, output_dir, gpu=0):
    """Step 4b (alternative to Step 4): protein-marker-driven cell typing, in three stages:
    predict multiplex (mIF) channels from the H&E with MIPHEI-ViT, detect cells (binary, no
    typing) with CellViT-plus-plus, then type each detected cell against MIPHEI_MULTIPLEX_HIERARCHY
    using its mean predicted marker intensities (calibrated, then thresholded per marker).
    """
    logger.info("Step 4b: MIPHEI multiplex cell type prediction...")

    logger.info(f"Step 4b.1: MIPHEI-ViT multiplex channel prediction (checkpoint_dir={MIPHEI_CHECKPOINT_DIR})...")
    ome_tiff_path = miphei_utils.predict_multiplex_channels(
        slide_path=slide_path,
        checkpoint_dir=str(MIPHEI_CHECKPOINT_DIR),
        output_dir=output_dir,
        repo_root=MIPHEI_VIT_REPO_ROOT,
    )

    logger.info(f"Step 4b.2: CellViT-plus-plus binary cell detection (model_path={CELLVIT_CHECKPOINT_PATH})...")
    cellvit_json_path = cellvit_utils.detect_cells_binary(
        slide_path=slide_path,
        model_path=str(CELLVIT_CHECKPOINT_PATH),
        output_dir=output_dir,
        repo_root=CELLVIT_REPO_ROOT,
        slide_mpp=mpp,
        gpu=gpu,
    )
    with open(cellvit_json_path) as f:
        detected_cells = json.load(f)["cells"]
    logger.info(f"CellViT-plus-plus detected {len(detected_cells)} cells.")

    logger.info("Step 4b.3: Marker extraction, calibration, and hierarchy-based cell typing...")
    contours = [cell["contour"] for cell in detected_cells]
    labeled_cell_mask = convert_contours_to_labeled_mask(contours, width=slide_width, height=slide_height)

    cell_means = extract_mean_marker_intensities(
        ome_tiff_path=ome_tiff_path,
        labeled_cell_mask=labeled_cell_mask,
        marker_channels=MIPHEI_MARKER_CHANNELS_FULL,
    )
    binarized = binarize_markers(
        cell_means=cell_means,
        calibration_classifier_path=str(MIPHEI_CALIBRATION_CLASSIFIER_PATH),
        marker_channels=MIPHEI_MARKER_CHANNELS_FULL,
        nucleus_marker=MIPHEI_NUCLEUS_MARKER,
    )
    annotated = annotate_cell_hierarchy(binarized, MIPHEI_MULTIPLEX_HIERARCHY)

    # Merge level_1/level_2/.../full_path back onto each detected cell by CellID (1-based index
    # into `detected_cells`, matching the order contours were passed to
    # convert_contours_to_labeled_mask).
    level_columns = [c for c in annotated.columns if c.startswith("level_") or c == "full_path"]
    cells_output = []
    for cell_id, cell in enumerate(detected_cells, start=1):
        if cell_id not in annotated.index:
            continue  # dropped upstream (e.g. by a nuclear-marker filter)
        row = annotated.loc[cell_id]
        cells_output.append({
            "centroid": cell["centroid"],
            "contour": cell["contour"],
            "score": cell.get("score", cell.get("type_prob")),
            **{col: row[col] for col in level_columns},
        })

    miphei_cell_json_path = os.path.join(output_dir, "miphei_cell_type_predictions.json")
    with open(miphei_cell_json_path, "w") as f:
        json.dump({"cells": cells_output}, f)

    level_1_counts = pd.Series([c["level_1"] for c in cells_output]).value_counts().to_dict()
    logger.info(f"MIPHEI multiplex cell type prediction done. {len(cells_output)} cells typed: {level_1_counts}")
    logger.info(f"Cell predictions saved: {miphei_cell_json_path}")
    return miphei_cell_json_path


def compute_spatial_features(slide_path, tissue_mask_path, malignant_mask_path, geojson_path, mpp, output_dir, cell_json_path=None, cell_hierarchy_json_path=None):
    """Step 5: compute spatial TIME (tumor immune microenvironment) features and generate the CSV + PDF report.

    Exactly one of `cell_json_path` (Step 4, morphology_based cell typing) or
    `cell_hierarchy_json_path` (Step 4b, miphei_multiplex marker-based cell typing) must be given.
    """
    logger.info("Step 5: Spatial TIME feature computation + report generation...")
    logger.info(f"Using tissue mask: {tissue_mask_path}" + (f", malignant mask: {malignant_mask_path}" if malignant_mask_path else " (no malignant mask; skipping that visualization)"))

    extra_kwargs = {}
    if cell_hierarchy_json_path is not None:
        level_categories = hierarchy_level_categories(MIPHEI_MULTIPLEX_HIERARCHY)
        extra_kwargs = dict(
            tumor_cell_label="Tumour",
            level_categories=level_categories,
        )

    computer = SpatialFeatureComputer(
        slide_path=slide_path,
        cell_json_path=cell_json_path,
        cell_hierarchy_json_path=cell_hierarchy_json_path,
        tissue_mask_path=tissue_mask_path,
        output_directory=output_dir,
        mpp=mpp,
        malignant_mask_path=malignant_mask_path,
        tissue_contour_geojson_path=geojson_path,
        **extra_kwargs,
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
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run the whole pipeline on (Trident, tissue compartment, cell type, miphei_multiplex).")
    parser.add_argument(
        "--use_malignant_region", action=argparse.BooleanOptionalAction, default=True,
        help="Run malignant/non-malignant region identification (Step 2) and restrict the tissue-compartment "
             "mask to it. Pass --no-use_malignant_region to skip Step 2 and use the raw tissue-compartment "
             "mask directly.",
    )
    parser.add_argument(
        "--cell_type_method", choices=["morphology_based", "miphei_multiplex", "both"], default="morphology_based",
        help="Which cell-typing method(s) to run as Step 4: 'morphology_based' (default) - "
             "morphology-based typing from the H&E via Mask2Former; 'miphei_multiplex' - "
             "protein-marker-based typing via MIPHEI-ViT (virtual multiplex staining) + "
             "CellViT-plus-plus (binary cell detection) + marker-hierarchy annotation; "
             "'both' - run both methods, and Step 5 twice, writing to "
             "spatial_features_morphology_based/ and spatial_features_miphei_multiplex/ subdirectories.",
    )
    args = parser.parse_args()

    # GPU selection: CUDA_VISIBLE_DEVICES was already set (at module import time, above) to
    # restrict the whole process - and everything it forks/spawns, e.g. the Ray actors used by
    # the miphei_multiplex path - to a single physical GPU (--gpu, or whatever the caller had
    # already restricted visibility to, e.g. batch_infer.py's own per-subprocess masking).
    # Every step below can therefore just address "cuda:0" - the one GPU visible - without
    # needing an explicit absolute device index anywhere. This also sidesteps a failure mode
    # explicit "cuda:N" strings can't handle: CellViT-plus-plus's Ray-based postprocessing
    # spawns separate actor processes with their own restricted GPU view, where a tensor
    # created on an absolute non-zero device index in the main process can't be deserialized
    # there ("Attempting to deserialize object on CUDA device N but
    # torch.cuda.device_count() is 1").

    slide_name = Path(args.slide_path).stem
    output_dir = args.output_dir if args.output_dir is not None else os.path.join("output", slide_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = setup_logging(output_dir)
    logger.info(f"Logging to console and {log_path}")
    logger.info(f"Arguments: {vars(args)}")
    timing_tracker = TimingTracker(logger)

    device = "cuda:0" if torch.cuda.is_available() else None
    logger.info(
        f"CUDA available: {torch.cuda.is_available()}"
        + (f" (using GPU {args.gpu}, masked to {device}: {torch.cuda.get_device_name(0)}, uuid={torch.cuda.get_device_properties(0).uuid})" if torch.cuda.is_available() else "")
    )

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
    geojson_path, conch_features_path = preprocess_with_trident(slide_path, output_dir, final_mpp, gpu=0)
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
    cell_json_path = None
    cell_hierarchy_json_path = None
    num_cells_detected = 0

    if args.cell_type_method in ("morphology_based", "both"):
        timing_tracker.start_step("4. Cell type prediction (morphology_based)")
        cell_json_path, centroids, contours, labels, scores = predict_cell_type(slide_path, geojson_path, final_mpp, device, output_dir)
        num_cells_detected = len(centroids)
        timing_tracker.end_step("4. Cell type prediction (morphology_based)")

    if args.cell_type_method in ("miphei_multiplex", "both"):
        timing_tracker.start_step("4b. Cell type prediction (miphei_multiplex)")
        cell_hierarchy_json_path = predict_miphei_multiplex_cell_type(slide_path, width, height, final_mpp, output_dir, gpu=0)
        with open(cell_hierarchy_json_path) as f:
            num_cells_detected = max(num_cells_detected, len(json.load(f)["cells"]))
        timing_tracker.end_step("4b. Cell type prediction (miphei_multiplex)")

    # Step 5
    timing_tracker.start_step("5. Spatial feature computation")
    if args.cell_type_method == "both":
        compute_spatial_features(
            slide_path=slide_path,
            cell_json_path=cell_json_path,
            tissue_mask_path=tissue_mask_combined_path,
            malignant_mask_path=malignant_mask_path,
            geojson_path=geojson_path,
            mpp=final_mpp,
            output_dir=os.path.join(output_dir, "spatial_features_morphology_based"),
        )
        compute_spatial_features(
            slide_path=slide_path,
            cell_hierarchy_json_path=cell_hierarchy_json_path,
            tissue_mask_path=tissue_mask_combined_path,
            malignant_mask_path=malignant_mask_path,
            geojson_path=geojson_path,
            mpp=final_mpp,
            output_dir=os.path.join(output_dir, "spatial_features_miphei_multiplex"),
        )
    else:
        compute_spatial_features(
            slide_path=slide_path,
            cell_json_path=cell_json_path,
            cell_hierarchy_json_path=cell_hierarchy_json_path,
            tissue_mask_path=tissue_mask_combined_path,
            malignant_mask_path=malignant_mask_path,
            geojson_path=geojson_path,
            mpp=final_mpp,
            output_dir=output_dir,
        )
    timing_tracker.end_step("5. Spatial feature computation")

    logger.info("Pipeline complete.")
    logger.info(f"Tissue compartment labels: {TISSUE_ID2LABEL}")
    logger.info(f"Cell-type method(s): {args.cell_type_method}" + (f" (morphology_based labels: {CELL_ID2LABEL})" if cell_json_path else ""))
    logger.info(f"Detected cells: {num_cells_detected}")
    logger.info(f"All outputs saved to: {output_dir}")
    timing_tracker.log_summary()


if __name__ == "__main__":
    main()
