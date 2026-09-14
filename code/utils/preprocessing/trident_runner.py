import logging
import os

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

logger = logging.getLogger(__name__)


def run_trident_preprocessing(
    slide_path,
    job_dir,
    mpp,
    patch_encoder_ckpt_path,
    gpu=0,
    patch_encoder="conch_v1",
    patch_size=512,
    mag=20,
    segmenter="hest",
    seg_conf_thresh=0.5,
    remove_holes=True,
    overlap=0,
    batch_size=32,
):
    """
    Step 1 of the pipeline: tissue/background segmentation + patch-level
    foundation-model feature extraction via Trident, for a single WSI.

    Mirrors the invocation used in process.py:
        run_single_slide --segmenter hest --remove_holes --patch_encoder conch_v1
                          --patch_size 512 --mag 20

    Args:
        slide_path (str): Path to an OpenSlide-readable WSI.
        job_dir (str): Directory to store Trident outputs (contours, coords, features).
        mpp (float): Microns-per-pixel of the slide.
        patch_encoder_ckpt_path (str): Local checkpoint path for the patch encoder
            (loaded only from model_weights/, never downloaded).
        gpu (int): GPU index to run segmentation/encoding on.
        patch_encoder (str): Trident patch encoder name (default: "conch_v1").
        patch_size (int): Patch size at which coords/features are extracted.
        mag (int): Magnification at which patches/features are extracted.
        segmenter (str): Tissue-vs-background segmenter ("hest" or "grandqc").
        seg_conf_thresh (float): Confidence threshold for the tissue segmenter.
        remove_holes (bool): Whether holes inside tissue contours are excluded from tissue.
        overlap (int): Absolute patch overlap in pixels.
        batch_size (int): Batch size for feature extraction.

    Returns:
        Tuple[str, str]:
            geojson_path: tissue contour GeoJSON, used to restrict later inference to tissue.
            features_path: H5 file with patch coords + encoder features (for malignant-region prediction).
    """
    logger.info(f"Loading slide for Trident preprocessing: {slide_path} (mpp={mpp})")
    slide = load_wsi(slide_path=slide_path, lazy_init=False, mpp=mpp)

    logger.info(f"Running tissue/background segmentation (segmenter={segmenter}, conf_thresh={seg_conf_thresh})...")
    segmentation_model = segmentation_model_factory(
        model_name=segmenter,
        confidence_thresh=seg_conf_thresh,
    )
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=job_dir,
        device=f"cuda:{gpu}",
        holes_are_tissue=not remove_holes,
    )
    geojson_path = os.path.join(job_dir, "contours_geojson", f"{slide.name}.geojson")
    logger.info(f"Tissue/background segmentation done. Contours saved: {geojson_path}")

    logger.info(f"Extracting tissue patch coordinates (mag={mag}x, patch_size={patch_size}px)...")
    save_coords = os.path.join(job_dir, f"{mag}x_{patch_size}px_{overlap}px_overlap")
    slide.extract_tissue_coords(
        target_mag=mag,
        patch_size=patch_size,
        save_coords=save_coords,
    )

    logger.info(f"Extracting patch features with {patch_encoder} encoder (batch_size={batch_size})...")
    encoder = encoder_factory(patch_encoder, weights_path=patch_encoder_ckpt_path)
    encoder.eval()
    encoder.to(f"cuda:{gpu}")
    features_dir = os.path.join(save_coords, f"features_{patch_encoder}")
    slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=os.path.join(save_coords, "patches", f"{slide.name}_patches.h5"),
        save_features=features_dir,
        device=f"cuda:{gpu}",
        batch_limit=batch_size,
    )
    features_path = os.path.join(features_dir, f"{slide.name}.h5")
    logger.info(f"Patch feature extraction done. Features saved: {features_path}")

    return geojson_path, features_path
