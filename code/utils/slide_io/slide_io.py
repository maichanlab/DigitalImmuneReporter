import logging
from pathlib import Path

import openslide
import pyvips

from .. import pyvips_pyramid_tiffsave_patch
from .czi_reader import CZIReader
from .tiff_wsi_reader import TiffWSIReader

logger = logging.getLogger(__name__)

# Several models in this pipeline assume roughly this MPP or finer (MIPHEI-ViT's
# mpp_target=0.5, CellViT's resolution=0.25, the mmseg/mmdet checkpoints trained at 0.5mpp).
# A slide coarser than this gets upscaled to it (see _rescale_to_target_mpp below) so those
# assumptions hold regardless of the input's native/requested resolution.
TARGET_MPP_IF_COARSER = 0.5


def prepare_openslide_wsi(input_file_path, input_dir, user_input_mpp=None):
    """
    Ensure a slide is readable by OpenSlide, converting it first if needed.

    Mirrors process.py's "Slide format checking and conversion" step: try
    OpenSlide directly (covers .svs, .ome.tiff/.tiff, .ndpi, etc.); if that
    fails, convert to a pyramidal SVS via CZIReader (.czi) or TiffWSIReader
    (any other raster format, e.g. plain .tiff/.png/.jpg) and reopen. If the
    resolved MPP is coarser than TARGET_MPP_IF_COARSER, the slide is then
    upscaled to that resolution (see _rescale_to_target_mpp).

    Every downstream step (Trident preprocessing, malignant-region,
    tissue-compartment, cell-type prediction) then operates on the same
    OpenSlide-readable file and pixel grid.

    Args:
        input_file_path (str): Path to the input slide/image file.
        input_dir (str): Directory to write a converted .svs into, if conversion is needed.
        user_input_mpp (float, optional): User-provided microns-per-pixel override.

    Returns:
        Tuple[str, float]: (path to an OpenSlide-readable file, resolved MPP)
    """
    input_file_path = Path(input_file_path)
    try:
        logger.info(f"Attempting to open slide with OpenSlide: {input_file_path}")
        slide = openslide.OpenSlide(str(input_file_path))
        logger.info("Slide opened successfully with OpenSlide")
    except Exception as exc:
        logger.warning(f"OpenSlide could not open {input_file_path}: {exc}")
        logger.info("Converting slide to SVS format...")
        suffix = input_file_path.suffix.lower()

        if suffix == ".czi":
            logger.info("Detected CZI format, converting...")
            reader = CZIReader(str(input_file_path))
            reader.save_as_svs(output_dir=input_dir)
        else:
            logger.info(f"Detected {suffix or 'unknown'} format, converting...")
            reader = TiffWSIReader(str(input_file_path))
            try:
                custom_mpp = float(user_input_mpp)
            except (TypeError, ValueError):
                custom_mpp = 0.25  # default MPP
            reader.save_as_svs(output_dir=input_dir, custom_mpp=custom_mpp)

        input_file_path = (Path(input_dir) / input_file_path.name).with_suffix(".svs")
        logger.info(f"Opening converted slide: {input_file_path}")
        slide = openslide.OpenSlide(str(input_file_path))

    micron_per_pixel = slide.properties.get("aperio.MPP", "Unknown")
    try:
        final_mpp = float(user_input_mpp)
        logger.info(f"Using user-provided MPP: {final_mpp}")
    except (TypeError, ValueError):
        if micron_per_pixel != "Unknown":
            final_mpp = float(micron_per_pixel)
            logger.info(f"Using image MPP: {final_mpp}")
        else:
            final_mpp = 0.25
            logger.info(f"Using default MPP: {final_mpp}")

    slide.close()

    if final_mpp > TARGET_MPP_IF_COARSER:
        logger.warning(
            f"Slide MPP ({final_mpp}) is coarser than {TARGET_MPP_IF_COARSER}. Several models in "
            f"this pipeline (MIPHEI-ViT, CellViT, the tissue-compartment/cell-type segmentation "
            f"checkpoints) assume roughly {TARGET_MPP_IF_COARSER} MPP or finer input. Upscaling "
            f"the slide to {TARGET_MPP_IF_COARSER} MPP so downstream steps run at a consistent, "
            f"expected resolution - this resamples the existing pixels, it does not add real "
            f"detail beyond what the slide was originally scanned at."
        )
        input_file_path, final_mpp = _rescale_to_target_mpp(
            input_file_path, final_mpp, input_dir, TARGET_MPP_IF_COARSER
        )

    return str(input_file_path), final_mpp


def _rescale_to_target_mpp(slide_path, current_mpp, output_dir, target_mpp):
    """
    Resample a slide so its MPP becomes `target_mpp` (only called when `current_mpp` is coarser,
    i.e. this always upscales), writing a new pyramidal SVS and returning its path.

    Uses `pyvips.Image.openslideload` explicitly (not the generic `pyvips.Image.new_from_file`,
    whose own format auto-detection has repeatedly picked the wrong loader for slides converted
    elsewhere in this module - see slidevips_mpp_patch.py) so this reads via the same
    `libopenslide` backend that already proved capable of opening `slide_path` above.

    Returns:
        Tuple[Path, float]: (path to the rescaled .svs, target_mpp)
    """
    pyvips_pyramid_tiffsave_patch.apply()

    scale = current_mpp / target_mpp
    output_path = Path(output_dir) / f"{Path(slide_path).stem}_mpp{target_mpp}.svs"

    vips_slide = pyvips.Image.openslideload(str(slide_path), level=0, access="sequential")
    if vips_slide.bands > 3:
        vips_slide = vips_slide[:3]  # drop the alpha band OpenSlide adds, matching every other
        # conversion path in this module (CZIReader/TiffWSIReader both write plain RGB)

    logger.info(f"Rescaling slide {scale:.3f}x ({vips_slide.width}x{vips_slide.height} at {current_mpp} MPP)...")
    scaled = vips_slide.resize(scale)

    # pixels per cm: 1 cm = 10,000 microns (see tiff_wsi_reader.py's identical derivation).
    xres = 10000 / target_mpp
    yres = 10000 / target_mpp
    scaled.tiffsave(
        str(output_path),
        compression="jpeg",
        Q=80,
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        bigtiff=True,
        properties=True,
        xres=xres,
        yres=yres,
        resunit="cm",
    )
    logger.info(f"Rescaled slide saved: {output_path} ({scaled.width}x{scaled.height} at {target_mpp} MPP)")
    return output_path, target_mpp
