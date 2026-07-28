import logging
from pathlib import Path

import openslide

from .czi_reader import CZIReader
from .tiff_wsi_reader import TiffWSIReader


def prepare_openslide_wsi(input_file_path, input_dir, user_input_mpp=None):
    """
    Ensure a slide is readable by OpenSlide, converting it first if needed.

    Mirrors process.py's "Slide format checking and conversion" step: try
    OpenSlide directly (covers .svs, .ome.tiff/.tiff, .ndpi, etc.); if that
    fails, convert to a pyramidal SVS via CZIReader (.czi) or TiffWSIReader
    (any other raster format, e.g. plain .tiff/.png/.jpg) and reopen.

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
        logging.info(f"Attempting to open slide with OpenSlide: {input_file_path}")
        slide = openslide.OpenSlide(str(input_file_path))
        logging.info("Slide opened successfully with OpenSlide")
    except Exception as exc:
        logging.warning(f"OpenSlide could not open {input_file_path}: {exc}")
        logging.info("Converting slide to SVS format...")
        suffix = input_file_path.suffix.lower()

        if suffix == ".czi":
            logging.info("Detected CZI format, converting...")
            reader = CZIReader(str(input_file_path))
            reader.save_as_svs(output_dir=input_dir)
        else:
            logging.info(f"Detected {suffix or 'unknown'} format, converting...")
            reader = TiffWSIReader(str(input_file_path))
            try:
                custom_mpp = float(user_input_mpp)
            except (TypeError, ValueError):
                custom_mpp = 0.25  # default MPP
            reader.save_as_svs(output_dir=input_dir, custom_mpp=custom_mpp)

        input_file_path = (Path(input_dir) / input_file_path.name).with_suffix(".svs")
        logging.info(f"Opening converted slide: {input_file_path}")
        slide = openslide.OpenSlide(str(input_file_path))

    micron_per_pixel = slide.properties.get("aperio.MPP", "Unknown")
    try:
        final_mpp = float(user_input_mpp)
        logging.info(f"Using user-provided MPP: {final_mpp}")
    except (TypeError, ValueError):
        if micron_per_pixel != "Unknown":
            final_mpp = float(micron_per_pixel)
            logging.info(f"Using image MPP: {final_mpp}")
        else:
            final_mpp = 0.25
            logging.info(f"Using default MPP: {final_mpp}")

    slide.close()
    return str(input_file_path), final_mpp
