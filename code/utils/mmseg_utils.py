import os
from pathlib import Path

import numpy as np
from PIL import Image
from mmseg.apis.inference import init_model, inference_model

from .wsi_patcher import WSIPatcher


def model_fn(
    config_file_path: str,
    checkpoint_file_path: str,
    device: str = "cuda:0",
):
    """
    Initialize the tissue-compartment segmentation model for inference.

    Args:
        config_file_path (str): Path to the model configuration file.
        checkpoint_file_path (str): Path to the model checkpoint file.
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.

    Returns:
        model: Initialized model ready for inference.
    """
    return init_model(config_file_path, checkpoint_file_path, device=device)


def infer_single_wsi(wsi_path, model, tile_size=1024, output_dir="output", contours_geojson_path=None):
    """
    Performs semantic segmentation inference on a whole slide image (WSI) using a tile-by-tile
    approach, reconstructs the full-resolution prediction mask, and saves the result as a PNG.

    Returns:
        np.ndarray: Single-channel mask where pixel values are predicted tissue-compartment
        class indices, or None if the output file already existed and was skipped.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, Path(wsi_path).stem + ".png")
    if os.path.exists(output_file_path):
        print("Output file path exists, ending processing.")
        return None

    tile_processor = WSIPatcher(wsi_path, tile_size, contours_geojson_path=contours_geojson_path)
    width, height = tile_processor.width, tile_processor.height

    mask = np.zeros((height, width), dtype=np.uint8)

    for idx in range(len(tile_processor)):
        tile, coord = tile_processor[idx]
        x1, y1, x2, y2, pad_right, pad_bottom = coord

        result = inference_model(model, tile)
        tile_mask = result.pred_sem_seg.data.cpu().numpy().astype("uint8")[0]

        mask[y1:y2, x1:x2] = tile_mask[:tile_mask.shape[0] - pad_bottom, :tile_mask.shape[1] - pad_right]

    Image.fromarray(mask).save(output_file_path)

    return mask
