import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Mask2Former checkpoints store extra non-tensor objects; torch>=2.6 defaults
# to `weights_only=True` on load and refuses to unpickle them.
_real_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from torch.utils.data import DataLoader

from mmcv.ops import batched_nms
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import SampleList

from .wsi_patcher import WSIPatcher


def model_fn(
    config_file_path: str,
    checkpoint_file_path: str,
    device: str = "cuda:0",
):
    """
    Initialize the cell-type (instance segmentation) model for inference.

    Args:
        config_file_path (str): Path to the model configuration file.
        checkpoint_file_path (str): Path to the model checkpoint file.
        device (str): Device to run the model on, e.g., 'cuda:0' or 'cpu'.

    Returns:
        model: Initialized model ready for inference.
    """
    model = init_detector(config_file_path, checkpoint_file_path, device=device)
    max_per_image_setting = model.cfg.model.test_cfg["max_per_image"]
    if max_per_image_setting < 300:
        print(f"WARNING! Max proposals per image: {max_per_image_setting}")
    return model


def predict(
    model,
    imgs: np.ndarray,
    nms_cfg=dict(iou_threshold=0.5, type="nms"),
):
    """
    Run inference on a single image or a batch of images.

    Args:
        model: The initialized model.
        imgs (np.ndarray): A single image (H, W, C) or a batch (N, H, W, C).

    Returns:
        result(s): Inference result(s) from the model, after NMS.
    """
    if not isinstance(imgs, np.ndarray):
        raise ValueError("Input imgs should be a numpy array.")

    if imgs.ndim == 3:
        result = inference_detector(model, imgs)
        _, keeps = batched_nms(
            boxes=result.pred_instances.bboxes,
            scores=result.pred_instances.scores,
            idxs=torch.zeros_like(result.pred_instances.labels),
            nms_cfg=nms_cfg)
        result.pred_instances = result.pred_instances[keeps]
        return result
    elif imgs.ndim != 4:
        raise ValueError("Input imgs should be a 3D or 4D array.")

    from mmdet.utils import get_test_pipeline_cfg
    from mmcv.transforms import Compose

    cfg = model.cfg
    cfg = cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

    test_pipeline = Compose(test_pipeline)

    data = {
        "inputs": [],
        "data_samples": [],
    }

    for _, img in enumerate(imgs):
        data_ = dict(img=img, img_id=0)
        data_ = test_pipeline(data_)

        data["inputs"].append(data_["inputs"])
        data["data_samples"].append(data_["data_samples"])

    with torch.no_grad():
        results = model.test_step(data)

    for result in results:
        _, keeps = batched_nms(
            boxes=result.pred_instances.bboxes,
            scores=result.pred_instances.scores,
            idxs=torch.zeros_like(result.pred_instances.labels),
            nms_cfg=dict(iou_threshold=0.5, type="nms"))
        result.pred_instances = result.pred_instances[keeps]

    return results


def convert_masks_to_contours(masks: np.ndarray) -> List[np.ndarray]:
    """
    Convert a batch of binary masks to their largest external contours.

    Args:
        masks (np.ndarray): Binary masks of shape (N, H, W). Non-zero pixels are treated as foreground.

    Returns:
        List[np.ndarray]: A list of contours (each of shape (n_points, 2)) for each mask that has at least one contour.
                          Masks without contours are skipped.
    """
    contours_list = []
    valid_mask_indices = []

    for i in range(masks.shape[0]):
        mask = masks[i]
        mask_uint8 = (mask.astype(np.uint8)) * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour = largest_contour.squeeze(1)

        if len(largest_contour) < 3:
            continue

        contours_list.append(largest_contour)
        valid_mask_indices.append(i)

    return contours_list, valid_mask_indices


def get_centroid_from_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        raise ValueError

    return [cX, cY]


def shift_bboxes(bboxes: Union[np.ndarray, list], offset: Sequence[int]):
    """
    Shift bboxes w.r.t offset. Bboxes are in the format (x1, y1, x2, y2).
    """
    dx, dy = offset

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    elif isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

    shifted = bboxes.copy()
    shifted[:, 0] += dx
    shifted[:, 1] += dy
    shifted[:, 2] += dx
    shifted[:, 3] += dy

    if isinstance(bboxes, list):
        return shifted.tolist()
    return shifted


def shift_contours(contours: Union[np.ndarray, list], offset: Sequence[int]):
    """
    Shift contours (each a list/array of (x, y) points) w.r.t offset.
    """
    dx, dy = offset

    shifted_contours = []
    for contour in contours:
        if isinstance(contour, list):
            contour = np.array(contour)

        shifted = contour.copy()
        shifted[:, 0] += dx
        shifted[:, 1] += dy

        shifted_contours.append(shifted)

    return shifted_contours


def postprocess_results(
    results: SampleList,
    coords: Sequence[Tuple[int, int, int, int, int, int]],
    nms_cfg: dict = dict(iou_threshold=0.5, type="nms"),
) -> Dict:
    """
    Postprocess instance segmentation results from tiled inference on whole-slide images (WSIs).

    Consolidates predictions from multiple image tiles into a unified set of cell detections:
    1. Tile-level NMS & filtering (discard predictions in padded areas, classify cells as
       "margin" vs "center").
    2. Cross-tile merging (NMS on margin cells to remove duplicate detections at overlapping
       tile borders).
    3. Global overlap cleaning (`OverlapCellCleaner`) to remove duplicated cells across tiles.
    4. Output formatting into a JSON-style dict with centroids, bboxes, contours, labels, scores.
    """
    json_output = {
        "cells": [],
    }

    def _touch_margin(bbox, tile_size):
        x1, y1, x2, y2 = bbox
        H, W = tile_size

        margin_x = W / 8
        margin_y = H / 8

        if x1 <= margin_x or x2 >= W - margin_x or y1 <= margin_y or y2 >= H - margin_y:
            return True
        return False

    # Step 1
    all_cells = {
        "bboxes": [],
        "contours": [],
        "labels": [],
        "scores": [],
        "cell_status": [],
    }

    for det_data_sample, coord in zip(results, coords):
        pred_inst = det_data_sample.pred_instances
        x1, y1, x2, y2, pad_right, pad_bottom = coord

        pred_masks = pred_inst["masks"].numpy()
        pred_labels = pred_inst["labels"].numpy()
        pred_scores = pred_inst["scores"].numpy()
        pred_bboxes = pred_inst["bboxes"].numpy()

        image_height, image_width = pred_masks[0].shape

        cell_status = np.array([_touch_margin(bbox, (image_height, image_width)) for bbox in pred_bboxes]).astype(int)

        valid_mask = (pred_bboxes[:, 0] < image_width - pad_right) & (pred_bboxes[:, 1] < image_height - pad_bottom) & (pred_bboxes[:, 2] < image_width - pad_right) & (pred_bboxes[:, 3] < image_height - pad_bottom) & (pred_bboxes[:, 0] < pred_bboxes[:, 2]) & (pred_bboxes[:, 1] < pred_bboxes[:, 3])
        pred_masks = pred_masks[valid_mask]
        pred_labels = pred_labels[valid_mask]
        pred_scores = pred_scores[valid_mask]
        pred_bboxes = pred_bboxes[valid_mask]
        cell_status = cell_status[valid_mask]

        if "masks" in pred_inst and pred_inst.masks.numel() > 0:
            contours, valid_mask_indices = convert_masks_to_contours(pred_masks)
            all_cells["contours"].append(shift_contours(contours, (x1, y1)))

            all_cells["bboxes"].append(shift_bboxes(pred_bboxes[valid_mask_indices], (x1, y1)))
            all_cells["labels"].append(pred_labels[valid_mask_indices])
            all_cells["scores"].append(pred_scores[valid_mask_indices])
            all_cells["cell_status"].append(cell_status[valid_mask_indices])

    if len(all_cells["bboxes"]) == 0:
        return json_output

    for key, value in all_cells.items():
        if key == "contours":
            all_cells["contours"] = [item for sublist in value for item in sublist]
        else:
            all_cells[key] = np.concatenate(value, axis=0)

    # Step 2
    cells_at_margin = {
        "bboxes": all_cells["bboxes"][all_cells["cell_status"] == 1],
        "contours": [c for idx, c in enumerate(all_cells["contours"]) if all_cells["cell_status"][idx] == 1],
        "labels": all_cells["labels"][all_cells["cell_status"] == 1],
        "scores": all_cells["scores"][all_cells["cell_status"] == 1],
        "cell_status": all_cells["cell_status"][all_cells["cell_status"] == 1],
    }
    cells_kept = {
        "bboxes": all_cells["bboxes"][all_cells["cell_status"] == 0],
        "contours": [c for idx, c in enumerate(all_cells["contours"]) if all_cells["cell_status"][idx] == 0],
        "labels": all_cells["labels"][all_cells["cell_status"] == 0],
        "scores": all_cells["scores"][all_cells["cell_status"] == 0],
        "cell_status": all_cells["cell_status"][all_cells["cell_status"] == 0],
    }

    if len(cells_at_margin["bboxes"]):
        print("Running nms...")
        _, keeps = batched_nms(
            boxes=torch.Tensor(cells_at_margin["bboxes"]),
            scores=torch.Tensor(cells_at_margin["scores"]),
            idxs=torch.zeros(cells_at_margin["labels"].shape),
            nms_cfg=nms_cfg)

        for key, value in cells_kept.items():
            if key == "contours":
                cells_kept["contours"] += [cells_at_margin["contours"][i] for i in keeps]
            else:
                if len(keeps) == 1:
                    cells_kept[key] = np.concatenate([cells_kept[key], [cells_at_margin[key][keeps]]], axis=0)
                else:
                    cells_kept[key] = np.concatenate([cells_kept[key], cells_at_margin[key][keeps]], axis=0)

    print(f"After nms: {len(cells_kept['bboxes'])} cells remaining from {len(all_cells['bboxes'])}")
    del all_cells
    del cells_at_margin

    # Step 3
    from .overlap_cell_cleaner import OverlapCellCleaner
    from logging import Logger

    cell_list = []

    for bbox, contour, label, score, status in zip(
        cells_kept["bboxes"],
        cells_kept["contours"],
        cells_kept["labels"],
        cells_kept["scores"],
        cells_kept["cell_status"],
    ):
        if len(contour) < 3:
            continue
        cell_list.append({
            "contour": contour,
            "bbox": bbox,
            "type": label,
            "type_prob": score,
            "cell_status": status,
            "patch_coordinates": (0, 0),  # dummy
            "edge_position": 0,  # dummy
        })

    del cells_kept

    if len(cell_list) > 0:
        print("Cleaning overlapping cells...")
        overlap_cell_cleaner = OverlapCellCleaner(
            cell_list=cell_list,
            logger=Logger(name=""),
        )
        cleaned_cells = overlap_cell_cleaner.clean_detected_cells()
        keep_idx = list(cleaned_cells.index.values)
        cleaned_cell_list = [cell_list[idx_c] for idx_c in keep_idx]
        print(f"After cleaning overlap: {len(cleaned_cell_list)} cells remaining from {len(cell_list)}")
    else:
        cleaned_cell_list = []

    for cell_info in cleaned_cell_list:
        try:
            centroid = get_centroid_from_contour(cell_info["contour"])

            json_output["cells"].append({
                "centroid": centroid,
                "bbox": cell_info["bbox"].tolist(),
                "contour": cell_info["contour"].tolist(),
                "label": cell_info["type"].tolist(),
                "score": cell_info["type_prob"].tolist()
            })
        except ValueError:
            continue

    print(f"Final valid cells: {len(json_output['cells'])} cells.")

    return json_output


def infer_single_wsi(wsi_path, model, tile_size=512, output_dir="output", batch_size=16, chunk_size=2048, contours_geojson_path=None):
    """
    Performs instance segmentation inference on a whole slide image (WSI) using a tiled sliding-window approach,
    aggregates model predictions across tiles, and writes the final cell-level outputs to a JSON file.

    Returns:
        str: Path to the final JSON file containing aggregated cell-level predictions
             (each entry has "centroid", "bbox", "contour", "label", "score").
    """

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, Path(wsi_path).stem + ".json")
    temp_output_file_path = os.path.join(output_dir, Path(wsi_path).stem + "_temp.jsonl")

    if os.path.exists(output_file_path):
        print("Output file path exists, ending processing.")
        return output_file_path
    if os.path.exists(temp_output_file_path):
        os.remove(temp_output_file_path)

    tile_processor = WSIPatcher(wsi_path, tile_size, overlap=tile_size // 8, contours_geojson_path=contours_geojson_path)

    coords = []
    results = []

    chunk_number = 1
    tile_dataloader = DataLoader(tile_processor, batch_size=batch_size, shuffle=False)
    for batch in tile_dataloader:
        batch_tiles, batch_coords = batch
        batch_results = predict(model, batch_tiles.numpy())

        coords.extend(batch_coords.tolist())
        results.extend([r.cpu() for r in batch_results])

        if len(coords) >= chunk_size:
            print(f"Postprocessing chunk {chunk_number}...")
            json_output = postprocess_results(results[:-tile_processor.number_of_tiles_per_column], coords[:-tile_processor.number_of_tiles_per_column])

            print(f"Saving temp output chunk {chunk_number}...")
            with open(temp_output_file_path, "a") as f:
                f.write(json.dumps(json_output) + "\n")

            chunk_number += 1

            coords = coords[-tile_processor.number_of_tiles_per_column:]
            results = results[-tile_processor.number_of_tiles_per_column:]

    if coords:
        print(f"Postprocessing last chunk {chunk_number}...")
        json_output = postprocess_results(results, coords)

        print(f"Saving temp output last chunk {chunk_number}...")
        with open(temp_output_file_path, "a") as f:
            f.write(json.dumps(json_output) + "\n")

    print("Saving output...")
    json_output = {
        "cells": []
    }
    with open(temp_output_file_path, "r") as f:
        for line in f:
            json_output["cells"].extend(json.loads(line)["cells"])

    with open(output_file_path, "w") as f:
        json.dump(json_output, f)

    if os.path.exists(temp_output_file_path):
        os.remove(temp_output_file_path)

    return output_file_path
