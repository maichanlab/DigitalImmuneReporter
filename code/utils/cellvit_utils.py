"""
Thin wrapper around CellViT-plus-plus (binary cell detection), imported in-process from its
repo checkout (no packaging, so the repo root is registered on `sys.path` rather than
pip-installed).
"""
import logging
import os
import sys
from pathlib import Path

from . import pathopatch_pydantic_patch, cellvit_shapely2_patch, pathopatch_no_tissue_filter_patch, cellvit_logging_patch

logger = logging.getLogger(__name__)


def _register_repo_on_path(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def detect_cells_binary(
    slide_path: str,
    model_path: str,
    output_dir: str,
    repo_root: str,
    slide_mpp: float,
    gpu: int = 0,
    resolution: float = 0.25,
    batch_size: int = 8,
    patch_size: int = 1024,
    overlap: int = 64,
) -> str:
    """
    Detect cells (binary: cell vs. background, no typing) on an H&E slide using CellViT-plus-plus.

    `resolution` is the fixed target resolution CellViT resamples tiles to before running the
    model (must be 0.25 or 0.5, a model-calibration constant); `slide_mpp` is the slide's own
    MPP, used to compute the resampling ratio relative to `resolution`.

    Returns:
        str: Path to the written `{slide_stem}_cells.json` (cells with `centroid`, `contour`,
        `bbox`, `type` (always 1="Cell" in binary mode), `type_prob`).
    """
    _register_repo_on_path(repo_root)
    from cellvit.inference.inference_memory import CellViTInferenceMemory
    pathopatch_pydantic_patch.apply()
    cellvit_shapely2_patch.apply()
    pathopatch_no_tissue_filter_patch.apply()
    cellvit_logging_patch.apply()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"Running CellViT-plus-plus binary cell detection "
        f"(model_path={model_path}, resolution={resolution}, slide_mpp={slide_mpp}, patch_size={patch_size}, overlap={overlap})"
    )

    inferer = CellViTInferenceMemory(
        model_path=model_path,
        gpu=gpu,
        outdir=output_dir,
        classifier_path=None,
        binary=True,
        batch_size=batch_size,
        patch_size=patch_size,
        overlap=overlap,
    )

    inferer.process_wsi(
        wsi_path=Path(slide_path),
        # "magnification" is required by CellViT-plus-plus's load_wsi_meta() - it raises
        # NotImplementedError if neither this nor the slide's own "openslide.objective-power"
        # metadata is present (missing on some scanners/converted formats). It's otherwise
        # unused: only target_mpp (derived from slide_mpp/resolution) drives actual processing,
        # so the standard mpp->magnification approximation (~0.25 MPP <-> 40x, ~0.5 MPP <-> 20x)
        # is safe here even though it's not exact for every scanner.
        wsi_properties={"slide_mpp": slide_mpp, "magnification": 10.0 / slide_mpp},
        resolution=resolution,
        min_intersection_ratio=0.0,  # pathopatch_no_tissue_filter_patch already forces every
        # grid tile "interesting"; this also disables the separate per-patch background-ratio
        # discard check in __getitem__, so nothing gets dropped there either.
    )

    output_stem = Path(slide_path).name.split(".")[0]
    cells_json_path = Path(output_dir) / f"{output_stem}_cells.json"
    if not cells_json_path.exists():
        raise FileNotFoundError(f"CellViT-plus-plus did not produce the expected output: {cells_json_path}")
    logger.info(f"CellViT-plus-plus binary cell detection done: {cells_json_path}")
    return str(cells_json_path)
