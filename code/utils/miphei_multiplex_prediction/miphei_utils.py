"""
Thin wrapper around MIPHEI-ViT (virtual multiplex immunofluorescence staining from H&E),
imported in-process from its repo checkout (no packaging, so the repo root is registered
on `sys.path` rather than pip-installed).
"""
import logging
from pathlib import Path

from . import slidevips_mpp_patch

logger = logging.getLogger(__name__)


def predict_multiplex_channels(
    slide_path: str,
    checkpoint_dir: str,
    output_dir: str,
    repo_root: str,
    mpp_target: float = 0.5,
    level: int = 0,
    tile_size: int = 256,
    tile_overlap: int = 10,
    batch_size: int = 4,
    num_workers: int = 0,
) -> str:
    """
    Predict multiplex immunofluorescence (mIF) channels from an H&E slide using MIPHEI-ViT.

    `mpp_target` is the fixed resampling resolution MIPHEI-ViT tiles are fed to the model at
    (a model-calibration constant, not the slide's own MPP) — leave it at the default unless
    the checkpoint was specifically trained at a different target resolution.

    `num_workers` defaults to 0 (no worker subprocesses), not MIPHEI-ViT's own `-1` sentinel
    (which resolves to `os.cpu_count() - 1` DataLoader workers) or any other positive value.
    By the time this runs, `infer.py` has already initialized CUDA in-process for earlier steps
    (Trident, tissue-compartment, cell-type prediction); `run_wsi_inference.py`'s DataLoader is
    constructed with no `multiprocessing_context`, so `num_workers > 0` forks worker processes
    with the parent's CUDA context already live. That's a documented PyTorch/CUDA hazard, and it
    doesn't just risk a crash - it manifests as forked workers silently deadlocking at 0% CPU
    after the very first batch (confirmed via a stuck production batch run: three concurrent
    slides all hung at tile 1/N for 13-24h, each with 8 `pt_data_worker` children parked on a
    futex, GPU utilization at 0%). It's timing-dependent - more likely under concurrent
    multi-process GPU load - which is why a single-slide test run may not reproduce it. 0 avoids
    the fork entirely by loading tiles on the main process; pass an explicit positive value only
    if you've verified it's safe in your deployment (e.g. always single-slide, no prior CUDA use).

    Returns:
        str: Path to the written pyramidal OME-TIFF (`{slide_stem}.ome.tiff`), with channels
        named per the checkpoint's `config.yaml: data.targ_channel_names`.
    """
    slidevips_mpp_patch.apply(repo_root)
    from run_wsi_inference import wsi_inference

    logger.info(
        f"Running MIPHEI-ViT multiplex channel prediction "
        f"(checkpoint_dir={checkpoint_dir}, mpp_target={mpp_target}, tile_size={tile_size}, tile_overlap={tile_overlap})"
    )
    wsi_inference(
        slide_path=slide_path,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        num_workers=num_workers,
        level=level,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        mpp_target=mpp_target,
    )

    ome_tiff_path = str(Path(output_dir) / f"{Path(slide_path).stem}.ome.tiff")
    if not Path(ome_tiff_path).exists():
        raise FileNotFoundError(f"MIPHEI-ViT did not produce the expected output: {ome_tiff_path}")
    logger.info(f"MIPHEI-ViT multiplex prediction done: {ome_tiff_path}")
    return ome_tiff_path
