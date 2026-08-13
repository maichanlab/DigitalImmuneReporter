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
    num_workers: int = 8,
) -> str:
    """
    Predict multiplex immunofluorescence (mIF) channels from an H&E slide using MIPHEI-ViT.

    `mpp_target` is the fixed resampling resolution MIPHEI-ViT tiles are fed to the model at
    (a model-calibration constant, not the slide's own MPP) — leave it at the default unless
    the checkpoint was specifically trained at a different target resolution.

    `num_workers` defaults to a fixed, modest value rather than MIPHEI-ViT's own `-1` sentinel
    (which resolves to `os.cpu_count() - 1` DataLoader workers): on many-core machines that
    spawns hundreds of worker processes all competing to read the same slide file, which adds
    far more contention/overhead than it saves — pass an explicit value to override.

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
