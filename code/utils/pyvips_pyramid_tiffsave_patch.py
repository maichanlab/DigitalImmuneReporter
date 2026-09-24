"""
Patches `pyvips.Image.tiffsave` to fall back to a non-pyramidal write if a pyramidal one fails.

MIPHEI-ViT's `run_wsi_inference.py` always writes its output multiplex OME-TIFF with
`pyramid=True, subifd=True, tile_width=512, tile_height=512` (needed so large WSIs stay viewable
at multiple zoom levels in QuPath etc.). For a small input - e.g. a single H&E tile run through
this pipeline, converted to a tiny "WSI" by slide_io.TiffWSIReader - the output channel mosaic
can end up small relative to that fixed tile size, and libvips/libtiff's SubIFD-based pyramid
writer breaks outright (`TIFFWriteDirectorySec: Cannot find SubIFD tag`, `unable to call
tiffsave`) instead of just writing zero extra levels. A pyramid is meaningless for an image that
small anyway (there's nothing to zoom out to), so retrying without `pyramid`/`subifd` on that
specific failure produces a correct, equivalent flat tiled TIFF instead of crashing.

Rather than editing the external repo checkout directly (it's re-fetched by setup_env.sh, so an
in-place edit wouldn't survive a fresh clone on another machine), this monkey-patches
`pyvips.Image.tiffsave` at runtime - same "patch before use" approach as `torch_load_patch.py` /
`slidevips_mpp_patch.py`. Call `apply()` before any pyramidal `tiffsave()` that might hit this -
currently `miphei_utils.predict_multiplex_channels()` (before its call into
`run_wsi_inference.wsi_inference()`) and `slide_io.prepare_openslide_wsi()`'s coarse-MPP rescale
step (same risk: the rescaled output can still be small relative to its own tile size).
Lives at the top level of `utils/` (not under a specific step's subpackage) since it's a generic
pyvips fix needed by more than one of them.
"""
import logging

logger = logging.getLogger(__name__)

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return

    import pyvips

    original_tiffsave = pyvips.Image.tiffsave

    def _tiffsave_with_pyramid_fallback(self, *args, **kwargs):
        try:
            return original_tiffsave(self, *args, **kwargs)
        except pyvips.error.Error:
            if not kwargs.get("pyramid"):
                raise
            logger.warning(
                "Pyramidal tiffsave failed (image likely too small relative to the requested "
                "tile size to generate any pyramid levels) - retrying as a flat (non-pyramidal) "
                "TIFF instead."
            )
            fallback_kwargs = {k: v for k, v in kwargs.items() if k not in ("pyramid", "subifd")}
            return original_tiffsave(self, *args, **fallback_kwargs)

    pyvips.Image.tiffsave = _tiffsave_with_pyramid_fallback
    _applied = True
