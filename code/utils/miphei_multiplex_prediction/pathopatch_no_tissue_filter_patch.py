"""
Disables `pathopatch`'s built-in tissue pre-filtering (used by CellViT-plus-plus's WSI
dataloader) so every patch in the slide's tiling grid gets processed, instead of only the
patches its Otsu-thresholding-based tissue detector considers "tissue".

That Otsu heuristic (a single global brightness threshold on a downscaled thumbnail) turned
out to be far more conservative than the dedicated tissue-segmentation model (Trident's HEST
segmenter) used elsewhere in this pipeline: on a benchmark slide it kept only 75 of 368 grid
tiles (~20%), causing CellViT to detect ~16k cells instead of the ~96k found when every tile
is processed — verified to be an exact match against an independently-generated reference
built by running CellViT without this restriction (see git history / PR discussion for the
comparison this patch was introduced to fix).

`min_intersection_ratio=0.0` alone isn't enough to get this: it only loosens the acceptance
threshold *against* the Otsu mask, it doesn't stop Otsu thresholding from running in the first
place (the code path that would skip it entirely requires an internal `tissue_region is None`
state that isn't reachable through any `LivePatchWSIConfig`/`process_wsi()` kwarg). So this
monkey-patches `compute_interesting_patches` itself - same "patch before use" approach as
`torch_load_patch.py` / `slidevips_mpp_patch.py` / `pathopatch_pydantic_patch.py` /
`cellvit_shapely2_patch.py`. Call `apply()` before constructing any `CellViTInferenceMemory`
(i.e. before `cellvit_utils.detect_cells_binary`'s call into `CellViTInferenceMemory.process_wsi`).
Callers should also pass `min_intersection_ratio=0.0` to `process_wsi()` so the per-patch
background-ratio discard check (a separate mechanism, evaluated per-tile in `__getitem__`)
doesn't drop anything either.
"""
_applied = False


def _all_grid_coordinates(slide, tiles, target_level, **kwargs):
    n_cols, n_rows = tiles.level_tiles[target_level]
    return [(row, col, 0.0) for row in range(n_rows) for col in range(n_cols)], {}, {}


def apply() -> None:
    global _applied
    if _applied:
        return
    import pathopatch.patch_extraction.dataset as pp_dataset

    pp_dataset.compute_interesting_patches = _all_grid_coordinates
    _applied = True
