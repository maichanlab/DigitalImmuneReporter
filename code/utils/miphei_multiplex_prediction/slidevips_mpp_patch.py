"""
Patches MIPHEI-ViT's vendored `slidevips` WSI reader (installed from its
`slidevips-python/` subdirectory — see setup_env.sh) to fall back to deriving MPP from the
TIFF `xres`/`yres` tags when a `.svs`/`.ndpi` slide lacks `openslide.mpp-x`/`openslide.mpp-y`
metadata, instead of raising `KeyError`. `get_pyramid_pyvips()` upstream reads those two
fields unconditionally; some slides (older scanners, re-exported/anonymized files) don't
carry them.

Rather than editing the external repo checkout directly (it's re-fetched by setup_env.sh, so
an in-place edit wouldn't survive a fresh clone on another machine), this monkey-patches the
more defensive function in at runtime — same approach as `torch_load_patch.py` for
mmdet/mmseg/mmcv. Call `apply()` (which also registers the repo root on `sys.path`) before
importing anything from MIPHEI-ViT that reads a WSI (i.e. before
`run_wsi_inference.wsi_inference`).
"""
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _patched_get_pyramid_pyvips(filename: str, channel_idxs: Optional[List[int]] = None, mode: str = "RGB") -> Tuple[List, dict]:
    import ome_types
    import pyvips

    pyramid_image = []
    image_format = Path(filename).suffix
    image = pyvips.Image.new_from_file(filename, access="sequential")
    fields = {field: image.get(field) for field in image.get_fields()}
    if mode == "RGB" and image.bands == 4:
        channel_idxs = [0, 1, 2]

    if image_format in [".ndpi", ".svs"]:
        if "openslide.mpp-x" in fields:
            mppx = float(fields["openslide.mpp-x"])
        else:
            xres = float(fields.get("xres", 1))
            mppx = 10000 / xres
        if "openslide.mpp-y" in fields:
            mppy = float(fields["openslide.mpp-y"])  # fixed: was `mppx = fields["openslide.mpp-x"]`
        else:
            yres = float(fields.get("yres", 1))  # fixed: was reading "xres" into mppx again
            mppy = 10000 / yres
        try:
            n_levels = int(image.get("openslide.level-count"))
        except Exception:
            n_levels = int(fields.get("n-pages"))
        for level in range(n_levels):
            try:
                image = pyvips.Image.new_from_file(filename, level=level, access="sequential")
            except pyvips.error.Error:
                # Not every ".svs"/".ndpi"-suffixed file is actually openslide-loadable - a
                # slide converted from a plain image (e.g. a single H&E tile, via this
                # pipeline's slide_io.TiffWSIReader) is really just a generic pyramidal TIFF
                # wearing an ".svs" extension, so pyvips's own format auto-detection picks its
                # plain TIFF loader instead of its openslide one - which selects a pyramid
                # level via "page", not "level" (same convention the .tif/.tiff branch below
                # already uses).
                image = pyvips.Image.new_from_file(filename, page=level, access="sequential")
            if channel_idxs is not None:
                image = image[channel_idxs]
            pyramid_image.append(image)

    elif image_format in [".tif", ".tiff", ".ome.tiff", ".ome.tif"]:
        pixels_metadata = ome_types.from_xml(fields["image-description"]).images[0].pixels
        mppx = pixels_metadata.physical_size_x
        mppy = pixels_metadata.physical_size_x
        n_pages, n_subifds = image.get("n-pages"), image.get("n-subifds")
        del image
        for level in range(-1, n_subifds):
            if channel_idxs is None:
                channels = [pyvips.Image.new_from_file(filename, subifd=level, page=channel, access="sequential") for channel in range(n_pages)]
            else:
                channels = [pyvips.Image.new_from_file(filename, subifd=level, page=channel, access="sequential") for channel in channel_idxs]
            image = channels[0].bandjoin(channels[1:])
            pyramid_image.append(image)
    elif image_format == ".qptiff":
        mppx = 1 / fields["xres"]
        mppy = 1 / fields["yres"]
        if fields["resolution-unit"] == "cm":
            mppx *= 1000
            mppy *= 1000
        else:
            raise ValueError("Unknown resolution unit")
        n_pages = image.get("n-pages")
        del image

        area2channels = defaultdict(list)
        for page in range(n_pages):
            channel_level = pyvips.Image.new_from_file(filename, page=page, access="sequential")
            if channel_level.bands > 1:
                del channel_level
                continue
            area = channel_level.width * channel_level.height
            area2channels[area].append(channel_level)

        pyramid_image = []
        areas = list(area2channels.keys())
        nb_bands = len(area2channels[areas[0]])
        for area in sorted(areas, reverse=True):
            channels = area2channels[area]
            assert len(channels) == nb_bands
            if channel_idxs is not None:
                channels = [channels[channel_idx] for channel_idx in channel_idxs]
            image = channels[0].bandjoin(channels[1:])
            pyramid_image.append(image)
    elif image_format == "qptiff":
        raise NotImplementedError
    else:
        raise NotImplementedError

    del image
    assert np.abs(mppx - mppy) < 1e-3  # fixed: was `np.abs(mppx - mppy < 1e-3)` (wrong operator precedence)
    mpp = (mppx + mppy) / 2
    assert 0 < mpp < 15
    fields["mpp"] = mpp
    return pyramid_image, fields


def apply(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import slidevips.read_pyramid as read_pyramid_module
    import slidevips.reader as reader_module

    read_pyramid_module.get_pyramid_pyvips = _patched_get_pyramid_pyvips
    reader_module.get_pyramid_pyvips = _patched_get_pyramid_pyvips
