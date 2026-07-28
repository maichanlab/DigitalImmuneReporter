from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image
import cv2
import pyvips

class TiffWSIReader:

    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.tf = tiff.TiffFile(self.image_path)
        s = self.tf.series[0]

        self._arr = None

        self.height = s.shape[0]
        self.width = s.shape[1]
        self.dimensions = (self.width, self.height)

    def _downsample(self, level=None, width=None, height=None):
        """
        Downsample by either:
        - level: power-of-two factor (scale = 1 / 2**level), OR
        - width/height: target size to downsample image to.

        Returns:
            np.ndarray
        """
        h, w = self._arr.shape[:2]

        # argument validation
        if level is not None and (width is not None or height is not None):
            raise ValueError("Specify either 'level' or 'width/height', not both.")
        if level is None and width is None and height is None:
            raise ValueError("Provide one of: level OR width/height.")

        if level is not None:
            if not isinstance(level, int) or level < 0:
                raise ValueError("'level' must be a non-negative integer.")

            factor = 2 ** level
            new_w = max(1, w // factor)
            new_h = max(1, h // factor)
            downsample_info = f"level={level} (factor {factor}x)"
        else:
            new_w = int(width) if width is not None else w
            new_h = int(height) if height is not None else h
            downsample_info = f" size {new_w}x{new_h}"

        # use appropriate interp method: AREA for shrinking, LINEAR for enlarging
        shrink = (new_w < w) or (new_h < h)
        interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR

        new_shape = (int(new_w), int(new_h))
        print(
            f"Resizing to {downsample_info}: new shape {new_shape} "
            f"(original shape: ({w}, {h})) using {'INTER_AREA' if shrink else 'INTER_LINEAR'}"
        )
        return cv2.resize(self._arr, new_shape, interpolation=interp)

    def _ensure_loaded(self):
        """Load image."""
        if self._arr is not None:
            return
        self._arr = tiff.imread(self.image_path)

    def get_thumbnail_np(self, size):

        width, height = size[0], size[1]
        """Return thumbnail as numpy."""
        arr = self.get_full_image_np()

        thumbnail = self._downsample(width=width, height=height)

        return thumbnail

    def get_thumbnail(self, size) -> Image.Image:
        return Image.fromarray(self.get_thumbnail_np(size))

    def read_region_np(self, location, level, size):

        self._ensure_loaded()

        x, y = location[0], location[1]
        width, height = size[0], size[1]

        if level != 0:
            # downsample tiff image
            arr = self._downsample(level=level)
        else:
            arr = self._arr

        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(arr.shape[0], y0 + height), min(arr.shape[1], x0 + width)

        out = arr[y0:y1, x0:x1, :]

        return out

    def read_region(self, location, level, size):
        return Image.fromarray(self.read_region_np(location, level, size))

    def get_full_image_np(self):
        self._ensure_loaded()

        return self._arr

    def save_as_svs(
        self,
        output_dir: str,
        custom_mpp: float,
    ) -> None:
        """
        Save the Tiff image as a pyramidal SVS (BigTIFF) file using pyvips.

        This function exports the image as an SVS-compatible tiled TIFF. It can either save the
        entire mosaic as a single file or export each scene separately. The output is stored in
        a pyramidal format with tiling for efficient visualization in digital pathology viewers.

        Parameters
        ----------
        output_dir : str
            Directory to save the output SVS files.
        separate_scenes : bool, optional
            If True, saves each scene as an individual SVS file; otherwise, saves the full mosaic as one file. (default: False)
        c : int, optional
            Channel index to read from the CZI image. Defaults to 0.

        Returns
        -------
        None
            The function saves image files to disk and does not return any value.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / (Path(self.image_path).stem + ".svs")

        image = self.get_full_image_np()
        vips_image = pyvips.Image.new_from_memory(
            image.tobytes(),
            image.shape[1], # width
            image.shape[0], # height
            bands=image.shape[2] if image.ndim == 3 else 1,
            format=pyvips.BandFormat.UCHAR if image.dtype == np.uint8 else pyvips.BandFormat.FLOAT
        )
        xres = 1000 / custom_mpp  # pixels per cm
        yres = 1000 / custom_mpp

        vips_image.tiffsave(
            output_path,
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

        print("Saved:", output_path)
