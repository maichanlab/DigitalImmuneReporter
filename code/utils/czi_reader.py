from aicspylibczi import CziFile
import cv2
import numpy as np
from PIL import Image
import warnings
from pathlib import Path
import pyvips

class CZIReader:
    """
    Class for reading and extracting image data from Zeiss .CZI whole-slide images using the aicspylibczi library.

    This class provides functionalities similar to OpenSlide for handling CZI files, including retrieving image metadata,
    reading thumbnails, extracting specific image regions, and accessing full-resolution images as NumPy arrays or PIL Images.

    Attributes
    ----------
    image_path : str
        Path to the input .CZI image file.
    CziFileObj : aicspylibczi.CziFile
        CZI file object for reading image data.
    bbox : BoundingBox
        Bounding box of the full mosaic image.
    x0, y0 : int
        Top-left coordinates of the mosaic bounding box.
    width, height : int
        Dimensions of the full mosaic image.
    dimensions : tuple[int, int]
        (width, height) of the image.
    dimension_size : dict
        Mapping of CZI dimensions (e.g., 'X', 'Y', 'C', 'Z', etc.) to their respective sizes.
    level_count : int
        Number of pyramid levels (fixed as 1 for CZI mosaic readout).
    level_dimensions : tuple[int, int]
        Dimensions for each level (single level for now).
    level_downsamples : tuple[float]
        Downsample factors for each level (1.0 for level 0).
    properties : dict
        Image properties including microns per pixel (MPP) and objective magnification.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.CziFileObj = CziFile(self.image_path)

        # Retrieve information from the image
        self.bbox = self.CziFileObj.get_mosaic_bounding_box()
        self.x0 = self.bbox.x
        self.y0 = self.bbox.y
        self.width = self.bbox.w
        self.height = self.bbox.h
        self.dimensions = self.width, self.height
        self.dimension_size = {d:s for d,s in zip(self.CziFileObj.dims, self.CziFileObj.size)}
        self.scene_bboxes = self.CziFileObj.get_all_scene_bounding_boxes()
        print(f"Number of scenes: {len(self.scene_bboxes)}")

        self.level_count = 1
        self.level_dimensions = (self.dimensions)
        self.level_downsamples = (1.0,)

        objective_elem = self.CziFileObj.meta.find(".//ImageScaling/ScalingComponent[@Name='MTBObjectiveChanger']")

        if objective_elem is not None:
            magnification = float(objective_elem.attrib['Magnification'])
        else:
            magnification = None
            print("Objective magnification not found in metadata")


        self.properties = {
            "openslide.mirax.MPP": float(self.CziFileObj.meta.find(".//Scaling/Items/Distance[@Id='X']/Value").text) * 1e6,
            "openslide.objective-power": magnification,
        }
        self.mpp = self.properties["openslide.mirax.MPP"]

    def get_thumbnail_np(self, size: tuple[int, int], c: int = 0) -> np.ndarray:
        """
        Generate a thumbnail of the CZI image as a NumPy array.

        Parameters
        ----------
        size : tuple[int, int]
            Maximum size (width, height) of the thumbnail.
        c : int, optional
            Channel index to use (default: 0).

        Returns
        -------
        np.ndarray
            RGB thumbnail image as a NumPy array.
        """
        if self.dimension_size['C'] > 1:
            warnings.warn("Image has multiple channels, using channel 0")
        scale_factor = min(size[0] / self.width, size[1] / self.height)
        print(f"Getting thumbnail with scale factor {scale_factor}")
        image_np = self.CziFileObj.read_mosaic(scale_factor=scale_factor, C=c).squeeze(0)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np

    def get_thumbnail(self, size: tuple[int, int], c: int = 0) -> Image:
        """
        Generate a thumbnail of the CZI image as a PIL Image.

        Parameters
        ----------
        size : tuple[int, int]
            Maximum size (width, height) of the thumbnail.
        c : int, optional
            Channel index to use (default: 0).

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail image as a PIL Image object.
        """
        image_np = self.get_thumbnail_np(size=size, c=c)
        return Image.fromarray(image_np)

    def read_region_np(self, location: tuple[int, int], level:int, size: tuple[int, int], c: int = 0) -> np.ndarray:
        """
        Read a specific rectangular region from the CZI image as a NumPy array.

        Parameters
        ----------
        location : tuple[int, int]
            (x, y) coordinates of the top-left pixel in the level 0 reference frame.
        level : int
            Pyramid level to read from (currently only level 0 supported).
        size : tuple[int, int]
            (width, height) of the region to extract.
        c : int, optional
            Channel index to read (default: 0).

        Returns
        -------
        np.ndarray
            Extracted RGB image region as a NumPy array.
        """
        if self.dimension_size['C'] > 1:
            warnings.warn("Image has multiple channels, using channel 0")

        size = (min(self.width - location[0], size[0]), min(self.height - location[1], size[1]))

        if size[0] < 30000 and size[1] < 30000:
            image_np = self.CziFileObj.read_mosaic((location[0] + self.x0, location[1] + self.y0, size[0], size[1]), C=c).squeeze(0)
        else:
            # For very large images, read in tiles to avoid memory issues
            image_np = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for y in range(location[1], location[1] + size[1] + self.y0, 30000):
                for x in range(location[0], location[0] + size[0] + self.x0, 30000):
                    w = min(30000, self.width - x)
                    h = min(30000, self.height - y)
                    tile = self.CziFileObj.read_mosaic((x + self.x0, y + self.y0, w, h), C=c).squeeze(0)
                    image_np[y:y+h, x:x+w, :] = tile

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np

    def read_region(self, location: tuple[int, int], level:int, size: tuple[int, int], c: int = 0) -> Image:
        """
        Read a specific rectangular region from the CZI image as a PIL Image.

        Parameters
        ----------
        location : tuple[int, int]
            (x, y) coordinates of the top-left pixel in the level 0 reference frame.
        level : int
            Pyramid level to read from (currently only level 0 supported).
        size : tuple[int, int]
            (width, height) of the region to extract.
        c : int, optional
            Channel index to read (default: 0).

        Returns
        -------
        PIL.Image.Image
            Extracted RGB image region as a PIL Image object.
        """
        image_np = self.read_region_np(
            location=location,
            level=level,
            size=size,
            c=c,
        )
        return Image.fromarray(image_np)

    def get_full_image_np(self, c: int = 0) -> np.ndarray:
        """
        Read the full-resolution mosaic image from the CZI file as a NumPy array.

        Parameters
        ----------
        c : int, optional
            Channel index to read (default: 0).

        Returns
        -------
        np.ndarray
            Full RGB mosaic image as a NumPy array.
        """
        if self.width < 30000 and self.height < 30000:
            image_np = self.CziFileObj.read_mosaic(C=c).squeeze(0)
        else:
            # For very large images, read in tiles to avoid memory issues
            image_np = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for y in range(0, self.height, 30000):
                for x in range(0, self.width, 30000):
                    w = min(30000, self.width - x)
                    h = min(30000, self.height - y)
                    tile = self.CziFileObj.read_mosaic((x + self.x0, y + self.y0, w, h), C=c).squeeze(0)
                    image_np[y:y+h, x:x+w, :] = tile
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np

    def save_as_svs(
        self,
        output_dir: str,
        separate_scenes: bool = False,
        c: int = 0
    ) -> None:
        """
        Save the CZI image as a pyramidal SVS (BigTIFF) file using pyvips.

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
        if not separate_scenes:
            output_path = Path(output_dir) / (Path(self.image_path).stem + ".svs")

            image = self.get_full_image_np(c=c)
            vips_image = pyvips.Image.new_from_memory(
                image.tobytes(),
                image.shape[1], # width
                image.shape[0], # height
                bands=image.shape[2] if image.ndim == 3 else 1,
                format=pyvips.BandFormat.UCHAR if image.dtype == np.uint8 else pyvips.BandFormat.FLOAT
            )
            mpp = self.properties["openslide.mirax.MPP"]
            xres = 1000 / mpp  # pixels per cm
            yres = 1000 / mpp

            # Save pyramidal tiled BigTIFF (SVS-compatible for most tools)
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
        else:
            for scene_number, bbox in self.scene_bboxes.items():
                output_path = Path(output_dir) / f"{Path(self.image_path).stem}_scene{scene_number}.svs"

                image = self.read_region_np(
                    location=(bbox.x - self.x0, bbox.y - self.y0),
                    level=0,
                    size=(bbox.w, bbox.h)
                )

                vips_image = pyvips.Image.new_from_memory(
                    image.tobytes(),
                    image.shape[1], # width
                    image.shape[0], # height
                    bands=image.shape[2] if image.ndim == 3 else 1,
                    format=pyvips.BandFormat.UCHAR if image.dtype == np.uint8 else pyvips.BandFormat.FLOAT
                )
                mpp = self.properties["openslide.mirax.MPP"]
                xres = 1000 / mpp  # pixels per cm
                yres = 1000 / mpp

                # Save pyramidal tiled BigTIFF (SVS-compatible for most tools)
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
