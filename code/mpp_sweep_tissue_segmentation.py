"""
Run the tissue-compartment segmentation model (Step 3 of infer.py) on a single
H&E image at several MPP (microns-per-pixel) values and plot the resulting
masks side by side, to see how sensitive the prediction is to MPP.

Mirrors infer.py's predict_tissue_compartment() step, minus Trident
preprocessing / malignant-region restriction (Steps 1-2) which aren't needed
just to inspect the raw tissue-compartment prediction.

Usage: edit SLIDE_PATH / MPP_VALUES below and run as a script, or paste into
a notebook cell.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/data_g1/home/tanweikit/DigitalImmuneReporter")
sys.path.insert(0, str(REPO_ROOT / "code"))  # so `from utils....` resolves, same as infer.py

import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch

from utils.slide_io import prepare_openslide_wsi
from utils.mmseg_utils import model_fn as load_tissue_model, infer_single_wsi as infer_tissue_wsi
from utils.mask_utils import convert_label_mask_to_rgb

TISSUE_MODEL_CKPT = REPO_ROOT / "model_weights" / "tissue_compartment_segmentation" / "iter_40000.pth"
TISSUE_MODEL_CONFIG = REPO_ROOT / "model_weights" / "tissue_compartment_segmentation" / "segformer_b3_40k_2xb4_tcgacrc_tissue_augment.py"

TISSUE_ID2LABEL = {1: "Tumor", 2: "Stroma", 3: "Necrosis", 4: "Other"}
# Same color convention as SpatialFeatureComputer.tissue_mask_thumbnail()
TISSUE_COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (255, 255, 0)}

SLIDE_PATH = "/data_g1/AI_projects/raw_data/PCITB_CRC/HE_raw_images/Tumor/TMA1/TMA 1 Level 16_Core[1,1,10]_[4381,47584].tif"
MPP_VALUES = [0.25, 0.5, 1.0]
OUTPUT_DIR = str(REPO_ROOT / "output" / f"mpp_sweep_{Path(SLIDE_PATH).stem}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert the raw TIFF to an OpenSlide-readable file once. custom_mpp here only
# sets metadata on the converted file -- it doesn't resample pixels -- so the
# same converted slide is reused for every MPP value tested below.
slide_path, _ = prepare_openslide_wsi(SLIDE_PATH, OUTPUT_DIR, user_input_mpp=MPP_VALUES[0])
print(f"Slide ready for inference: {slide_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading tissue-compartment model on {device}...")
model = load_tissue_model(str(TISSUE_MODEL_CONFIG), str(TISSUE_MODEL_CKPT), device=device)

slide = openslide.OpenSlide(slide_path)
thumbnail = slide.get_thumbnail((512, 512))
slide.close()

masks = {}
for mpp in MPP_VALUES:
    tile_size = int(1024 * 0.25 / mpp)  # same formula as infer.py's predict_tissue_compartment()
    mpp_output_dir = os.path.join(OUTPUT_DIR, f"mpp_{mpp}")
    os.makedirs(mpp_output_dir, exist_ok=True)

    print(f"Running tissue segmentation at mpp={mpp} (tile_size={tile_size}px)...")
    mask = infer_tissue_wsi(slide_path, model, tile_size=tile_size, output_dir=mpp_output_dir)
    masks[mpp] = mask

    unique, counts = np.unique(mask, return_counts=True)
    breakdown = {
        TISSUE_ID2LABEL.get(int(label_id), f"id={label_id}"): f"{count / mask.size * 100:.1f}%"
        for label_id, count in zip(unique, counts)
    }
    print(f"  mpp={mpp}: {breakdown}")

# ---- plot: H&E thumbnail + one predicted mask per MPP value ----
fig, axes = plt.subplots(1, len(MPP_VALUES) + 1, figsize=(5 * (len(MPP_VALUES) + 1), 5.5))

axes[0].imshow(thumbnail)
axes[0].set_title("H&E")
axes[0].axis("off")

for ax, mpp in zip(axes[1:], MPP_VALUES):
    rgb_mask = convert_label_mask_to_rgb(masks[mpp], label_colors=TISSUE_COLORS, ignore_unseen_labels=True)
    ax.imshow(rgb_mask)
    ax.set_title(f"mpp={mpp} (tile={int(1024 * 0.25 / mpp)}px)")
    ax.axis("off")

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(color) / 255) for color in TISSUE_COLORS.values()]
fig.legend(legend_handles, TISSUE_ID2LABEL.values(), loc="lower center", ncol=len(TISSUE_ID2LABEL))
fig.suptitle(Path(SLIDE_PATH).name)
plt.tight_layout(rect=[0, 0.05, 1, 1])

plot_path = os.path.join(OUTPUT_DIR, "mpp_sweep_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_path}")
plt.show()
