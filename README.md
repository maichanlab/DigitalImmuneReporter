# Digital Immune Reporter

Prerequisite: The pipeline inference requires a CUDA GPU to run. Visit https://crc.digitalimmunereporter.com/ for cloud-based DIR platform.
1. Set up the virtual environment and install packages.
   ```bash
   source setup_env.sh
   ```
   Note: running `setup_env.sh` will activate/deactivate within the script's own subshell only.
   Pytorch version `torch==2.6.0+cu124` will be installed for compatibility with `mmcv`/`mmdet`/`mmsegmentation`. `mmcv` will be built from source, expect up to 20 minutes. This requires a CUDA toolkit/compiler (`nvcc`, `g++`) on the host in addition to the GPU driver.

2. Download models weights.

   2a. CONCH (used for malignant-region patch features) is a **gated** model on Hugging Face — you need to request access before you can download it:

       1. Create/use a Hugging Face account with your **institutional email** (personal addresses like @gmail/@hotmail are rejected), then request access at https://huggingface.co/MahmoodLab/CONCH and agree to its CC-BY-NC-ND 4.0 license (non-commercial academic use only). Approval isn't instant.
       2. Authenticate locally with a token from https://huggingface.co/settings/tokens (`huggingface_hub` is already installed via `setup_env.sh`):
          ```bash
          huggingface-cli login
          ```
       3. Download `pytorch_model.bin` directly into `model_weights/malignant_region_identification/`:
          ```bash
          huggingface-cli download MahmoodLab/CONCH pytorch_model.bin --local-dir model_weights/malignant_region_identification
          ```

   2b. Download our in-house trained model checkpoints from **[TODO: Zenodo record URL/DOI, not yet published]** and place them at:

       ```
       model_weights/
         malignant_region_identification/
           logreg_conch_model_20260204.json
         tissue_compartment_segmentation/
           iter_40000.pth
           segformer_b3_40k_2xb4_tcgacrc_tissue_augment.py
         cell_type_prediction/
           epoch_36.pth
           mask2former_swin-s-3x_dataset_tcga_lizard_class_weight_log_count.py
       ```

       This section will be replaced with the actual Zenodo link and download command once the record is published.

3. Run the pipeline on a slide:

   ```bash
   python code/infer.py --slide_path /path/to/slide.svs
   ```

   Parameters:

   | Argument       | Required | Default    | Description                                                                                       |
   |----------------|----------|------------|---------------------------------------------------------------------------------------------------|
   | `--slide_path` | Yes      | —          | Path to the input slide. Supports OpenSlide formats (`.svs`, `.tiff`/`.ome.tiff`, `.ndpi`, ...), `.czi`, and plain raster images (`.png`, `.jpg`, ...) — non-OpenSlide formats are converted to `.svs` automatically. |
   | `--output_dir` | No       | `output/<slide_name>` | Directory where Trident outputs and all files below are written.                        |
   | `--mpp`        | No       | slide's own MPP, else `0.25` | Microns-per-pixel override, used if the slide doesn't carry MPP metadata or it should be overridden. |
   | `--gpu`        | No       | `0`        | GPU index used for the Trident preprocessing step.                                                |
   | `--use_malignant_region` / `--no-use_malignant_region` | No | `--use_malignant_region` (enabled) | Whether to run Step 2 (malignant-region identification) and restrict tissue-compartment predictions to it. With `--no-use_malignant_region`, Step 2 is skipped entirely (no `malignant_region_mask.npy`, and no CONCH weights needed for it) and Step 3 uses the raw tissue-compartment mask directly. |

   The script runs 5 steps in order and writes into `<output_dir>`, including `pipeline.log` — a complete, timestamped log of the run (everything printed to the console, plus a step-by-step breakdown and a final per-step timing summary, is mirrored here):
   1. `preprocess_with_trident` — tissue/background segmentation + CONCH patch feature extraction
   2. `predict_malignant_region` — malignant/non-malignant patch classification → `malignant_region_mask.npy` (skipped if `--no-use_malignant_region`)
   3. `predict_tissue_compartment` — tumor/stroma/necrosis/other segmentation → `tissue_compartment_mask_raw.tif` (before combining with the malignant region, compressed) and `tissue_compartment_mask_combined.npy` (restricted to the malignant region — the final result; identical to the raw mask if Step 2 was skipped)
   4. `predict_cell_type` — cell-level instance segmentation and typing → `cell_type_predictions.json`
   5. `compute_spatial_features` — spatial TIME (tumor immune microenvironment) feature computation + report generation:
      - `features.csv` — every computed feature: cell densities per tissue region, G-cross tumor–immune proximity AUCs, CT (core tumor) / PT (peritumoral) relative abundance and CT/PT ratios, tumor–stroma percentage, cell counts/abundance, neutrophil/lymphocyte ratio
      - `digital_immune_report.pdf` — H&E/tissue/malignant-region thumbnails, cell count & tissue area tables, per-tissue-region density table, CT/PT segmentation plot and relative-abundance table
      - `area_table.csv`, `cell_table.csv` — supporting per-tissue-area and per-cell intermediate tables

4. Batch-process multiple slides from a CSV file, optionally in parallel across GPUs:

   ```bash
   python code/batch_infer.py --csv_path slides.csv
   ```

   CSV format (one row per slide):

   | Column                 | Required | Description                                                                 |
   |------------------------|----------|------------------------------------------------------------------------------|
   | `slide_path`           | Yes      | Path to the slide/image.                                                    |
   | `output_dir`           | No       | Defaults to `output/<slide_name>`, same as `infer.py`.                       |
   | `mpp`                  | No       | Microns-per-pixel override.                                                  |
   | `use_malignant_region` | No       | `true`/`false` (also accepts `1`/`0`, `yes`/`no`). Defaults to `true`.       |

   Example:
   ```csv
   slide_path,output_dir,mpp,use_malignant_region
   /data/slideA.svs,,,true
   /data/slideB.svs,output/slideB_run2,0.25,false
   ```

   Parameters:

   | Argument        | Required | Default                          | Description                                                                 |
   |-----------------|----------|-----------------------------------|-------------------------------------------------------------------------------|
   | `--csv_path`    | Yes      | —                                 | CSV file as described above.                                                 |
   | `--gpu_ids`     | No       | all GPUs detected                 | Comma-separated GPU indices to use, e.g. `0,1,2`.                            |
   | `--num_workers` | No       | number of `--gpu_ids`             | Number of slides processed concurrently. Set higher than the number of GPUs to share a GPU across workers (only if you have the VRAM for it). |
   | `--results_csv` | No       | `<csv_path stem>_results.csv`     | Where to write the summary (per-slide status, elapsed time, log path).      |

   Each slide runs as its own `infer.py` subprocess, pinned to a GPU via `CUDA_VISIBLE_DEVICES` (so all 5 steps land on that device) and isolated from the others — one slide crashing or running out of memory doesn't affect the rest of the batch. Per-slide stdout/stderr is captured to `<output_dir>/batch_run.log`, which duplicates (and, if `infer.py` crashes very early, may capture slightly more than) that slide's own `<output_dir>/pipeline.log`.