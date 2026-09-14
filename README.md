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

   2c. `--cell_type_method miphei_multiplex`/`both` needs two more checkpoints that are **not**
       part of our in-house release — they're published by MIPHEI-ViT and CellViT-plus-plus
       themselves, from their own repos/hosts:

       - **MIPHEI-ViT** (`model_weights/miphei_multiplex_prediction/`) — predicts multiplex (mIF)
         channels from H&E. Its encoder backbone (`H-optimus-0`) is a **gated** model on Hugging
         Face, so request access first:
         1. Log in to Hugging Face and request access at https://huggingface.co/bioptimus/H-optimus-0
            (Apache-2.0, but you must accept its conditions before downloading — approval isn't
            instant). Re-use the same account/token you set up for CONCH in step 2a if you already
            did that.
         2. `huggingface-cli login` (skip if already logged in from step 2a).
         3. Download the MIPHEI-ViT release assets (`model.safetensors`, `config.yaml`, and the
            calibration classifier `logreg.pth` used to binarize its per-cell marker predictions)
            via the script `setup_env.sh` already cloned into `code/external/MIPHEI-ViT/`:
            ```bash
            python code/external/MIPHEI-ViT/scripts/download_miphei.py --out-dir model_weights/miphei_multiplex_prediction
            ```
            (This also downloads `LICENSE`/`model.py`/`requirements.txt` alongside the three files
            above — harmless extras from the same GitHub release, safe to leave in place or delete.)

       - **CellViT-plus-plus** (`model_weights/cellvit_binary_cell_detection/`) — binary (cell vs.
         background) nuclei detection. Its checkpoints are hosted on Google Drive, not
         script-downloadable (and the maintainers note they "cannot share all checkpoints due to
         their license", so only specific files are available): open
         https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79 , download
         `CellViT-SAM-H-x40-AMP.pth`, and place it at
         `model_weights/cellvit_binary_cell_detection/CellViT-SAM-H-x40-AMP.pth`.

   2d. `--cell_type_method miphei_multiplex`/`both` also needs the MIPHEI-ViT and CellViT-plus-plus
       source repos (their code is imported in-process, not pip-packaged; this is also where step
       2c's `download_miphei.py` script lives). `setup_env.sh` clones both into `code/external/`
       (matching `infer.py`'s `MIPHEI_VIT_REPO_ROOT` / `CELLVIT_REPO_ROOT` constants) and installs
       their runtime dependencies into the same virtual environment as the rest of the pipeline
       (mmdet/mmseg/trident/CONCH); see its comments for version-conflict notes. For best
       performance on large slides, MIPHEI-ViT recommends jemalloc (`sudo apt-get install
       libjemalloc2`, then run with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`) —
       optional, only affects memory growth during Step 4b, not correctness.

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
   | `--gpu`        | No       | `0`        | GPU index used for the whole pipeline (Trident, tissue compartment, cell type, miphei_multiplex). |
   | `--use_malignant_region` / `--no-use_malignant_region` | No | `--use_malignant_region` (enabled) | Whether to run Step 2 (malignant-region identification) and restrict tissue-compartment predictions to it. With `--no-use_malignant_region`, Step 2 is skipped entirely (no `malignant_region_mask.tif`, and no CONCH weights needed for it) and Step 3 uses the raw tissue-compartment mask directly. |
   | `--cell_type_method` | No | `morphology_based` | Which cell-typing method(s) to run as Step 4: `morphology_based` — typing from the H&E (Mask2Former, 6 fixed classes); `miphei_multiplex` — protein-marker-based typing (MIPHEI-ViT virtual multiplex staining + CellViT-plus-plus binary cell detection + marker-hierarchy annotation); `both` — run both, and Step 5 twice (see below). |

   The script runs 5 steps in order (Step 4 has two mutually-selectable variants) and writes into `<output_dir>`, including `pipeline.log` — a complete, timestamped log of the run (everything printed to the console, plus a step-by-step breakdown and a final per-step timing summary, is mirrored here):
   1. `preprocess_with_trident` — tissue/background segmentation + CONCH patch feature extraction
   2. `predict_malignant_region` — malignant/non-malignant patch classification → `malignant_region_mask.tif` (compressed, skipped if `--no-use_malignant_region`)
   3. `predict_tissue_compartment` — tumor/stroma/necrosis/other segmentation → `tissue_compartment_mask_raw.tif` (before combining with the malignant region, compressed) and `tissue_compartment_mask_combined.tif` (restricted to the malignant region — the final result; identical to the raw mask if Step 2 was skipped; compressed)
   4. `predict_cell_type` (`--cell_type_method morphology_based`, the default) — cell-level instance segmentation and typing (6 fixed classes: tumor, lymphocyte, neutrophil, plasmacell, eosinophil, other) → `cell_type_predictions.json`
   4b. `predict_miphei_multiplex_cell_type` (`--cell_type_method miphei_multiplex`) — protein-marker-driven cell typing, in three sub-steps: (i) predict multiplex (mIF) channels from the H&E via MIPHEI-ViT, (ii) detect cells (binary, no typing) via CellViT-plus-plus, (iii) type each detected cell by thresholding its mean predicted marker intensity against a fixed hierarchy (Tumour[Pan-CK+]; T Cell[CD3e+] → CytotoxicT[CD8a+]/Helper T[CD4+]; B Cell[CD20+]; Macrophage[CD68+]) → `miphei_cell_type_predictions.json`. With `--cell_type_method both`, both Step 4 and Step 4b run.
   5. `compute_spatial_features` — spatial TIME (tumor immune microenvironment) feature computation + report generation, from whichever Step 4 output(s) ran (with `--cell_type_method both`, this runs twice, into `spatial_features_morphology_based/` and `spatial_features_miphei_multiplex/` subdirectories):
      - `features.csv` — every computed feature: cell densities per tissue region (per cell type, and per marker-hierarchy level when using `miphei_multiplex`), G-cross tumor–immune proximity AUCs, CT (core tumor) / PT (peritumoral) relative abundance and CT/PT ratios, tumor–stroma percentage, cell counts/abundance, and (morphology_based taxonomy only) neutrophil/lymphocyte ratio
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
   | `mpp`                  | No       | Microns-per-pixel override.                                                  |
   | `use_malignant_region` | No       | `true`/`false` (also accepts `1`/`0`, `yes`/`no`). Defaults to `true`.       |
   | `cell_type_method`     | No       | `morphology_based`/`miphei_multiplex`/`both`. Defaults to `morphology_based`. |

   There's no per-slide output directory column — every slide's output goes to `<output_folder>/<slide_name>` (see `--output_folder` below), so a batch's outputs always live under one root.

   Example:
   ```csv
   slide_path,mpp,use_malignant_region,cell_type_method
   /data/slideA.svs,,true,morphology_based
   /data/slideB.svs,0.25,false,miphei_multiplex
   ```

   Parameters:

   | Argument        | Required | Default                          | Description                                                                 |
   |-----------------|----------|-----------------------------------|-------------------------------------------------------------------------------|
   | `--csv_path`    | Yes      | —                                 | CSV file as described above.                                                 |
   | `--output_folder` | No    | `output`                          | Root directory for every slide's output; each slide is written to `<output_folder>/<slide_name>`. |
   | `--gpu_ids`     | No       | all GPUs detected                 | Comma-separated GPU indices to use, e.g. `0,1,2`.                            |
   | `--num_workers` | No       | number of `--gpu_ids`             | Number of slides processed concurrently. Set higher than the number of GPUs to share a GPU across workers (only if you have the VRAM for it). |
   | `--results_csv` | No       | `<csv_path stem>_results.csv`     | Where to write the summary (per-slide status, elapsed time, log path).      |
   | `--mpp`         | No       | —                                 | Batch-wide `mpp` override, applied to every slide in the batch. |
   | `--use_malignant_region` / `--no-use_malignant_region` | No | — | Batch-wide `use_malignant_region` override, applied to every slide in the batch. |
   | `--cell_type_method` | No  | —                                 | Batch-wide `cell_type_method` override, applied to every slide in the batch. |

   The last three let you apply a non-default value across an entire batch without editing the CSV — useful when the CSV only really needs to vary `slide_path`. Each is **mutually exclusive with its CSV column at the batch level**: if `--mpp` is passed, the CSV's `mpp` column must be entirely empty (no row may set it) — if any row does, the script aborts before starting any subprocess and reports the conflict so you can clear the column or drop the CLI flag and re-run:
   ```bash
   python code/batch_infer.py --csv_path slides.csv --output_folder /data/batch_run_1 --cell_type_method miphei_multiplex --mpp 0.25
   ```

   Each slide runs as its own `infer.py` subprocess, pinned to a GPU via `CUDA_VISIBLE_DEVICES` (so all 5 steps land on that device) and isolated from the others — one slide crashing or running out of memory doesn't affect the rest of the batch. Per-slide stdout/stderr is captured to `<output_folder>/<slide_name>/batch_run.log`, which duplicates (and, if `infer.py` crashes very early, may capture slightly more than) that slide's own `<output_folder>/<slide_name>/pipeline.log`.