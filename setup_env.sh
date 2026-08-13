#!/usr/bin/env bash
#
# Sets up the full development environment for this codebase: creates a
# Python virtual environment, activates it, installs a torch build pinned to
# a version known to work with this pipeline's mmlab stack (falls back to
# CPU if no NVIDIA GPU / CUDA driver is detected), builds mmcv against it,
# installs mmsegmentation/mmdet, trident, CONCH, the MIPHEI-ViT/CellViT-plus-plus
# runtime dependencies used by the miphei_multiplex cell-typing path, and this
# project's requirements.txt.
#
# Why torch is pinned instead of "always newest": mmcv==2.1.0 (required by
# mmdet==3.3.0 / mmsegmentation==1.2.2) only ships prebuilt compiled-ops
# wheels for torch up to ~2.4 (see https://download.openmmlab.com/mmcv/dist/).
# Installing the newest CUDA-matching torch build breaks mmcv's ops with an
# ABI mismatch (undefined symbol errors) at import time. torch 2.6.0 / CUDA
# 12.4 is the combination used by this pipeline's AWS Dockerfile
# (ai4he-crc-cell-tissuetype-processing-v3), so we pin to that and build
# mmcv from source against it (no prebuilt wheel exists for it either).
#
# Usage:
#   ./setup_env.sh [venv_dir]
#
# venv_dir defaults to ".venv". Run with `source` if you want the venv to
# stay active in your current shell after the script finishes:
#   source ./setup_env.sh

set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# MIPHEI-ViT / CellViT-plus-plus source checkouts (must match infer.py's MIPHEI_VIT_REPO_ROOT /
# CELLVIT_REPO_ROOT) — cloned below into a fixed, repo-relative location so nothing in this
# codebase references any particular machine's paths.
EXTERNAL_REPOS_DIR="code/external"
MIPHEI_VIT_REPO_ROOT="${EXTERNAL_REPOS_DIR}/MIPHEI-ViT"
CELLVIT_REPO_ROOT="${EXTERNAL_REPOS_DIR}/CellViT-plus-plus"

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
CUDA_TAG="cu124"

echo "==> Creating virtual environment in '${VENV_DIR}'"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

# Detect GPU(s) and their compute capability, so mmcv's CUDA extension is
# built for the hardware actually present (with PTX for forward-compat).
HAS_GPU=0
TORCH_CUDA_ARCH_LIST=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=1
    GPU_ARCHES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u)"
    TORCH_CUDA_ARCH_LIST="$(echo "${GPU_ARCHES}" | sed 's/$/+PTX/' | tr '\n' ' ' | sed 's/ *$//')"
    echo "==> Detected GPU(s) with compute capability: $(echo "${GPU_ARCHES}" | tr '\n' ' ')"
fi

if [[ "${HAS_GPU}" -eq 1 ]]; then
    echo "==> Installing torch==${TORCH_VERSION}+${CUDA_TAG} / torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}"
    pip install "torch==${TORCH_VERSION}+${CUDA_TAG}" "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
    echo "==> No NVIDIA GPU / driver detected; installing CPU-only torch==${TORCH_VERSION} / torchvision==${TORCHVISION_VERSION}"
    echo "    Note: this pipeline's mmdet/mmseg-based steps (tissue compartment, cell type) require a CUDA GPU."
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Verifying torch installation"
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "==> Installing openmim and mmengine"
pip install openmim
mim install "mmengine==0.10.7"

echo "==> Building mmcv==2.1.0 from source against torch ${TORCH_VERSION}"
echo "    (no prebuilt mmcv wheel exists for this torch version, so it must be compiled locally)"
# mmcv's setup.py needs pkg_resources, which setuptools>=81 no longer bundles.
# Pin an older setuptools in this env and disable build isolation so mmcv's
# build uses it instead of a fresh, incompatible isolated setuptools.
pip install "setuptools<81" wheel
export FORCE_CUDA="${HAS_GPU}"
export MMCV_WITH_OPS=1
if [[ -n "${TORCH_CUDA_ARCH_LIST}" ]]; then
    export TORCH_CUDA_ARCH_LIST
fi
export MAX_JOBS="$(nproc 2>/dev/null || echo 4)"
if (( MAX_JOBS > 8 )); then
    MAX_JOBS=8
fi
pip install mmcv==2.1.0 --no-binary mmcv --no-cache-dir --no-build-isolation
unset FORCE_CUDA MMCV_WITH_OPS TORCH_CUDA_ARCH_LIST MAX_JOBS

echo "==> Verifying mmcv installation"
python -c "import mmcv; from mmcv.ops import batched_nms; print('mmcv', mmcv.__version__, 'ops OK')"

echo "==> Installing mmsegmentation and mmdet"
mim install mmsegmentation==1.2.2 mmdet==3.3.0

echo "==> Installing trident"
pip install git+https://github.com/mahmoodlab/trident.git

echo "==> Installing CONCH"
pip install git+https://github.com/Mahmoodlab/CONCH.git

echo "==> Cloning MIPHEI-ViT and CellViT-plus-plus (used in-process, not pip-packaged)"
mkdir -p "${EXTERNAL_REPOS_DIR}"
if [[ ! -d "${MIPHEI_VIT_REPO_ROOT}/.git" ]]; then
    git clone --depth 1 https://github.com/sanofi-public/miphei-vit.git "${MIPHEI_VIT_REPO_ROOT}"
else
    echo "    ${MIPHEI_VIT_REPO_ROOT} already exists, skipping clone"
fi
if [[ ! -d "${CELLVIT_REPO_ROOT}/.git" ]]; then
    git clone --depth 1 https://github.com/tio-ikim/cellvit-plus-plus.git "${CELLVIT_REPO_ROOT}"
else
    echo "    ${CELLVIT_REPO_ROOT} already exists, skipping clone"
fi

echo "==> Installing MIPHEI-ViT + CellViT-plus-plus runtime dependencies"
# Only the packages actually exercised by MIPHEI-ViT's run_wsi_inference.wsi_inference() and
# CellViT-plus-plus's cellvit.inference.inference_memory.CellViTInferenceMemory (both imported
# in-process, registered on sys.path from their repo checkouts — see infer.py's
# MIPHEI_VIT_REPO_ROOT / CELLVIT_REPO_ROOT) — not each repo's full training/dev requirements.txt,
# which pulls in unrelated heavy packages (tensorflow, jupyterlab, xgboost, scikit-survival, ...).
pip install \
    albumentations hydra-core omegaconf pytorch_lightning torchmetrics wandb "timm>=1.0.15" \
    colorama pyyaml geojson pathopatch torchstain ray ujson python-snappy numba cupy-cuda12x

# The install above pulls in pathopatch's declared (and, verified empirically, overly
# conservative) numpy<2/pydantic<2/Shapely<=2.0.5 pins, silently downgrading the numpy/pydantic/
# shapely that mmcv/mmdet/trident/CONCH were just installed against — the Shapely downgrade in
# particular breaks Trident's own Step 1 tissue segmentation (WSIPatcher._compute_masked calls
# geopandas' union_all(), which needs Shapely>=2.1). The specific pathopatch code
# CellViT-plus-plus actually imports (LivePatchWSIDataset/Dataloader/Config) works fine with
# numpy>=2/pydantic>=2/Shapely>=2.1 despite the pins, and albumentations/wandb genuinely require
# pydantic>=2 to import at all — so force all three back up afterwards.
pip install "pydantic>=2.6" "numpy>=2" "shapely>=2.1"

echo "==> Installing MIPHEI-ViT's slidevips WSI reader (editable, from its repo checkout)"
pip install -e "${MIPHEI_VIT_REPO_ROOT}/slidevips-python"

echo "==> Downgrading segmentation-models-pytorch for MIPHEI-ViT's generator code"
# src/generators/smp_unet.py imports CenterBlock from segmentation_models_pytorch.decoders.unet.decoder,
# which was removed in segmentation-models-pytorch>=0.5. trident's own (unpinned) smp usage
# works fine with 0.4.0 too, verified empirically.
pip install "segmentation-models-pytorch==0.4.0"

echo "==> Installing project requirements.txt"
pip install -r requirements.txt

echo "==> Done. Virtual environment '${VENV_DIR}' is active in this shell."
echo "    (If you ran this script with 'bash setup_env.sh' instead of 'source setup_env.sh',"
echo "     the venv will be deactivated when the script exits — run 'source ${VENV_DIR}/bin/activate' to re-enter it.)"
