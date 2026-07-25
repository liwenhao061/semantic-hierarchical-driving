#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-deps ninja

if ! command -v nvcc >/dev/null 2>&1 \
  || ! nvcc --version | grep -q "release 11.8" \
  || [[ ! -f "${CONDA_PREFIX:-}/include/cuda_runtime.h" ]] \
  || [[ ! -f "${CONDA_PREFIX:-}/include/cublas_v2.h" ]] \
  || [[ ! -f "${CONDA_PREFIX:-}/include/cusparse.h" ]] \
  || [[ ! -f "${CONDA_PREFIX:-}/include/cusolverDn.h" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CUDA nvcc 11.8 is required to build NATTEN 0.14.6." >&2
    exit 1
  fi
  CONDA_NO_PLUGINS=true conda install \
    --solver=classic --override-channels \
    -c nvidia/label/cuda-11.8.0 \
    cuda-nvcc=11.8 cuda-cudart-dev=11.8 \
    libcublas-dev=11.11.3.6 \
    libcusparse-dev=11.7.5.86 \
    libcusolver-dev=11.4.1.48 \
    -y
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
else
  CUDA_NVCC_PATH="$(command -v nvcc)"
  export CUDA_HOME="$(dirname "$(dirname "${CUDA_NVCC_PATH}")")"
fi

export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install \
  --no-build-isolation --no-deps natten==0.14.6
