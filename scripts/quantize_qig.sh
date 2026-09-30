#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QIG_SOURCE_DIR="${QIG_SOURCE_DIR:-${REPOSITORY_ROOT}/.third_party/QIG}"
QIG_VENV_DIR="${QIG_VENV_DIR:-${REPOSITORY_ROOT}/.venv-qig}"
PYTHON311="${PYTHON311:-python3.11}"
LLAVA_SOURCE_DIR="${QIG_SOURCE_DIR}/3rdparty/LLaVA-NeXT"
LMMS_EVAL_SOURCE_DIR="${QIG_SOURCE_DIR}/3rdparty/lmms-eval"

if [[ ! -d "${QIG_SOURCE_DIR}/.git" ]]; then
    if [[ -e "${QIG_SOURCE_DIR}" ]]; then
        echo "QIG source path exists but is not a git checkout: ${QIG_SOURCE_DIR}" >&2
        exit 1
    fi
    mkdir -p "$(dirname -- "${QIG_SOURCE_DIR}")"
    echo "Cloning QIG source into ${QIG_SOURCE_DIR}"
    git clone https://github.com/ucas-xiang/QIG.git "${QIG_SOURCE_DIR}"
else
    echo "Using existing QIG checkout: ${QIG_SOURCE_DIR}"
fi

if [[ ! -f "${QIG_SOURCE_DIR}/main_quant.py" ]]; then
    echo "The QIG checkout is incomplete: ${QIG_SOURCE_DIR}" >&2
    exit 1
fi

clone_companion() {
    local repository_url="$1"
    local checkout_dir="$2"
    if [[ -f "${checkout_dir}/setup.py" || -f "${checkout_dir}/pyproject.toml" ]]; then
        echo "Using companion checkout: ${checkout_dir}"
        return
    fi
    if [[ -e "${checkout_dir}" ]]; then
        echo "Companion source path exists but is incomplete: ${checkout_dir}" >&2
        exit 1
    fi
    mkdir -p "$(dirname -- "${checkout_dir}")"
    git clone "${repository_url}" "${checkout_dir}"
}

# QIG documents these companion repositories; they are not Git submodules.
clone_companion "https://github.com/LSY-noya/LLaVA-NeXT.git" "${LLAVA_SOURCE_DIR}"
clone_companion "https://github.com/LSY-noya/lmms-eval.git" "${LMMS_EVAL_SOURCE_DIR}"

if [[ ! -x "${QIG_VENV_DIR}/bin/python" ]]; then
    if ! command -v "${PYTHON311}" >/dev/null 2>&1; then
        echo "Python 3.11 was not found (${PYTHON311}); QIG documents Python 3.11." >&2
        exit 1
    fi
    echo "Creating isolated QIG environment: ${QIG_VENV_DIR}"
    "${PYTHON311}" -m venv "${QIG_VENV_DIR}"
fi

QIG_PYTHON="${QIG_VENV_DIR}/bin/python"
QIG_SETUP_MARKER="${QIG_VENV_DIR}/.qig-dependencies-source-selected-v2"
if [[ ! -f "${QIG_SETUP_MARKER}" ]]; then
    "${QIG_PYTHON}" -m pip install --upgrade pip
    "${QIG_PYTHON}" -m pip install -r "${REPOSITORY_ROOT}/requirements-qig.txt"
    "${QIG_PYTHON}" -m pip install -e "${LLAVA_SOURCE_DIR}" --no-deps
    "${QIG_PYTHON}" -m pip install -e "${LMMS_EVAL_SOURCE_DIR}" --no-deps
    "${QIG_PYTHON}" -m pip install -e "${QIG_SOURCE_DIR}" --no-deps
    "${QIG_PYTHON}" -m pip install -e "${REPOSITORY_ROOT}" --no-deps
    "${QIG_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available to this PyTorch install"; print(f"PyTorch {torch.__version__}; CUDA device: {torch.cuda.get_device_name(0)}")'
    touch "${QIG_SETUP_MARKER}"
else
    echo "Using installed QIG environment: ${QIG_VENV_DIR}"
fi

export QIG_SOURCE_DIR
cd "${QIG_SOURCE_DIR}"
exec "${QIG_PYTHON}" "${REPOSITORY_ROOT}/scripts/quantize_qig.py" "$@"
