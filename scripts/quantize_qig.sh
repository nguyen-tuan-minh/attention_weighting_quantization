#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QIG_SOURCE_DIR="${QIG_SOURCE_DIR:-${REPOSITORY_ROOT}/.third_party/QIG}"
VENV_DIR="${VENV_DIR:-${REPOSITORY_ROOT}/.venv}"
PYTHON_BIN="${PYTHON:-python3.11}"
LLAVA_SOURCE_DIR="${QIG_SOURCE_DIR}/3rdparty/LLaVA-NeXT"
LMMS_EVAL_SOURCE_DIR="${QIG_SOURCE_DIR}/3rdparty/lmms-eval"

if [[ "${VENV_DIR}" != /* ]]; then
    VENV_DIR="${REPOSITORY_ROOT}/${VENV_DIR}"
fi
SETUP_ONLY=false
FORWARD_ARGS=()

for arg in "$@"; do
    if [[ "${arg}" == "--setup-only" ]]; then
        SETUP_ONLY=true
    else
        FORWARD_ARGS+=("${arg}")
    fi
done

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

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python 3.11 was not found (${PYTHON_BIN}); this project's shared environment uses Python 3.11." >&2
        exit 1
    fi
    echo "Creating project environment: ${VENV_DIR}"
    mkdir -p "$(dirname -- "${VENV_DIR}")"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PROJECT_PYTHON="${VENV_DIR}/bin/python"
PYTHON_VERSION="$(${PROJECT_PYTHON} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
    echo "The shared .venv uses Python ${PYTHON_VERSION}; recreate it with Python 3.11 before installing the quantization packages." >&2
    exit 1
fi

SETUP_MARKER="${VENV_DIR}/.project-dependencies-source-selected-v5"
if [[ ! -f "${SETUP_MARKER}" ]]; then
    "${PROJECT_PYTHON}" -m pip install --upgrade pip
    "${PROJECT_PYTHON}" -m pip install -r "${REPOSITORY_ROOT}/requirements.txt"
    "${PROJECT_PYTHON}" -m pip install -e "${LLAVA_SOURCE_DIR}" --no-deps
    "${PROJECT_PYTHON}" -m pip install -e "${LMMS_EVAL_SOURCE_DIR}" --no-deps
    "${PROJECT_PYTHON}" -m pip install -e "${QIG_SOURCE_DIR}" --no-deps
    "${PROJECT_PYTHON}" -m pip install -e "${REPOSITORY_ROOT}" --no-deps
    "${PROJECT_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available to this PyTorch install"; print(f"PyTorch {torch.__version__}; CUDA device: {torch.cuda.get_device_name(0)}")'
    touch "${SETUP_MARKER}"
else
    echo "Using installed project environment: ${VENV_DIR}"
fi

if [[ "${SETUP_ONLY}" == true ]]; then
    echo "Project environment is ready: ${VENV_DIR}"
    exit 0
fi

export QIG_SOURCE_DIR
cd "${QIG_SOURCE_DIR}"
exec "${PROJECT_PYTHON}" "${REPOSITORY_ROOT}/scripts/quantize_qig.py" "${FORWARD_ARGS[@]}"
