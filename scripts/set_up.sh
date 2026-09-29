#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPOSITORY_ROOT}/.venv}"

if [[ "${VENV_DIR}" != /* ]]; then
    VENV_DIR="${REPOSITORY_ROOT}/${VENV_DIR}"
fi

if VENV_PARENT="$(cd -- "$(dirname -- "${VENV_DIR}")" 2>/dev/null && pwd)"; then
    VENV_DIR="${VENV_PARENT}/$(basename -- "${VENV_DIR}")"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if ACTIVE_ENV="$(cd -- "${VIRTUAL_ENV}" 2>/dev/null && pwd)" && [[ "${ACTIVE_ENV}" == "${VENV_DIR}" ]]; then
        echo "Already running in the requested environment: ${VENV_DIR}"
        exit 0
    fi
fi

PYTHON_BIN="${PYTHON:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    mkdir -p "$(dirname -- "${VENV_DIR}")"
    echo "Creating virtual environment: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Virtual environment Python was not found: ${VENV_PYTHON}" >&2
    exit 1
fi

"${VENV_PYTHON}" -m pip install -r "${REPOSITORY_ROOT}/requirements.txt"
"${VENV_PYTHON}" -m pip install -e "${REPOSITORY_ROOT}"

echo "Setup complete. Activate the environment with:"
echo "  source \"${VENV_DIR}/bin/activate\""
