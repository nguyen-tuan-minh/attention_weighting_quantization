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
        ACTIVE_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
        if [[ "${ACTIVE_PYTHON_VERSION}" != "3.11" ]]; then
            echo "The active .venv uses Python ${ACTIVE_PYTHON_VERSION}; recreate it with Python 3.11 before setup." >&2
            exit 1
        fi
        echo "Already running in the requested environment: ${VENV_DIR}"
        exit 0
    fi
fi

export VENV_DIR
exec "${SCRIPT_DIR}/quantize_qig.sh" --setup-only
