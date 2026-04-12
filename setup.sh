#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_VERSION="3.11"
VENV_DIR=".venv"
KERNEL_NAME="neural-ssm"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required but was not found on PATH."
    echo "Install uv first, then rerun this script."
    exit 1
fi

echo "Installing Python ${PYTHON_VERSION} with uv if needed..."
uv python install "${PYTHON_VERSION}"

echo "Creating ${VENV_DIR} with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

echo "Syncing project dependencies into ${VENV_DIR}..."
uv sync

echo "Verifying ${VENV_DIR} is using Python ${PYTHON_VERSION}..."
ACTUAL_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${ACTUAL_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]]; then
    echo "error: ${VENV_DIR} is using Python ${ACTUAL_PYTHON_VERSION}, expected ${PYTHON_VERSION}."
    echo "Remove ${VENV_DIR} and rerun setup."
    exit 1
fi

echo "Installing Git pre-commit hook..."
"${VENV_DIR}/bin/python" -m pre_commit install --install-hooks

echo "Registering Jupyter kernel '${KERNEL_NAME}'..."
"${VENV_DIR}/bin/python" -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_NAME}"

cat <<EOF

Setup complete.

Next steps:
  source ${VENV_DIR}/bin/activate
  jupyter lab

In Jupyter, select the '${KERNEL_NAME}' kernel.
EOF
