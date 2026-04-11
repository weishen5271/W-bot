#!/usr/bin/env bash

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_UV_VERSION="${INSTALL_UV_VERSION:-}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: uv is not installed and curl is required to install it automatically." >&2
    echo "Please install curl first, then rerun this script." >&2
    exit 1
  fi

  echo "Installing uv..."
  if [[ -n "${INSTALL_UV_VERSION}" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_VERSION="${INSTALL_UV_VERSION}" sh
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi

  export PATH="${HOME}/.local/bin:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv installation completed, but 'uv' is still not available in PATH." >&2
    echo "Try running: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
  fi
}

main() {
  ensure_uv

  echo "Using uv: $(command -v uv)"
  echo "Creating virtual environment in ${VENV_DIR} with Python ${PYTHON_VERSION}..."
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

  echo "Installing project in editable mode..."
  uv pip install --python "${VENV_DIR}/bin/python" -e .

  echo
  echo "Bootstrap complete."
  echo "Activate the environment with:"
  echo "  source ${VENV_DIR}/bin/activate"
}

main "$@"
