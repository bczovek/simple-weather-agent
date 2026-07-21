#!/usr/bin/env bash
# Sets up an isolated virtual environment for the weather agent and runs its
# interactive CLI. Nothing is installed globally: everything lives under
# ./.venv, which is created next to this script on first run and reused
# afterwards.
#
# Configuration (OPENAI_API_KEY, etc.) is read from a .env file in this
# directory - copy .env.example to .env and fill in your values before
# running this script.
#
# Usage:
#   ./run.sh
#   ./run.sh --verbose
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet -e "${SCRIPT_DIR}"

python -m weather_agent.main "$@"
