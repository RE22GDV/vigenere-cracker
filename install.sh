#!/usr/bin/env bash
# One-shot setup for Linux/macOS: virtual environment, dependencies, language data.
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " Vigenere Cracker - setup"
echo "============================================================"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[!] $PY not found. Install Python 3.9 or newer."
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[1/4] Creating the virtual environment..."
    "$PY" -m venv .venv
else
    echo "[1/4] Virtual environment already exists."
fi

echo "[2/4] Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt

echo
echo "[3/4] Optional: CUDA support (about 2.5 GB, ~10x faster brute force)."
read -r -p "    Install PyTorch with CUDA? [y/N]: " CUDA
if [[ "${CUDA:-n}" =~ ^[Yy]$ ]]; then
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

echo
echo "[4/4] Language data (word lists and corpora, about 280 MB)."
read -r -p "    Download now? [Y/n]: " DATA
if [[ ! "${DATA:-y}" =~ ^[Nn]$ ]]; then
    .venv/bin/python -m vigenere.download_data
fi

echo
echo "============================================================"
echo " Done. Start the app with ./run_gui.sh"
echo "============================================================"
