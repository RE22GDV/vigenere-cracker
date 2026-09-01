#!/usr/bin/env bash
# Launch the graphical interface, preferring the local virtual environment.
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
    exec .venv/bin/python -m vigenere.gui "$@"
fi
exec "${PYTHON:-python3}" -m vigenere.gui "$@"
