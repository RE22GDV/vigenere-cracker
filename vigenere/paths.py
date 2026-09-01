# -*- coding: utf-8 -*-
"""Locations of downloadable language data and cached artefacts.

Data lives outside the package so that the repository stays small: corpora and
word lists are tens of megabytes and are downloaded on demand.

Resolution order for the data directory:
  1. ``$VIGENERE_DATA`` if set;
  2. ``<repository root>/data`` when the package is used from a source checkout;
  3. ``~/.vigenere-cracker/data`` for installed copies.
"""

from __future__ import annotations

import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)


def _default_data_dir() -> str:
    env = os.environ.get("VIGENERE_DATA")
    if env:
        return os.path.abspath(env)
    local = os.path.join(REPO_DIR, "data")
    if os.path.isdir(local) or os.access(REPO_DIR, os.W_OK):
        return local
    return os.path.join(os.path.expanduser("~"), ".vigenere-cracker", "data")


DATA_DIR = _default_data_dir()


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)
