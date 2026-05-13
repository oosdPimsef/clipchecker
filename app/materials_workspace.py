# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
from pathlib import Path


MATERIALS_DIR_NAME = "Clipchecker_materials"


def prepare_materials_dir(preview_dir_path: str | Path) -> Path:
    """Recreate the generated materials directory inside the selected preview folder."""
    preview_dir = Path(preview_dir_path).resolve()
    materials_dir = (preview_dir / MATERIALS_DIR_NAME).resolve()

    if materials_dir.name != MATERIALS_DIR_NAME or materials_dir.parent != preview_dir:
        raise RuntimeError(f"Unsafe materials directory path: {materials_dir}")

    if materials_dir.exists():
        if materials_dir.is_dir():
            shutil.rmtree(materials_dir)
        else:
            materials_dir.unlink()

    materials_dir.mkdir(parents=True, exist_ok=True)
    return materials_dir


def ensure_materials_dir(preview_dir_path: str | Path) -> Path:
    """Create the generated materials directory without removing existing files."""
    preview_dir = Path(preview_dir_path).resolve()
    materials_dir = (preview_dir / MATERIALS_DIR_NAME).resolve()

    if materials_dir.name != MATERIALS_DIR_NAME or materials_dir.parent != preview_dir:
        raise RuntimeError(f"Unsafe materials directory path: {materials_dir}")

    materials_dir.mkdir(parents=True, exist_ok=True)
    return materials_dir
