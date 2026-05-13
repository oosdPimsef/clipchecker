import shutil
import tempfile
import unittest
from pathlib import Path

from app.materials_workspace import MATERIALS_DIR_NAME, ensure_materials_dir, prepare_materials_dir


class MaterialsWorkspaceTests(unittest.TestCase):
    def test_recreates_existing_materials_directory(self):
        tmp = tempfile.mkdtemp()
        try:
            preview_dir = Path(tmp) / "preview"
            old_dir = preview_dir / MATERIALS_DIR_NAME
            nested_dir = old_dir / "nested"
            nested_dir.mkdir(parents=True)
            (old_dir / "old.txt").write_text("old", encoding="utf-8")
            (nested_dir / "nested.txt").write_text("old", encoding="utf-8")

            materials_dir = prepare_materials_dir(preview_dir)

            self.assertEqual(materials_dir, old_dir.resolve())
            self.assertTrue(materials_dir.is_dir())
            self.assertFalse((materials_dir / "old.txt").exists())
            self.assertFalse(nested_dir.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_replaces_file_with_materials_directory(self):
        tmp = tempfile.mkdtemp()
        try:
            preview_dir = Path(tmp) / "preview"
            preview_dir.mkdir(parents=True)
            materials_path = preview_dir / MATERIALS_DIR_NAME
            materials_path.write_text("not a directory", encoding="utf-8")

            materials_dir = prepare_materials_dir(preview_dir)

            self.assertTrue(materials_dir.is_dir())
            self.assertEqual(list(materials_dir.iterdir()), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ensure_materials_directory_keeps_existing_files(self):
        tmp = tempfile.mkdtemp()
        try:
            preview_dir = Path(tmp) / "preview"
            materials_dir = preview_dir / MATERIALS_DIR_NAME
            materials_dir.mkdir(parents=True)
            old_file = materials_dir / "old.txt"
            old_file.write_text("old", encoding="utf-8")

            ensured_dir = ensure_materials_dir(preview_dir)

            self.assertEqual(ensured_dir, materials_dir.resolve())
            self.assertEqual(old_file.read_text(encoding="utf-8"), "old")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
