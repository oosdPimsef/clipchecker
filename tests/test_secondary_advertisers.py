# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook
from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.secondary_advertisers import (
    BRAND_BEARING_CV_LABELS,
    evaluate_secondary_advertisers,
    load_secondary_brand_database,
)


def make_ocr_line(text: str, box: tuple[int, int, int, int] = (120, 80, 700, 140)) -> dict:
    x1, y1, x2, y2 = box
    vertices = [
        {"x": str(x1), "y": str(y1)},
        {"x": str(x1), "y": str(y2)},
        {"x": str(x2), "y": str(y2)},
        {"x": str(x2), "y": str(y1)},
    ]
    return {
        "boundingBox": {"vertices": vertices},
        "words": [{"text": word, "boundingBox": {"vertices": vertices}} for word in text.split()],
    }


def make_ocr_log(frame_lines: dict[str, list[str]]) -> dict:
    data = {}
    for frame, lines in frame_lines.items():
        data[frame] = {
            "results": [
                {
                    "results": [
                        {
                            "textDetection": {
                                "pages": [{"blocks": [{"lines": [make_ocr_line(line) for line in lines]}]}]
                            }
                        }
                    ]
                }
            ]
        }
    return data


def make_brand_database(path: Path, brands: list[str]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Brands"
    for row, brand in enumerate(brands, start=1):
        worksheet.cell(row=row, column=1, value=brand)
    workbook.save(path)


def make_result_dir(ocr_log: dict | None = None, documents_text: str = ""):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    frame_names = list(ocr_log or {"frame_001.jpg": []})
    for frame_name in frame_names:
        Image.new("RGB", (1000, 500), "white").save(frames / frame_name)
    if ocr_log is not None:
        (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    if documents_text:
        (base / "Documents_Texts.txt").write_text(documents_text, encoding="utf-8")
    return tmp, base


class SecondaryAdvertisersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brand_path = Path(self.tmp.name) / "brands.xlsx"
        make_brand_database(self.brand_path, ["Основной бренд", "PepsiCo", "X5 RETAIL GROUP", "АВТОВАЗ (LADA)"])
        self.brand_patch = patch("app.secondary_advertisers.BRAND_DATABASE_PATH", self.brand_path)
        self.cv_patch = patch(
            "app.secondary_advertisers.analyze_secondary_advertisers_from_cv",
            return_value={
                "cv_enabled": False,
                "cv_model_path": "",
                "cv_error": "YOLO test disabled",
                "cv_brand_mentions": [],
                "cv_brand_bearing_objects": [],
            },
        )
        self.brand_patch.start()
        self.cv_patch.start()

    def tearDown(self):
        self.cv_patch.stop()
        self.brand_patch.stop()
        self.tmp.cleanup()

    def test_loads_brand_database_and_parentheses_variants(self):
        records = load_secondary_brand_database(self.brand_path)
        lada = [record for record in records if record.name == "АВТОВАЗ (LADA)"][0]
        self.assertIn("LADA", lada.variants)
        self.assertIn("АВТОВАЗ", lada.variants)

    def test_brand_database_is_read_strictly_from_brands_sheet(self):
        workbook = Workbook()
        workbook.active.title = "Other"
        workbook.active["A1"] = "PepsiCo"
        strict_path = Path(self.tmp.name) / "strict_brands.xlsx"
        workbook.save(strict_path)

        records = load_secondary_brand_database(strict_path)

        self.assertEqual(records, [])

    def test_passes_when_only_primary_advertiser_is_found(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Основной бренд"], "frame_002.jpg": ["Основной бренд"]}),
            documents_text="Рекламодатель: Основной бренд",
        )
        try:
            result = evaluate_secondary_advertisers(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["primary_advertiser"], "Основной бренд")
        self.assertIn('style="color:#16a34a;font-weight:800"', result["message_html"])

    def test_fails_and_counts_seconds_for_secondary_advertisers(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                {
                    "frame_001.jpg": ["Основной бренд"],
                    "frame_002.jpg": ["PepsiCo"],
                    "frame_003.jpg": ["PepsiCo"],
                }
            ),
            documents_text="Рекламодатель: Основной бренд",
        )
        try:
            result = evaluate_secondary_advertisers(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["primary_advertiser"], "Основной бренд")
        self.assertEqual(result["secondary_advertisers"][0]["brand"], "PepsiCo")
        self.assertEqual(result["secondary_advertisers"][0]["duration_sec"], 2)
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])

    def test_cv_brand_detection_can_find_secondary_without_ocr(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Основной бренд"]}), documents_text="Основной бренд")
        fake_cv = {
            "cv_enabled": True,
            "cv_model_path": "model.pt",
            "cv_error": "",
            "cv_brand_mentions": [
                {
                    "brand": "PepsiCo",
                    "seconds": ["1сек."],
                    "frames": ["frame_001.jpg"],
                    "sources": ["cv"],
                    "cv_labels": ["PepsiCo"],
                    "duration_sec": 1,
                }
            ],
            "cv_brand_bearing_objects": [],
        }
        try:
            with patch("app.secondary_advertisers.analyze_secondary_advertisers_from_cv", return_value=fake_cv):
                result = evaluate_secondary_advertisers(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["secondary_advertisers"][0]["brand"], "PepsiCo")

    def test_warns_when_packaging_or_label_zone_is_unrecognized(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["нейтральный текст"]}))
        fake_cv = {
            "cv_enabled": True,
            "cv_model_path": "model.pt",
            "cv_error": "",
            "cv_brand_mentions": [],
            "cv_brand_bearing_objects": [
                {"label": "product packaging", "raw_label": "product packaging", "frame": "frame_001.jpg", "second": "1сек.", "confidence": 0.8}
            ],
        }
        try:
            with patch("app.secondary_advertisers.analyze_secondary_advertisers_from_cv", return_value=fake_cv):
                result = evaluate_secondary_advertisers(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("упаковки", result["message"])

    def test_evaluates_item_id_220_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Основной бренд"], "frame_002.jpg": ["X5 RETAIL GROUP"]}),
            documents_text="Основной бренд",
        )
        view_model = {
            "ok": True,
            "blocks": [{"name": "Видеоряд", "items": [{"id": "220", "number": "220", "text": "2rd"}]}],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "fail")
        self.assertIn("X5 RETAIL GROUP", item["message"])

    def test_cv_generic_labels_include_packaging_and_labels(self):
        normalized = {label.lower() for label in BRAND_BEARING_CV_LABELS}
        self.assertIn("product packaging", normalized)
        self.assertIn("label", normalized)
        self.assertIn("brand logo", normalized)


if __name__ == "__main__":
    unittest.main()
