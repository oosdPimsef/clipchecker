# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.frame_safety import (
    analyze_frame_safety,
    classify_substantial_text,
    evaluate_all_text_safety,
    evaluate_frame_safety,
    evaluate_legal_disclaimer_safety,
    evaluate_logo_safety,
)


def make_ocr_line(text: str, box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = box
    vertices = [
        {"x": str(x1), "y": str(y1)},
        {"x": str(x1), "y": str(y2)},
        {"x": str(x2), "y": str(y2)},
        {"x": str(x2), "y": str(y1)},
    ]
    return {
        "boundingBox": {"vertices": vertices},
        "words": [{"text": text, "boundingBox": {"vertices": vertices}}],
    }


def make_ocr_log(text: str, box: tuple[int, int, int, int], frame_name: str = "frame_001.jpg") -> dict:
    return {
        frame_name: {
            "results": [
                {
                    "results": [
                        {
                            "textDetection": {
                                "pages": [
                                    {
                                        "blocks": [
                                            {
                                                "lines": [make_ocr_line(text, box)],
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    }


def make_ocr_log_many(items: list[tuple[str, tuple[int, int, int, int], str]]) -> dict:
    data: dict = {}
    for text, box, frame_name in items:
        data.setdefault(
            frame_name,
            {
                "results": [
                    {
                        "results": [
                            {
                                "textDetection": {
                                    "pages": [
                                        {
                                            "blocks": [
                                                {
                                                    "lines": [],
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
        )
        data[frame_name]["results"][0]["results"][0]["textDetection"]["pages"][0]["blocks"][0]["lines"].append(
            make_ocr_line(text, box)
        )
    return data


def make_result_dir(ocr_log: dict):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    for frame_name in ocr_log:
        Image.new("RGB", (1000, 500), "white").save(frames / frame_name)
    (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class FrameSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cv_patcher = patch(
            "app.frame_safety.detect_cv_objects_in_frames",
            return_value={"enabled": False, "model_path": "", "error": "test", "detections": []},
        )
        self.cv_patcher.start()

    def tearDown(self):
        self.cv_patcher.stop()

    def test_classifies_age_mark_and_price_as_substantial(self):
        self.assertEqual(classify_substantial_text("18+"), "age_mark")
        self.assertEqual(classify_substantial_text("1 999 руб."), "price")
        self.assertEqual(classify_substantial_text("SOKOLOV"), "text")
        self.assertEqual(classify_substantial_text("2 = 1"), "text")
        self.assertIsNone(classify_substantial_text("15"))
        self.assertIsNone(classify_substantial_text("*"))

    def test_analyze_frame_safety_passes_when_text_is_inside_green_frame(self):
        tmp, base = make_result_dir(make_ocr_log("18+", (850, 40, 890, 80)))
        try:
            result = evaluate_frame_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_count"], 1)
        self.assertEqual(result["violation_count"], 0)

    def test_analyze_frame_safety_fails_when_text_is_outside_green_frame(self):
        tmp, base = make_result_dir(make_ocr_log("SOKOLOV", (20, 40, 90, 80)))
        try:
            result = evaluate_frame_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checked_count"], 1)
        self.assertEqual(result["violation_count"], 1)
        self.assertIn("SOKOLOV", result["message"])

    def test_evaluates_item_id_3_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log("Рекламодатель ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430)))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "3", "number": "3", "text": "рамка"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        self.assertEqual(evaluated["blocks"][0]["items"][0]["status"], "pass")

    def test_legal_disclaimer_scope_ignores_top_logo(self):
        tmp, base = make_result_dir(
            make_ocr_log_many(
                [
                    ("SOKOLOV", (20, 40, 90, 80), "frame_001.jpg"),
                    ("Рекламодатель ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430), "frame_001.jpg"),
                ]
            )
        )
        try:
            result = evaluate_legal_disclaimer_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_count"], 1)

    def test_legal_disclaimer_scope_fails_for_bottom_legal_text_outside_frame(self):
        tmp, base = make_result_dir(make_ocr_log("Рекламодатель ООО Ромашка, ОГРН 1234567890", (20, 400, 700, 430)))
        try:
            result = evaluate_legal_disclaimer_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("Рекламодатель", result["message"])

    def test_logo_scope_uses_repeated_short_brand_text(self):
        tmp, base = make_result_dir(
            make_ocr_log_many(
                [
                    ("SOKOLOV", (20, 40, 90, 80), "frame_001.jpg"),
                    ("SOKOLOV", (20, 40, 90, 80), "frame_002.jpg"),
                    ("НА ВСЕ", (20, 120, 90, 150), "frame_001.jpg"),
                ]
            )
        )
        try:
            result = evaluate_logo_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checked_count"], 2)
        self.assertIn("SOKOLOV", result["message"])

    def test_all_text_scope_checks_price_age_and_legal_text(self):
        tmp, base = make_result_dir(
            make_ocr_log_many(
                [
                    ("18+", (850, 40, 890, 80), "frame_001.jpg"),
                    ("Цена 1 999 руб.", (20, 100, 200, 130), "frame_001.jpg"),
                    ("Рекламодатель ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430), "frame_001.jpg"),
                ]
            )
        )
        try:
            result = evaluate_all_text_safety(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checked_count"], 3)
        self.assertIn("Цена 1 999 руб.", result["message"])

    def test_logo_scope_fails_for_visual_logo_outside_green_frame(self):
        tmp, base = make_result_dir(make_ocr_log("Новая коллекция", (120, 100, 260, 130)))
        self.cv_patcher.stop()
        fake_cv = {
            "enabled": True,
            "model_path": "model.pt",
            "error": "",
            "detections": [
                {
                    "label": "logo",
                    "raw_label": "logo",
                    "confidence": 0.91,
                    "frame": "frame_001.jpg",
                    "second": "1сек.",
                    "bbox": [20, 30, 120, 90],
                }
            ],
        }
        try:
            with patch("app.frame_safety.detect_cv_objects_in_frames", return_value=fake_cv):
                result = evaluate_logo_safety(base)
        finally:
            tmp.cleanup()
            self.cv_patcher.start()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["cv_checked_count"], 1)
        self.assertEqual(result["cv_violation_count"], 1)
        self.assertIn("CV: logo", result["message"])


if __name__ == "__main__":
    unittest.main()
