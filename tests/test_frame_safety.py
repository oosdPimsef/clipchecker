# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.frame_safety import analyze_frame_safety, classify_substantial_text, evaluate_frame_safety


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


def make_result_dir(ocr_log: dict):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    Image.new("RGB", (1000, 500), "white").save(frames / "frame_001.jpg")
    (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class FrameSafetyTests(unittest.TestCase):
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
        tmp, base = make_result_dir(make_ocr_log("Цена 1 999 руб.", (120, 40, 300, 80)))
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


if __name__ == "__main__":
    unittest.main()
