# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.foreign_words import WARNING_TEXT, evaluate_foreign_words, extract_foreign_words


def make_ocr_line(text: str, box: tuple[int, int, int, int] = (120, 80, 500, 120)) -> dict:
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


def make_ocr_log(lines: list[str]) -> dict:
    return {
        "frame_001.jpg": {
            "results": [
                {
                    "results": [
                        {
                            "textDetection": {
                                "pages": [
                                    {
                                        "blocks": [
                                            {
                                                "lines": [make_ocr_line(line) for line in lines],
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


class ForeignWordsTests(unittest.TestCase):
    def test_extract_foreign_words_ignores_urls(self):
        self.assertEqual(extract_foreign_words("SOKOLOV sale https www"), ["SOKOLOV", "sale"])

    def test_evaluate_foreign_words_warns_and_formats_html(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV sale", "Реклама"]))
        try:
            result = evaluate_foreign_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn(WARNING_TEXT, result["message"])
        self.assertIn('<strong class="foreign-word">SOKOLOV</strong>', result["message_html"])
        self.assertIn('<strong class="foreign-word">sale</strong>', result["message_html"])

    def test_evaluate_foreign_words_passes_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама акция"]))
        try:
            result = evaluate_foreign_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_evaluates_item_id_8_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV"]))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Набивка", "items": [{"id": "8", "number": "8", "text": "иностранные слова"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "warning")
        self.assertIn("message_html", item)
        self.assertIn("SOKOLOV", item["message"])


if __name__ == "__main__":
    unittest.main()
