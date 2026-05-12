# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.grammar_checks import (
    check_word,
    evaluate_legal_disclaimer_grammar,
    evaluate_non_legal_grammar,
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


def make_ocr_log(items: list[tuple[str, tuple[int, int, int, int], str]]) -> dict:
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


class GrammarChecksTests(unittest.TestCase):
    def test_check_word_detects_local_issues(self):
        self.assertIsNone(check_word("SOKOLOV"))
        self.assertIsNone(check_word("ООО"))
        self.assertIsNone(check_word("пр-зд"))
        self.assertIn("смешаны", check_word("реклaма")["issue"])
        self.assertIn("информация", check_word("инфоормация")["issue"])
        self.assertIn("повторение", check_word("скидкаааа")["issue"])

    def test_non_legal_grammar_checks_only_words_outside_disclaimer(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                [
                    ("Большая инфоормация о скидке", (120, 80, 420, 110), "frame_001.jpg"),
                    ("Рекламодатель ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430), "frame_001.jpg"),
                ]
            )
        )
        try:
            result = evaluate_non_legal_grammar(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("инфоормация", result["message"])

    def test_legal_disclaimer_grammar_checks_only_bottom_disclaimer(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                [
                    ("Большая информация о скидке", (120, 80, 420, 110), "frame_001.jpg"),
                    ("Рекламодателль ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430), "frame_001.jpg"),
                ]
            )
        )
        try:
            result = evaluate_legal_disclaimer_grammar(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("Рекламодателль", result["message"])

    def test_grammar_ids_6_and_7_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                [
                    ("Большая инфоормация о скидке", (120, 80, 420, 110), "frame_001.jpg"),
                    ("Рекламодателль ООО Ромашка, ОГРН 1234567890", (120, 400, 700, 430), "frame_001.jpg"),
                ]
            )
        )
        view_model = {
            "ok": True,
            "blocks": [
                {
                    "name": "Набивка",
                    "items": [
                        {"id": "6", "number": "6", "text": "грамматика вне набивки"},
                        {"id": "7", "number": "7", "text": "грамматика набивки"},
                    ],
                }
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        self.assertEqual(evaluated["blocks"][0]["items"][0]["status"], "fail")
        self.assertIn("инфоормация", evaluated["blocks"][0]["items"][0]["message"])
        self.assertEqual(evaluated["blocks"][0]["items"][1]["status"], "fail")
        self.assertIn("Рекламодателль", evaluated["blocks"][0]["items"][1]["message"])


if __name__ == "__main__":
    unittest.main()
