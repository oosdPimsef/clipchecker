# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.comparative_claims import (
    COMPARATIVE_TERMS,
    NO_COMPARATIVE_WORDS_MESSAGE,
    evaluate_comparative_claims,
    extract_comparative_terms,
)


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


def make_result_dir(ocr_log: dict):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    for frame_name in ocr_log:
        Image.new("RGB", (1000, 500), "white").save(frames / frame_name)
    (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class ComparativeClaimsTests(unittest.TestCase):
    def test_dictionary_has_at_least_50_terms(self):
        self.assertGreaterEqual(len(COMPARATIVE_TERMS), 50)

    def test_extract_comparative_terms_finds_variants(self):
        terms = extract_comparative_terms("Выгоднее и дешевле конкурентов")
        self.assertIn("выгоднее", terms)
        self.assertIn("дешевле", terms)
        self.assertIn("конкурент", terms)

    def test_evaluate_comparative_claims_passes_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Новая коллекция украшений"]}))
        try:
            result = evaluate_comparative_claims(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_COMPARATIVE_WORDS_MESSAGE)

    def test_evaluate_comparative_claims_fails_and_formats_html(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                {
                    "frame_001.jpg": ["Лучшее предложение"],
                    "frame_003.jpg": ["Дешевле конкурентов"],
                }
            )
        )
        try:
            result = evaluate_comparative_claims(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("лучшее", result["message"].lower())
        self.assertIn("1сек.", result["message"])
        self.assertIn("3сек.", result["message"])
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])
        self.assertIn("лучшее", result["message_html"].lower())
        self.assertIn("дешевле", result["message_html"].lower())

    def test_evaluates_item_id_16_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Самая низкая цена"]}))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "16", "number": "16", "text": "сравнение"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "fail")
        self.assertIn("message_html", item)
        self.assertIn("низкая цена", item["message"].lower())


if __name__ == "__main__":
    unittest.main()
