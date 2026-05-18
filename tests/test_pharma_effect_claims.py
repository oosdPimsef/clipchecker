# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.pharma_effect_claims import (
    NO_PHARMA_EFFECT_CLAIMS_MESSAGE,
    PHARMA_EFFECT_TERMS,
    evaluate_pharma_effect_claims,
    extract_pharma_effect_terms,
)


def make_ocr_line(text: str, box: tuple[int, int, int, int] = (120, 80, 800, 130)) -> dict:
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


class PharmaEffectClaimsTests(unittest.TestCase):
    def test_dictionary_has_at_least_30_terms(self):
        self.assertGreaterEqual(len(PHARMA_EFFECT_TERMS), 30)

    def test_extracts_inflected_effect_terms(self):
        terms = extract_pharma_effect_terms("Лечит простуду, спасает от боли и гарантированный результат")
        self.assertIn("лечить", terms)
        self.assertIn("спасти", terms)
        self.assertIn("гарантировать", terms)
        self.assertIn("результат", terms)

    def test_passes_when_claims_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Новая упаковка препарата"]}))
        try:
            result = evaluate_pharma_effect_claims(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_PHARMA_EFFECT_CLAIMS_MESSAGE)

    def test_fails_and_formats_html_when_claims_found(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                {
                    "frame_001.jpg": ["Лечит и быстро помогает"],
                    "frame_003.jpg": ["Гарантированный результат"],
                }
            )
        )
        try:
            result = evaluate_pharma_effect_claims(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("лечить", result["message"])
        self.assertIn("гарантировать", result["message"])
        self.assertIn("1сек.", result["message"])
        self.assertIn("3сек.", result["message"])
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])

    def test_evaluates_item_id_21_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Средство устраняет симптомы"]}))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "21", "number": "21", "text": "гарантии действия"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "fail")
        self.assertIn("устранить", item["message"])


if __name__ == "__main__":
    unittest.main()
