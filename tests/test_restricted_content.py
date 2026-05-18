# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.restricted_content import (
    RESTRICTED_CONTENT_CHECKS,
    evaluate_alcohol_references,
    evaluate_drug_references,
    evaluate_restricted_content,
    evaluate_smoking_references,
    evaluate_weapon_violence_references,
    extract_restricted_terms,
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


def make_result_dir(ocr_log: dict | None = None):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    frame_names = list(ocr_log or {"frame_001.jpg": []})
    for frame_name in frame_names:
        Image.new("RGB", (1000, 500), "white").save(frames / frame_name)
    if ocr_log is not None:
        (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class RestrictedContentTests(unittest.TestCase):
    def test_dictionaries_have_practical_size(self):
        for check_id in ("249", "269", "270", "271"):
            self.assertGreaterEqual(len(RESTRICTED_CONTENT_CHECKS[check_id]["terms"]), 35)

    def test_extracts_terms_for_each_category(self):
        self.assertIn("вино", extract_restricted_terms("Красное вино в бокале", "249"))
        self.assertIn("сигарета", extract_restricted_terms("Электронная сигарета и дым", "269"))
        self.assertIn("наркотик", extract_restricted_terms("Наркотики запрещены", "270"))
        self.assertIn("пистолет", extract_restricted_terms("Пистолет и патроны", "271"))

    def test_evaluate_alcohol_fails_and_formats_html(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Бокал вина"], "frame_003.jpg": ["бутылка пива"]}))
        try:
            result = evaluate_alcohol_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("вина", result["message"].lower())
        self.assertIn("3сек.", result["message"])
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])

    def test_evaluate_smoking_fails(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["кальян и табак"]}))
        try:
            result = evaluate_smoking_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("кальян", result["message"].lower())

    def test_evaluate_drugs_fails(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["запрещенные наркотики"]}))
        try:
            result = evaluate_drug_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("наркотик", result["message"].lower())

    def test_evaluate_weapons_fails(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["нож и пистолет"]}))
        try:
            result = evaluate_weapon_violence_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("пистолет", result["message"].lower())

    def test_passes_when_terms_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Коллекция украшений"]}))
        try:
            result = evaluate_restricted_content(base, "249")
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("cv_enabled", result)

    def test_cv_detection_fails_even_without_ocr_terms(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Коллекция украшений"]}))
        fake_cv = {
            "cv_enabled": True,
            "cv_model_path": "model.pt",
            "cv_error": "",
            "cv_detections": [
                {
                    "label": "bottle",
                    "raw_label": "bottle",
                    "confidence": 0.91,
                    "frame": "frame_001.jpg",
                    "second": "1сек.",
                }
            ],
            "cv_restricted_mentions": [
                {
                    "label": "bottle",
                    "raw_label": "bottle",
                    "confidence": 0.91,
                    "frame": "frame_001.jpg",
                    "second": "1сек.",
                }
            ],
        }
        try:
            with patch("app.restricted_content.analyze_restricted_content_from_cv", return_value=fake_cv):
                result = evaluate_alcohol_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("CV", result["message"])
        self.assertIn("bottle", result["message"])

    def test_cv_model_absence_is_reported_in_details(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Коллекция украшений"]}))
        try:
            result = evaluate_alcohol_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["cv_enabled"])
        self.assertIn("YOLO", result["cv_error"])

    def test_evaluates_item_ids_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                {
                    "frame_001.jpg": ["вино", "сигарета", "наркотик", "пистолет"],
                }
            )
        )
        view_model = {
            "ok": True,
            "blocks": [
                {
                    "name": "Видеоряд",
                    "items": [
                        {"id": "249", "number": "249", "text": "алкоголь"},
                        {"id": "269", "number": "269", "text": "курение"},
                        {"id": "270", "number": "270", "text": "наркотики"},
                        {"id": "271", "number": "271", "text": "оружие"},
                    ],
                },
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        statuses = [item["status"] for item in evaluated["blocks"][0]["items"]]
        self.assertEqual(statuses, ["fail", "fail", "fail", "fail"])
        for item in evaluated["blocks"][0]["items"]:
            self.assertIn("message_html", item)


if __name__ == "__main__":
    unittest.main()
