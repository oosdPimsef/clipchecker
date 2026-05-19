# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.visual_reference_checks import (
    VISUAL_REFERENCE_CHECKS,
    evaluate_car_license_plate,
    evaluate_disease_references,
    evaluate_doctor_image_references,
    evaluate_traffic_violation_references,
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


class VisualReferenceChecksTests(unittest.TestCase):
    def setUp(self):
        self.cv_patcher = patch(
            "app.visual_reference_checks.analyze_visual_reference_from_cv",
            return_value={
                "cv_enabled": False,
                "cv_model_path": "",
                "cv_error": "YOLO test disabled",
                "cv_detections": [],
                "cv_mentions": [],
            },
        )
        self.plate_cv_patcher = patch(
            "app.visual_reference_checks.analyze_license_plate_from_cv",
            return_value={
                "cv_enabled": False,
                "cv_model_path": "",
                "cv_error": "YOLO test disabled",
                "cv_detections": [],
                "cv_plate_mentions": [],
            },
        )
        self.cv_patcher.start()
        self.plate_cv_patcher.start()

    def tearDown(self):
        self.cv_patcher.stop()
        self.plate_cv_patcher.stop()

    def test_dictionaries_have_at_least_20_terms(self):
        for check_id in ("23", "250", "24"):
            self.assertGreaterEqual(len(VISUAL_REFERENCE_CHECKS[check_id]["terms"]), 20)
            self.assertGreaterEqual(len(VISUAL_REFERENCE_CHECKS[check_id]["cv_labels"]), 20)

    def test_doctor_image_references_fail_on_ocr_terms(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["врач в белом халате со стетоскопом"]}))
        try:
            result = evaluate_doctor_image_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("образу врача", result["message"])
        self.assertIn("белом халате", result["message"].lower())
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])

    def test_disease_references_fail_on_ocr_terms(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_002.jpg": ["кашель температура больное горло"]}))
        try:
            result = evaluate_disease_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("заболеванию", result["message"])
        self.assertIn("2сек.", result["message"])

    def test_traffic_violation_references_fail_on_ocr_terms(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_003.jpg": ["гонки и превышение скорости на красный свет"]}))
        try:
            result = evaluate_traffic_violation_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("дорожного движения", result["message"])
        self.assertIn("превышение скорости", result["message"].lower())

    def test_visual_reference_cv_detection_fails_without_ocr_terms(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["нейтральный ролик"]}))
        fake_cv = {
            "cv_enabled": True,
            "cv_model_path": "model.pt",
            "cv_error": "",
            "cv_detections": [{"label": "doctor", "raw_label": "doctor", "confidence": 0.9, "frame": "frame_001.jpg", "second": "1сек."}],
            "cv_mentions": [{"label": "doctor", "raw_label": "doctor", "confidence": 0.9, "frame": "frame_001.jpg", "second": "1сек."}],
        }
        try:
            with patch("app.visual_reference_checks.analyze_visual_reference_from_cv", return_value=fake_cv):
                result = evaluate_doctor_image_references(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("CV", result["message"])
        self.assertIn("doctor", result["message"])

    def test_license_plate_readable_fails(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["автомобиль А123ВС777"]}))
        try:
            result = evaluate_car_license_plate(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("Требуется согласие", result["message"])
        self.assertIn("А123ВС777", result["message_html"])

    def test_license_plate_blurred_passes(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["номер автомобиля заблюрен"]}))
        try:
            result = evaluate_car_license_plate(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("Номер автомобиля есть и заблюрен", result["message"])

    def test_license_plate_cv_detection_without_blur_fails(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["автомобиль"]}))
        fake_cv = {
            "cv_enabled": True,
            "cv_model_path": "model.pt",
            "cv_error": "",
            "cv_detections": [],
            "cv_plate_mentions": [
                {
                    "label": "license plate",
                    "raw_label": "license plate",
                    "confidence": 0.9,
                    "frame": "frame_001.jpg",
                    "second": "1сек.",
                    "blur_score": 60.0,
                    "blurred": False,
                }
            ],
        }
        try:
            with patch("app.visual_reference_checks.analyze_license_plate_from_cv", return_value=fake_cv):
                result = evaluate_car_license_plate(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("Требуется согласие", result["message"])

    def test_evaluates_new_ids_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                {
                    "frame_001.jpg": [
                        "врач в белом халате",
                        "кашель и температура",
                        "гонки на красный свет",
                        "номер А123ВС777",
                    ]
                }
            )
        )
        view_model = {
            "ok": True,
            "blocks": [
                {
                    "name": "Видеоряд",
                    "items": [
                        {"id": "23", "number": "23", "text": "doctor"},
                        {"id": "250", "number": "250", "text": "disease"},
                        {"id": "24", "number": "24", "text": "traffic"},
                        {"id": "25", "number": "25", "text": "plate"},
                    ],
                }
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        statuses = [item["status"] for item in evaluated["blocks"][0]["items"]]
        self.assertEqual(statuses, ["fail", "fail", "fail", "fail"])


if __name__ == "__main__":
    unittest.main()
