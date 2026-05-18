# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.distance_sales import evaluate_distance_sales_disclaimer


def make_ocr_line(text: str, box: tuple[int, int, int, int] = (120, 400, 800, 430)) -> dict:
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


class DistanceSalesTests(unittest.TestCase):
    def test_passes_when_distance_sales_words_found_in_legal_disclaimer(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Реклама. Условия дистанционных продаж на сайте рекламодателя"]})
        )
        try:
            result = evaluate_distance_sales_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("Дистанционные продажи", result["message"])
        self.assertEqual(result["distance_sales_count"], 1)

    def test_warns_when_distance_sales_words_missing(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Реклама. Условия акции на сайте рекламодателя"]}))
        try:
            result = evaluate_distance_sales_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("не найдена", result["message"])

    def test_ignores_same_words_outside_legal_disclaimer(self):
        ocr_log = make_ocr_log({"frame_001.jpg": ["Реклама. Условия акции на сайте"]})
        ocr_log["frame_001.jpg"]["results"][0]["results"][0]["textDetection"]["pages"][0]["blocks"][0]["lines"].append(
            make_ocr_line("Дистанционные продажи", (120, 80, 320, 120))
        )
        tmp, base = make_result_dir(ocr_log)
        try:
            result = evaluate_distance_sales_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["distance_sales_count"], 0)

    def test_evaluates_item_id_246_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Реклама. Условия дистанционным продажам на сайте рекламодателя"]})
        )
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Набивка", "items": [{"id": "246", "number": "246", "text": "дистанционные продажи"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn("Дистанционные продажи", item["message"])


if __name__ == "__main__":
    unittest.main()
