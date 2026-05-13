# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.jewelry_check import (
    JEWELRY_FOUND_MESSAGE,
    JEWELRY_NOT_FOUND_MESSAGE,
    evaluate_jewelry_presence,
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


def make_result_dir(ocr_log: dict | None = None):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    Image.new("RGB", (1000, 500), "white").save(frames / "frame_001.jpg")
    if ocr_log is not None:
        (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class JewelryCheckTests(unittest.TestCase):
    def test_evaluate_jewelry_presence_passes_when_keyword_found(self):
        tmp, base = make_result_dir(make_ocr_log(["Золотое кольцо с бриллиантом"]))
        try:
            result = evaluate_jewelry_presence(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn(JEWELRY_FOUND_MESSAGE, result["message"])
        self.assertIn("кольцо", result["message"].lower())

    def test_evaluate_jewelry_presence_warns_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log(["Новая коллекция одежды"]))
        try:
            result = evaluate_jewelry_presence(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["message"], JEWELRY_NOT_FOUND_MESSAGE)

    def test_evaluates_item_id_12_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["Серьги из серебра"]))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "12", "number": "12", "text": "изображение ювелирного изделия"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn(JEWELRY_FOUND_MESSAGE, item["message"])


if __name__ == "__main__":
    unittest.main()
