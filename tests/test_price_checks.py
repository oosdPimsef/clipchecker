# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from app.approval_checks import evaluate_approval_view_model
from app.price_checks import NO_PRICE_MESSAGE, evaluate_ruble_prices, extract_ruble_prices


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
    (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


class PriceChecksTests(unittest.TestCase):
    def test_extract_ruble_prices(self):
        self.assertEqual(extract_ruble_prices("Цена 1 999 руб."), ["1 999 руб."])
        self.assertEqual(extract_ruble_prices("Стоимость 1999 ₽"), ["1999 ₽"])
        self.assertEqual(extract_ruble_prices("Цена 2 500 р."), ["2 500 р."])
        self.assertEqual(extract_ruble_prices("г. Москва, стр. 1, ОГРН 123456"), [])

    def test_evaluate_ruble_prices_outputs_large_red_html(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 1 999 руб.", "Реклама"]))
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("1 999 руб.", result["message"])
        self.assertIn('<strong class="price-value">1 999 руб.</strong>', result["message_html"])

    def test_evaluate_ruble_prices_fails_when_missing(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама без цены"]))
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["message"], NO_PRICE_MESSAGE)

    def test_evaluates_item_id_10_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["Стоимость 1999 ₽"]))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "10", "number": "10", "text": "цена"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn("message_html", item)
        self.assertIn("1999 ₽", item["message"])


if __name__ == "__main__":
    unittest.main()
