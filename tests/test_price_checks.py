# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.price_checks import (
    NO_DOCUMENT_PRICE_MESSAGE,
    NO_PRICE_MESSAGE,
    evaluate_document_prices_match_video,
    evaluate_ruble_prices,
    analyze_document_prices,
    extract_document_prices,
    extract_ruble_prices,
    is_price_letter_title,
    normalize_price_value,
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


def make_ocr_log(lines: list[str | tuple[str, tuple[int, int, int, int]]]) -> dict:
    ocr_lines = []
    for item in lines:
        if isinstance(item, tuple):
            text, box = item
        else:
            text, box = item, (120, 80, 500, 120)
        ocr_lines.append(make_ocr_line(text, box))

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
                                                "lines": ocr_lines,
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


def write_documents_text(base: Path, text: str):
    (base / "Documents_Texts.txt").write_text(text, encoding="utf-8")


class PriceChecksTests(unittest.TestCase):
    def test_extract_ruble_prices(self):
        self.assertEqual(extract_ruble_prices("Цена 1 999 руб."), ["1 999 руб."])
        self.assertEqual(extract_ruble_prices("Стоимость 1999 ₽"), ["1999 ₽"])
        self.assertEqual(extract_ruble_prices("Цена 2 500 р."), ["2 500 р."])
        self.assertEqual(extract_ruble_prices("установлено в размере 49 990 (сорок девять тысяч девятьсот девяносто) рублей"), ["49 990 рублей"])
        self.assertEqual(extract_ruble_prices("г. Москва, стр. 1, ОГРН 123456"), [])
        self.assertEqual(normalize_price_value("1 999 руб."), "1999")
        self.assertEqual(normalize_price_value("1999 ₽"), "1999")

    def test_extract_document_prices_uses_price_context(self):
        text = (
            "Письмо рекламодателя. Сообщаем, что цена товара составляет 1 999 рублей. "
            "Доверенность действует на сумму не более 10 000 рублей."
        )

        prices = extract_document_prices(text)

        self.assertEqual([item["price"] for item in prices], ["1 999 рублей"])

    def test_price_letter_title_has_priority(self):
        documents_text = (
            "Документ 1. Доверенность.docx\n"
            "Доверенность на совершение договоров стоимостью не более 10 000 рублей.\n\n"
            "Документ 2. Письмо о ценах.docx\n"
            "Ценовое предложение на изделие, артикул 9010269-3-5, установлено в размере "
            "49 990 (сорок девять тысяч девятьсот девяносто) рублей.\n"
            "Ценовое предложение на изделие, артикул 019686-3, установлено в размере "
            "9 990 (девять тысяч девятьсот девяносто) рублей.\n"
        )

        result = analyze_document_prices(documents_text)

        self.assertTrue(is_price_letter_title("Письмо о ценах.docx"))
        self.assertTrue(result["searched_price_letter_first"])
        self.assertEqual(result["document_price_values"], ["49990", "9990"])
        self.assertEqual(result["document_price_source_titles"], ["Письмо о ценах.docx"])

    def test_evaluate_ruble_prices_outputs_large_red_html(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 1 999 руб.", "Реклама"]))
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("1 999 руб.", result["message"])
        self.assertIn('<strong class="price-value">1 999 руб.</strong>', result["message_html"])

    def test_evaluate_ruble_prices_passes_when_missing(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама без цены"]))
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_PRICE_MESSAGE)

    def test_evaluate_ruble_prices_fails_when_frame_and_disclaimer_prices_mismatch(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                [
                    ("Цена 1 999 руб.", (120, 80, 500, 120)),
                    ("Рекламодатель ООО Ромашка, условия акции, цена 2 999 руб.", (120, 400, 820, 430)),
                ]
            )
        )
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["price_mismatch"])
        self.assertIn("не совпадает", result["message"])

    def test_evaluate_ruble_prices_passes_when_frame_and_disclaimer_prices_match(self):
        tmp, base = make_result_dir(
            make_ocr_log(
                [
                    ("Цена 1 999 руб.", (120, 80, 500, 120)),
                    ("Рекламодатель ООО Ромашка, условия акции, цена 1999 ₽", (120, 400, 820, 430)),
                ]
            )
        )
        try:
            result = evaluate_ruble_prices(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["price_mismatch"])

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

    def test_document_price_passes_when_only_document_has_price(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама без цены"]))
        write_documents_text(base, "Письмо. Стоимость товара составляет 1 999 руб.")
        try:
            result = evaluate_document_prices_match_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("Цена найдена в документах", result["message"])
        self.assertIn('<strong class="price-value">1 999 руб.</strong>', result["message_html"])

    def test_document_price_fails_when_video_price_mismatch(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 2 999 руб."]))
        write_documents_text(base, "Письмо в свободной форме: цена товара 1 999 рублей.")
        try:
            result = evaluate_document_prices_match_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["price_mismatch"])
        self.assertIn("не совпадает", result["message"])

    def test_document_price_passes_when_video_price_matches(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 1999 ₽"]))
        write_documents_text(base, "Письмо: акционная цена товара - 1 999 рублей.")
        try:
            result = evaluate_document_prices_match_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["price_mismatch"])

    def test_document_price_passes_when_missing(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 1999 ₽"]))
        write_documents_text(base, "Письмо без сведений о цене.")
        try:
            result = evaluate_document_prices_match_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_DOCUMENT_PRICE_MESSAGE)

    def test_evaluates_item_id_11_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["Цена 1999 ₽"]))
        write_documents_text(base, "Письмо: стоимость товара 1999 руб.")
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Документы", "items": [{"id": "11", "number": "11", "text": "цена в письме"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn("message_html", item)


if __name__ == "__main__":
    unittest.main()
