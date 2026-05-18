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
    analyze_claim_disclaimers,
    analyze_claim_document_support,
    evaluate_claim_support,
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
                                "pages": [{"blocks": [{"lines": [line if isinstance(line, dict) else make_ocr_line(line) for line in lines]}]}]
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


def add_documents_text(base: Path, text: str) -> None:
    (base / "Documents_Texts.txt").write_text(text, encoding="utf-8")


class ComparativeClaimsTests(unittest.TestCase):
    def test_dictionary_has_at_least_50_terms(self):
        self.assertGreaterEqual(len(COMPARATIVE_TERMS), 50)

    def test_extract_comparative_terms_finds_variants(self):
        terms = extract_comparative_terms("Выгоднее и дешевле конкурентов")
        self.assertIn("выгоднее", terms)
        self.assertIn("дешевле", terms)
        self.assertIn("конкурент", terms)

    def test_extract_claim_terms_finds_new_claim_types(self):
        terms = extract_comparative_terms("№1 официальный выбор, скидка до 50%, цена от 999, результаты исследований")
        self.assertIn("№1", terms)
        self.assertIn("официальный", terms)
        self.assertIn("скидка до", terms)
        self.assertIn("цена от", terms)
        self.assertIn("результаты исследований", terms)

    def test_evaluate_comparative_claims_passes_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Коллекция украшений"]}))
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
        self.assertIn("claims", result["message"].lower())
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

    def test_analyze_claim_disclaimers_finds_bottom_explanation(self):
        ocr_log = make_ocr_log(
            {
                "frame_001.jpg": [
                    "Скидка до 50%",
                    make_ocr_line("*Скидка действует на товары, участвующие в акции, подробности на сайте", (80, 420, 920, 470)),
                ]
            }
        )
        tmp, base = make_result_dir(ocr_log)
        try:
            disclaimers = analyze_claim_disclaimers(ocr_log, base / "frames")
        finally:
            tmp.cleanup()

        self.assertEqual(len(disclaimers), 1)
        self.assertIn("Скидка действует", disclaimers[0]["text"])

    def test_analyze_claim_document_support_finds_explanation(self):
        claims = [{"term": "№1", "seconds": ["1сек."], "frames": ["frame_001.jpg"], "examples": ["№1"]}]
        support = analyze_claim_document_support(
            "Документ 1. Исследование\nСогласно данным исследования бренд является №1 в категории по продажам.",
            claims,
        )

        self.assertEqual(len(support), 1)
        self.assertIn("№1", support[0]["text"])

    def test_id_235_passes_when_claims_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Коллекция украшений"]}))
        try:
            result = evaluate_claim_support(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_id_235_fails_when_claim_has_no_support(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Лучший выбор"]}))
        try:
            result = evaluate_claim_support(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("лучший", result["message"])
        self.assertIn("не найдены", result["message"])

    def test_id_235_warns_when_disclaimer_exists(self):
        ocr_log = make_ocr_log(
            {
                "frame_001.jpg": [
                    "Цена от 999 руб",
                    make_ocr_line("*Указана минимальная цена товара, подробности и условия акции на сайте", (80, 420, 920, 470)),
                ]
            }
        )
        tmp, base = make_result_dir(ocr_log)
        try:
            result = evaluate_claim_support(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("Указана минимальная цена", result["message"])
        self.assertIn("message_html", result)

    def test_id_235_warns_when_document_exists(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Рекомендовано экспертами"]}))
        add_documents_text(base, "Документ 1. Заключение\nСогласно экспертному заключению товар рекомендовано использовать для ухода.")
        try:
            result = evaluate_claim_support(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("экспертному заключению", result["message"])

    def test_evaluates_item_id_235_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Результаты исследований"]}))
        add_documents_text(base, "Документ 1. Исследование\nИсследование подтверждает результаты исследований по продукту.")
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "235", "number": "235", "text": "документы на claims"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "warning")
        self.assertIn("message_html", item)


if __name__ == "__main__":
    unittest.main()
