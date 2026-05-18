# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.foreign_words import (
    WARNING_TEXT,
    analyze_star_translations,
    analyze_trademark_documents,
    evaluate_foreign_words,
    evaluate_foreign_words_translation_or_trademark,
    extract_foreign_words,
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


def make_result_dir(ocr_log: dict):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    Image.new("RGB", (1000, 500), "white").save(frames / "frame_001.jpg")
    (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


def add_documents_text(base: Path, text: str) -> None:
    (base / "Documents_Texts.txt").write_text(text, encoding="utf-8")


class ForeignWordsTests(unittest.TestCase):
    def test_extract_foreign_words_ignores_urls(self):
        self.assertEqual(extract_foreign_words("SOKOLOV sale https www"), ["SOKOLOV", "sale"])

    def test_evaluate_foreign_words_warns_and_formats_html(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV sale", "Реклама"]))
        try:
            result = evaluate_foreign_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn(WARNING_TEXT, result["message"])
        self.assertIn('<strong class="foreign-word">SOKOLOV</strong>', result["message_html"])
        self.assertIn('<strong class="foreign-word">sale</strong>', result["message_html"])

    def test_evaluate_foreign_words_passes_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама акция"]))
        try:
            result = evaluate_foreign_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_evaluates_item_id_8_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV"]))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Набивка", "items": [{"id": "8", "number": "8", "text": "иностранные слова"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "warning")
        self.assertIn("message_html", item)
        self.assertIn("SOKOLOV", item["message"])

    def test_analyze_star_translations_finds_asterisk_lines(self):
        translations = analyze_star_translations(make_ocr_log(["SOKOLOV sale", "* sale - распродажа"]))

        self.assertEqual(len(translations), 1)
        self.assertIn("распродажа", translations[0]["text"])

    def test_analyze_trademark_documents_extracts_brand(self):
        analysis = analyze_trademark_documents(
            'Документ 1. Свидетельство\nПредоставлены права на товарный знак "SOKOLOV" для рекламы.'
        )

        self.assertEqual(analysis["trademark_document_count"], 1)
        self.assertEqual(analysis["trademark_brands"][0]["brand"], "SOKOLOV")

    def test_id_262_passes_when_foreign_words_not_found(self):
        tmp, base = make_result_dir(make_ocr_log(["Реклама акция"]))
        try:
            result = evaluate_foreign_words_translation_or_trademark(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_id_262_warns_when_translation_exists(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV sale", "* sale - распродажа"]))
        try:
            result = evaluate_foreign_words_translation_or_trademark(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("распродажа", result["message"])
        self.assertIn("message_html", result)

    def test_id_262_warns_when_trademark_document_exists(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV"]))
        add_documents_text(base, 'Документ 1. Свидетельство\nПредоставлены права на товарный знак "SOKOLOV".')
        try:
            result = evaluate_foreign_words_translation_or_trademark(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("SOKOLOV", result["message"])
        self.assertEqual(result["trademark_brands"][0]["brand"], "SOKOLOV")

    def test_id_262_fails_when_foreign_words_have_no_support(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV sale"]))
        try:
            result = evaluate_foreign_words_translation_or_trademark(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("не найдены", result["message"])

    def test_evaluates_item_id_262_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log(["SOKOLOV"]))
        add_documents_text(base, 'Документ 1. Свидетельство\nПравообладатель имеет товарный знак «SOKOLOV».')
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Набивка", "items": [{"id": "262", "number": "262", "text": "перевод или товарный знак"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "warning")
        self.assertIn("message_html", item)
        self.assertIn("SOKOLOV", item["message"])


if __name__ == "__main__":
    unittest.main()
