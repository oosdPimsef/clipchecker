# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.non_russian_words import (
    classify_non_russian_word,
    evaluate_non_russian_words,
    evaluate_non_russian_words_translation,
    extract_non_russian_words,
    filter_star_translations_for_words,
    is_russian_dictionary_word,
    morphology_available,
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


def make_ocr_log(lines: list[str]) -> dict:
    return {
        "frame_001.jpg": {
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
    }


def make_result_dir(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames"
    frames.mkdir()
    Image.new("RGB", (1000, 500), "white").save(frames / "frame_001.jpg")
    (base / "OCR_Log.json").write_text(json.dumps(make_ocr_log(lines), ensure_ascii=False), encoding="utf-8")
    return tmp, base


class NonRussianWordsTests(unittest.TestCase):
    def test_classifies_latin_mixed_and_anglicism_words(self):
        self.assertEqual(classify_non_russian_word("sale"), "латиница")
        self.assertEqual(classify_non_russian_word("SОKOLOV"), "смешаны латиница и кириллица")
        self.assertEqual(classify_non_russian_word("кэшбэк"), "англицизм/неологизм")
        self.assertIsNone(classify_non_russian_word("скидка"))
        self.assertIsNone(classify_non_russian_word("BCE"))
        self.assertIsNone(classify_non_russian_word("HA"))
        self.assertIsNone(classify_non_russian_word("ОГРН"))
        self.assertIsNone(classify_non_russian_word("пр-зд"))
        self.assertIsNone(classify_non_russian_word("товаров-исключений"))

    def test_morphology_dictionary_recognizes_russian_inflections(self):
        self.assertTrue(morphology_available())
        self.assertTrue(is_russian_dictionary_word("продажам"))
        self.assertTrue(is_russian_dictionary_word("товарами"))

    def test_morphology_dictionary_flags_unknown_cyrillic_word(self):
        self.assertEqual(classify_non_russian_word("фывапролдж"), "не найдено в русском морфологическом словаре")

    def test_extracts_non_russian_words(self):
        words = extract_non_russian_words("Скидка sale и кэшбэк")
        self.assertEqual([item["word"] for item in words], ["sale", "кэшбэк"])

    def test_id_252_passes_when_words_not_found(self):
        tmp, base = make_result_dir(["Реклама. Скидка на все товары"])
        try:
            result = evaluate_non_russian_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_id_252_fails_and_formats_words(self):
        tmp, base = make_result_dir(["SOKOLOV sale и кэшбэк"])
        try:
            result = evaluate_non_russian_words(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("SOKOLOV", result["message"])
        self.assertIn("кэшбэк", result["message"])
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])

    def test_id_258_passes_when_words_not_found(self):
        tmp, base = make_result_dir(["Реклама. Скидка на все товары"])
        try:
            result = evaluate_non_russian_words_translation(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_id_258_warns_when_translation_exists(self):
        tmp, base = make_result_dir(["SOKOLOV sale", "* sale - распродажа"])
        try:
            result = evaluate_non_russian_words_translation(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertIn("Найден перевод", result["message"])

    def test_translation_must_reference_found_word(self):
        translations = [{"text": "* Кроме товаров-исключений.", "seconds": ["1сек."], "frames": ["frame_001.jpg"]}]
        words = [{"word": "sale"}]

        self.assertEqual(filter_star_translations_for_words(translations, words), [])

    def test_id_258_fails_when_translation_missing(self):
        tmp, base = make_result_dir(["SOKOLOV sale"])
        try:
            result = evaluate_non_russian_words_translation(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("Перевод со звёздочкой", result["message"])

    def test_evaluates_item_ids_252_and_258_inside_view_model(self):
        tmp, base = make_result_dir(["SOKOLOV sale", "* sale - распродажа"])
        view_model = {
            "ok": True,
            "blocks": [
                {
                    "name": "Набивка",
                    "items": [
                        {"id": "252", "number": "252", "text": "англицизмы"},
                        {"id": "258", "number": "258", "text": "перевод"},
                    ],
                },
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        items = evaluated["blocks"][0]["items"]
        self.assertEqual(items[0]["status"], "fail")
        self.assertEqual(items[1]["status"], "warning")


if __name__ == "__main__":
    unittest.main()
