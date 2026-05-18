# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.contact_info import (
    evaluate_contact_info_disclaimer,
    extract_phone_contacts,
    extract_web_contacts,
)


def make_ocr_line(text: str, box: tuple[int, int, int, int] = (120, 400, 900, 430)) -> dict:
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


class ContactInfoTests(unittest.TestCase):
    def test_extracts_web_contacts_and_social_pages(self):
        self.assertEqual(extract_web_contacts("Подробности на www.sokolov.ru/sale и t.me/sokolov"), ["www.sokolov.ru/sale", "t.me/sokolov"])
        self.assertEqual(extract_web_contacts("Сроки акции 01.12.2024 - 28.01.2025"), [])

    def test_extracts_phone_contacts(self):
        self.assertEqual(extract_phone_contacts("Тел. +7 (495) 123-45-67"), ["+7 (495) 123-45-67"])
        self.assertEqual(extract_phone_contacts("8 800 555 35 35"), ["8 800 555 35 35"])
        self.assertEqual(extract_phone_contacts("по телефону: 8 (800) 1000-750"), ["8 (800) 1000-750"])

    def test_passes_and_formats_site_in_red_bold_when_contact_found(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Реклама. Подробности на сайте sokolov.ru/sale"]})
        )
        try:
            result = evaluate_contact_info_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("sokolov.ru/sale", result["message"])
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])
        self.assertEqual(result["web_contacts"][0]["value"], "sokolov.ru/sale")

    def test_passes_when_only_phone_found(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Реклама. Телефон рекламодателя 8 800 555 35 35"]})
        )
        try:
            result = evaluate_contact_info_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["phone_contacts"][0]["value"], "8 800 555 35 35")

    def test_warns_when_site_and_phone_are_missing(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Реклама. Условия акции у рекламодателя"]}))
        try:
            result = evaluate_contact_info_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["message"], "Сайт и номер телефона в тексте набивки не найдены.")

    def test_ignores_contacts_outside_legal_disclaimer(self):
        ocr_log = make_ocr_log({"frame_001.jpg": ["Реклама. Условия акции у рекламодателя"]})
        ocr_log["frame_001.jpg"]["results"][0]["results"][0]["textDetection"]["pages"][0]["blocks"][0]["lines"].append(
            make_ocr_line("sokolov.ru", (120, 80, 320, 120))
        )
        tmp, base = make_result_dir(ocr_log)
        try:
            result = evaluate_contact_info_disclaimer(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["contact_count"], 0)

    def test_evaluates_item_id_247_inside_view_model(self):
        tmp, base = make_result_dir(
            make_ocr_log({"frame_001.jpg": ["Реклама. Подробности акции на сайте example.ru"]})
        )
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Набивка", "items": [{"id": "247", "number": "247", "text": "сайт или телефон"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn("example.ru", item["message"])
        self.assertIn("message_html", item)


if __name__ == "__main__":
    unittest.main()
