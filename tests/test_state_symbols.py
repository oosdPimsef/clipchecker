# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.state_symbols import (
    NO_STATE_SYMBOLS_MESSAGE,
    STATE_SYMBOL_DEFINITIONS,
    detect_tricolor_frame,
    evaluate_state_symbols,
    extract_state_symbol_terms,
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
        Image.new("RGB", (300, 180), "white").save(frames / frame_name)
    if ocr_log is not None:
        (base / "OCR_Log.json").write_text(json.dumps(ocr_log, ensure_ascii=False), encoding="utf-8")
    return tmp, base


def save_tricolor(path: Path) -> None:
    img = Image.new("RGB", (300, 180), "white")
    pixels = img.load()
    for y in range(60, 120):
        for x in range(300):
            pixels[x, y] = (0, 57, 166)
    for y in range(120, 180):
        for x in range(300):
            pixels[x, y] = (213, 43, 30)
    img.save(path)


class StateSymbolsTests(unittest.TestCase):
    def test_has_at_least_15_definitions(self):
        self.assertGreaterEqual(len(STATE_SYMBOL_DEFINITIONS), 15)

    def test_extract_state_symbol_terms(self):
        terms = extract_state_symbol_terms("На кадре герб города и двуглавый орел")
        definitions = {item["definition"] for item in terms}
        self.assertIn("герб", definitions)
        self.assertIn("герб города", definitions)
        self.assertIn("двуглавый орел", definitions)

    def test_detect_tricolor_frame(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "flag.jpg"
        try:
            save_tricolor(path)
            self.assertTrue(detect_tricolor_frame(path))
        finally:
            tmp.cleanup()

    def test_evaluate_state_symbols_passes_when_not_found(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Новая коллекция украшений"]}))
        try:
            result = evaluate_state_symbols(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_STATE_SYMBOLS_MESSAGE)

    def test_evaluate_state_symbols_fails_and_formats_html(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Герб города"], "frame_002.jpg": ["Флаг России"]}))
        try:
            result = evaluate_state_symbols(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("герб", result["message"].lower())
        self.assertIn('style="color:#dc2626;font-weight:800"', result["message_html"])
        self.assertIn("1сек.", result["message"])
        self.assertIn("2сек.", result["message"])

    def test_evaluates_item_id_251_inside_view_model(self):
        tmp, base = make_result_dir(make_ocr_log({"frame_001.jpg": ["Георгиевская лента"]}))
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "251", "number": "251", "text": "госсимволы"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "fail")
        self.assertIn("message_html", item)
        self.assertIn("георгиевская", item["message"].lower())


if __name__ == "__main__":
    unittest.main()
