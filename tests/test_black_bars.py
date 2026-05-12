# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.approval_checks import evaluate_approval_view_model
from app.black_bars import analyze_frame_black_bars, evaluate_black_side_bars


def make_result_dir(with_left_bar: bool = False, with_right_bar: bool = False):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames_pdf_original"
    frames.mkdir()

    image = Image.new("RGB", (1000, 500), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    if with_left_bar:
        draw.rectangle([0, 0, 29, 499], fill=(0, 0, 0))
    if with_right_bar:
        draw.rectangle([970, 0, 999, 499], fill=(0, 0, 0))
    image.save(frames / "frame_001.jpg")
    return tmp, base


class BlackBarsTests(unittest.TestCase):
    def test_analyze_frame_black_bars_passes_without_side_bars(self):
        tmp, base = make_result_dir()
        try:
            result = analyze_frame_black_bars(base / "frames_pdf_original" / "frame_001.jpg")
        finally:
            tmp.cleanup()

        self.assertFalse(result["detected"])

    def test_analyze_frame_black_bars_detects_left_and_right_bars(self):
        tmp, base = make_result_dir(with_left_bar=True, with_right_bar=True)
        try:
            result = analyze_frame_black_bars(base / "frames_pdf_original" / "frame_001.jpg")
        finally:
            tmp.cleanup()

        self.assertTrue(result["detected"])
        self.assertTrue(result["left_detected"])
        self.assertTrue(result["right_detected"])

    def test_evaluate_black_side_bars_passes_when_clean(self):
        tmp, base = make_result_dir()
        try:
            result = evaluate_black_side_bars(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")

    def test_evaluate_black_side_bars_fails_when_bars_exist(self):
        tmp, base = make_result_dir(with_left_bar=True)
        try:
            result = evaluate_black_side_bars(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("слева", result["message"])

    def test_evaluates_item_id_9_inside_view_model(self):
        tmp, base = make_result_dir(with_right_bar=True)
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "9", "number": "9", "text": "черные полосы"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "fail")
        self.assertIn("справа", item["message"])


if __name__ == "__main__":
    unittest.main()
