# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.approval_checks import evaluate_approval_view_model
from app.frame_quality import (
    analyze_blank_frames,
    analyze_repeated_frames,
    blank_frame_score,
    evaluate_blank_frames,
    evaluate_repeated_frames,
    frame_similarity,
    is_blank_frame,
)


def make_result_dir(images: list[Image.Image]):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frames = base / "frames_pdf_original"
    frames.mkdir()
    for index, image in enumerate(images, start=1):
        image.save(frames / f"frame_{index:03d}.jpg")
    return tmp, base


def patterned_image(color: tuple[int, int, int] = (220, 220, 220)) -> Image.Image:
    image = Image.new("RGB", (320, 180), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 30, 180, 120], fill=(40, 100, 180))
    draw.ellipse([210, 55, 270, 115], fill=(220, 40, 60))
    return image


class FrameQualityTests(unittest.TestCase):
    def test_frame_similarity_detects_identical_frames(self):
        image = patterned_image()
        tmp, base = make_result_dir([image, image.copy()])
        try:
            similarity = frame_similarity(
                base / "frames_pdf_original" / "frame_001.jpg",
                base / "frames_pdf_original" / "frame_002.jpg",
            )
        finally:
            tmp.cleanup()

        self.assertGreaterEqual(similarity, 0.90)

    def test_analyze_repeated_frames_fails_on_consecutive_duplicate(self):
        image = patterned_image()
        tmp, base = make_result_dir([image, image.copy(), patterned_image((80, 180, 120))])
        try:
            result = evaluate_repeated_frames(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["repeat_count"], 1)
        self.assertEqual(result["repeats"][0]["previous_frame"], "frame_001.jpg")
        self.assertEqual(result["repeats"][0]["current_frame"], "frame_002.jpg")

    def test_analyze_repeated_frames_passes_when_frames_differ(self):
        tmp, base = make_result_dir([patterned_image((220, 220, 220)), patterned_image((60, 150, 200))])
        try:
            result = evaluate_repeated_frames(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["repeat_count"], 0)

    def test_blank_frame_score_detects_any_uniform_color(self):
        tmp, base = make_result_dir([Image.new("RGB", (320, 180), (15, 15, 15)), Image.new("RGB", (320, 180), (240, 240, 240))])
        try:
            black_path = base / "frames_pdf_original" / "frame_001.jpg"
            white_path = base / "frames_pdf_original" / "frame_002.jpg"
            self.assertTrue(is_blank_frame(black_path))
            self.assertTrue(is_blank_frame(white_path))
            self.assertLessEqual(blank_frame_score(white_path)["mean_deviation"], 3.0)
        finally:
            tmp.cleanup()

    def test_evaluate_blank_frames_fails_on_uniform_frame(self):
        tmp, base = make_result_dir([patterned_image(), Image.new("RGB", (320, 180), (45, 45, 45))])
        try:
            result = evaluate_blank_frames(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["blank_count"], 1)
        self.assertEqual(result["blank_frames"][0]["frame"], "frame_002.jpg")

    def test_evaluate_blank_frames_passes_when_frames_have_content(self):
        tmp, base = make_result_dir([patterned_image(), patterned_image((120, 200, 160))])
        try:
            result = evaluate_blank_frames(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["blank_count"], 0)

    def test_evaluates_item_ids_228_and_229_inside_view_model(self):
        image = patterned_image()
        tmp, base = make_result_dir([image, image.copy(), Image.new("RGB", (320, 180), (0, 0, 0))])
        view_model = {
            "ok": True,
            "blocks": [
                {
                    "name": "Видеоряд",
                    "items": [
                        {"id": "228", "number": "228", "text": "стоп-кадры"},
                        {"id": "229", "number": "229", "text": "пустые кадры"},
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
        self.assertEqual(items[1]["status"], "fail")
        self.assertEqual(items[0]["details"]["repeat_count"], 1)
        self.assertEqual(items[1]["details"]["blank_count"], 1)


if __name__ == "__main__":
    unittest.main()
