# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from app.approval_checks import evaluate_approval_view_model, evaluate_duration_multiple_of_five


def make_result_dir(frame_count: int | None):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    if frame_count is not None:
        frames_dir = base / "frames"
        frames_dir.mkdir()
        for idx in range(1, frame_count + 1):
            (frames_dir / f"frame_{idx:03d}.jpg").write_bytes(b"fake")
    return tmp, base


class ApprovalChecksTests(unittest.TestCase):
    def test_duration_multiple_of_five_passes(self):
        tmp, base = make_result_dir(15)
        try:
            result = evaluate_duration_multiple_of_five(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["duration_sec"], 15)
        self.assertIn("15 секунд", result["message"])

    def test_duration_not_multiple_of_five_fails(self):
        tmp, base = make_result_dir(14)
        try:
            result = evaluate_duration_multiple_of_five(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["duration_sec"], 14)
        self.assertIn("14 секунд", result["message"])

    def test_missing_frames_is_pending(self):
        tmp, base = make_result_dir(None)
        try:
            result = evaluate_duration_multiple_of_five(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pending")
        self.assertIn("после предобработки", result["message"])

    def test_evaluates_item_id_1_inside_view_model(self):
        tmp, base = make_result_dir(10)
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "1", "number": "1", "text": "длительность"}]},
                {"name": "Набивка", "items": [{"id": "26", "number": "1", "text": "размер набивки"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        self.assertEqual(evaluated["blocks"][0]["items"][0]["status"], "pass")
        self.assertEqual(evaluated["blocks"][0]["items"][0]["message"], "Длительность ролика 10 секунд.")
        self.assertEqual(evaluated["blocks"][1]["items"][0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()

