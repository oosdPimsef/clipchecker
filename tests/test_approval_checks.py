# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from app.approval_checks import (
    evaluate_approval_view_model,
    evaluate_booking_form_duration_matches_video,
    evaluate_duration_multiple_of_five,
)


def make_result_dir(frame_count: int | None):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    if frame_count is not None:
        frames_dir = base / "frames"
        frames_dir.mkdir()
        for idx in range(1, frame_count + 1):
            (frames_dir / f"frame_{idx:03d}.jpg").write_bytes(b"fake")
    return tmp, base


def write_documents_text(base: Path, text: str):
    (base / "Documents_Texts.txt").write_text(text, encoding="utf-8")


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

    def test_booking_form_missing_fails(self):
        tmp, base = make_result_dir(15)
        write_documents_text(base, "Документ 1. Письмо.docx\nОбычное письмо рекламодателя.")
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["message"], "Бланк-заявки нет в документах.")

    def test_booking_form_duration_matches_video_passes(self):
        tmp, base = make_result_dir(15)
        write_documents_text(base, "Документ 1. Бланк-заявка.xlsx\nХронометраж ролика: 15 сек.")
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["duration_sec"], 15)
        self.assertEqual(result["booking_duration_sec"], 15)
        self.assertIn("Длительность ролика 15 секунд", result["message"])

    def test_booking_form_detected_by_usage_information_phrase(self):
        tmp, base = make_result_dir(5)
        write_documents_text(
            base,
            (
                "Документ 1. БЗ_гп.(5с)_гендеры.docx\n"
                "(обработано с помощью: python-docx)\n"
                "СВЕДЕНИЯ ОБ ИСПОЛЬЗОВАНИИ ПРОИЗВЕДЕНИЙ РОССИЙСКИХ И ИНОСТРАННЫХ АВТОРОВ\n"
                "Продолжительность ролика | 5 сек."
            ),
        )
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["duration_sec"], 5)
        self.assertEqual(result["booking_duration_sec"], 5)
        self.assertEqual(result["booking_form_title"], "БЗ_гп.(5с)_гендеры.docx")

    def test_booking_form_duration_prefers_duration_field_over_timecode(self):
        tmp, base = make_result_dir(5)
        write_documents_text(
            base,
            (
                "Документ 1. БЗ_гп.(5с)_гендеры.docx\n"
                "СВЕДЕНИЯ ОБ ИСПОЛЬЗОВАНИИ ПРОИЗВЕДЕНИЙ\n"
                "Продолжительность ролика | 5 сек. "
                "Тайм-код ролика на кассете | Начало 00:00:00:00 | Конец 00:00:00:00"
            ),
        )
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["booking_duration_sec"], 5)

    def test_booking_form_duration_mismatch_fails(self):
        tmp, base = make_result_dir(15)
        write_documents_text(base, "Документ 1. БЗ.docx\nДлительность 20 секунд.")
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["duration_sec"], 15)
        self.assertEqual(result["booking_duration_sec"], 20)

    def test_booking_form_without_duration_fails(self):
        tmp, base = make_result_dir(15)
        write_documents_text(base, "Документ 1. Бланк-заявка.docx\nМатериалы к размещению без указания хронометража.")
        try:
            result = evaluate_booking_form_duration_matches_video(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn("длительность ролика в ней не найдена", result["message"])

    def test_evaluates_item_id_2_inside_view_model(self):
        tmp, base = make_result_dir(15)
        write_documents_text(base, "Документ 1. БЗ.docx\nХронометраж 15 сек.")
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "2", "number": "2", "text": "длительность в БЗ"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        self.assertEqual(evaluated["blocks"][0]["items"][0]["status"], "pass")
        self.assertIn("В БЗ указано 15 секунд", evaluated["blocks"][0]["items"][0]["message"])


if __name__ == "__main__":
    unittest.main()
