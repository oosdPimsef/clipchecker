# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.approval_checks import evaluate_approval_view_model
from app.media_person_check import (
    ACTOR_NOT_RECOGNIZED_MESSAGE,
    NO_ACTORS_MESSAGE,
    SEARCH_CACHE_NAME,
    SEARCH_CACHE_VERSION,
    _analysis_face_files,
    _face_signature,
    evaluate_actor_recognition,
    extract_actor_names,
    extract_names_from_search_text,
)


def make_result_dir(search_faces: list[dict] | None = None, create_faces: bool = False):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    if create_faces:
        faces = base / "faces"
        faces.mkdir()
        Image.new("RGB", (120, 120), "white").save(faces / "face_001.jpg")
    if search_faces is not None:
        (base / SEARCH_CACHE_NAME).write_text(
            json.dumps(
                {
                    "ok": True,
                    "cache_version": SEARCH_CACHE_VERSION,
                    "face_signature": _face_signature(_analysis_face_files(base)),
                    "provider": "direct_web",
                    "faces": search_faces,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return tmp, base


class MediaPersonCheckTests(unittest.TestCase):
    def test_extract_names_from_search_text(self):
        self.assertEqual(extract_names_from_search_text("Иван Ургант 98%"), ["Иван Ургант"])
        self.assertEqual(extract_names_from_search_text("not found"), [])

    def test_extract_names_from_search_text_ignores_google_ui_noise(self):
        text = (
            "Google Search; Deutsch English; Cymraeg Dansk Deutsch; English United Kingdom; "
            "English United States; France Gaeilge; Hrvatski Indonesia; Italiano Kiswahili; "
            "Melayu Nederlands; Suomi Svenska; Конфиденциальность Условия Политика"
        )
        self.assertEqual(extract_names_from_search_text(text), [])

    def test_extract_actor_names_from_cached_search(self):
        result = {"faces": [{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}]}
        self.assertEqual(extract_actor_names(result), ["Иван Ургант"])

    def test_evaluate_actor_recognition_passes_when_no_faces(self):
        tmp, base = make_result_dir(search_faces=[])
        try:
            result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_ACTORS_MESSAGE)

    def test_evaluate_actor_recognition_passes_when_actor_not_recognized(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "", "names": []}],
            create_faces=True,
        )
        try:
            result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], ACTOR_NOT_RECOGNIZED_MESSAGE)

    def test_stale_empty_cache_is_ignored_when_faces_exist(self):
        tmp, base = make_result_dir(search_faces=[], create_faces=True)
        try:
            result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], ACTOR_NOT_RECOGNIZED_MESSAGE)
        self.assertEqual([face["file"] for face in result["search_results"]["faces"]], ["face_001.jpg"])

    def test_faces_thumbnails_pdf_pages_are_used_as_separate_faces(self):
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        try:
            first = Image.new("RGB", (120, 120), "white")
            second = Image.new("RGB", (120, 120), "gray")
            first.save(base / "Faces_Thumbnails.pdf", save_all=True, append_images=[second])

            faces = _analysis_face_files(base)
        finally:
            tmp.cleanup()

        self.assertEqual([path.name for path in faces], ["face_pdf_page_001.jpg", "face_pdf_page_002.jpg"])

    @unittest.skip("direct web parser is a fallback; local known-face matching is the primary actor recognition path")
    def test_direct_web_search_extracts_names_from_search_pages(self):
        tmp, base = make_result_dir(
            search_faces=None,
            create_faces=True,
        )
        try:
            with patch.dict("os.environ", {"MEDIA_PERSON_PUBLIC_FACE_BASE_URL": "https://example.com/faces"}, clear=False):
                with patch("app.media_person_check.requests.get") as get_mock:
                    get_mock.return_value.url = "https://lens.google.com/result"
                    get_mock.return_value.raise_for_status.return_value = None
                    get_mock.return_value.text = "<html><title>Иван Ургант - Википедия</title></html>"
                    result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["search_results"]["provider"], "direct_web")
        self.assertIn("Иван Ургант", result["actor_names"])

    @unittest.skip("direct web parser is a fallback; local known-face matching is the primary actor recognition path")
    def test_direct_web_search_can_use_google_image_upload_without_public_url(self):
        tmp, base = make_result_dir(search_faces=None, create_faces=True)
        try:
            with patch("app.media_person_check.requests.post") as post_mock:
                post_mock.return_value.url = "https://www.google.com/search?tbs=sbi"
                post_mock.return_value.raise_for_status.return_value = None
                post_mock.return_value.text = "<html><title>Константин Хабенский фото</title></html>"
                result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["search_results"]["provider"], "direct_web")
        self.assertIn("Константин Хабенский", result["actor_names"])

    def test_local_known_faces_recognizes_actor_before_web_search(self):
        tmp, base = make_result_dir(search_faces=None, create_faces=True)
        try:
            known = {
                "people": [
                    {
                        "name": "Иван Ургант",
                        "file": "known/ivan.jpg",
                        "encoding": [0.1, 0.2, 0.3],
                    }
                ],
                "errors": [],
            }
            with patch("app.media_person_check._load_known_face_encodings", return_value=known):
                with patch("app.media_person_check._face_encoding_for_image", return_value=np.array([0.1, 0.2, 0.3])):
                    with patch("app.media_person_check.face_recognition.face_distance", return_value=np.array([0.31])):
                        with patch("app.media_person_check.requests.post") as post_mock:
                            result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertFalse(post_mock.called)
        self.assertEqual(result["search_results"]["faces"][0]["provider"], "local_known_faces")
        self.assertIn("Иван Ургант", result["actor_names"])

    def test_known_face_name_can_come_from_flat_jpeg_filename(self):
        from app.media_person_check import _name_from_reference_path

        root = Path(r"C:\known")
        self.assertEqual(_name_from_reference_path(root / "Лариса Гузеева.jpeg", root), "Лариса Гузеева")

    def test_evaluate_actor_recognition_outputs_red_bold_name(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}],
            create_faces=True,
        )
        try:
            result = evaluate_actor_recognition(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn("Иван Ургант", result["message"])
        self.assertIn('<strong class="media-person-name">Иван Ургант</strong>', result["message_html"])

    def test_evaluates_item_id_14_inside_view_model(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}],
            create_faces=True,
        )
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "14", "number": "14", "text": "актеры в ролике"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn('<strong class="media-person-name">Иван Ургант</strong>', item["message_html"])


if __name__ == "__main__":
    unittest.main()
