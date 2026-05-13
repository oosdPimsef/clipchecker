# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from app.approval_checks import evaluate_approval_view_model
from app.media_person_check import (
    CONTRACT_FOUND_MESSAGE,
    CONTRACT_REQUIRED_MESSAGE,
    NO_MEDIA_PERSONS_MESSAGE,
    SEARCH_CACHE_NAME,
    evaluate_media_person_contracts,
    extract_media_person_names_from_text,
    find_contracts_for_person,
)


def make_result_dir(search_faces: list[dict] | None = None, documents_text: str = ""):
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    if search_faces is not None:
        (base / SEARCH_CACHE_NAME).write_text(
            json.dumps({"ok": True, "pending": False, "faces": search_faces}, ensure_ascii=False),
            encoding="utf-8",
        )
    if documents_text:
        (base / "Documents_Texts.txt").write_text(documents_text, encoding="utf-8")
    return tmp, base


class MediaPersonCheckTests(unittest.TestCase):
    def test_extract_media_person_names_from_text(self):
        self.assertEqual(extract_media_person_names_from_text("face: Иван Ургант 98%"), ["Иван Ургант"])
        self.assertEqual(extract_media_person_names_from_text("Публичная личность не установлена"), [])

    def test_find_contracts_for_person_by_surname(self):
        documents = (
            "Документ 1. Договор_Ургант.pdf\n"
            "Договор оказания услуг с Иваном Ургантом.\n\n"
            "Документ 2. Письмо.txt\n"
            "Обычное письмо."
        )
        self.assertEqual(find_contracts_for_person("Иван Ургант", documents), ["Договор_Ургант.pdf"])

    def test_evaluate_media_person_contracts_fails_when_contract_missing(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}],
            documents_text="Документ 1. Письмо.txt\nМатериалы к ролику.",
        )
        try:
            result = evaluate_media_person_contracts(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "fail")
        self.assertIn(CONTRACT_REQUIRED_MESSAGE, result["message"])
        self.assertIn('<strong class="media-person-name">Иван Ургант</strong>', result["message_html"])

    def test_evaluate_media_person_contracts_passes_when_contract_found(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}],
            documents_text="Документ 1. Договор_Ургант.pdf\nДоговор с Иваном Ургантом.",
        )
        try:
            result = evaluate_media_person_contracts(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertIn(CONTRACT_FOUND_MESSAGE, result["message"])
        self.assertIn('<strong class="media-person-name">Иван Ургант</strong>', result["message_html"])

    def test_evaluate_media_person_contracts_passes_when_persons_not_found(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Публичная личность не установлена", "names": []}],
        )
        try:
            result = evaluate_media_person_contracts(base)
        finally:
            tmp.cleanup()

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["message"], NO_MEDIA_PERSONS_MESSAGE)

    def test_evaluates_item_id_14_inside_view_model(self):
        tmp, base = make_result_dir(
            search_faces=[{"file": "face_001.jpg", "raw_result": "Иван Ургант", "names": ["Иван Ургант"]}],
            documents_text="Документ 1. Договор_Ургант.pdf\nДоговор с Иваном Ургантом.",
        )
        view_model = {
            "ok": True,
            "blocks": [
                {"name": "Видеоряд", "items": [{"id": "14", "number": "14", "text": "договор с медийной личностью"}]},
            ],
        }
        try:
            evaluated = evaluate_approval_view_model(view_model, base)
        finally:
            tmp.cleanup()

        item = evaluated["blocks"][0]["items"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn(CONTRACT_FOUND_MESSAGE, item["message"])


if __name__ == "__main__":
    unittest.main()
