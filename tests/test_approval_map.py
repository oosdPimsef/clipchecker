# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

from openpyxl import Workbook

from app.approval_map import build_approval_view_model, load_approval_categories, load_approval_checklist


def make_workbook():
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)

    wb = Workbook()
    ws = wb.active
    ws.title = "РМ"
    ws.append(["", "", "", "", "Категория"])
    ws.append(["", "Блок проверки", "номер", "id", "Проверка", "Ювелирная", "Ритейл", "Фарма"])
    ws.append(["", "Видеоряд", 1, "V-001", "длительность ролика кратна 5 секундам", 1, 1, 1])
    ws.append(["", "Видеоряд", 2, "V-002", "в видеоряде есть изображение ювелирного изделия", 1, "", ""])
    ws.append(["", "Набивка", 3, "N-003", "есть возрастная маркировка", 1, "", 1])
    ws.append(["", "Документы", 4, "D-004", "есть письмо о сайте", "", 1, ""])
    wb.save(path)
    wb.close()
    return path


class ApprovalMapTests(unittest.TestCase):
    def test_load_categories_from_header(self):
        path = make_workbook()
        try:
            self.assertEqual(load_approval_categories(path), ["Ювелирная", "Ритейл", "Фарма"])
        finally:
            os.remove(path)

    def test_filters_items_by_selected_category(self):
        path = make_workbook()
        try:
            checklist = load_approval_checklist("Ювелирная", path)
            self.assertEqual(list(checklist.keys()), ["Видеоряд", "Набивка"])
            self.assertEqual([item.id for item in checklist["Видеоряд"]], ["V-001", "V-002"])
            self.assertEqual([item.number for item in checklist["Видеоряд"]], ["1", "2"])
            self.assertEqual(checklist["Набивка"][0].text, "есть возрастная маркировка")
        finally:
            os.remove(path)

    def test_view_model_keeps_blocks_and_selected_category(self):
        path = make_workbook()
        try:
            model = build_approval_view_model("Ритейл", path)
            self.assertTrue(model["ok"])
            self.assertEqual(model["selected_category"], "Ритейл")
            self.assertEqual([block["name"] for block in model["blocks"]], ["Видеоряд", "Документы"])
            self.assertEqual(model["blocks"][0]["items"][0]["id"], "V-001")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
