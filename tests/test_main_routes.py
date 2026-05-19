# -*- coding: utf-8 -*-

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.main import app


class MainRoutesTests(unittest.TestCase):
    def test_post_start_redirects_to_result_page(self):
        fake_process = Mock()
        fake_process.poll.return_value = None

        with patch("app.main._launch_analysis_background", return_value=(fake_process, 1.0)) as launch:
            with patch("app.main._wait_for_analysis_start") as wait_start:
                response = app.test_client().post(
                    "/",
                    data={
                        "preview": r"C:\preview",
                        "docs": r"C:\docs",
                        "broadcast": r"C:\broadcast",
                        "approval_category": "test",
                        "openai_model": "",
                    },
                )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/result")
        launch.assert_called_once()
        wait_start.assert_called_once_with(r"C:\preview", fake_process, 1.0)

    def test_result_page_has_collapsible_approval_blocks_filters_and_ready_status(self):
        template_data = {
            "analysis_done": True,
            "runtime_log": "done",
            "approval": {
                "ok": True,
                "selected_category": "test",
                "blocks": [
                    {
                        "name": "Видеоряд",
                        "items": [
                            {"id": "1", "text": "Длительность", "status": "pass", "message": "OK"},
                            {"id": "2", "text": "БЗ", "status": "fail", "message": "Ошибка"},
                            {"id": "3", "text": "Набивка", "status": "warning", "message": "Проверить"},
                        ],
                    }
                ],
            },
            "tech_validation_data": [],
            "face_recognition": "",
            "speech_fasterwhisper": "",
            "speech_yandex": "",
            "music_info": "",
            "music_shazam": "",
            "ocr_text": "",
            "overlay_report": "",
            "document_texts": "",
            "broadcast_params": "",
            "openai_review": "",
            "legal_analysis_full": "",
        }

        with patch("app.main._collect_results_for_template", return_value=template_data):
            response = app.test_client().get("/result")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('<details class="approval-block" open>', html)
        self.assertIn('data-approval-filter="green"', html)
        self.assertIn('data-approval-filter="orange"', html)
        self.assertIn('data-approval-filter="red"', html)
        self.assertIn('data-approval-status="green"', html)
        self.assertIn('data-approval-status="red"', html)
        self.assertIn('data-approval-status="orange"', html)
        self.assertIn('id="analysis-progress"', html)
        self.assertIn('id="analysis-ready"', html)
        self.assertIn("анализ готов", html)


if __name__ == "__main__":
    unittest.main()
