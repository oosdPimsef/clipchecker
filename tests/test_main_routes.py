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


if __name__ == "__main__":
    unittest.main()
