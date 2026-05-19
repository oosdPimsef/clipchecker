# -*- coding: utf-8 -*-

import re
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_ANALYSIS_PATH = ROOT_DIR / "app" / "run_analysis.py"


class RunAnalysisOpenAIGateTests(unittest.TestCase):
    def test_openai_materials_review_runs_by_default(self):
        source = RUN_ANALYSIS_PATH.read_text(encoding="utf-8")
        self.assertNotIn("ENABLE_OPENAI", source)
        self.assertIn('os.getenv("DISABLE_OPENAI"', source)

        main_block = source.split('if __name__ == "__main__":', 1)[1]
        openai_block = main_block.split('print("✅ Последовательный анализ завершён.")', 1)[0]
        self.assertIn("run_openai_connectivity_test_safe(result_dir)", openai_block)
        self.assertIn("write_openai_tokens_status(result_dir)", openai_block)
        self.assertIn("run_openai_materials_review_safe(result_dir)", openai_block)
        self.assertIsNone(
            re.search(r"#\s*(run_openai_connectivity_test_safe|write_openai_tokens_status|run_openai_materials_review_safe)", openai_block)
        )


if __name__ == "__main__":
    unittest.main()
