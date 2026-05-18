# -*- coding: utf-8 -*-

import unittest

from app.cv_detection import filter_cv_detections, normalize_cv_label, unique_cv_labels


class CvDetectionTests(unittest.TestCase):
    def test_normalizes_labels_for_open_vocabulary_matching(self):
        self.assertEqual(normalize_cv_label("Brand-Logo"), "brand logo")
        self.assertEqual(normalize_cv_label("wine_bottle"), "wine bottle")

    def test_unique_cv_labels_keeps_first_spelling(self):
        self.assertEqual(unique_cv_labels(["logo", "Logo", "brand-logo", "brand logo"]), ["logo", "brand-logo"])

    def test_filter_cv_detections_matches_normalized_labels(self):
        detections = [
            {"label": "brand logo", "raw_label": "brand-logo"},
            {"label": "bottle", "raw_label": "bottle"},
        ]

        result = filter_cv_detections(detections, ["brand logo"])

        self.assertEqual(result, [{"label": "brand logo", "raw_label": "brand-logo"}])


if __name__ == "__main__":
    unittest.main()
