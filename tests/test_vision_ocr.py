# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

import requests

from app.vision_ocr import recognize_image_file


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


def write_temp_image():
    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(b"fake image bytes")
    return path


def vision_payload_with_text(*words):
    return {
        "results": [
            {
                "results": [
                    {
                        "textDetection": {
                            "pages": [
                                {
                                    "blocks": [
                                        {
                                            "lines": [
                                                {
                                                    "words": [
                                                        {"text": word}
                                                        for word in words
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


class VisionOcrTests(unittest.TestCase):
    def test_retries_request_error_and_returns_text(self):
        path = write_temp_image()
        calls = {"count": 0}

        def fake_post(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise requests.ConnectionError("proxy closed connection")
            return FakeResponse(payload=vision_payload_with_text("TEST", "123"))

        try:
            result = recognize_image_file(
                path,
                api_key="key",
                folder_id="folder",
                attempts=3,
                post=fake_post,
                sleep=lambda _: None,
            )
        finally:
            os.remove(path)

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "TEST 123")
        self.assertEqual(result.attempts, 2)

    def test_retries_retryable_http_status(self):
        path = write_temp_image()
        responses = [
            FakeResponse(status_code=503, text="temporary unavailable"),
            FakeResponse(payload=vision_payload_with_text("OK")),
        ]

        def fake_post(*args, **kwargs):
            return responses.pop(0)

        try:
            result = recognize_image_file(
                path,
                api_key="key",
                folder_id="folder",
                attempts=2,
                post=fake_post,
                sleep=lambda _: None,
            )
        finally:
            os.remove(path)

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.attempts, 2)

    def test_returns_structured_api_error_without_raise(self):
        path = write_temp_image()

        def fake_post(*args, **kwargs):
            return FakeResponse(payload={"results": [{"error": {"message": "billing disabled"}}]})

        try:
            result = recognize_image_file(
                path,
                api_key="key",
                folder_id="folder",
                attempts=3,
                post=fake_post,
                sleep=lambda _: None,
            )
        finally:
            os.remove(path)

        self.assertFalse(result.ok)
        self.assertIn("billing disabled", result.error)
        self.assertEqual(result.attempts, 1)


if __name__ == "__main__":
    unittest.main()

