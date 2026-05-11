# -*- coding: utf-8 -*-
"""Small Yandex Vision OCR helpers with retryable network handling."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import requests


VISION_OCR_URL = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"


@dataclass
class OcrResult:
    ok: bool
    text: str
    raw: dict
    attempts: int
    error: str | None = None


def extract_text_from_vision_response(result: dict) -> str:
    blocks = result["results"][0]["results"][0]["textDetection"]["pages"][0].get("blocks", [])
    texts: list[str] = []
    for block in blocks:
        for line in block.get("lines", []):
            line_text = " ".join(word.get("text", "") for word in line.get("words", []))
            if line_text.strip():
                texts.append(line_text.strip())
    return "\n".join(texts).strip()


def _build_payload(encoded_image: str, folder_id: str) -> dict:
    return {
        "folderId": folder_id,
        "analyze_specs": [
            {
                "content": encoded_image,
                "features": [
                    {
                        "type": "TEXT_DETECTION",
                        "textDetectionConfig": {"languageCodes": ["*"]},
                    }
                ],
            }
        ],
    }


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 425, 429, 500, 502, 503, 504}


def recognize_image_file(
    image_path: str,
    *,
    api_key: str,
    folder_id: str,
    attempts: int = 3,
    timeout: int = 90,
    backoff_seconds: float = 1.5,
    post: Callable[..., requests.Response] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> OcrResult:
    """Recognize text in one image and return a structured result instead of raising."""
    post = post or requests.post
    sleep = sleep or time.sleep
    attempts = max(1, int(attempts or 1))
    timeout = max(1, int(timeout or 1))

    try:
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode("utf-8")
    except Exception as exc:
        return OcrResult(
            ok=False,
            text="",
            raw={"error": {"stage": "read_file", "message": str(exc)}},
            attempts=0,
            error=f"read_file: {exc}",
        )

    payload = _build_payload(encoded, folder_id)
    headers = {"Authorization": f"Api-Key {api_key}"}
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            response = post(VISION_OCR_URL, headers=headers, json=payload, timeout=timeout)
            status_code = getattr(response, "status_code", None)
            if status_code and status_code >= 400:
                body = getattr(response, "text", "") or ""
                last_error = f"HTTP {status_code}: {body[:500]}"
                if attempt < attempts and _is_retryable_status(status_code):
                    sleep(backoff_seconds * attempt)
                    continue
                return OcrResult(
                    ok=False,
                    text="",
                    raw={
                        "error": {
                            "stage": "http",
                            "status_code": status_code,
                            "message": body[:2000],
                        }
                    },
                    attempts=attempt,
                    error=last_error,
                )

            try:
                result = response.json()
            except Exception as exc:
                last_error = f"json_decode: {exc}"
                if attempt < attempts:
                    sleep(backoff_seconds * attempt)
                    continue
                return OcrResult(
                    ok=False,
                    text="",
                    raw={"error": {"stage": "json_decode", "message": str(exc)}},
                    attempts=attempt,
                    error=last_error,
                )

            api_error = result.get("results", [{}])[0].get("error") if isinstance(result, dict) else None
            if api_error:
                message = api_error.get("message") if isinstance(api_error, dict) else str(api_error)
                return OcrResult(
                    ok=False,
                    text="",
                    raw=result,
                    attempts=attempt,
                    error=f"api_error: {message}",
                )

            try:
                text = extract_text_from_vision_response(result)
            except Exception as exc:
                return OcrResult(
                    ok=False,
                    text="",
                    raw=result if isinstance(result, dict) else {"raw": str(result)},
                    attempts=attempt,
                    error=f"parse_response: {exc}",
                )

            return OcrResult(ok=True, text=text, raw=result, attempts=attempt)

        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                sleep(backoff_seconds * attempt)
                continue
            return OcrResult(
                ok=False,
                text="",
                raw={"error": {"stage": "request", "message": last_error}},
                attempts=attempt,
                error=last_error,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            return OcrResult(
                ok=False,
                text="",
                raw={"error": {"stage": "unexpected", "message": last_error}},
                attempts=attempt,
                error=last_error,
            )

    return OcrResult(
        ok=False,
        text="",
        raw={"error": {"stage": "unknown", "message": last_error}},
        attempts=attempts,
        error=last_error or "unknown error",
    )

