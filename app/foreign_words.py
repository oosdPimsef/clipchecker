# -*- coding: utf-8 -*-
"""Detect foreign OCR words on video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


WARNING_TEXT = "Иностранные слова должны быть переведены или предоставлены документы на товарный знак!"
IGNORED_LATIN_WORDS = {"http", "https", "www"}


def extract_foreign_words(text: str) -> list[str]:
    words = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z'-]*\b", text or ""):
        word = match.group(0).strip("-'")
        if len(word) < 2 or word.lower() in IGNORED_LATIN_WORDS:
            continue
        words.append(word)
    return words


def analyze_foreign_words(ocr_data: dict) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        for word in extract_foreign_words(item["text"]):
            total += 1
            key = word.lower()
            if key not in grouped:
                grouped[key] = {
                    "word": word,
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 2:
                grouped[key]["examples"].append(normalize_text(item["text"]))

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "foreign_word_count": total,
        "foreign_words": list(grouped.values()),
    }


def _format_words_html(words: list[dict]) -> str:
    parts = []
    for item in words:
        seconds = ", ".join(item["seconds"][:5])
        safe_word = html.escape(item["word"])
        safe_seconds = html.escape(seconds)
        parts.append(f'<strong class="foreign-word">{safe_word}</strong> ({safe_seconds})')
    return "; ".join(parts)


def evaluate_foreign_words(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка иностранных слов будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка иностранных слов будет выполнена после извлечения кадров.",
        }

    analysis = analyze_foreign_words(load_ocr_log(ocr_path))
    words = analysis["foreign_words"]
    if not words:
        return {
            "status": "pass",
            "message": "Иностранные слова на кадрах не найдены.",
            **analysis,
        }

    plain_words = "; ".join(f"{item['word']} ({', '.join(item['seconds'][:5])})" for item in words)
    message = f"{WARNING_TEXT} Найдены иностранные слова: {plain_words}."
    message_html = f"{html.escape(WARNING_TEXT)} Найдены иностранные слова: {_format_words_html(words)}."
    return {
        "status": "warning",
        "message": message,
        "message_html": message_html,
        **analysis,
    }
