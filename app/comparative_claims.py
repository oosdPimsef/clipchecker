# -*- coding: utf-8 -*-
"""Detect comparative advertising claims in OCR text from video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


NO_COMPARATIVE_WORDS_MESSAGE = "Сравнительных слов не обнаружено"

COMPARATIVE_TERMS = [
    "выгодно",
    "выгоднее",
    "самый выгодный",
    "лучшая выгода",
    "дешево",
    "дешевле",
    "дешевый",
    "дешевле чем",
    "низкая цена",
    "ниже цена",
    "самая низкая цена",
    "минимальная цена",
    "лучше",
    "лучший",
    "лучшая",
    "лучшее",
    "лучшие",
    "самый лучший",
    "лучший выбор",
    "лучшее предложение",
    "лучше конкурентов",
    "выше качество",
    "качественнее",
    "быстрее",
    "быстрый",
    "быстрейший",
    "мощнее",
    "мощный",
    "сильнее",
    "эффективнее",
    "эффективный",
    "надежнее",
    "надежный",
    "прочнее",
    "экономичнее",
    "экономия",
    "больше выгоды",
    "больше возможностей",
    "больше пользы",
    "меньше затрат",
    "меньше переплат",
    "больше",
    "меньше",
    "дольше",
    "длительнее",
    "ярче",
    "чище",
    "свежее",
    "вкуснее",
    "удобнее",
    "проще",
    "легче",
    "комфортнее",
    "безопаснее",
    "уникальный",
    "уникальная",
    "единственный",
    "единственная",
    "первый",
    "первая",
    "номер 1",
    "№1",
    "лидер",
    "лидер рынка",
    "лидирующий",
    "топ",
    "топовый",
    "премиальный",
    "эксклюзивный",
    "не имеет аналогов",
    "нет аналогов",
    "превосходит",
    "превосходство",
    "опережает",
    "обходит",
    "конкурент",
    "конкуренты",
    "по сравнению",
    "в сравнении",
    "сравни",
    "сравните",
    "аналогичный",
    "аналоги",
    "альтернатива",
    "самый",
    "самая",
    "самое",
    "самые",
    "максимальный",
    "максимальная",
    "максимум",
    "минимум",
]


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term.lower().replace("ё", "е"))
    escaped = escaped.replace(r"\ ", r"\s+")
    if re.search(r"[a-zа-я0-9]$", term.lower(), flags=re.IGNORECASE):
        escaped = escaped + r"[а-яa-z]*"
    return re.compile(rf"(?<![a-zа-я0-9]){escaped}(?![a-zа-я0-9])", flags=re.IGNORECASE)


COMPARATIVE_PATTERNS = [(term, _term_pattern(term)) for term in COMPARATIVE_TERMS]


def extract_comparative_terms(text: str) -> list[str]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    found = []
    seen = set()
    for term, pattern in COMPARATIVE_PATTERNS:
        if pattern.search(normalized):
            key = term.lower().replace("ё", "е")
            if key not in seen:
                seen.add(key)
                found.append(term)
    return found


def analyze_comparative_claims(ocr_data: dict) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        for term in extract_comparative_terms(text):
            total += 1
            key = term.lower().replace("ё", "е")
            if key not in grouped:
                grouped[key] = {
                    "term": term,
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 3:
                grouped[key]["examples"].append(text)

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "comparative_word_count": total,
        "comparative_terms": list(grouped.values()),
        "dictionary_size": len(COMPARATIVE_TERMS),
    }


def _format_terms_html(terms: list[dict]) -> str:
    parts = []
    for item in terms:
        seconds = ", ".join(item["seconds"][:8])
        safe_term = html.escape(item["term"])
        safe_seconds = html.escape(seconds)
        parts.append(
            f'<strong style="color:#dc2626;font-weight:800">{safe_term}</strong> '
            f"({safe_seconds})"
        )
    return "; ".join(parts)


def evaluate_comparative_claims(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка сравнительных слов будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка сравнительных слов будет выполнена после извлечения кадров.",
        }

    analysis = analyze_comparative_claims(load_ocr_log(ocr_path))
    terms = analysis["comparative_terms"]
    if not terms:
        return {
            "status": "pass",
            "message": NO_COMPARATIVE_WORDS_MESSAGE,
            **analysis,
        }

    plain_terms = "; ".join(f"{item['term']} ({', '.join(item['seconds'][:8])})" for item in terms)
    message = f"Обнаружены слова/элементы, похожие на сравнение с конкурентами: {plain_terms}."
    message_html = (
        "Обнаружены слова/элементы, похожие на сравнение с конкурентами: "
        f"{_format_terms_html(terms)}."
    )
    return {
        "status": "fail",
        "message": message,
        "message_html": message_html,
        **analysis,
    }
