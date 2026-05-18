# -*- coding: utf-8 -*-
"""Detect anglicisms, neologisms and invented non-Russian OCR words."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from foreign_words import analyze_star_translations
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .foreign_words import analyze_star_translations
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)?")
IGNORED_LATIN_WORDS = {"http", "https", "www", "ru", "com", "net", "org"}
COMMON_RUSSIAN_WORDS = {
    "акции",
    "акция",
    "в",
    "все",
    "для",
    "до",
    "и",
    "из",
    "на",
    "не",
    "новая",
    "новые",
    "о",
    "об",
    "от",
    "по",
    "подробности",
    "при",
    "продажи",
    "реклама",
    "рекламодатель",
    "с",
    "сайте",
    "сайт",
    "скидка",
    "скидки",
    "телефон",
    "товар",
    "товара",
    "товары",
    "условия",
    "участвующие",
}
ANGLICISM_TERMS = {
    "айфон",
    "аккаунт",
    "апгрейд",
    "аутлет",
    "бестселлер",
    "блог",
    "блогер",
    "бренд",
    "брендовый",
    "вайб",
    "веб",
    "видео",
    "гаджет",
    "дедлайн",
    "дизайн",
    "диджитал",
    "драйв",
    "ивент",
    "инсайт",
    "интернет",
    "кейс",
    "кешбэк",
    "клик",
    "коллаборация",
    "контент",
    "кэшбэк",
    "лайв",
    "лайк",
    "лайфстайл",
    "лайфхак",
    "лендинг",
    "логин",
    "лук",
    "маркет",
    "маркетинг",
    "маркетплейс",
    "мерч",
    "мессенджер",
    "онлайн",
    "офлайн",
    "премиум",
    "промо",
    "релиз",
    "репост",
    "сейл",
    "сервис",
    "скилл",
    "смарт",
    "софт",
    "сторис",
    "стрим",
    "тренд",
    "фейк",
    "фешн",
    "фидбек",
    "фит",
    "флеш",
    "хайп",
    "хит",
    "худи",
    "шопинг",
}
ANGLICISM_PARTS = (
    "кешбэк",
    "кэшбэк",
    "маркетплейс",
    "лайфстайл",
    "диджитал",
    "онлайн",
    "офлайн",
    "шопинг",
    "фешн",
    "фэшн",
    "хайп",
)
LATIN_TO_CYRILLIC_OCR = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
    }
)


def _normalize_word(word: str) -> str:
    return word.lower().replace("ё", "е").strip("-'")


def _has_latin(word: str) -> bool:
    return bool(re.search(r"[A-Za-z]", word))


def _has_cyrillic(word: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", word))


def _is_ignored_token(word: str) -> bool:
    normalized = _normalize_word(word)
    if len(normalized) <= 1:
        return True
    if normalized in IGNORED_LATIN_WORDS or normalized in COMMON_RUSSIAN_WORDS:
        return True
    if re.fullmatch(r"[ivxlcdm]+", normalized):
        return True
    if _has_latin(word) and not _has_cyrillic(word):
        cyrillic_like = _normalize_word(word.translate(LATIN_TO_CYRILLIC_OCR))
        if cyrillic_like in COMMON_RUSSIAN_WORDS:
            return True
    return False


def classify_non_russian_word(word: str) -> str | None:
    normalized = _normalize_word(word)
    if _is_ignored_token(word):
        return None

    has_latin = _has_latin(word)
    has_cyrillic = _has_cyrillic(word)
    if has_latin and has_cyrillic:
        return "смешаны латиница и кириллица"
    if has_latin:
        return "латиница"

    if normalized in {term.replace("ё", "е") for term in ANGLICISM_TERMS}:
        return "англицизм/неологизм"
    if any(part in normalized for part in ANGLICISM_PARTS):
        return "англицизм/неологизм"

    # A conservative OCR-friendly signal for invented words: repeated uncommon
    # consonant clusters in Cyrillic. This avoids marking all brand names.
    if len(normalized) >= 8 and re.search(r"[бвгджзйклмнпрстфхцчшщ]{5,}", normalized):
        return "похоже на придуманное слово или OCR-шум"

    return None


def extract_non_russian_words(text: str) -> list[dict]:
    output = []
    for match in TOKEN_RE.finditer(text or ""):
        word = match.group(0)
        reason = classify_non_russian_word(word)
        if reason:
            output.append({"word": word, "reason": reason})
    return output


def analyze_non_russian_words(ocr_data: dict) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        for found in extract_non_russian_words(text):
            total += 1
            key = _normalize_word(found["word"])
            if key not in grouped:
                grouped[key] = {
                    "word": found["word"],
                    "reason": found["reason"],
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 2:
                grouped[key]["examples"].append(text)

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "non_russian_word_count": total,
        "non_russian_words": list(grouped.values()),
        "dictionary_size": len(ANGLICISM_TERMS),
    }


def _format_words_html(words: list[dict]) -> str:
    parts = []
    for item in words:
        safe_word = html.escape(item["word"])
        safe_seconds = html.escape(", ".join(item["seconds"][:6]))
        safe_reason = html.escape(item.get("reason", ""))
        parts.append(
            f'<strong style="color:#dc2626;font-weight:800">{safe_word}</strong> '
            f"({safe_reason}; {safe_seconds})"
        )
    return "; ".join(parts)


def _format_words_plain(words: list[dict]) -> str:
    return "; ".join(
        f"{item['word']} ({item.get('reason', '')}; {', '.join(item['seconds'][:6])})"
        for item in words
    )


def filter_star_translations_for_words(translations: list[dict], words: list[dict]) -> list[dict]:
    targets = {_normalize_word(item["word"]) for item in words}
    matched = []
    for item in translations:
        text = normalize_text(item.get("text", ""))
        text_words = {_normalize_word(match.group(0)) for match in TOKEN_RE.finditer(text)}
        if targets & text_words:
            matched.append(item)
    return matched


def evaluate_non_russian_words(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка англицизмов, неологизмов и придуманных слов будет выполнена после OCR кадров.",
        }
    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка англицизмов, неологизмов и придуманных слов будет выполнена после извлечения кадров.",
        }

    analysis = analyze_non_russian_words(load_ocr_log(ocr_path))
    words = analysis["non_russian_words"]
    if not words:
        return {
            "status": "pass",
            "message": "Англицизмы, неологизмы и придуманные слова на кадрах не найдены.",
            **analysis,
        }

    return {
        "status": "fail",
        "message": f"Найдены слова, не принадлежащие русскому языку: {_format_words_plain(words)}.",
        "message_html": f"Найдены слова, не принадлежащие русскому языку: {_format_words_html(words)}.",
        **analysis,
    }


def evaluate_non_russian_words_translation(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка перевода англицизмов и неологизмов будет выполнена после OCR кадров.",
        }
    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка перевода англицизмов и неологизмов будет выполнена после извлечения кадров.",
        }

    ocr_data = load_ocr_log(ocr_path)
    analysis = analyze_non_russian_words(ocr_data)
    words = analysis["non_russian_words"]
    if not words:
        return {
            "status": "pass",
            "message": "Англицизмы, неологизмы и придуманные слова на кадрах не найдены.",
            **analysis,
        }

    translations = filter_star_translations_for_words(analyze_star_translations(ocr_data), words)
    result = {**analysis, "translations": translations}
    plain_words = _format_words_plain(words)
    if translations:
        translations_plain = "; ".join(f"{item['text']} ({', '.join(item['seconds'][:6])})" for item in translations[:8])
        translations_html = "; ".join(
            f'<strong style="color:#dc2626;font-weight:800">{html.escape(item["text"])}</strong> '
            f'({html.escape(", ".join(item["seconds"][:6]))})'
            for item in translations[:8]
        )
        return {
            "status": "warning",
            "message": (
                f"Найдены слова, не принадлежащие русскому языку: {plain_words}. "
                f"Найден перевод со звёздочкой: {translations_plain}."
            ),
            "message_html": (
                f"Найдены слова, не принадлежащие русскому языку: {_format_words_html(words)}. "
                f"Найден перевод со звёздочкой: {translations_html}."
            ),
            **result,
        }

    return {
        "status": "fail",
        "message": (
            f"Найдены слова, не принадлежащие русскому языку: {plain_words}. "
            "Перевод со звёздочкой на кадрах не найден."
        ),
        "message_html": (
            f"Найдены слова, не принадлежащие русскому языку: {_format_words_html(words)}. "
            "Перевод со звёздочкой на кадрах не найден."
        ),
        **result,
    }
