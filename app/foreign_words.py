# -*- coding: utf-8 -*-
"""Detect foreign OCR words on video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
    from price_checks import read_text_file, split_document_sections
except ImportError:
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
    from .price_checks import read_text_file, split_document_sections


WARNING_TEXT = "Иностранные слова должны быть переведены или предоставлены документы на товарный знак!"
IGNORED_LATIN_WORDS = {"http", "https", "www"}
TRADEMARK_CONTEXT_RE = re.compile(
    r"(?:товарн[а-яё]*\s+знак[а-яё]*|знак[а-яё]*\s+обслуживан[а-яё]*|trademark|trade\s*mark)",
    flags=re.IGNORECASE,
)
QUOTED_BRAND_RE = re.compile(r"[«\"]([^»\"\n]{2,80})[»\"]")
LATIN_BRAND_RE = re.compile(r"\b[A-Z][A-Z0-9&'.-]{1,40}(?:\s+[A-Z][A-Z0-9&'.-]{1,40}){0,4}\b")
CYRILLIC_BRAND_RE = re.compile(r"\b[А-ЯЁ][А-ЯЁ0-9&'.-]{2,40}(?:\s+[А-ЯЁ][А-ЯЁ0-9&'.-]{2,40}){0,4}\b")


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


def analyze_star_translations(ocr_data: dict) -> list[dict]:
    translations: dict[str, dict] = {}
    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        if "*" not in text:
            continue
        if not re.search(r"[А-Яа-яЁё]", text):
            continue

        key = text.lower()
        if key not in translations:
            translations[key] = {
                "text": text,
                "seconds": [],
                "frames": [],
            }
        translations[key]["seconds"].append(frame_second_label(item["frame"]))
        translations[key]["frames"].append(item["frame"])

    for item in translations.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))
    return list(translations.values())


def _clean_brand(candidate: str) -> str:
    value = normalize_text(candidate)
    value = re.sub(r"^(?:товарный|словесный|комбинированный|изобразительный)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:()[]")
    return value


def _extract_brands_from_context(context: str) -> list[str]:
    brands = []

    for regex in (QUOTED_BRAND_RE, LATIN_BRAND_RE, CYRILLIC_BRAND_RE):
        for match in regex.finditer(context):
            brand = _clean_brand(match.group(1) if regex is QUOTED_BRAND_RE else match.group(0))
            lower = brand.lower()
            if not brand or lower in {"товарный знак", "знак обслуживания", "свидетельство", "приложение"}:
                continue
            if TRADEMARK_CONTEXT_RE.fullmatch(lower):
                continue
            if brand not in brands:
                brands.append(brand)

    return brands


def analyze_trademark_documents(documents_text: str) -> dict:
    sections = split_document_sections(documents_text)
    matches = []
    grouped_brands: dict[str, dict] = {}

    for title, section_text in sections:
        haystack = f"{title}\n{section_text}"
        for match in TRADEMARK_CONTEXT_RE.finditer(haystack):
            context = normalize_text(haystack[max(0, match.start() - 180): min(len(haystack), match.end() + 220)])
            brands = _extract_brands_from_context(context)
            if not brands:
                brands = ["документ о товарном знаке без автоматически определённого названия"]

            item = {
                "title": title,
                "context": context,
                "brands": brands,
            }
            matches.append(item)
            for brand in brands:
                key = brand.lower()
                if key not in grouped_brands:
                    grouped_brands[key] = {
                        "brand": brand,
                        "titles": [],
                        "contexts": [],
                    }
                if title and title not in grouped_brands[key]["titles"]:
                    grouped_brands[key]["titles"].append(title)
                if len(grouped_brands[key]["contexts"]) < 2:
                    grouped_brands[key]["contexts"].append(context)

    return {
        "trademark_document_count": len(matches),
        "trademark_brands": list(grouped_brands.values()),
        "trademark_matches": matches,
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


def _format_support_html(translations: list[dict], trademark_brands: list[dict]) -> str:
    parts = []
    if translations:
        translation_parts = []
        for item in translations:
            safe_text = html.escape(item["text"])
            safe_seconds = html.escape(", ".join(item["seconds"][:5]))
            translation_parts.append(f'<strong style="color:#dc2626;font-weight:800">{safe_text}</strong> ({safe_seconds})')
        parts.append("Найдены переводы со звёздочкой: " + "; ".join(translation_parts))

    if trademark_brands:
        brand_parts = []
        for item in trademark_brands:
            safe_brand = html.escape(item["brand"])
            titles = ", ".join(item["titles"][:3]) if item["titles"] else "документы"
            safe_titles = html.escape(titles)
            brand_parts.append(f'<strong style="color:#dc2626;font-weight:800">{safe_brand}</strong> ({safe_titles})')
        parts.append("Найдены документы по товарному знаку: " + "; ".join(brand_parts))

    return ". ".join(parts) + "."


def evaluate_foreign_words_translation_or_trademark(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка переводов и документов на товарный знак будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка переводов и документов на товарный знак будет выполнена после извлечения кадров.",
        }

    ocr_data = load_ocr_log(ocr_path)
    foreign_analysis = analyze_foreign_words(ocr_data)
    foreign_words = foreign_analysis["foreign_words"]
    if not foreign_words:
        return {
            "status": "pass",
            "message": "Иностранные слова на кадрах не найдены.",
            **foreign_analysis,
        }

    translations = analyze_star_translations(ocr_data)
    trademark_analysis = analyze_trademark_documents(read_text_file(base / "Documents_Texts.txt"))
    trademark_brands = trademark_analysis["trademark_brands"]
    support_found = bool(translations or trademark_brands)
    foreign_plain = "; ".join(f"{item['word']} ({', '.join(item['seconds'][:5])})" for item in foreign_words)

    result = {
        **foreign_analysis,
        "translations": translations,
        **trademark_analysis,
    }

    if support_found:
        support_parts = []
        if translations:
            support_parts.append(
                "найдены переводы со звёздочкой: "
                + "; ".join(f"{item['text']} ({', '.join(item['seconds'][:5])})" for item in translations)
            )
        if trademark_brands:
            support_parts.append(
                "найдены документы по товарному знаку: "
                + "; ".join(item["brand"] for item in trademark_brands)
            )
        return {
            "status": "warning",
            "message": f"Иностранные слова найдены: {foreign_plain}. " + " ".join(support_parts) + ".",
            "message_html": (
                f"Иностранные слова найдены: {_format_words_html(foreign_words)}. "
                + _format_support_html(translations, trademark_brands)
            ),
            **result,
        }

    return {
        "status": "fail",
        "message": (
            f"Иностранные слова найдены: {foreign_plain}. Перевод со звёздочкой и документы на товарный знак не найдены."
        ),
        "message_html": (
            f"Иностранные слова найдены: {_format_words_html(foreign_words)}. "
            "Перевод со звёздочкой и документы на товарный знак не найдены."
        ),
        **result,
    }
