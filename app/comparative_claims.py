# -*- coding: utf-8 -*-
"""Detect advertising claims in OCR text and supporting materials."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import (
        find_frames_dir,
        frame_second_label,
        frame_size,
        is_legal_disclaimer_text,
        iter_ocr_lines,
        load_ocr_log,
        normalize_text,
    )
    from price_checks import read_text_file, split_document_sections
except ImportError:
    from .frame_safety import (
        find_frames_dir,
        frame_second_label,
        frame_size,
        is_legal_disclaimer_text,
        iter_ocr_lines,
        load_ocr_log,
        normalize_text,
    )
    from .price_checks import read_text_file, split_document_sections


NO_COMPARATIVE_WORDS_MESSAGE = "Проверяемые claims не обнаружены"

COMPARATIVE_TERMS = [
    "№1",
    "номер 1",
    "номер один",
    "n1",
    "no 1",
    "no. 1",
    "number one",
    "топ 1",
    "топ-1",
    "первый",
    "первая",
    "первые",
    "лидер",
    "лидер рынка",
    "лидирующий",
    "лучший",
    "лучшая",
    "лучшее",
    "лучшие",
    "самый лучший",
    "самая лучшая",
    "самое лучшее",
    "самый",
    "самая",
    "самое",
    "самые",
    "максимальный",
    "максимальная",
    "максимум",
    "минимальный",
    "минимальная",
    "минимум",
    "эффективный",
    "эффективная",
    "эффективнее",
    "результативный",
    "результат уже",
    "доказанный результат",
    "клинически доказано",
    "доказано",
    "подтверждено",
    "результаты исследований",
    "исследования показали",
    "по данным исследования",
    "безопасный",
    "безопасная",
    "безопаснее",
    "надежный",
    "надежная",
    "гарантированный",
    "гарантированная",
    "гарантия результата",
    "рекомендовано",
    "рекомендуют",
    "одобрено",
    "проверено экспертами",
    "экспертный выбор",
    "официальный",
    "официальная",
    "официально",
    "сертифицированный",
    "сертифицированная",
    "новинка",
    "впервые",
    "инновационный",
    "инновационная",
    "уникальный",
    "уникальная",
    "эксклюзивный",
    "эксклюзивная",
    "не имеет аналогов",
    "нет аналогов",
    "скидка до",
    "до 50%",
    "до 70%",
    "до 90%",
    "цена от",
    "от 999",
    "выгода до",
    "экономия до",
    "дешевле",
    "дешевле чем",
    "выгоднее",
    "выгодно",
    "низкая цена",
    "самая низкая цена",
    "лучшее предложение",
    "лучший выбор",
    "бестселлер",
    "хит продаж",
    "популярный",
    "самый популярный",
    "выбор покупателей",
    "больше",
    "меньше",
    "быстрее",
    "быстрый",
    "быстрейший",
    "мощнее",
    "мощный",
    "сильнее",
    "прочнее",
    "экономичнее",
    "больше выгоды",
    "больше возможностей",
    "меньше затрат",
    "меньше переплат",
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
CLAIM_SUPPORT_RE = re.compile(
    r"(?:по\s+данным|согласно|источник|исследован[а-яё]*|опрос[а-яё]*|рейтинг[а-яё]*|"
    r"подтвержден[а-яё]*|сертификат[а-яё]*|деклараци[а-яё]*|протокол[а-яё]*|"
    r"отчет[а-яё]*|заключени[а-яё]*|справк[а-яё]*|письм[а-яё]*|договор[а-яё]*|"
    r"прав[а-яё]*\s+на|лицензи[а-яё]*|услови[а-яё]*\s+акци[а-яё]*|подробност[а-яё]*|"
    r"период[а-яё]*\s+акци[а-яё]*|количеств[а-яё]*\s+ограничен[а-яё]*|"
    r"сравнен[а-яё]*|категори[а-яё]*|рынк[а-яё]*|в\s+сети|на\s+сайт[а-яё]*|"
    r"не\s+является|может\s+отличаться|указан[а-яё]*\s+цена|цена\s+от|скидк[а-яё]*\s+до)",
    flags=re.IGNORECASE,
)
DOCUMENT_SUPPORT_TITLE_RE = re.compile(
    r"(?:исследован[а-яё]*|рейтинг[а-яё]*|сертификат[а-яё]*|деклараци[а-яё]*|протокол[а-яё]*|"
    r"заключени[а-яё]*|справк[а-яё]*|письм[а-яё]*|подтверждени[а-яё]*|отчет[а-яё]*|"
    r"услови[а-яё]*\s+акци[а-яё]*|claim|клейм)",
    flags=re.IGNORECASE,
)


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


def _line_is_legal_disclaimer(item: dict, frames_dir: Path | None) -> bool:
    if frames_dir is None:
        return False
    size = frame_size(frames_dir / item["frame"])
    if size is None:
        return False
    _, height = size
    return is_legal_disclaimer_text(item["text"], item["bbox"], height)


def analyze_claim_disclaimers(ocr_data: dict, frames_dir: str | Path | None) -> list[dict]:
    frames_base = Path(frames_dir) if frames_dir else None
    disclaimers: dict[str, dict] = {}

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        if not _line_is_legal_disclaimer(item, frames_base):
            continue
        if not (CLAIM_SUPPORT_RE.search(text) or extract_comparative_terms(text) or "*" in text):
            continue

        key = text.lower().replace("ё", "е")
        if key not in disclaimers:
            disclaimers[key] = {
                "text": text,
                "seconds": [],
                "frames": [],
            }
        disclaimers[key]["seconds"].append(frame_second_label(item["frame"]))
        disclaimers[key]["frames"].append(item["frame"])

    for item in disclaimers.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))
    return list(disclaimers.values())


def _context_around(text: str, start: int, end: int, radius: int = 220) -> str:
    return normalize_text(text[max(0, start - radius): min(len(text), end + radius)])


def analyze_claim_document_support(documents_text: str, claim_terms: list[dict]) -> list[dict]:
    sections = split_document_sections(documents_text)
    if not sections and (documents_text or "").strip():
        sections = [("", documents_text)]

    support: dict[str, dict] = {}
    claim_names = [item["term"] for item in claim_terms]
    normalized_claims = [term.lower().replace("ё", "е") for term in claim_names]

    for title, section_text in sections:
        section = section_text or ""
        haystack = f"{title}\n{section}"
        normalized = normalize_text(haystack).lower().replace("ё", "е")
        title_has_support = bool(DOCUMENT_SUPPORT_TITLE_RE.search(title or ""))

        candidate_contexts: list[tuple[str, str]] = []
        for term, normalized_term in zip(claim_names, normalized_claims):
            for match in _term_pattern(normalized_term).finditer(normalized):
                context = _context_around(haystack, match.start(), match.end())
                if CLAIM_SUPPORT_RE.search(context) or title_has_support:
                    candidate_contexts.append((term, context))

        if not candidate_contexts and title_has_support and CLAIM_SUPPORT_RE.search(haystack):
            candidate_contexts.append(("подтверждающий документ", _context_around(haystack, 0, min(len(haystack), 220), radius=0)))

        for term, context in candidate_contexts:
            key = f"{title}\n{context}".lower()
            if key not in support:
                support[key] = {
                    "title": title,
                    "claim": term,
                    "text": context,
                }

    return list(support.values())


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
            "message": "Проверка claims будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка claims будет выполнена после извлечения кадров.",
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
    message = f"Обнаружены проверяемые claims: {plain_terms}."
    message_html = (
        "Обнаружены проверяемые claims: "
        f"{_format_terms_html(terms)}."
    )
    return {
        "status": "fail",
        "message": message,
        "message_html": message_html,
        **analysis,
    }


def _format_support_items_html(items: list[dict], text_key: str = "text") -> str:
    parts = []
    for item in items:
        text = item.get(text_key, "")
        suffix = ""
        if item.get("seconds"):
            suffix = f" ({html.escape(', '.join(item['seconds'][:6]))})"
        elif item.get("title"):
            suffix = f" ({html.escape(item['title'])})"
        parts.append(f'<strong style="color:#dc2626;font-weight:800">{html.escape(text)}</strong>{suffix}')
    return "; ".join(parts)


def evaluate_claim_support(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка документов и набивок по claims будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка документов и набивок по claims будет выполнена после извлечения кадров.",
        }

    ocr_data = load_ocr_log(ocr_path)
    claims_analysis = analyze_comparative_claims(ocr_data)
    claims = claims_analysis["comparative_terms"]
    if not claims:
        return {
            "status": "pass",
            "message": "Проверяемые claims не обнаружены.",
            **claims_analysis,
        }

    disclaimers = analyze_claim_disclaimers(ocr_data, frames_dir)
    document_support = analyze_claim_document_support(read_text_file(base / "Documents_Texts.txt"), claims)
    support_found = bool(disclaimers or document_support)
    plain_claims = "; ".join(f"{item['term']} ({', '.join(item['seconds'][:8])})" for item in claims)

    details = {
        **claims_analysis,
        "claim_disclaimers": disclaimers,
        "claim_document_support": document_support,
        "claim_support_found": support_found,
    }

    if support_found:
        parts = []
        if disclaimers:
            parts.append(
                "найдена объясняющая набивка: "
                + "; ".join(f"{item['text']} ({', '.join(item['seconds'][:6])})" for item in disclaimers[:8])
            )
        if document_support:
            parts.append(
                "найдены подтверждающие фразы в документах: "
                + "; ".join(item["text"] for item in document_support[:8])
            )
        message_html = (
            f"Claims: {_format_terms_html(claims)}. "
            + (
                "Объясняющая набивка: "
                + _format_support_items_html(disclaimers[:8])
                + ". "
                if disclaimers
                else ""
            )
            + (
                "Фразы в документах: "
                + _format_support_items_html(document_support[:8])
                + "."
                if document_support
                else ""
            )
        )
        return {
            "status": "warning",
            "message": f"Claims найдены: {plain_claims}. " + " ".join(parts) + ".",
            "message_html": message_html,
            **details,
        }

    return {
        "status": "fail",
        "message": (
            f"Claims найдены: {plain_claims}. Объясняющая набивка и подтверждающие документы не найдены."
        ),
        "message_html": (
            f"Claims: {_format_terms_html(claims)}. "
            "Объясняющая набивка и подтверждающие документы не найдены."
        ),
        **details,
    }
