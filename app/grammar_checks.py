# -*- coding: utf-8 -*-
"""Local OCR word checks split by legal disclaimer scope."""

from __future__ import annotations

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


KNOWN_WORDS = {
    "реклама",
    "рекламодатель",
    "акция",
    "акции",
    "условия",
    "сайт",
    "телефон",
    "москва",
    "приложение",
    "приложении",
    "доставка",
    "продажи",
    "товаров",
    "организатор",
    "скидка",
    "скидки",
    "цена",
    "руб",
    "сек",
}

COMMON_MISSPELLINGS = {
    "инфоормация": "информация",
    "ошибкок": "ошибок",
    "рекламодателль": "рекламодатель",
    "доставкка": "доставка",
    "приложениии": "приложении",
    "условвия": "условия",
    "скидкка": "скидка",
}

IGNORED_WORDS = {
    "ооо",
    "оао",
    "ао",
    "зао",
    "пао",
    "огрн",
    "инн",
    "кпп",
    "рф",
    "тв",
    "https",
    "http",
    "www",
    "г",
    "д",
    "стр",
    "пр-зд",
    "офис",
}


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)?", text or "")


def find_abbreviation_notes(text: str) -> list[str]:
    notes: list[str] = []
    for match in re.finditer(r"\b[А-Яа-яЁёA-Za-z]{1,4}\.", text or ""):
        notes.append(match.group(0))
    for word in tokenize_words(text):
        normalized = _normalized_word(word)
        if "-" in word and len(normalized) >= 3:
            notes.append(word)
    return notes


def _has_cyrillic(word: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", word))


def _has_latin(word: str) -> bool:
    return bool(re.search(r"[A-Za-z]", word))


def _is_brand_or_abbreviation(word: str) -> bool:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", word)
    if len(letters) <= 1:
        return True
    upper = [char for char in letters if char.upper() == char and char.lower() != char]
    return len(upper) == len(letters)


def _normalized_word(word: str) -> str:
    return word.lower().replace("ё", "е").strip("-'")


def check_word(word: str) -> dict | None:
    normalized = _normalized_word(word)
    if len(normalized) <= 2 or normalized in IGNORED_WORDS:
        return None
    if re.search(r"\d", word) or word.startswith(("http", "www")):
        return None
    if _is_brand_or_abbreviation(word):
        return None

    has_cyr = _has_cyrillic(word)
    has_lat = _has_latin(word)
    if has_cyr and has_lat:
        return {"word": word, "issue": "смешаны кириллица и латиница"}

    if normalized in COMMON_MISSPELLINGS:
        return {"word": word, "issue": f"возможная ошибка, ожидается «{COMMON_MISSPELLINGS[normalized]}»"}

    if re.search(r"([а-яё])\1{3,}", normalized):
        return {"word": word, "issue": "подозрительное повторение букв"}

    if has_cyr and len(normalized) >= 5 and not re.search(r"[аеёиоуыэюя]", normalized):
        return {"word": word, "issue": "в русском слове не найдены гласные"}

    if has_cyr and len(normalized) >= 8:
        for known in KNOWN_WORDS:
            if normalized.startswith(known) and normalized != known and normalized not in COMMON_MISSPELLINGS:
                extra = normalized[len(known):]
                if len(extra) <= 2 and re.fullmatch(r"([а-яё])\1+", extra):
                    return {"word": word, "issue": f"возможная лишняя буква после «{known}»"}

    return None


def _line_is_legal(item: dict, frames_dir: Path) -> bool:
    size = frame_size(frames_dir / item["frame"])
    if size is None:
        return False
    _, height = size
    return is_legal_disclaimer_text(item["text"], item["bbox"], height)


def _matches_scope(item: dict, frames_dir: Path, scope: str) -> bool:
    is_legal = _line_is_legal(item, frames_dir)
    if scope == "legal_disclaimer":
        return is_legal
    if scope == "non_legal":
        return not is_legal
    return True


def analyze_grammar(ocr_data: dict, frames_dir: str | Path, scope: str) -> dict:
    frames_base = Path(frames_dir)
    checked_words = 0
    issues = []
    recommendation_notes = []

    for item in iter_ocr_lines(ocr_data):
        if not _matches_scope(item, frames_base, scope):
            continue
        for note in find_abbreviation_notes(item["text"]):
            recommendation_notes.append(
                {
                    "word": note,
                    "text": item["text"],
                    "frame": item["frame"],
                    "second": frame_second_label(item["frame"]),
                }
            )
        for word in tokenize_words(item["text"]):
            checked_words += 1
            issue = check_word(word)
            if issue:
                issues.append(
                    {
                        **issue,
                        "text": item["text"],
                        "frame": item["frame"],
                        "second": frame_second_label(item["frame"]),
                    }
                )

    grouped = {}
    for issue in issues:
        key = (_normalized_word(issue["word"]), issue["issue"])
        if key not in grouped:
            grouped[key] = {
                "word": issue["word"],
                "issue": issue["issue"],
                "seconds": [],
                "frames": [],
                "examples": [],
            }
        grouped[key]["seconds"].append(issue["second"])
        grouped[key]["frames"].append(issue["frame"])
        if len(grouped[key]["examples"]) < 2:
            grouped[key]["examples"].append(normalize_text(issue["text"]))

    for issue in grouped.values():
        issue["seconds"] = sorted(set(issue["seconds"]), key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0)
        issue["frames"] = sorted(set(issue["frames"]))

    grouped_notes = {}
    for note in recommendation_notes:
        key = _normalized_word(note["word"])
        if key not in grouped_notes:
            grouped_notes[key] = {
                "word": note["word"],
                "seconds": [],
                "frames": [],
                "examples": [],
            }
        grouped_notes[key]["seconds"].append(note["second"])
        grouped_notes[key]["frames"].append(note["frame"])
        if len(grouped_notes[key]["examples"]) < 2:
            grouped_notes[key]["examples"].append(normalize_text(note["text"]))

    for note in grouped_notes.values():
        note["seconds"] = sorted(set(note["seconds"]), key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0)
        note["frames"] = sorted(set(note["frames"]))

    return {
        "checked_words": checked_words,
        "issue_count": len(issues),
        "issues": list(grouped.values()),
        "recommendation_count": len(recommendation_notes),
        "recommendations": list(grouped_notes.values()),
        "scope": scope,
    }


SCOPE_LABELS = {
    "non_legal": "слова вне юридической набивки",
    "legal_disclaimer": "слова юридической набивки",
}


def evaluate_grammar(result_dir: str | Path, scope: str) -> dict:
    base = Path(result_dir)
    label = SCOPE_LABELS.get(scope, "слова OCR")
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": f"Проверка орфографии для группы «{label}» будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": f"Проверка орфографии для группы «{label}» будет выполнена после извлечения кадров.",
        }

    analysis = analyze_grammar(load_ocr_log(ocr_path), frames_dir, scope)
    if analysis["checked_words"] == 0:
        return {
            "status": "pending",
            "message": f"Слова группы «{label}» для проверки орфографии не найдены.",
            **analysis,
        }

    if analysis["issue_count"] == 0:
        recommendation = _build_recommendation_text(analysis)
        message = f"Группа «{label}»: явные орфографические/OCR-ошибки не найдены. Проверено слов: {analysis['checked_words']}."
        if recommendation:
            message += f" Рекомендация: {recommendation}"
        return {
            "status": "pass",
            "message": message,
            **analysis,
        }

    preview = []
    for issue in analysis["issues"][:4]:
        preview.append(f"{issue['word']} - {issue['issue']} ({', '.join(issue['seconds'][:4])})")
    recommendation = _build_recommendation_text(analysis)
    message = f"Группа «{label}»: найдены возможные ошибки: {'; '.join(preview)}."
    if recommendation:
        message += f" Рекомендация: {recommendation}"

    return {
        "status": "fail",
        "message": message,
        **analysis,
    }


def _build_recommendation_text(analysis: dict) -> str:
    notes = analysis.get("recommendations") or []
    if not notes:
        return ""
    preview = []
    for note in notes[:5]:
        seconds = ", ".join(note["seconds"][:3])
        preview.append(f"{note['word']} ({seconds})")
    return f"вручную проверьте сокращения и написания через дефис: {'; '.join(preview)}."


def evaluate_non_legal_grammar(result_dir: str | Path) -> dict:
    return evaluate_grammar(result_dir, scope="non_legal")


def evaluate_legal_disclaimer_grammar(result_dir: str | Path) -> dict:
    return evaluate_grammar(result_dir, scope="legal_disclaimer")
