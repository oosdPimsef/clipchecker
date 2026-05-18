# -*- coding: utf-8 -*-
"""Detect state, municipal, religious, and tricolor symbols in video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image, ImageStat

try:
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


NO_STATE_SYMBOLS_MESSAGE = "Государственные символы не обнаружены"

STATE_SYMBOL_DEFINITIONS = [
    ("флаг", ["флаг", "знамя", "триколор"]),
    ("государственный флаг", ["государственный флаг", "флаг россии", "российский флаг"]),
    ("герб", ["герб", "геральдика", "геральдический"]),
    ("государственный герб", ["государственный герб", "герб россии", "герб рф"]),
    ("герб города", ["герб города", "городской герб", "герб москвы", "герб санкт-петербурга"]),
    ("герб региона", ["герб региона", "герб области", "герб края", "герб республики"]),
    ("двуглавый орел", ["двуглавый орел", "двуглавый орёл", "орел", "орёл"]),
    ("гимн", ["гимн", "государственный гимн"]),
    ("штандарт", ["штандарт", "президентский штандарт"]),
    ("эмблема госоргана", ["эмблема", "минобороны", "мвд", "мчс", "фсб", "россия"]),
    ("символы власти", ["кремль", "правительство", "президент", "госдума", "совет федерации"]),
    ("георгиевская лента", ["георгиевская лента", "георгиевская ленточка"]),
    ("военная символика", ["звезда", "красная звезда", "победа", "вечный огонь"]),
    ("религиозный крест", ["крест", "православный крест", "распятие"]),
    ("религиозный полумесяц", ["полумесяц", "исламский полумесяц"]),
    ("звезда давида", ["звезда давида", "иудейская звезда"]),
    ("икона", ["икона", "лик святого", "святой образ"]),
]
STATE_SYMBOL_CONTEXT_EXCLUSIONS = [
    r"\bфлагман[а-яa-z]*\b",
]


def _variant_pattern(variant: str) -> re.Pattern:
    escaped = re.escape(variant.lower().replace("ё", "е"))
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-zа-я0-9]){escaped}[а-яa-z]*(?![a-zа-я0-9])", flags=re.IGNORECASE)


STATE_SYMBOL_PATTERNS = [
    (definition, variant, _variant_pattern(variant))
    for definition, variants in STATE_SYMBOL_DEFINITIONS
    for variant in variants
]


def extract_state_symbol_terms(text: str) -> list[dict]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    for exclusion in STATE_SYMBOL_CONTEXT_EXCLUSIONS:
        normalized = re.sub(exclusion, " ", normalized, flags=re.IGNORECASE)
    found = []
    seen = set()
    for definition, variant, pattern in STATE_SYMBOL_PATTERNS:
        if not pattern.search(normalized):
            continue
        key = (definition, variant)
        if key in seen:
            continue
        seen.add(key)
        found.append({"definition": definition, "term": variant})
    return found


def _avg_rgb(img: Image.Image) -> tuple[float, float, float]:
    stat = ImageStat.Stat(img.convert("RGB"))
    r, g, b = stat.mean[:3]
    return r, g, b


def _is_white(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return min(r, g, b) >= 155 and max(r, g, b) - min(r, g, b) <= 70


def _is_blue(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return b >= 85 and b >= r + 20 and b >= g + 10


def _is_red(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return r >= 110 and r >= g + 30 and r >= b + 30


def detect_tricolor_frame(frame_path: str | Path) -> bool:
    try:
        with Image.open(frame_path) as original:
            img = original.convert("RGB").resize((90, 60))
    except Exception:
        return False

    width, height = img.size
    horizontal = [
        _avg_rgb(img.crop((0, 0, width, height // 3))),
        _avg_rgb(img.crop((0, height // 3, width, 2 * height // 3))),
        _avg_rgb(img.crop((0, 2 * height // 3, width, height))),
    ]
    vertical = [
        _avg_rgb(img.crop((0, 0, width // 3, height))),
        _avg_rgb(img.crop((width // 3, 0, 2 * width // 3, height))),
        _avg_rgb(img.crop((2 * width // 3, 0, width, height))),
    ]

    return (
        _is_white(horizontal[0]) and _is_blue(horizontal[1]) and _is_red(horizontal[2])
    ) or (
        _is_white(vertical[0]) and _is_blue(vertical[1]) and _is_red(vertical[2])
    )


def _add_grouped(grouped: dict[str, dict], definition: str, term: str, frame: str, example: str) -> None:
    key = f"{definition}|{term}".lower().replace("ё", "е")
    if key not in grouped:
        grouped[key] = {
            "definition": definition,
            "term": term,
            "seconds": [],
            "frames": [],
            "examples": [],
        }
    grouped[key]["seconds"].append(frame_second_label(frame))
    grouped[key]["frames"].append(frame)
    if example and len(grouped[key]["examples"]) < 3:
        grouped[key]["examples"].append(example)


def analyze_state_symbols(ocr_data: dict | None, frames_dir: str | Path | None) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    if ocr_data:
        for item in iter_ocr_lines(ocr_data):
            text = normalize_text(item["text"])
            for match in extract_state_symbol_terms(text):
                total += 1
                _add_grouped(grouped, match["definition"], match["term"], item["frame"], text)

    frames_base = Path(frames_dir) if frames_dir else None
    if frames_base and frames_base.is_dir():
        for frame_path in sorted(
            path
            for path in frames_base.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ):
            if detect_tricolor_frame(frame_path):
                total += 1
                _add_grouped(grouped, "триколор", "визуальный триколор", frame_path.name, "визуальная эвристика")

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "state_symbol_count": total,
        "state_symbols": list(grouped.values()),
        "dictionary_size": len(STATE_SYMBOL_DEFINITIONS),
    }


def _format_symbols_html(symbols: list[dict]) -> str:
    parts = []
    for item in symbols:
        seconds = ", ".join(item["seconds"][:8])
        safe_term = html.escape(item["term"])
        safe_definition = html.escape(item["definition"])
        safe_seconds = html.escape(seconds)
        parts.append(
            f'<strong style="color:#dc2626;font-weight:800">{safe_term}</strong> '
            f"({safe_definition}; {safe_seconds})"
        )
    return "; ".join(parts)


def evaluate_state_symbols(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    frames_dir = find_frames_dir(base)

    if not ocr_path.is_file() and frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка государственных символов будет выполнена после OCR и извлечения кадров.",
        }

    ocr_data = load_ocr_log(ocr_path) if ocr_path.is_file() else None
    analysis = analyze_state_symbols(ocr_data, frames_dir)
    symbols = analysis["state_symbols"]
    if not symbols:
        return {
            "status": "pass",
            "message": NO_STATE_SYMBOLS_MESSAGE,
            **analysis,
        }

    plain_symbols = "; ".join(
        f"{item['term']} ({item['definition']}; {', '.join(item['seconds'][:8])})"
        for item in symbols
    )
    message = f"Обнаружены возможные государственные/религиозные символы: {plain_symbols}."
    message_html = (
        "Обнаружены возможные государственные/религиозные символы: "
        f"{_format_symbols_html(symbols)}."
    )
    return {
        "status": "fail",
        "message": message,
        "message_html": message_html,
        **analysis,
    }
