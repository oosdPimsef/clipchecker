# -*- coding: utf-8 -*-
"""Check website/page or phone contacts in legal disclaimer overlays."""

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


URL_RE = re.compile(
    r"(?<![\w@])(?:https?://|www\.)?[a-zа-яё0-9][a-zа-яё0-9-]{1,63}"
    r"(?:\.[a-zа-яё0-9][a-zа-яё0-9-]{1,63})*"
    r"\.[a-zа-яё]{2,24}"
    r"(?:/[^\s,;]*)?",
    flags=re.IGNORECASE,
)
SOCIAL_PAGE_RE = re.compile(
    r"(?<![\w@])(?:vk|vkontakte|ok|t\.me|telegram|wa\.me|youtube|rutube|instagram)\s*[:/\\]\s*[a-zа-яё0-9_.-]{2,}",
    flags=re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\s*7|8)\s*(?:\(?\s*\d{3}\s*\)?[\s.-]*)"
    r"(?:\d[\s.-]*){7}(?!\d)"
)


def _line_is_legal_disclaimer(item: dict, frames_dir: Path | None) -> bool:
    if frames_dir is None:
        return False
    size = frame_size(frames_dir / item["frame"])
    if size is None:
        return False
    _, height = size
    return is_legal_disclaimer_text(item["text"], item["bbox"], height)


def _clean_contact(value: str) -> str:
    return normalize_text(value).strip(".,;:!?)(")


def extract_web_contacts(text: str) -> list[str]:
    contacts = []
    for regex in (URL_RE, SOCIAL_PAGE_RE):
        for match in regex.finditer(text or ""):
            value = _clean_contact(match.group(0))
            if value and value not in contacts:
                contacts.append(value)
    return contacts


def extract_phone_contacts(text: str) -> list[str]:
    phones = []
    for match in PHONE_RE.finditer(text or ""):
        value = _clean_contact(match.group(0))
        digits = re.sub(r"\D", "", value)
        if len(digits) < 11:
            continue
        if value and value not in phones:
            phones.append(value)
    return phones


def _add_grouped(grouped: dict[str, dict], kind: str, value: str, frame: str, text: str) -> None:
    key = f"{kind}|{value}".lower().replace("ё", "е")
    if key not in grouped:
        grouped[key] = {
            "kind": kind,
            "value": value,
            "seconds": [],
            "frames": [],
            "examples": [],
        }
    grouped[key]["seconds"].append(frame_second_label(frame))
    grouped[key]["frames"].append(frame)
    if len(grouped[key]["examples"]) < 2:
        grouped[key]["examples"].append(text)


def analyze_contact_info_disclaimer(ocr_data: dict, frames_dir: str | Path | None) -> dict:
    frames_base = Path(frames_dir) if frames_dir else None
    checked_lines = 0
    grouped: dict[str, dict] = {}

    for item in iter_ocr_lines(ocr_data):
        if not _line_is_legal_disclaimer(item, frames_base):
            continue

        checked_lines += 1
        text = normalize_text(item["text"])
        for value in extract_web_contacts(text):
            _add_grouped(grouped, "site", value, item["frame"], text)
        for value in extract_phone_contacts(text):
            _add_grouped(grouped, "phone", value, item["frame"], text)

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    contacts = list(grouped.values())
    return {
        "checked_legal_disclaimer_lines": checked_lines,
        "contact_count": sum(len(item["frames"]) for item in contacts),
        "contacts": contacts,
        "web_contacts": [item for item in contacts if item["kind"] == "site"],
        "phone_contacts": [item for item in contacts if item["kind"] == "phone"],
    }


def _format_contacts_plain(contacts: list[dict]) -> str:
    return "; ".join(f"{item['value']} ({', '.join(item['seconds'][:6])})" for item in contacts)


def _format_contacts_html(contacts: list[dict]) -> str:
    parts = []
    for item in contacts:
        safe_value = html.escape(item["value"])
        safe_seconds = html.escape(", ".join(item["seconds"][:6]))
        parts.append(f'<strong style="color:#dc2626;font-weight:800">{safe_value}</strong> ({safe_seconds})')
    return "; ".join(parts)


def evaluate_contact_info_disclaimer(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка сайта и телефона в набивке будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка сайта и телефона в набивке будет выполнена после извлечения кадров.",
        }

    analysis = analyze_contact_info_disclaimer(load_ocr_log(ocr_path), frames_dir)
    contacts = analysis["contacts"]
    if contacts:
        plain_contacts = _format_contacts_plain(contacts)
        return {
            "status": "pass",
            "message": f"В тексте набивки найдены сайт/страница или номер телефона: {plain_contacts}.",
            "message_html": (
                "В тексте набивки найдены сайт/страница или номер телефона: "
                f"{_format_contacts_html(contacts)}."
            ),
            **analysis,
        }

    return {
        "status": "warning",
        "message": "Сайт и номер телефона в тексте набивки не найдены.",
        **analysis,
    }
