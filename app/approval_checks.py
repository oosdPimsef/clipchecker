# -*- coding: utf-8 -*-
"""Python evaluators for approval checklist items keyed by Excel item id."""

from __future__ import annotations

import re
from pathlib import Path


def _count_frame_seconds(result_dir: str | Path) -> int | None:
    base = Path(result_dir)
    for folder_name in ("frames_pdf_original", "frames"):
        frames_dir = base / folder_name
        if not frames_dir.is_dir():
            continue
        count = len(
            [
                p
                for p in frames_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        if count > 0:
            return count
    return None


def evaluate_duration_multiple_of_five(result_dir: str | Path) -> dict:
    duration_sec = _count_frame_seconds(result_dir)
    if duration_sec is None:
        return {
            "status": "pending",
            "message": "Длительность ролика будет рассчитана после предобработки кадров.",
        }

    ok = duration_sec % 5 == 0
    return {
        "status": "pass" if ok else "fail",
        "message": f"Длительность ролика {duration_sec} секунд.",
        "duration_sec": duration_sec,
    }


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _split_document_sections(documents_text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^Документ\s+\d+\.\s*(.+?)\s*$", documents_text))
    if not matches:
        return [("", documents_text)] if documents_text.strip() else []

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(documents_text)
        title = match.group(1).strip()
        sections.append((title, documents_text[start:end]))
    return sections


def _is_booking_form_document(title: str, text: str) -> bool:
    haystack = f"{title}\n{text}".lower().replace("ё", "е")
    patterns = [
        r"\bбз\b",
        r"бланк\s*[-–—]?\s*заявк",
        r"бланк\s+заявк",
    ]
    return any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in patterns)


def _extract_duration_from_booking_form(text: str) -> int | None:
    normalized = text.lower().replace("ё", "е")

    time_match = re.search(r"\b(?:[01]?\d|2[0-3]):([0-5]\d):([0-5]\d)\b", normalized)
    if time_match:
        return int(time_match.group(1)) * 60 + int(time_match.group(2))

    minute_second_match = re.search(
        r"\b(\d{1,2})\s*(?:мин|минут[а-я]*)\s*(\d{1,2})\s*(?:сек|секунд[а-я]*)\b",
        normalized,
    )
    if minute_second_match:
        return int(minute_second_match.group(1)) * 60 + int(minute_second_match.group(2))

    keyword_match = re.search(
        r"(?:длительность|продолжительность|хронометраж|хрон[-\s]*ж)[^\d]{0,40}(\d{1,4})\s*(?:сек|с\.|секунд[а-я]*)?",
        normalized,
    )
    if keyword_match:
        return int(keyword_match.group(1))

    seconds_match = re.search(r"\b(\d{1,4})\s*(?:сек|с\.|секунд[а-я]*)\b", normalized)
    if seconds_match:
        return int(seconds_match.group(1))

    return None


def evaluate_booking_form_duration_matches_video(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    video_duration = _count_frame_seconds(base)
    documents_text = _read_text(base / "Documents_Texts.txt")
    sections = _split_document_sections(documents_text)
    booking_sections = [(title, text) for title, text in sections if _is_booking_form_document(title, text)]

    if not booking_sections:
        return {
            "status": "fail",
            "message": "Бланк-заявки нет в документах.",
            "video_duration_sec": video_duration,
        }

    title, booking_text = booking_sections[0]
    booking_duration = _extract_duration_from_booking_form(booking_text)
    if booking_duration is None:
        return {
            "status": "fail",
            "message": f"Бланк-заявка найдена ({title}), но длительность ролика в ней не найдена.",
            "video_duration_sec": video_duration,
            "booking_form_title": title,
        }

    if video_duration is None:
        return {
            "status": "pending",
            "message": f"В БЗ указана длительность {booking_duration} секунд. Длительность ролика будет рассчитана после предобработки кадров.",
            "booking_duration_sec": booking_duration,
            "booking_form_title": title,
        }

    ok = booking_duration == video_duration
    return {
        "status": "pass" if ok else "fail",
        "message": f"Длительность ролика {video_duration} секунд. В БЗ указано {booking_duration} секунд.",
        "duration_sec": video_duration,
        "booking_duration_sec": booking_duration,
        "booking_form_title": title,
    }


EVALUATORS = {
    "1": evaluate_duration_multiple_of_five,
    "2": evaluate_booking_form_duration_matches_video,
}


def evaluate_approval_view_model(view_model: dict, result_dir: str | Path) -> dict:
    if not view_model.get("ok"):
        return view_model

    for block in view_model.get("blocks", []):
        for item in block.get("items", []):
            evaluator = EVALUATORS.get(str(item.get("id", "")).strip())
            if evaluator is None:
                item["status"] = "pending"
                item["message"] = "Оценка Python: будет добавлена на следующем этапе."
                continue

            result = evaluator(result_dir)
            item["status"] = result.get("status", "pending")
            item["message"] = result.get("message", "")
            item["details"] = {k: v for k, v in result.items() if k not in {"status", "message"}}

    return view_model
