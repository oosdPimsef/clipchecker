# -*- coding: utf-8 -*-
"""Check required distance sales wording in legal disclaimer overlays."""

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


DISTANCE_SALES_RE = re.compile(
    r"(?<![а-яa-z])дистанционн[а-яё]*\s+продаж[а-яё]*(?![а-яa-z])",
    flags=re.IGNORECASE,
)


def _line_is_legal_disclaimer(item: dict, frames_dir: Path | None) -> bool:
    if frames_dir is None:
        return False
    size = frame_size(frames_dir / item["frame"])
    if size is None:
        return False
    _, height = size
    return is_legal_disclaimer_text(item["text"], item["bbox"], height)


def analyze_distance_sales_disclaimer(ocr_data: dict, frames_dir: str | Path | None) -> dict:
    frames_base = Path(frames_dir) if frames_dir else None
    checked_lines = 0
    matches: dict[str, dict] = {}

    for item in iter_ocr_lines(ocr_data):
        if not _line_is_legal_disclaimer(item, frames_base):
            continue

        checked_lines += 1
        text = normalize_text(item["text"])
        if not DISTANCE_SALES_RE.search(text.lower().replace("ё", "е")):
            continue

        key = text.lower().replace("ё", "е")
        if key not in matches:
            matches[key] = {
                "text": text,
                "seconds": [],
                "frames": [],
            }
        matches[key]["seconds"].append(frame_second_label(item["frame"]))
        matches[key]["frames"].append(item["frame"])

    for item in matches.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "checked_legal_disclaimer_lines": checked_lines,
        "distance_sales_count": sum(len(item["frames"]) for item in matches.values()),
        "distance_sales_mentions": list(matches.values()),
    }


def evaluate_distance_sales_disclaimer(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка фразы «Дистанционные продажи» будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка фразы «Дистанционные продажи» будет выполнена после извлечения кадров.",
        }

    analysis = analyze_distance_sales_disclaimer(load_ocr_log(ocr_path), frames_dir)
    mentions = analysis["distance_sales_mentions"]
    if mentions:
        found = "; ".join(f"{item['text']} ({', '.join(item['seconds'][:6])})" for item in mentions[:6])
        return {
            "status": "pass",
            "message": f"В тексте набивки найдена фраза «Дистанционные продажи»: {found}.",
            **analysis,
        }

    return {
        "status": "warning",
        "message": "В тексте набивки фраза «Дистанционные продажи» не найдена.",
        **analysis,
    }
