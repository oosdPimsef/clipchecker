# -*- coding: utf-8 -*-
"""Detect jewelry-related evidence in preprocessed video frame materials."""

from __future__ import annotations

import re
from pathlib import Path

try:
    from black_bars import find_source_frames_dir
    from cv_detection import detect_cv_objects_in_frames, filter_cv_detections
    from frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .black_bars import find_source_frames_dir
    from .cv_detection import detect_cv_objects_in_frames, filter_cv_detections
    from .frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


JEWELRY_FOUND_MESSAGE = "Рекламодатель должен стоять С, а так же предоставить документы, подтверждающее это"
JEWELRY_NOT_FOUND_MESSAGE = (
    "Необходимо проверить наличие ювелирных изделий в видеоряде и предоставить документы "
    "о постановке на спецучет в ГИИС ДМДК"
)
JEWELRY_TAGS_REQUIRED_MESSAGE = "Необходимо предосатвить бирки на ювелирную продукцию в ролике"
JEWELRY_TAGS_NOT_REQUIRED_MESSAGE = "Ювелирных изделий в видеоряд не найдено"

JEWELRY_KEYWORDS = [
    "ювелир",
    "кольцо",
    "кольца",
    "серьги",
    "сережки",
    "браслет",
    "цепочка",
    "цепь",
    "подвеска",
    "кулон",
    "колье",
    "ожерелье",
    "золото",
    "золотой",
    "золотая",
    "серебро",
    "серебряный",
    "серебряная",
    "бриллиант",
    "бриллианты",
    "алмаз",
    "камень",
    "карат",
    "diamond",
    "jewelry",
    "jewellery",
    "ring",
    "rings",
    "earrings",
    "bracelet",
    "necklace",
    "pendant",
]
JEWELRY_CV_LABELS = [
    "jewelry",
    "jewellery",
    "ring",
    "earring",
    "earrings",
    "bracelet",
    "necklace",
    "pendant",
    "chain",
    "gold ring",
    "diamond ring",
    "diamond",
    "gemstone",
    "brooch",
    "watch",
]


KEYWORD_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(word) for word in JEWELRY_KEYWORDS) + r")([\w-]*)",
    flags=re.IGNORECASE,
)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_keywords(text: str) -> list[str]:
    found = []
    for match in KEYWORD_RE.finditer(text or ""):
        word = match.group(0).strip(" .,;:!?()[]{}\"'")
        if word and word.lower() not in {item.lower() for item in found}:
            found.append(word)
    return found


def analyze_jewelry_mentions_from_ocr(ocr_data: dict) -> dict:
    mentions: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item.get("text", ""))
        keywords = _extract_keywords(text)
        if not keywords:
            continue

        total += len(keywords)
        frame = item.get("frame", "")
        second = frame_second_label(frame)
        for keyword in keywords:
            key = keyword.lower()
            if key not in mentions:
                mentions[key] = {
                    "keyword": keyword,
                    "frames": [],
                    "seconds": [],
                    "examples": [],
                }
            mentions[key]["frames"].append(frame)
            mentions[key]["seconds"].append(second)
            if len(mentions[key]["examples"]) < 2:
                mentions[key]["examples"].append(text)

    for item in mentions.values():
        item["frames"] = sorted(set(item["frames"]))
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )

    return {
        "jewelry_keyword_count": total,
        "jewelry_mentions": list(mentions.values()),
    }


def analyze_jewelry_mentions_from_text(text: str) -> dict:
    keywords = _extract_keywords(text)
    return {
        "jewelry_keyword_count": len(keywords),
        "jewelry_mentions": [
            {"keyword": keyword, "frames": [], "seconds": [], "examples": []}
            for keyword in keywords
        ],
    }


def _analyze_jewelry_from_result_dir(result_dir: str | Path) -> tuple[dict, bool]:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    all_text_path = base / "All_Frames_Text.txt"
    frames_dir = base / "frames"

    if ocr_path.is_file():
        analysis = analyze_jewelry_mentions_from_ocr(load_ocr_log(ocr_path))
    else:
        analysis = analyze_jewelry_mentions_from_text(_read_text(all_text_path))

    cv_analysis = analyze_jewelry_mentions_from_cv(base)
    analysis = {**analysis, **cv_analysis}
    has_materials = frames_dir.is_dir() or (base / "frames_pdf_original").is_dir() or all_text_path.is_file() or ocr_path.is_file()
    return analysis, has_materials


def analyze_jewelry_mentions_from_cv(result_dir: str | Path) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "cv_enabled": False,
            "cv_model_path": "",
            "cv_error": "Кадры для CV-проверки не найдены.",
            "cv_detections": [],
            "cv_jewelry_mentions": [],
        }
    cv_result = detect_cv_objects_in_frames(frames_dir, labels=JEWELRY_CV_LABELS)
    filtered = filter_cv_detections(cv_result["detections"], JEWELRY_CV_LABELS)
    return {
        "cv_enabled": cv_result["enabled"],
        "cv_model_path": cv_result["model_path"],
        "cv_error": cv_result["error"],
        "cv_detections": cv_result["detections"],
        "cv_jewelry_mentions": filtered,
    }


def evaluate_jewelry_presence(result_dir: str | Path) -> dict:
    analysis, has_materials = _analyze_jewelry_from_result_dir(result_dir)
    mentions = analysis["jewelry_mentions"]
    cv_mentions = analysis.get("cv_jewelry_mentions", [])
    if mentions or cv_mentions:
        keywords = ", ".join(item["keyword"] for item in mentions[:8])
        seconds = sorted(
            {second for item in mentions for second in item.get("seconds", [])},
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        cv_labels = ", ".join(
            str(item.get("raw_label") or item.get("label")) for item in cv_mentions[:8]
        )
        place_parts = []
        if keywords:
            place_parts.append(f"текст: {keywords}")
        if cv_labels:
            place_parts.append(f"CV: {cv_labels}")
        place = f" Найденные признаки: {'; '.join(place_parts)}."
        if seconds:
            place += f" Кадры/секунды: {', '.join(seconds[:10])}."
        return {
            "status": "pass",
            "message": f"{JEWELRY_FOUND_MESSAGE}.{place}",
            **analysis,
        }

    if not has_materials:
        return {
            "status": "pending",
            "message": "Проверка наличия ювелирных изделий будет выполнена после извлечения кадров и OCR.",
            **analysis,
        }

    return {
        "status": "warning",
        "message": JEWELRY_NOT_FOUND_MESSAGE,
        **analysis,
    }


def evaluate_jewelry_tags_required(result_dir: str | Path) -> dict:
    analysis, has_materials = _analyze_jewelry_from_result_dir(result_dir)
    mentions = analysis["jewelry_mentions"]
    cv_mentions = analysis.get("cv_jewelry_mentions", [])
    if mentions or cv_mentions:
        return {
            "status": "warning",
            "message": JEWELRY_TAGS_REQUIRED_MESSAGE,
            **analysis,
        }

    if not has_materials:
        return {
            "status": "pending",
            "message": "Проверка наличия бирок будет выполнена после извлечения кадров и OCR.",
            **analysis,
        }

    return {
        "status": "pass",
        "message": JEWELRY_TAGS_NOT_REQUIRED_MESSAGE,
        **analysis,
    }
