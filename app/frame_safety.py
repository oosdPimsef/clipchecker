# -*- coding: utf-8 -*-
"""Frame safety checks for substantial OCR-detected ad elements."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


SAFE_MARGIN_X = 0.10
SAFE_MARGIN_Y = 0.05


def load_ocr_log(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_frames_dir(result_dir: str | Path) -> Path | None:
    base = Path(result_dir)
    for folder_name in ("frames_pdf_original", "frames"):
        candidate = base / folder_name
        if candidate.is_dir():
            return candidate
    return None


def frame_size(frame_path: str | Path) -> tuple[int, int] | None:
    try:
        with Image.open(frame_path) as img:
            return img.size
    except Exception:
        return None


def safe_box(width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(width * SAFE_MARGIN_X),
        int(height * SAFE_MARGIN_Y),
        int(width * (1 - SAFE_MARGIN_X)),
        int(height * (1 - SAFE_MARGIN_Y)),
    )


def bbox_from_vertices(vertices: list[dict]) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for vertex in vertices or []:
        try:
            if "x" in vertex:
                xs.append(int(float(vertex["x"])))
            if "y" in vertex:
                ys.append(int(float(vertex["y"])))
        except (TypeError, ValueError):
            continue
    if len(xs) < 2 or len(ys) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int], margin: int = 2) -> bool:
    return (
        outer[0] - margin <= inner[0]
        and inner[2] <= outer[2] + margin
        and outer[1] - margin <= inner[1]
        and inner[3] <= outer[3] + margin
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalized_key(text: str) -> str:
    return normalize_text(text).lower().replace("ё", "е")


def classify_substantial_text(text: str) -> str | None:
    cleaned = normalize_text(text)
    if not cleaned:
        return None

    lower = normalized_key(cleaned)
    if re.search(r"(?<!\d)(?:0|6|12|16|18)\s*\+(?!\d)", lower):
        return "age_mark"
    if re.search(r"(?:₽|руб\.?|р\.|\b\d[\d\s,.]*\s*%|\bскидк[а-я]*)", lower):
        return "price"
    if len(cleaned) >= 2 and re.search(r"[a-zа-я]", lower):
        return "text"
    if re.search(r"\d\s*[=+\-/%]|[=+\-/%]\s*\d", lower):
        return "text"
    return None


def frame_second_label(frame_name: str) -> str:
    match = re.search(r"(\d+)(?=\.[^.]+$)", frame_name)
    if not match:
        return frame_name
    return f"{int(match.group(1))}сек."


def is_legal_disclaimer_text(text: str, bbox: tuple[int, int, int, int], frame_height: int) -> bool:
    lower = normalized_key(text)
    y_center = (bbox[1] + bbox[3]) / 2
    is_lower_screen = y_center >= frame_height * 0.55
    legal_keywords = (
        "реклама",
        "рекламодатель",
        "огрн",
        "инн",
        "акци",
        "услов",
        "организатор",
        "сайт",
        "телефон",
        "приложени",
        "доставка",
        "продаж",
    )
    return is_lower_screen and (len(lower) >= 45 or any(keyword in lower for keyword in legal_keywords))


def is_logo_text(text: str, category: str, frequency: int) -> bool:
    cleaned = normalize_text(text)
    if category != "text" or frequency < 2:
        return False

    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", cleaned)
    if not words or len(words) > 3 or len(cleaned) > 40:
        return False

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned)
    if not letters:
        return False

    upper_letters = [char for char in letters if char.upper() == char and char.lower() != char]
    has_latin = bool(re.search(r"[A-Za-z]", cleaned))
    uppercase_share = len(upper_letters) / len(letters)
    return has_latin or uppercase_share >= 0.6


def _line_text(line: dict[str, Any]) -> str:
    return normalize_text(" ".join(str(word.get("text", "")) for word in line.get("words", [])))


def _line_bbox(line: dict[str, Any]) -> tuple[int, int, int, int] | None:
    line_box = bbox_from_vertices(line.get("boundingBox", {}).get("vertices", []))
    if line_box:
        return line_box

    boxes = [
        bbox_from_vertices(word.get("boundingBox", {}).get("vertices", []))
        for word in line.get("words", [])
    ]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def iter_ocr_lines(ocr_data: dict) -> list[dict]:
    lines_out: list[dict] = []
    for frame_name, result in ocr_data.items():
        try:
            blocks = result["results"][0]["results"][0]["textDetection"]["pages"][0].get("blocks", [])
        except Exception:
            continue
        for block in blocks:
            for line in block.get("lines", []):
                text = _line_text(line)
                category = classify_substantial_text(text)
                bbox = _line_bbox(line)
                if category and bbox:
                    lines_out.append(
                        {
                            "frame": frame_name,
                            "text": text,
                            "category": category,
                            "bbox": bbox,
                        }
                    )
    return lines_out


def _matches_scope(item: dict, scope: str, frame_height: int, frequency: int) -> bool:
    if scope == "all_text":
        return True
    if scope == "legal_disclaimer":
        return is_legal_disclaimer_text(item["text"], item["bbox"], frame_height)
    if scope == "logos":
        return is_logo_text(item["text"], item["category"], frequency)
    return True


def analyze_frame_safety(ocr_data: dict, frames_dir: str | Path | None, scope: str = "all_text") -> dict:
    frames_base = Path(frames_dir) if frames_dir else None
    checked = 0
    violations: list[dict] = []
    all_items = iter_ocr_lines(ocr_data)
    frequencies = Counter(normalized_key(item["text"]) for item in all_items)

    for item in all_items:
        if not frames_base:
            continue
        size = frame_size(frames_base / item["frame"])
        if size is None:
            continue

        width, height = size
        frequency = frequencies[normalized_key(item["text"])]
        if not _matches_scope(item, scope, height, frequency):
            continue

        checked += 1
        box = safe_box(width, height)
        if not bbox_inside(item["bbox"], box):
            violations.append({**item, "safe_box": box, "second": frame_second_label(item["frame"])})

    grouped: dict[str, dict] = {}
    frames_with_violations = set()
    for violation in violations:
        key = normalized_key(violation["text"])
        frames_with_violations.add(violation["frame"])
        if key not in grouped:
            grouped[key] = {
                "text": violation["text"],
                "category": violation["category"],
                "frames": [],
                "seconds": [],
            }
        grouped[key]["frames"].append(violation["frame"])
        grouped[key]["seconds"].append(violation["second"])

    for group in grouped.values():
        group["frames"] = sorted(set(group["frames"]))
        group["seconds"] = sorted(set(group["seconds"]), key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0)

    return {
        "checked_count": checked,
        "violation_count": len(violations),
        "frames_with_violations_count": len(frames_with_violations),
        "violations": list(grouped.values()),
        "scope": scope,
    }


SCOPE_LABELS = {
    "legal_disclaimer": "текст юридических набивок",
    "logos": "логотипы рекламодателя, распознанные OCR",
    "all_text": "все надписи на экране",
}


def evaluate_frame_safety(result_dir: str | Path, scope: str = "all_text") -> dict:
    base = Path(result_dir)
    label = SCOPE_LABELS.get(scope, "существенные OCR-элементы")
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": f"Проверка рамки для группы «{label}» будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": f"Проверка рамки для группы «{label}» будет выполнена после извлечения кадров.",
        }

    analysis = analyze_frame_safety(load_ocr_log(ocr_path), frames_dir, scope=scope)
    if analysis["checked_count"] == 0:
        return {
            "status": "pending",
            "message": f"Элементы группы «{label}» для проверки рамки не найдены.",
            **analysis,
        }

    if analysis["violation_count"] == 0:
        return {
            "status": "pass",
            "message": f"Группа «{label}»: все элементы в зеленой рамке. Проверено элементов: {analysis['checked_count']}.",
            **analysis,
        }

    preview = []
    for violation in analysis["violations"][:3]:
        seconds = ", ".join(violation["seconds"][:5])
        preview.append(f"{violation['text']} ({seconds})")

    return {
        "status": "fail",
        "message": (
            f"Группа «{label}»: за зеленой рамкой найдено {analysis['violation_count']} элементов "
            f"на {analysis['frames_with_violations_count']} кадрах: {'; '.join(preview)}."
        ),
        **analysis,
    }


def evaluate_legal_disclaimer_safety(result_dir: str | Path) -> dict:
    return evaluate_frame_safety(result_dir, scope="legal_disclaimer")


def evaluate_logo_safety(result_dir: str | Path) -> dict:
    return evaluate_frame_safety(result_dir, scope="logos")


def evaluate_all_text_safety(result_dir: str | Path) -> dict:
    return evaluate_frame_safety(result_dir, scope="all_text")
