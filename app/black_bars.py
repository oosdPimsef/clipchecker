# -*- coding: utf-8 -*-
"""Detect black vertical bars on video frames."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_source_frames_dir(result_dir: str | Path) -> Path | None:
    base = Path(result_dir)
    for folder_name in ("frames_pdf_original", "frames"):
        candidate = base / folder_name
        if candidate.is_dir():
            return candidate
    return None


def list_frame_files(frames_dir: str | Path) -> list[Path]:
    return sorted(
        path
        for path in Path(frames_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def frame_second_label(frame_name: str) -> str:
    match = re.search(r"(\d+)(?=\.[^.]+$)", frame_name)
    if not match:
        return frame_name
    return f"{int(match.group(1))}сек."


def _dark_share(gray: np.ndarray, threshold: int) -> float:
    if gray.size == 0:
        return 0.0
    return float((gray <= threshold).mean())


def _edge_bar_width(gray: np.ndarray, side: str, *, threshold: int, min_dark_share: float, max_scan_ratio: float) -> int:
    height, width = gray.shape
    max_scan = max(1, int(width * max_scan_ratio))
    bar_width = 0

    for offset in range(max_scan):
        column = gray[:, offset] if side == "left" else gray[:, width - offset - 1]
        if _dark_share(column, threshold) >= min_dark_share:
            bar_width += 1
        else:
            break

    return bar_width


def analyze_frame_black_bars(
    image_path: str | Path,
    *,
    threshold: int = 16,
    min_dark_share: float = 0.98,
    min_width_ratio: float = 0.02,
    max_scan_ratio: float = 0.20,
) -> dict:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"))

    height, width = gray.shape
    left_width = _edge_bar_width(gray, "left", threshold=threshold, min_dark_share=min_dark_share, max_scan_ratio=max_scan_ratio)
    right_width = _edge_bar_width(gray, "right", threshold=threshold, min_dark_share=min_dark_share, max_scan_ratio=max_scan_ratio)
    min_width_px = max(2, int(width * min_width_ratio))

    left_detected = left_width >= min_width_px
    right_detected = right_width >= min_width_px
    return {
        "frame": Path(image_path).name,
        "width": width,
        "height": height,
        "left_width_px": left_width,
        "right_width_px": right_width,
        "left_width_percent": round(100 * left_width / width, 2),
        "right_width_percent": round(100 * right_width / width, 2),
        "left_detected": left_detected,
        "right_detected": right_detected,
        "detected": left_detected or right_detected,
    }


def analyze_black_bars(frames_dir: str | Path) -> dict:
    frames = list_frame_files(frames_dir)
    violations = []
    checked = 0

    for frame in frames:
        try:
            result = analyze_frame_black_bars(frame)
        except Exception:
            continue
        checked += 1
        if result["detected"]:
            result["second"] = frame_second_label(result["frame"])
            violations.append(result)

    return {
        "checked_frames": checked,
        "violation_count": len(violations),
        "violations": violations,
    }


def evaluate_black_side_bars(result_dir: str | Path) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка черных полос будет выполнена после извлечения кадров.",
        }

    analysis = analyze_black_bars(frames_dir)
    if analysis["checked_frames"] == 0:
        return {
            "status": "pending",
            "message": "Кадры для проверки черных полос не найдены.",
            **analysis,
        }

    if analysis["violation_count"] == 0:
        return {
            "status": "pass",
            "message": f"Черные полосы по бокам кадров не найдены. Проверено кадров: {analysis['checked_frames']}.",
            **analysis,
        }

    preview = []
    for violation in analysis["violations"][:5]:
        sides = []
        if violation["left_detected"]:
            sides.append(f"слева {violation['left_width_percent']}%")
        if violation["right_detected"]:
            sides.append(f"справа {violation['right_width_percent']}%")
        preview.append(f"{violation['second']} ({', '.join(sides)})")

    return {
        "status": "fail",
        "message": (
            f"Найдены черные полосы по бокам на {analysis['violation_count']} кадрах: "
            f"{'; '.join(preview)}."
        ),
        **analysis,
    }
