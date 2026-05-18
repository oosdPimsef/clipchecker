# -*- coding: utf-8 -*-
"""Detect repeated and visually blank frames in preprocessed video frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from black_bars import find_source_frames_dir, frame_second_label, list_frame_files
except ImportError:
    from .black_bars import find_source_frames_dir, frame_second_label, list_frame_files


PREVIEW_LIMIT = 12


def _load_small_rgb(image_path: str | Path, *, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.int16)


def frame_similarity(first_path: str | Path, second_path: str | Path, *, pixel_tolerance: int = 10) -> float:
    """Return share of nearly identical pixels after normalizing frame size."""

    first = _load_small_rgb(first_path)
    second = _load_small_rgb(second_path)
    diff = np.abs(first - second).mean(axis=2)
    return float((diff <= pixel_tolerance).mean())


def analyze_repeated_frames(frames_dir: str | Path, *, similarity_threshold: float = 0.90) -> dict:
    frames = list_frame_files(frames_dir)
    repeats = []

    for previous, current in zip(frames, frames[1:]):
        try:
            similarity = frame_similarity(previous, current)
        except Exception:
            continue

        if similarity >= similarity_threshold:
            repeats.append(
                {
                    "previous_frame": previous.name,
                    "current_frame": current.name,
                    "previous_second": frame_second_label(previous.name),
                    "current_second": frame_second_label(current.name),
                    "similarity_percent": round(similarity * 100, 1),
                }
            )

    return {
        "checked_pairs": max(0, len(frames) - 1),
        "repeat_count": len(repeats),
        "repeats": repeats,
        "similarity_threshold_percent": round(similarity_threshold * 100, 1),
    }


def blank_frame_score(image_path: str | Path) -> dict:
    """Return color uniformity metrics for a frame.

    Mean absolute deviation catches flat frames of any color, not only black.
    """

    pixels = _load_small_rgb(image_path)
    mean_color = pixels.reshape(-1, 3).mean(axis=0)
    deviations = np.abs(pixels - mean_color).mean(axis=2)
    mean_deviation = float(deviations.mean())
    near_mean_share = float((deviations <= 5).mean())
    return {
        "mean_deviation": round(mean_deviation, 3),
        "near_mean_share": round(near_mean_share, 4),
        "mean_color": [int(round(value)) for value in mean_color],
    }


def is_blank_frame(
    image_path: str | Path,
    *,
    max_mean_deviation: float = 3.0,
    min_near_mean_share: float = 0.98,
) -> bool:
    score = blank_frame_score(image_path)
    return score["mean_deviation"] <= max_mean_deviation and score["near_mean_share"] >= min_near_mean_share


def analyze_blank_frames(frames_dir: str | Path) -> dict:
    frames = list_frame_files(frames_dir)
    blank_frames = []
    checked = 0

    for frame in frames:
        try:
            score = blank_frame_score(frame)
        except Exception:
            continue

        checked += 1
        if score["mean_deviation"] <= 3.0 and score["near_mean_share"] >= 0.98:
            blank_frames.append(
                {
                    "frame": frame.name,
                    "second": frame_second_label(frame.name),
                    **score,
                }
            )

    return {
        "checked_frames": checked,
        "blank_count": len(blank_frames),
        "blank_frames": blank_frames,
    }


def evaluate_repeated_frames(result_dir: str | Path) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка стоп-кадров будет выполнена после извлечения кадров.",
        }

    analysis = analyze_repeated_frames(frames_dir)
    if analysis["checked_pairs"] == 0:
        return {
            "status": "pending",
            "message": "Недостаточно кадров для проверки стоп-кадров.",
            **analysis,
        }

    if analysis["repeat_count"] == 0:
        return {
            "status": "pass",
            "message": f"Стоп-кадры и повторяющиеся кадры не обнаружены. Проверено пар кадров: {analysis['checked_pairs']}.",
            **analysis,
        }

    preview = [
        f"{item['previous_second']} и {item['current_second']} ({item['similarity_percent']}%)"
        for item in analysis["repeats"][:PREVIEW_LIMIT]
    ]
    more = ""
    if analysis["repeat_count"] > PREVIEW_LIMIT:
        more = f" Еще {analysis['repeat_count'] - PREVIEW_LIMIT} повторов не показано."

    return {
        "status": "fail",
        "message": f"Найдены повторяющиеся кадры: {'; '.join(preview)}.{more}",
        **analysis,
    }


def evaluate_blank_frames(result_dir: str | Path) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "status": "pending",
            "message": "Проверка пустых кадров будет выполнена после извлечения кадров.",
        }

    analysis = analyze_blank_frames(frames_dir)
    if analysis["checked_frames"] == 0:
        return {
            "status": "pending",
            "message": "Кадры для проверки пустых кадров не найдены.",
            **analysis,
        }

    if analysis["blank_count"] == 0:
        return {
            "status": "pass",
            "message": f"Пустые кадры не обнаружены. Проверено кадров: {analysis['checked_frames']}.",
            **analysis,
        }

    preview = [
        f"{item['second']} (средний цвет RGB {item['mean_color']})"
        for item in analysis["blank_frames"][:PREVIEW_LIMIT]
    ]
    more = ""
    if analysis["blank_count"] > PREVIEW_LIMIT:
        more = f" Еще {analysis['blank_count'] - PREVIEW_LIMIT} пустых кадров не показано."

    return {
        "status": "fail",
        "message": f"Найдены пустые однотонные кадры: {'; '.join(preview)}.{more}",
        **analysis,
    }
