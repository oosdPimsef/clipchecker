# -*- coding: utf-8 -*-
"""Python evaluators for approval checklist items keyed by Excel item id."""

from __future__ import annotations

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


EVALUATORS = {
    "1": evaluate_duration_multiple_of_five,
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

