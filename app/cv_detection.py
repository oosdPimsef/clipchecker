# -*- coding: utf-8 -*-
"""Shared local YOLO/YOLOWorld object detection helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from black_bars import list_frame_files
except ImportError:
    from .black_bars import list_frame_files


_YOLO_MODEL_CACHE = {"path": None, "model": None, "error": None}
_YOLO_DETECTIONS_CACHE: dict[tuple[str, str, float, tuple[str, ...]], dict] = {}


def normalize_cv_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label or "")).strip().lower().replace("_", " ").replace("-", " ")


def unique_cv_labels(labels: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    output = []
    seen = set()
    for label in labels:
        normalized = normalize_cv_label(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(str(label))
    return output


def configured_cv_model_path() -> Path | None:
    value = (
        os.getenv("RESTRICTED_CONTENT_YOLO_MODEL")
        or os.getenv("CLIPCHECKER_YOLO_MODEL")
        or ""
    ).strip().strip('"')
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _load_yolo_model(model_path: Path):
    cached_path = _YOLO_MODEL_CACHE.get("path")
    if cached_path == str(model_path):
        return _YOLO_MODEL_CACHE.get("model"), _YOLO_MODEL_CACHE.get("error")

    try:
        cache_root = Path(os.getenv("CLIPCHECKER_CV_CACHE") or (model_path.parent / "cv_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache_root / "ultralytics"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        # CLIP uses expanduser("~/.cache/clip"); corporate profiles can block it.
        os.environ["HOME"] = str(cache_root)
        os.environ["USERPROFILE"] = str(cache_root)
        from ultralytics import YOLO, YOLOWorld

        model_class = YOLOWorld if "world" in model_path.name.lower() else YOLO
        model = model_class(str(model_path))
        _YOLO_MODEL_CACHE.update({"path": str(model_path), "model": model, "error": None})
        return model, None
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        _YOLO_MODEL_CACHE.update({"path": str(model_path), "model": None, "error": error})
        return None, error


def _frame_second_label(frame_name: str) -> str:
    match = re.search(r"(\d+)(?=\.[^.]+$)", frame_name)
    if not match:
        return frame_name
    return f"{int(match.group(1))}сек."


def _extract_bbox(box) -> list[int] | None:
    try:
        values = box.xyxy[0].tolist()
        if len(values) < 4:
            return None
        return [int(round(float(value))) for value in values[:4]]
    except Exception:
        return None


def _extract_yolo_result_detections(result, frame_name: str, confidence_threshold: float) -> list[dict]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections = []
    for box in boxes:
        try:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
        except Exception:
            continue
        if confidence < confidence_threshold:
            continue

        label = str(names.get(cls_id, cls_id))
        item = {
            "label": normalize_cv_label(label),
            "raw_label": label,
            "confidence": round(confidence, 3),
            "frame": frame_name,
            "second": _frame_second_label(frame_name),
        }
        bbox = _extract_bbox(box)
        if bbox:
            item["bbox"] = bbox
        detections.append(item)
    return detections


def detect_cv_objects_in_frames(
    frames_dir: str | Path,
    *,
    labels: list[str] | tuple[str, ...] | set[str] | None = None,
    model_path: str | Path | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    if confidence_threshold is None:
        try:
            confidence_threshold = float(os.getenv("RESTRICTED_CONTENT_CV_CONFIDENCE", "0.55"))
        except ValueError:
            confidence_threshold = 0.55

    model_file = Path(model_path) if model_path else configured_cv_model_path()
    if model_file is None:
        return {
            "enabled": False,
            "model_path": "",
            "error": "Локальная YOLO-модель не задана. Укажите путь в CLIPCHECKER_YOLO_MODEL или RESTRICTED_CONTENT_YOLO_MODEL.",
            "detections": [],
        }

    labels_key = tuple(normalize_cv_label(label) for label in unique_cv_labels(labels or ()))
    cache_key = (str(Path(frames_dir).resolve()), str(model_file.resolve()), float(confidence_threshold), labels_key)
    if cache_key in _YOLO_DETECTIONS_CACHE:
        return _YOLO_DETECTIONS_CACHE[cache_key]

    model, error = _load_yolo_model(model_file)
    if model is None:
        return {
            "enabled": False,
            "model_path": str(model_file),
            "error": error or "YOLO-модель не загрузилась.",
            "detections": [],
        }

    labels_list = unique_cv_labels(labels or ())
    try:
        if labels_list and hasattr(model, "set_classes"):
            model.set_classes(labels_list)
    except Exception as exc:
        return {
            "enabled": False,
            "model_path": str(model_file),
            "error": f"{exc.__class__.__name__}: {exc}",
            "detections": [],
        }

    detections = []
    for frame in list_frame_files(frames_dir):
        try:
            results = model.predict(str(frame), verbose=False, conf=confidence_threshold)
        except Exception as exc:
            return {
                "enabled": False,
                "model_path": str(model_file),
                "error": f"{exc.__class__.__name__}: {exc}",
                "detections": detections,
            }
        for result in results:
            detections.extend(_extract_yolo_result_detections(result, frame.name, confidence_threshold))

    output = {
        "enabled": True,
        "model_path": str(model_file),
        "error": "",
        "detections": detections,
    }
    _YOLO_DETECTIONS_CACHE[cache_key] = output
    return output


def filter_cv_detections(detections: list[dict], target_labels: list[str] | tuple[str, ...] | set[str]) -> list[dict]:
    normalized_targets = {normalize_cv_label(label) for label in target_labels}
    return [
        item
        for item in detections
        if normalize_cv_label(item.get("label") or item.get("raw_label", "")) in normalized_targets
    ]
