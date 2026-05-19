# -*- coding: utf-8 -*-
"""OCR and CV checks for visual legal-risk references in video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

try:
    from black_bars import find_source_frames_dir
    from cv_detection import detect_cv_objects_in_frames, filter_cv_detections
    from frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
    from price_checks import read_text_file
except ImportError:
    from .black_bars import find_source_frames_dir
    from .cv_detection import detect_cv_objects_in_frames, filter_cv_detections
    from .frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
    from .price_checks import read_text_file


VISUAL_REFERENCE_CHECKS = {
    "23": {
        "label": "образ врача",
        "no_message": "Отсылки к образу врача в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к образу врача",
        "terms": [
            "врач",
            "доктор",
            "медик",
            "медицинский работник",
            "фармацевт",
            "провизор",
            "терапевт",
            "педиатр",
            "кардиолог",
            "хирург",
            "стоматолог",
            "дерматолог",
            "белый халат",
            "белом халате",
            "медицинский халат",
            "стетоскоп",
            "фонендоскоп",
            "шприц",
            "капельница",
            "скальпель",
            "медицинская маска",
            "медицинские перчатки",
            "клиника",
            "больница",
            "госпиталь",
            "палата",
            "операционная",
            "регистратура",
            "медкабинет",
            "лаборатория",
            "рецепт врача",
            "медицинская карта",
            "крест на халате",
        ],
        "cv_labels": [
            "doctor",
            "physician",
            "medical doctor",
            "nurse",
            "pharmacist",
            "dentist",
            "surgeon",
            "white coat",
            "lab coat",
            "stethoscope",
            "syringe",
            "medical syringe",
            "hospital bed",
            "hospital room",
            "operating room",
            "medical mask",
            "medical gloves",
            "ambulance",
            "wheelchair",
            "crutch",
            "medical cross",
        ],
    },
    "250": {
        "label": "заболевания",
        "no_message": "Отсылки к заболеваниям в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к заболеванию",
        "terms": [
            "болезнь",
            "заболевание",
            "недомогание",
            "симптом",
            "диагноз",
            "кашель",
            "кашляет",
            "насморк",
            "сопли",
            "простуда",
            "грипп",
            "орви",
            "температура",
            "жар",
            "лихорадка",
            "больное горло",
            "горло болит",
            "осиплость",
            "головная боль",
            "мигрень",
            "боль в животе",
            "тошнота",
            "аллергия",
            "сыпь",
            "воспаление",
            "инфекция",
            "вирус",
            "бактерия",
            "иммунитет",
            "пациент",
            "больной человек",
            "градусник",
            "термометр",
            "таблетки от",
            "постельный режим",
            "шарф на горле",
        ],
        "cv_labels": [
            "sick person",
            "coughing person",
            "person coughing",
            "person with sore throat",
            "thermometer",
            "medical thermometer",
            "fever",
            "tissue",
            "napkin",
            "scarf",
            "pill",
            "pills",
            "medicine",
            "capsule",
            "hospital bed",
            "patient",
            "runny nose",
            "red throat",
            "rash",
            "medical mask",
            "blanket",
        ],
    },
    "24": {
        "label": "нарушение ПДД",
        "no_message": "Отсылки к нарушению правил дорожного движения не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к нарушению правил дорожного движения",
        "terms": [
            "превышение скорости",
            "большая скорость",
            "высокая скорость",
            "гонка",
            "гонки",
            "стритрейсинг",
            "дрифт",
            "резкое торможение",
            "обгон",
            "обгон справа",
            "двойная сплошная",
            "на красный",
            "красный свет",
            "проезд на красный",
            "без ремня",
            "не пристегнут",
            "непристегнут",
            "телефон за рулем",
            "смартфон за рулем",
            "алкоголь за рулем",
            "пьяный водитель",
            "авария",
            "дтп",
            "занос",
            "встречная полоса",
            "выезд на встречку",
            "пешеходный переход",
            "не пропустил пешехода",
            "скорость 120",
            "скорость 140",
            "спидометр",
        ],
        "cv_labels": [
            "speedometer",
            "car speedometer",
            "racing car",
            "car racing",
            "drifting car",
            "car drift",
            "traffic light red",
            "red traffic light",
            "car accident",
            "traffic accident",
            "crashed car",
            "seat belt",
            "driver using phone",
            "mobile phone while driving",
            "overtaking car",
            "pedestrian crossing",
            "road sign",
            "police car",
            "broken car",
            "skid marks",
        ],
    },
}

LICENSE_PLATE_TERMS = [
    "номер автомобиля",
    "номерной знак",
    "государственный номер",
    "госномер",
    "автомобильный номер",
    "регистрационный знак",
    "license plate",
    "number plate",
]
LICENSE_PLATE_BLUR_TERMS = [
    "заблюрен",
    "заблюрено",
    "размыт",
    "размыто",
    "замазан",
    "замазано",
    "скрыт",
    "скрыто",
    "закрыт",
    "закрыто",
    "не читается",
]
LICENSE_PLATE_CV_LABELS = [
    "license plate",
    "number plate",
    "vehicle registration plate",
    "car license plate",
    "car number plate",
]


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term.lower().replace("ё", "е"))
    escaped = escaped.replace(r"\ ", r"\s+")
    if re.search(r"[a-zа-я0-9]$", term.lower(), flags=re.IGNORECASE):
        escaped += r"[а-яa-z]*"
    return re.compile(rf"(?<![a-zа-я0-9]){escaped}(?![a-zа-я0-9])", flags=re.IGNORECASE)


REFERENCE_PATTERNS = {
    check_id: [(term, _term_pattern(term)) for term in config["terms"]]
    for check_id, config in VISUAL_REFERENCE_CHECKS.items()
}
LICENSE_PLATE_PATTERNS = [(term, _term_pattern(term)) for term in LICENSE_PLATE_TERMS]
LICENSE_PLATE_BLUR_PATTERNS = [(term, _term_pattern(term)) for term in LICENSE_PLATE_BLUR_TERMS]
RUSSIAN_PLATE_RE = re.compile(
    r"(?<![a-zа-я0-9])(?:[авекмнорстухabekmhopctyx]\s*\d\s*\d\s*\d\s*[авекмнорстухabekmhopctyx]\s*[авекмнорстухabekmhopctyx]\s*\d{2,3})(?![a-zа-я0-9])",
    flags=re.IGNORECASE,
)


def _extract_terms(text: str, check_id: str) -> list[str]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    found = []
    seen = set()
    for term, pattern in REFERENCE_PATTERNS[check_id]:
        if pattern.search(normalized):
            key = term.lower().replace("ё", "е")
            if key not in seen:
                seen.add(key)
                found.append(term)
    return found


def _find_pattern_terms(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    found = []
    seen = set()
    for term, pattern in patterns:
        if pattern.search(normalized):
            key = term.lower().replace("ё", "е")
            if key not in seen:
                seen.add(key)
                found.append(term)
    return found


def _read_ocr_or_text(result_dir: Path) -> tuple[dict | None, str, bool]:
    ocr_path = result_dir / "OCR_Log.json"
    all_text_path = result_dir / "All_Frames_Text.txt"
    has_materials = (
        ocr_path.is_file()
        or all_text_path.is_file()
        or (result_dir / "frames").is_dir()
        or (result_dir / "frames_pdf_original").is_dir()
    )
    if ocr_path.is_file():
        return load_ocr_log(ocr_path), "", has_materials
    return None, read_text_file(all_text_path), has_materials


def _group_ocr_mentions(ocr_data: dict, check_id: str) -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        for term in _extract_terms(text, check_id):
            key = term.lower().replace("ё", "е")
            if key not in grouped:
                grouped[key] = {"term": term, "seconds": [], "frames": [], "examples": []}
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 3:
                grouped[key]["examples"].append(text)
    return _sorted_mentions(list(grouped.values()))


def _group_text_mentions(text: str, check_id: str) -> list[dict]:
    return [{"term": term, "seconds": [], "frames": [], "examples": []} for term in _extract_terms(text, check_id)]


def _sorted_mentions(mentions: list[dict]) -> list[dict]:
    for item in mentions:
        item["seconds"] = sorted(
            set(item.get("seconds", [])),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item.get("frames", [])))
    return mentions


def _all_reference_cv_labels() -> list[str]:
    labels = []
    seen = set()
    for config in VISUAL_REFERENCE_CHECKS.values():
        for label in config["cv_labels"]:
            key = label.lower().replace("_", " ").replace("-", " ")
            if key not in seen:
                seen.add(key)
                labels.append(label)
    return labels


def analyze_visual_reference_from_cv(result_dir: str | Path, check_id: str) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "cv_enabled": False,
            "cv_model_path": "",
            "cv_error": "Кадры для CV-проверки не найдены.",
            "cv_detections": [],
            "cv_mentions": [],
        }
    cv_result = detect_cv_objects_in_frames(frames_dir, labels=_all_reference_cv_labels())
    filtered = filter_cv_detections(cv_result["detections"], VISUAL_REFERENCE_CHECKS[check_id]["cv_labels"])
    return {
        "cv_enabled": cv_result["enabled"],
        "cv_model_path": cv_result["model_path"],
        "cv_error": cv_result["error"],
        "cv_detections": cv_result["detections"],
        "cv_mentions": filtered,
    }


def _format_mentions_html(mentions: list[dict]) -> str:
    parts = []
    for item in mentions:
        seconds = ", ".join(item.get("seconds", [])[:8])
        suffix = f" ({html.escape(seconds)})" if seconds else ""
        parts.append(f'<strong style="color:#dc2626;font-weight:800">{html.escape(item["term"])}</strong>{suffix}')
    return "; ".join(parts)


def _format_cv_mentions_html(mentions: list[dict]) -> str:
    parts = []
    for item in mentions:
        label = item.get("raw_label") or item.get("label", "")
        second = item.get("second", "")
        confidence = item.get("confidence")
        suffix = f" ({html.escape(second)}; {confidence})" if second else ""
        parts.append(f'<strong style="color:#dc2626;font-weight:800">{html.escape(str(label))}</strong>{suffix}')
    return "; ".join(parts)


def evaluate_visual_reference(result_dir: str | Path, check_id: str) -> dict:
    base = Path(result_dir)
    ocr_data, text, has_materials = _read_ocr_or_text(base)
    mentions = _group_ocr_mentions(ocr_data, check_id) if ocr_data is not None else _group_text_mentions(text, check_id)
    cv_analysis = analyze_visual_reference_from_cv(base, check_id)
    cv_mentions = cv_analysis["cv_mentions"]
    config = VISUAL_REFERENCE_CHECKS[check_id]

    if not mentions and not cv_mentions:
        if not has_materials:
            return {
                "status": "pending",
                "message": f"Проверка на категорию '{config['label']}' будет выполнена после извлечения кадров и OCR.",
                "reference_mentions": [],
                "dictionary_size": len(config["terms"]),
                **cv_analysis,
            }
        return {
            "status": "pass",
            "message": config["no_message"],
            "reference_mentions": [],
            "dictionary_size": len(config["terms"]),
            **cv_analysis,
        }

    plain_parts = []
    html_parts = []
    if mentions:
        plain_parts.append(
            "текст: "
            + "; ".join(
                f"{item['term']} ({', '.join(item.get('seconds', [])[:8])})" if item.get("seconds") else item["term"]
                for item in mentions
            )
        )
        html_parts.append(f"текст: {_format_mentions_html(mentions)}")
    if cv_mentions:
        plain_parts.append(
            "CV: "
            + "; ".join(
                f"{item.get('raw_label') or item.get('label')} ({item.get('second')}; {item.get('confidence')})"
                for item in cv_mentions
            )
        )
        html_parts.append(f"CV: {_format_cv_mentions_html(cv_mentions)}")

    return {
        "status": "fail",
        "message": f"{config['found_message']}: {'; '.join(plain_parts)}.",
        "message_html": f"{html.escape(config['found_message'])}: {'; '.join(html_parts)}.",
        "reference_mentions": mentions,
        "dictionary_size": len(config["terms"]),
        **cv_analysis,
    }


def evaluate_doctor_image_references(result_dir: str | Path) -> dict:
    return evaluate_visual_reference(result_dir, "23")


def evaluate_disease_references(result_dir: str | Path) -> dict:
    return evaluate_visual_reference(result_dir, "250")


def evaluate_traffic_violation_references(result_dir: str | Path) -> dict:
    return evaluate_visual_reference(result_dir, "24")


def _extract_license_plate_ocr_mentions(ocr_data: dict | None, text: str) -> dict:
    readable = []
    blurred = []
    term_mentions = []
    if ocr_data is not None:
        for item in iter_ocr_lines(ocr_data):
            line = normalize_text(item["text"])
            second = frame_second_label(item["frame"])
            for match in RUSSIAN_PLATE_RE.finditer(line.lower().replace("ё", "е")):
                readable.append({"term": match.group(0).upper().replace(" ", ""), "seconds": [second], "frames": [item["frame"]], "examples": [line]})
            if _find_pattern_terms(line, LICENSE_PLATE_PATTERNS):
                term_mentions.append({"term": "номерной знак", "seconds": [second], "frames": [item["frame"]], "examples": [line]})
            if _find_pattern_terms(line, LICENSE_PLATE_BLUR_PATTERNS) and _find_pattern_terms(line, LICENSE_PLATE_PATTERNS):
                blurred.append({"term": "номер автомобиля заблюрен", "seconds": [second], "frames": [item["frame"]], "examples": [line]})
    else:
        for match in RUSSIAN_PLATE_RE.finditer((text or "").lower().replace("ё", "е")):
            readable.append({"term": match.group(0).upper().replace(" ", ""), "seconds": [], "frames": [], "examples": []})
        if _find_pattern_terms(text, LICENSE_PLATE_PATTERNS):
            term_mentions.append({"term": "номерной знак", "seconds": [], "frames": [], "examples": []})
        if _find_pattern_terms(text, LICENSE_PLATE_BLUR_PATTERNS) and _find_pattern_terms(text, LICENSE_PLATE_PATTERNS):
            blurred.append({"term": "номер автомобиля заблюрен", "seconds": [], "frames": [], "examples": []})
    return {
        "readable_plate_mentions": _sorted_mentions(readable),
        "plate_term_mentions": _sorted_mentions(term_mentions),
        "blurred_plate_mentions": _sorted_mentions(blurred),
    }


def analyze_license_plate_from_cv(result_dir: str | Path) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "cv_enabled": False,
            "cv_model_path": "",
            "cv_error": "Кадры для CV-проверки не найдены.",
            "cv_detections": [],
            "cv_plate_mentions": [],
        }
    cv_result = detect_cv_objects_in_frames(frames_dir, labels=LICENSE_PLATE_CV_LABELS)
    filtered = filter_cv_detections(cv_result["detections"], LICENSE_PLATE_CV_LABELS)
    for item in filtered:
        blur_score = _license_plate_blur_score(frames_dir, item)
        item["blur_score"] = blur_score
        item["blurred"] = blur_score is not None and blur_score < 18.0
    return {
        "cv_enabled": cv_result["enabled"],
        "cv_model_path": cv_result["model_path"],
        "cv_error": cv_result["error"],
        "cv_detections": cv_result["detections"],
        "cv_plate_mentions": filtered,
    }


def _license_plate_blur_score(frames_dir: str | Path, detection: dict) -> float | None:
    bbox = detection.get("bbox")
    frame_name = detection.get("frame")
    if not bbox or len(bbox) < 4 or not frame_name:
        return None
    frame_path = Path(frames_dir) / str(frame_name)
    if not frame_path.is_file():
        return None
    try:
        with Image.open(frame_path) as image:
            x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
            width, height = image.size
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            crop = image.crop((x1, y1, x2, y2)).convert("L")
            edges = crop.filter(ImageFilter.FIND_EDGES)
            return round(float(ImageStat.Stat(edges).var[0]), 3)
    except Exception:
        return None


def evaluate_car_license_plate(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_data, text, has_materials = _read_ocr_or_text(base)
    ocr_analysis = _extract_license_plate_ocr_mentions(ocr_data, text)
    cv_analysis = analyze_license_plate_from_cv(base)

    readable = ocr_analysis["readable_plate_mentions"]
    blurred = ocr_analysis["blurred_plate_mentions"]
    plate_terms = ocr_analysis["plate_term_mentions"]
    cv_mentions = cv_analysis["cv_plate_mentions"]
    cv_blurred_mentions = [item for item in cv_mentions if item.get("blurred")]
    cv_readable_mentions = [item for item in cv_mentions if not item.get("blurred")]
    unblurred_plate_terms = plate_terms if not blurred else []

    if readable or cv_readable_mentions or unblurred_plate_terms:
        readable_html = _format_mentions_html(readable) if readable else ""
        cv_html = _format_cv_mentions_html(cv_readable_mentions) if cv_readable_mentions else ""
        term_html = _format_mentions_html(unblurred_plate_terms) if unblurred_plate_terms else ""
        details = "; ".join(part for part in [readable_html, f"CV: {cv_html}" if cv_html else "", term_html] if part)
        return {
            "status": "fail",
            "message": "Требуется согласие на использования номерных знаков от владельца",
            "message_html": (
                "Требуется согласие на использования номерных знаков от владельца. "
                f"Найден номер: {details}."
            ),
            **ocr_analysis,
            **cv_analysis,
        }

    if blurred or cv_blurred_mentions:
        details = []
        html_details = []
        if blurred:
            details.append("текст: номер автомобиля заблюрен")
            html_details.append(f"текст: {_format_mentions_html(blurred)}")
        if cv_blurred_mentions:
            details.append(
                "CV: "
                + "; ".join(
                    f"{item.get('raw_label') or item.get('label')} ({item.get('second')}; {item.get('confidence')}; blur={item.get('blur_score')})"
                    for item in cv_blurred_mentions
                )
            )
            html_details.append(f"CV: {_format_cv_mentions_html(cv_blurred_mentions)}")
        message = "Номер автомобиля есть и заблюрен"
        if details:
            message += f". {'; '.join(details)}."
        return {
            "status": "pass",
            "message": message,
            "message_html": f"Номер автомобиля есть и заблюрен. {'; '.join(html_details)}." if html_details else message,
            **ocr_analysis,
            **cv_analysis,
        }

    if not has_materials:
        return {
            "status": "pending",
            "message": "Проверка номера автомобиля будет выполнена после извлечения кадров и OCR.",
            **ocr_analysis,
            **cv_analysis,
        }

    return {
        "status": "pass",
        "message": "Номер автомобиля в видеоряде не обнаружен.",
        **ocr_analysis,
        **cv_analysis,
    }
