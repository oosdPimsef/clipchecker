# -*- coding: utf-8 -*-
"""Detect regulated or prohibited content mentions in OCR frame text."""

from __future__ import annotations

import html
import re
from pathlib import Path

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


RESTRICTED_CONTENT_CHECKS = {
    "249": {
        "label": "алкоголь",
        "no_message": "Признаки алкоголя в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к алкоголю",
        "terms": [
            "алкоголь",
            "алкогольный",
            "спиртное",
            "спирт",
            "этанол",
            "водка",
            "пиво",
            "пива",
            "пивной",
            "beer",
            "lager",
            "ale",
            "сидр",
            "cider",
            "вино",
            "вина",
            "винный",
            "винная",
            "wine",
            "шампанское",
            "champagne",
            "игристое",
            "просекко",
            "prosecco",
            "коньяк",
            "бренди",
            "brandy",
            "виски",
            "whisky",
            "whiskey",
            "ром",
            "rum",
            "джин",
            "gin",
            "текила",
            "tequila",
            "ликер",
            "liqueur",
            "коктейль",
            "cocktail",
            "бар",
            "паб",
            "pub",
            "рюмка",
            "бокал",
            "фужер",
            "стакан виски",
            "бутылка",
            "винная бутылка",
            "пивная бутылка",
            "жестяная банка",
            "алюминиевая банка",
            "банка пива",
            "опьянение",
            "пьяный",
            "трезвость",
            "безалкогольное пиво",
        ],
    },
    "269": {
        "label": "курение",
        "no_message": "Признаки курения и табака в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к курению или табаку",
        "terms": [
            "курение",
            "курить",
            "курит",
            "сигарета",
            "сигареты",
            "cigarette",
            "cigarettes",
            "табак",
            "табачный",
            "tobacco",
            "никотин",
            "nicotine",
            "папироса",
            "папиросы",
            "сигара",
            "cigar",
            "сигарилла",
            "вейп",
            "vape",
            "вейпинг",
            "электронная сигарета",
            "одноразка",
            "испаритель",
            "айкос",
            "iqos",
            "glo",
            "стик",
            "стики",
            "кальян",
            "hookah",
            "shisha",
            "дым",
            "дымит",
            "задымление",
            "затяжка",
            "пепел",
            "пепельница",
            "зажигалка",
            "спички",
            "мундштук",
            "курительная трубка",
            "смола",
        ],
    },
    "270": {
        "label": "запрещённые вещества",
        "no_message": "Признаки запрещённых веществ в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к запрещённым веществам",
        "terms": [
            "наркотик",
            "наркотики",
            "наркотический",
            "drug",
            "drugs",
            "substance",
            "психотроп",
            "психоактивный",
            "запрещенное вещество",
            "запрещённое вещество",
            "каннабис",
            "cannabis",
            "марихуана",
            "marijuana",
            "конопля",
            "гашиш",
            "hashish",
            "героин",
            "heroin",
            "кокаин",
            "cocaine",
            "амфетамин",
            "amphetamine",
            "метамфетамин",
            "экстази",
            "ecstasy",
            "mdma",
            "lsd",
            "лсд",
            "опиум",
            "opium",
            "морфин",
            "morphine",
            "спайс",
            "spice",
            "соль",
            "закладка",
            "доза",
            "дилер",
            "кайф",
            "трип",
            "шприц",
            "игла",
            "таблетка экстази",
            "порошок",
        ],
    },
    "271": {
        "label": "оружие и насилие",
        "no_message": "Признаки оружия и насилия в видеоряде не обнаружены.",
        "found_message": "Обнаружены возможные отсылки к оружию или насилию",
        "terms": [
            "оружие",
            "weapon",
            "пистолет",
            "gun",
            "револьвер",
            "автомат",
            "автомат калашникова",
            "ak-47",
            "винтовка",
            "rifle",
            "ружье",
            "ружьё",
            "дробовик",
            "shotgun",
            "пулемет",
            "пулемёт",
            "карабин",
            "ствол",
            "патрон",
            "патроны",
            "bullet",
            "пуля",
            "обойма",
            "магазин патронов",
            "снаряд",
            "снаряды",
            "граната",
            "grenade",
            "бомба",
            "bomb",
            "взрыв",
            "explosion",
            "нож",
            "knife",
            "кинжал",
            "мачете",
            "топор",
            "дубинка",
            "кастет",
            "стрельба",
            "выстрел",
            "прицел",
            "убить",
            "убийство",
            "насилие",
            "violent",
            "кровь",
            "кровавый",
            "драка",
            "удар",
            "нападение",
            "угроза",
            "война",
            "боеприпасы",
        ],
    },
}
RESTRICTED_CV_LABELS = {
    "249": {
        "bottle",
        "wine glass",
        "beer bottle",
        "wine bottle",
        "liquor bottle",
        "can",
        "beer can",
        "cup",
        "glass",
        "cocktail glass",
    },
    "269": {
        "cigarette",
        "cigar",
        "tobacco",
        "smoking",
        "smoke",
        "hookah",
        "shisha",
        "vape",
        "e-cigarette",
        "lighter",
        "ashtray",
    },
    "270": {
        "drug",
        "drugs",
        "pill",
        "pills",
        "tablet",
        "powder",
        "syringe",
        "needle",
        "cannabis",
        "marijuana",
    },
    "271": {
        "knife",
        "gun",
        "pistol",
        "rifle",
        "weapon",
        "bullet",
        "ammunition",
        "grenade",
        "bomb",
        "sword",
        "machete",
    },
}
def _all_cv_labels() -> list[str]:
    labels = []
    seen = set()
    for group in RESTRICTED_CV_LABELS.values():
        for label in group:
            normalized = label.lower().replace("_", " ").replace("-", " ")
            if normalized in seen:
                continue
            seen.add(normalized)
            labels.append(label)
    return labels


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term.lower().replace("ё", "е"))
    escaped = escaped.replace(r"\ ", r"\s+")
    if re.search(r"[a-zа-я0-9]$", term.lower(), flags=re.IGNORECASE):
        escaped = escaped + r"[а-яa-z]*"
    return re.compile(rf"(?<![a-zа-я0-9]){escaped}(?![a-zа-я0-9])", flags=re.IGNORECASE)


RESTRICTED_PATTERNS = {
    check_id: [(term, _term_pattern(term)) for term in config["terms"]]
    for check_id, config in RESTRICTED_CONTENT_CHECKS.items()
}


def extract_restricted_terms(text: str, check_id: str) -> list[str]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    found = []
    seen = set()
    for term, pattern in RESTRICTED_PATTERNS[check_id]:
        if not pattern.search(normalized):
            continue
        key = term.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        found.append(term)
    return found


def analyze_restricted_content_from_ocr(ocr_data: dict, check_id: str) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        for term in extract_restricted_terms(text, check_id):
            total += 1
            key = term.lower().replace("ё", "е")
            if key not in grouped:
                grouped[key] = {
                    "term": term,
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 3:
                grouped[key]["examples"].append(text)

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))

    return {
        "restricted_content_count": total,
        "restricted_mentions": list(grouped.values()),
        "dictionary_size": len(RESTRICTED_CONTENT_CHECKS[check_id]["terms"]),
        "check_label": RESTRICTED_CONTENT_CHECKS[check_id]["label"],
    }


def analyze_restricted_content_from_text(text: str, check_id: str) -> dict:
    grouped = []
    for term in extract_restricted_terms(text, check_id):
        grouped.append({"term": term, "seconds": [], "frames": [], "examples": []})
    return {
        "restricted_content_count": len(grouped),
        "restricted_mentions": grouped,
        "dictionary_size": len(RESTRICTED_CONTENT_CHECKS[check_id]["terms"]),
        "check_label": RESTRICTED_CONTENT_CHECKS[check_id]["label"],
    }


def analyze_restricted_content_from_cv(result_dir: str | Path, check_id: str) -> dict:
    frames_dir = find_source_frames_dir(result_dir)
    if frames_dir is None:
        return {
            "cv_enabled": False,
            "cv_model_path": "",
            "cv_error": "Кадры для CV-проверки не найдены.",
            "cv_detections": [],
            "cv_restricted_mentions": [],
        }

    cv_result = detect_cv_objects_in_frames(frames_dir, labels=_all_cv_labels())
    filtered = filter_cv_detections(cv_result["detections"], RESTRICTED_CV_LABELS[check_id])
    return {
        "cv_enabled": cv_result["enabled"],
        "cv_model_path": cv_result["model_path"],
        "cv_error": cv_result["error"],
        "cv_detections": cv_result["detections"],
        "cv_restricted_mentions": filtered,
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


def evaluate_restricted_content(result_dir: str | Path, check_id: str) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    all_text_path = base / "All_Frames_Text.txt"
    has_materials = ocr_path.is_file() or all_text_path.is_file() or (base / "frames").is_dir() or (base / "frames_pdf_original").is_dir()

    if ocr_path.is_file():
        analysis = analyze_restricted_content_from_ocr(load_ocr_log(ocr_path), check_id)
    else:
        analysis = analyze_restricted_content_from_text(read_text_file(all_text_path), check_id)
    cv_analysis = analyze_restricted_content_from_cv(base, check_id)
    analysis = {**analysis, **cv_analysis}

    config = RESTRICTED_CONTENT_CHECKS[check_id]
    mentions = analysis["restricted_mentions"]
    cv_mentions = analysis["cv_restricted_mentions"]
    if not mentions and not cv_mentions:
        if not has_materials:
            return {
                "status": "pending",
                "message": f"Проверка на категорию '{config['label']}' будет выполнена после извлечения кадров и OCR.",
                **analysis,
            }
        return {
            "status": "pass",
            "message": config["no_message"],
            **analysis,
        }

    plain = "; ".join(
        f"{item['term']} ({', '.join(item.get('seconds', [])[:8])})" if item.get("seconds") else item["term"]
        for item in mentions
    )
    cv_plain = "; ".join(
        f"{item.get('raw_label') or item.get('label')} ({item.get('second')}; {item.get('confidence')})"
        for item in cv_mentions
    )
    parts = []
    html_parts = []
    if plain:
        parts.append(f"текст: {plain}")
        html_parts.append(f"текст: {_format_mentions_html(mentions)}")
    if cv_plain:
        parts.append(f"CV: {cv_plain}")
        html_parts.append(f"CV: {_format_cv_mentions_html(cv_mentions)}")

    message = f"{config['found_message']}: {'; '.join(parts)}."
    message_html = f"{html.escape(config['found_message'])}: {'; '.join(html_parts)}."
    return {
        "status": "fail",
        "message": message,
        "message_html": message_html,
        **analysis,
    }


def evaluate_alcohol_references(result_dir: str | Path) -> dict:
    return evaluate_restricted_content(result_dir, "249")


def evaluate_smoking_references(result_dir: str | Path) -> dict:
    return evaluate_restricted_content(result_dir, "269")


def evaluate_drug_references(result_dir: str | Path) -> dict:
    return evaluate_restricted_content(result_dir, "270")


def evaluate_weapon_violence_references(result_dir: str | Path) -> dict:
    return evaluate_restricted_content(result_dir, "271")
