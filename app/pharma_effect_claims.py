# -*- coding: utf-8 -*-
"""Detect pharma positive-effect guarantee claims in OCR frame text."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .frame_safety import find_frames_dir, frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


NO_PHARMA_EFFECT_CLAIMS_MESSAGE = "Слова о гарантии положительного действия продукта не обнаружены."
NO_BAD_TREATMENT_CLAIMS_MESSAGE = "Слова и намёки на лечение в рекламе БАД не обнаружены."

PHARMA_EFFECT_TERMS = [
    ("лечить", r"леч(?:ит|ат|им|ишь|ите|у|ащ[а-яё]*|ебн[а-яё]*|ени[а-яё]*|ение|ением|ению|ений|ением|ен[а-яё]*)"),
    ("вылечить", r"вылеч[а-яё]*"),
    ("излечить", r"излеч[а-яё]*"),
    ("исцелить", r"исцел[а-яё]*"),
    ("спасти", r"спас(?:ает|ают|ем|ете|ешь|ти|ение|ением|ени[а-яё]*|ен[а-яё]*|ительн[а-яё]*)"),
    ("помочь", r"помо(?:жет|гут|гает|гают|г[а-яё]*|щи|щью)|помог[а-яё]*"),
    ("исчезнуть", r"исчез(?:нет|нут|ает|ают|новени[а-яё]*|нувш[а-яё]*)"),
    ("устранить", r"устран[а-яё]*"),
    ("убрать", r"убер[а-яё]*|убира[а-яё]*|убрать"),
    ("избавить", r"избав[а-яё]*"),
    ("снять симптом", r"снима[а-яё]*|снимет|снимут|снять|сняти[а-яё]*"),
    ("облегчить", r"облегч[а-яё]*"),
    ("предотвратить", r"предотврат[а-яё]*"),
    ("защитить", r"защит[а-яё]*"),
    ("восстановить", r"восстанов[а-яё]*"),
    ("улучшить", r"улучш[а-яё]*"),
    ("нормализовать", r"нормализ[а-яё]*"),
    ("снизить", r"сниж[а-яё]*|сниз[а-яё]*"),
    ("повысить", r"повыш[а-яё]*|повыс[а-яё]*"),
    ("стимулировать", r"стимулир[а-яё]*"),
    ("укрепить", r"укреп[а-яё]*"),
    ("регенерировать", r"регенерир[а-яё]*|регенераци[а-яё]*"),
    ("профилактика", r"профилактик[а-яё]*|профилактир[а-яё]*"),
    ("гарантировать", r"гарантир[а-яё]*|гарантированн[а-яё]*|гаранти[а-яё]*"),
    ("результат", r"результат[а-яё]*"),
    ("эффект", r"эффект[а-яё]*|эффективн[а-яё]*"),
    ("быстро действует", r"быстр[а-яё]*\s+(?:действ[а-яё]*|помог[а-яё]*|снима[а-яё]*)"),
    ("мгновенно", r"мгновенн[а-яё]*"),
    ("без боли", r"без\s+бол[а-яё]*|бол[а-яё]*\s+уйд[а-яё]*"),
    ("здоровье", r"здоров[а-яё]*"),
]

PHARMA_EFFECT_PATTERNS = [
    (term, re.compile(rf"(?<![a-zа-яё0-9]){pattern}(?![a-zа-яё0-9])", flags=re.IGNORECASE))
    for term, pattern in PHARMA_EFFECT_TERMS
]
BAD_TREATMENT_TERMS = PHARMA_EFFECT_TERMS + [
    ("лечение", r"лечени[а-яё]*|лечебн[а-яё]*|лечащ[а-яё]*"),
    ("болезнь", r"болезн[а-яё]*|заболевани[а-яё]*"),
    ("заболевание", r"заболева[а-яё]*|заболел[а-яё]*"),
    ("симптом", r"симптом[а-яё]*"),
    ("диагноз", r"диагноз[а-яё]*|диагностир[а-яё]*"),
    ("терапия", r"терапи[а-яё]*|терапевтическ[а-яё]*"),
    ("препарат", r"препарат[а-яё]*"),
    ("лекарство", r"лекарств[а-яё]*"),
    ("лекарственный", r"лекарственн[а-яё]*"),
    ("средство от", r"средств[а-яё]*\s+от\s+[а-яё]+"),
    ("против болезни", r"против\s+(?:бол[а-яё]*|простуд[а-яё]*|кашл[а-яё]*|воспалени[а-яё]*|инфекци[а-яё]*)"),
    ("боль", r"бол[а-яё]*|болев[а-яё]*"),
    ("воспаление", r"воспал[а-яё]*"),
    ("инфекция", r"инфекци[а-яё]*|антибактериальн[а-яё]*|противовирусн[а-яё]*"),
    ("простуда", r"простуд[а-яё]*|орви|грипп[а-яё]*"),
    ("кашель", r"кашл[а-яё]*|кашел[а-яё]*"),
    ("аллергия", r"аллерги[а-яё]*|аллерген[а-яё]*"),
    ("давление", r"давлени[а-яё]*|гипертони[а-яё]*"),
    ("сахар", r"сахар[а-яё]*|глюкоз[а-яё]*|диабет[а-яё]*"),
    ("холестерин", r"холестерин[а-яё]*"),
    ("сустав", r"сустав[а-яё]*|артрит[а-яё]*|артроз[а-яё]*"),
    ("сердце", r"сердц[а-яё]*|сердечн[а-яё]*"),
    ("печень", r"печен[а-яё]*|печеночн[а-яё]*"),
    ("желудок", r"желуд[а-яё]*|кишечник[а-яё]*|пищеварени[а-яё]*"),
    ("иммунитет", r"иммунитет[а-яё]*|иммунн[а-яё]*"),
    ("клинический", r"клиническ[а-яё]*"),
    ("доказано врачами", r"врач[а-яё]*\s+(?:доказ[а-яё]*|рекоменд[а-яё]*|подтвержд[а-яё]*)"),
    ("рекомендовано врачами", r"рекоменд[а-яё]*\s+врач[а-яё]*"),
    ("медицинский", r"медицинск[а-яё]*"),
    ("антиоксидантный эффект", r"антиоксидантн[а-яё]*\s+эффект[а-яё]*"),
]
BAD_TREATMENT_PATTERNS = [
    (term, re.compile(rf"(?<![a-zа-яё0-9]){pattern}(?![a-zа-яё0-9])", flags=re.IGNORECASE))
    for term, pattern in BAD_TREATMENT_TERMS
]


def _extract_terms(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    normalized = normalize_text(text or "").lower().replace("ё", "е")
    found = []
    seen = set()
    for term, pattern in patterns:
        if not pattern.search(normalized):
            continue
        key = term.lower().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        found.append(term)
    return found


def extract_pharma_effect_terms(text: str) -> list[str]:
    return _extract_terms(text, PHARMA_EFFECT_PATTERNS)


def extract_bad_treatment_terms(text: str) -> list[str]:
    return _extract_terms(text, BAD_TREATMENT_PATTERNS)


def _analyze_claims(ocr_data: dict, extractor, count_key: str, list_key: str, dictionary_size: int) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        text = normalize_text(item["text"])
        for term in extractor(text):
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
        count_key: total,
        list_key: list(grouped.values()),
        "dictionary_size": dictionary_size,
    }


def analyze_pharma_effect_claims(ocr_data: dict) -> dict:
    return _analyze_claims(
        ocr_data,
        extract_pharma_effect_terms,
        "pharma_effect_claim_count",
        "pharma_effect_claims",
        len(PHARMA_EFFECT_TERMS),
    )


def analyze_bad_treatment_claims(ocr_data: dict) -> dict:
    return _analyze_claims(
        ocr_data,
        extract_bad_treatment_terms,
        "bad_treatment_claim_count",
        "bad_treatment_claims",
        len(BAD_TREATMENT_TERMS),
    )


def _format_claims_html(claims: list[dict]) -> str:
    parts = []
    for item in claims:
        safe_term = html.escape(item["term"])
        safe_seconds = html.escape(", ".join(item["seconds"][:8]))
        parts.append(f'<strong style="color:#dc2626;font-weight:800">{safe_term}</strong> ({safe_seconds})')
    return "; ".join(parts)


def evaluate_pharma_effect_claims(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка слов о гарантии положительного действия продукта будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка слов о гарантии положительного действия продукта будет выполнена после извлечения кадров.",
        }

    analysis = analyze_pharma_effect_claims(load_ocr_log(ocr_path))
    claims = analysis["pharma_effect_claims"]
    if not claims:
        return {
            "status": "pass",
            "message": NO_PHARMA_EFFECT_CLAIMS_MESSAGE,
            **analysis,
        }

    plain_claims = "; ".join(f"{item['term']} ({', '.join(item['seconds'][:8])})" for item in claims)
    return {
        "status": "fail",
        "message": f"Обнаружены слова о гарантии положительного действия продукта: {plain_claims}.",
        "message_html": (
            "Обнаружены слова о гарантии положительного действия продукта: "
            f"{_format_claims_html(claims)}."
        ),
        **analysis,
    }


def evaluate_bad_treatment_claims(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка слов и намёков на лечение в рекламе БАД будет выполнена после OCR кадров.",
        }

    if find_frames_dir(base) is None:
        return {
            "status": "pending",
            "message": "Проверка слов и намёков на лечение в рекламе БАД будет выполнена после извлечения кадров.",
        }

    analysis = analyze_bad_treatment_claims(load_ocr_log(ocr_path))
    claims = analysis["bad_treatment_claims"]
    if not claims:
        return {
            "status": "pass",
            "message": NO_BAD_TREATMENT_CLAIMS_MESSAGE,
            **analysis,
        }

    plain_claims = "; ".join(f"{item['term']} ({', '.join(item['seconds'][:8])})" for item in claims)
    return {
        "status": "fail",
        "message": f"Обнаружены слова или намёки на лечение в рекламе БАД: {plain_claims}.",
        "message_html": (
            "Обнаружены слова или намёки на лечение в рекламе БАД: "
            f"{_format_claims_html(claims)}."
        ),
        **analysis,
    }
