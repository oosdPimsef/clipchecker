# -*- coding: utf-8 -*-
"""Detect ruble prices in OCR text from video frames."""

from __future__ import annotations

import html
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


NO_PRICE_MESSAGE = "Цена в видеоряде не указана"
NO_DOCUMENT_PRICE_MESSAGE = "Цена в документах не указана"
PRICE_RE = re.compile(
    r"(?<![\w])"
    r"(?P<amount>\d{1,3}(?:[\s\u00a0]\d{3})*(?:[,.]\d{1,2})?|\d+(?:[,.]\d{1,2})?)"
    r"\s*"
    r"(?P<currency>₽|руб(?:\.|лей|ля|ль|лю|лями)?|р\.)",
    flags=re.IGNORECASE,
)
PRICE_CONTEXT_RE = re.compile(
    r"(?:цена|стоимость|стоимость\s+товар[а-я]*|цена\s+товар[а-я]*|розничн[а-я]*\s+цен[а-я]*|"
    r"акционн[а-я]*\s+цен[а-я]*|по\s+цен[еуы]|прода[её]тся\s+по|составляет|руб(?:\.|лей|ля|ль|лю|лями)?)",
    flags=re.IGNORECASE,
)
NON_PRODUCT_PRICE_CONTEXT_RE = re.compile(
    r"(?:доверенност|не\s+более|до\s+сумм|на\s+сумм|сумм[ауы]\s+не\s+более|"
    r"базов[а-я]*\s+величин|договор[а-я]*\s+стоимостью|лимит)",
    flags=re.IGNORECASE,
)


def normalize_price(price: str) -> str:
    return re.sub(r"\s+", " ", price.replace("\u00a0", " ")).strip()


def normalize_price_value(price: str) -> str:
    match = PRICE_RE.search(price or "")
    if not match:
        return normalize_price(price).lower()
    amount = match.group("amount").replace("\u00a0", " ")
    amount = re.sub(r"\s+", "", amount).replace(",", ".")
    return amount


def extract_ruble_prices(text: str) -> list[str]:
    prices = []
    for match in PRICE_RE.finditer(text or ""):
        prices.append(normalize_price(match.group(0)))
    return prices


def _line_scope(item: dict, frames_dir: Path | None) -> str:
    if frames_dir is None:
        return "frame"
    size = frame_size(frames_dir / item["frame"])
    if size is None:
        return "frame"
    _, height = size
    return "legal_disclaimer" if is_legal_disclaimer_text(item["text"], item["bbox"], height) else "frame"


def analyze_prices(ocr_data: dict, frames_dir: str | Path | None = None) -> dict:
    frames_base = Path(frames_dir) if frames_dir else None
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        scope = _line_scope(item, frames_base)
        for price in extract_ruble_prices(item["text"]):
            total += 1
            key = normalize_price(price).lower()
            if key not in grouped:
                grouped[key] = {
                    "price": price,
                    "normalized_value": normalize_price_value(price),
                    "scopes": [],
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
            grouped[key]["scopes"].append(scope)
            grouped[key]["seconds"].append(frame_second_label(item["frame"]))
            grouped[key]["frames"].append(item["frame"])
            if len(grouped[key]["examples"]) < 2:
                grouped[key]["examples"].append(normalize_text(item["text"]))

    for item in grouped.values():
        item["seconds"] = sorted(
            set(item["seconds"]),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 0,
        )
        item["frames"] = sorted(set(item["frames"]))
        item["scopes"] = sorted(set(item["scopes"]))

    frame_values = {
        item["normalized_value"]
        for item in grouped.values()
        if "frame" in item["scopes"]
    }
    legal_values = {
        item["normalized_value"]
        for item in grouped.values()
        if "legal_disclaimer" in item["scopes"]
    }

    return {
        "price_count": total,
        "prices": list(grouped.values()),
        "frame_price_values": sorted(frame_values),
        "legal_disclaimer_price_values": sorted(legal_values),
        "price_mismatch": bool(frame_values and legal_values and frame_values != legal_values),
    }


def read_text_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def _price_context(text: str, start: int, end: int, radius: int = 80) -> str:
    return normalize_text(text[max(0, start - radius): min(len(text), end + radius)])


def _price_before_context(text: str, start: int, radius: int = 70) -> str:
    return normalize_text(text[max(0, start - radius): start])


def extract_document_prices(text: str) -> list[dict]:
    found = []
    for match in PRICE_RE.finditer(text or ""):
        context = _price_context(text, match.start(), match.end())
        if not PRICE_CONTEXT_RE.search(context):
            continue
        if NON_PRODUCT_PRICE_CONTEXT_RE.search(_price_before_context(text, match.start())):
            continue
        price = normalize_price(match.group(0))
        found.append(
            {
                "price": price,
                "normalized_value": normalize_price_value(price),
                "context": context,
            }
        )
    return found


def analyze_document_prices(documents_text: str) -> dict:
    grouped: dict[str, dict] = {}
    total = 0
    for item in extract_document_prices(documents_text):
        total += 1
        key = item["normalized_value"]
        if key not in grouped:
            grouped[key] = {
                "price": item["price"],
                "normalized_value": item["normalized_value"],
                "contexts": [],
            }
        if len(grouped[key]["contexts"]) < 3:
            grouped[key]["contexts"].append(item["context"])

    return {
        "document_price_count": total,
        "document_prices": list(grouped.values()),
        "document_price_values": sorted(grouped.keys()),
    }


def _format_prices_html(prices: list[dict]) -> str:
    parts = []
    for item in prices:
        seconds = ", ".join(item["seconds"][:5])
        safe_price = html.escape(item["price"])
        safe_seconds = html.escape(seconds)
        parts.append(f'<strong class="price-value">{safe_price}</strong> ({safe_seconds})')
    return "; ".join(parts)


def _format_document_prices_html(prices: list[dict]) -> str:
    parts = []
    for item in prices:
        parts.append(f'<strong class="price-value">{html.escape(item["price"])}</strong>')
    return "; ".join(parts)


def evaluate_ruble_prices(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка цены будет выполнена после OCR кадров.",
        }

    frames_dir = find_frames_dir(base)
    analysis = analyze_prices(load_ocr_log(ocr_path), frames_dir)
    prices = analysis["prices"]
    if not prices:
        return {
            "status": "pass",
            "message": NO_PRICE_MESSAGE,
            **analysis,
        }

    plain_prices = "; ".join(f"{item['price']} ({', '.join(item['seconds'][:5])})" for item in prices)
    mismatch = analysis["price_mismatch"]
    prefix = "Найдены цены в рублях, но цена в кадре не совпадает с ценой в набивке:" if mismatch else "Найдена цена в рублях:"
    message = f"{prefix} {plain_prices}."
    message_html = f"{html.escape(prefix)} {_format_prices_html(prices)}."
    return {
        "status": "fail" if mismatch else "pass",
        "message": message,
        "message_html": message_html,
        **analysis,
    }


def evaluate_document_prices_match_video(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    documents_analysis = analyze_document_prices(read_text_file(base / "Documents_Texts.txt"))

    ocr_path = base / "OCR_Log.json"
    video_analysis = {"prices": [], "frame_price_values": [], "legal_disclaimer_price_values": []}
    if ocr_path.is_file():
        video_analysis = analyze_prices(load_ocr_log(ocr_path), find_frames_dir(base))

    video_values = set(video_analysis.get("frame_price_values", [])) | set(video_analysis.get("legal_disclaimer_price_values", []))
    document_values = set(documents_analysis["document_price_values"])
    combined = {
        **documents_analysis,
        "video_prices": video_analysis.get("prices", []),
        "video_price_values": sorted(video_values),
        "price_mismatch": bool(video_values and document_values and video_values != document_values),
    }

    if not document_values:
        return {
            "status": "pass",
            "message": NO_DOCUMENT_PRICE_MESSAGE,
            **combined,
        }

    plain_doc_prices = "; ".join(item["price"] for item in documents_analysis["document_prices"])
    doc_html = _format_document_prices_html(documents_analysis["document_prices"])

    if not video_values:
        return {
            "status": "pass",
            "message": f"Цена найдена в документах: {plain_doc_prices}. Цена в видеоряде не указана.",
            "message_html": f"Цена найдена в документах: {doc_html}. Цена в видеоряде не указана.",
            **combined,
        }

    if combined["price_mismatch"]:
        video_plain = "; ".join(item["price"] for item in video_analysis.get("prices", []))
        video_html = _format_prices_html(video_analysis.get("prices", []))
        return {
            "status": "fail",
            "message": f"Цена в документах не совпадает с ценой в видеоряде. Документы: {plain_doc_prices}. Видеоряд: {video_plain}.",
            "message_html": f"Цена в документах не совпадает с ценой в видеоряде. Документы: {doc_html}. Видеоряд: {video_html}.",
            **combined,
        }

    return {
        "status": "pass",
        "message": f"Цена в документах совпадает с ценой в видеоряде: {plain_doc_prices}.",
        "message_html": f"Цена в документах совпадает с ценой в видеоряде: {doc_html}.",
        **combined,
    }
