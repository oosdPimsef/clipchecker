# -*- coding: utf-8 -*-
"""Detect ruble prices in OCR text from video frames."""

from __future__ import annotations

import html
import re
from pathlib import Path

try:
    from frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text
except ImportError:
    from .frame_safety import frame_second_label, iter_ocr_lines, load_ocr_log, normalize_text


NO_PRICE_MESSAGE = "Цена в видеоряде не указана"
PRICE_RE = re.compile(
    r"(?<![\w])"
    r"(?P<amount>\d{1,3}(?:[\s\u00a0]\d{3})*(?:[,.]\d{1,2})?|\d+(?:[,.]\d{1,2})?)"
    r"\s*"
    r"(?P<currency>₽|руб(?:\.|лей|ля|ль|лю|лями)?|р\.)",
    flags=re.IGNORECASE,
)


def normalize_price(price: str) -> str:
    return re.sub(r"\s+", " ", price.replace("\u00a0", " ")).strip()


def extract_ruble_prices(text: str) -> list[str]:
    prices = []
    for match in PRICE_RE.finditer(text or ""):
        prices.append(normalize_price(match.group(0)))
    return prices


def analyze_prices(ocr_data: dict) -> dict:
    grouped: dict[str, dict] = {}
    total = 0

    for item in iter_ocr_lines(ocr_data):
        for price in extract_ruble_prices(item["text"]):
            total += 1
            key = normalize_price(price).lower()
            if key not in grouped:
                grouped[key] = {
                    "price": price,
                    "seconds": [],
                    "frames": [],
                    "examples": [],
                }
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

    return {
        "price_count": total,
        "prices": list(grouped.values()),
    }


def _format_prices_html(prices: list[dict]) -> str:
    parts = []
    for item in prices:
        seconds = ", ".join(item["seconds"][:5])
        safe_price = html.escape(item["price"])
        safe_seconds = html.escape(seconds)
        parts.append(f'<strong class="price-value">{safe_price}</strong> ({safe_seconds})')
    return "; ".join(parts)


def evaluate_ruble_prices(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    ocr_path = base / "OCR_Log.json"
    if not ocr_path.is_file():
        return {
            "status": "pending",
            "message": "Проверка цены будет выполнена после OCR кадров.",
        }

    analysis = analyze_prices(load_ocr_log(ocr_path))
    prices = analysis["prices"]
    if not prices:
        return {
            "status": "fail",
            "message": NO_PRICE_MESSAGE,
            **analysis,
        }

    plain_prices = "; ".join(f"{item['price']} ({', '.join(item['seconds'][:5])})" for item in prices)
    message = f"Найдена цена в рублях: {plain_prices}."
    message_html = f"Найдена цена в рублях: {_format_prices_html(prices)}."
    return {
        "status": "pass",
        "message": message,
        "message_html": message_html,
        **analysis,
    }
