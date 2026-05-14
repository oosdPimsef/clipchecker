# -*- coding: utf-8 -*-
"""Recognize actors/media persons from extracted face images without Telegram."""

from __future__ import annotations

import html
import json
import os
import re
import time
from pathlib import Path

import requests


NO_ACTORS_MESSAGE = "Актеров в ролике нет"
ACTOR_NOT_RECOGNIZED_MESSAGE = "Актер не распознан"
SEARCH_CACHE_NAME = "Actor_Web_Search_Results.json"
SEARCH_ENDPOINT_ENV = "MEDIA_PERSON_WEB_SEARCH_ENDPOINT"

NAME_RE = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
)
NEGATIVE_RESULT_RE = re.compile(
    r"(?:не\s+установлен|не\s+найден|нет\s+совпад|unknown|not\s+found|no\s+match)",
    flags=re.IGNORECASE,
)


def _read_json(path: Path) -> dict | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _face_files(faces_dir: Path) -> list[Path]:
    if not faces_dir.is_dir():
        return []
    return sorted(
        path
        for path in faces_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def extract_names_from_search_text(text: str) -> list[str]:
    if not text or NEGATIVE_RESULT_RE.search(text):
        return []
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"\b\d{1,3}(?:[.,]\d+)?\s*%", " ", cleaned)
    cleaned = re.sub(r"[_*`[\](){}<>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    names = []
    seen = set()
    for candidate in NAME_RE.findall(cleaned):
        name = re.sub(r"\s+", " ", candidate).strip(" -:;,")
        key = name.lower().replace("ё", "е")
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _extract_names_from_response(data) -> list[str]:
    if not data:
        return []
    if isinstance(data, str):
        return extract_names_from_search_text(data)
    if isinstance(data, list):
        names = []
        for item in data:
            names.extend(_extract_names_from_response(item))
        return _dedupe_names(names)
    if isinstance(data, dict):
        for key in ("names", "persons", "actors", "celebrities"):
            if key in data:
                return _extract_names_from_response(data.get(key))
        for key in ("name", "person", "actor", "celebrity", "title", "text", "caption"):
            if key in data:
                return _extract_names_from_response(data.get(key))
    return []


def _dedupe_names(names: list[str]) -> list[str]:
    out = []
    seen = set()
    for name in names:
        normalized = str(name).strip()
        key = normalized.lower().replace("ё", "е")
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _search_face_via_endpoint(face_path: Path, endpoint: str) -> dict:
    with face_path.open("rb") as fh:
        response = requests.post(
            endpoint,
            files={"image": (face_path.name, fh, "application/octet-stream")},
            timeout=float(os.getenv("MEDIA_PERSON_WEB_SEARCH_TIMEOUT", "45")),
        )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
    else:
        payload = response.text
    return {
        "file": face_path.name,
        "raw_result": payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
        "names": _extract_names_from_response(payload),
    }


def search_actor_names(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    cache_path = base / SEARCH_CACHE_NAME
    cached = _read_json(cache_path)
    if cached and cached.get("ok"):
        return cached

    faces = _face_files(base / "faces")
    if not faces:
        result = {
            "ok": True,
            "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": "none",
            "faces": [],
        }
        _write_json(cache_path, result)
        return result

    endpoint = os.getenv(SEARCH_ENDPOINT_ENV, "").strip()
    results = []
    errors = []
    if endpoint:
        for face_path in faces:
            try:
                results.append(_search_face_via_endpoint(face_path, endpoint))
            except Exception as exc:
                errors.append({"file": face_path.name, "error": str(exc)})
                results.append({"file": face_path.name, "raw_result": "", "names": []})
    else:
        results = [{"file": face_path.name, "raw_result": "", "names": []} for face_path in faces]
        errors.append({"error": f"{SEARCH_ENDPOINT_ENV} is not configured"})

    result = {
        "ok": True,
        "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": endpoint or "not_configured",
        "faces": results,
        "errors": errors,
    }
    _write_json(cache_path, result)
    return result


def extract_actor_names(search_results: dict) -> list[str]:
    names = []
    for face in search_results.get("faces", []):
        names.extend(face.get("names") or extract_names_from_search_text(str(face.get("raw_result", ""))))
    return _dedupe_names(names)


def _format_names_html(names: list[str]) -> str:
    return "; ".join(
        f'<strong class="media-person-name">{html.escape(name)}</strong>'
        for name in names
    )


def evaluate_actor_recognition(result_dir: str | Path) -> dict:
    search_results = search_actor_names(result_dir)
    faces = search_results.get("faces", [])
    if not faces:
        return {
            "status": "pass",
            "message": NO_ACTORS_MESSAGE,
            "search_results": search_results,
            "actor_names": [],
        }

    names = extract_actor_names(search_results)
    if not names:
        return {
            "status": "pass",
            "message": ACTOR_NOT_RECOGNIZED_MESSAGE,
            "search_results": search_results,
            "actor_names": [],
        }

    names_text = "; ".join(names)
    return {
        "status": "pass",
        "message": f"Личность определена: {names_text}",
        "message_html": f"Личность определена: {_format_names_html(names)}",
        "search_results": search_results,
        "actor_names": names,
    }
