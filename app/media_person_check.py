# -*- coding: utf-8 -*-
"""Search media persons by face images and check contracts in documents."""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
from pathlib import Path


CONTRACT_FOUND_MESSAGE = "Договор с медийной личностью найден в документах"
CONTRACT_REQUIRED_MESSAGE = "Необходимо предоставить договор с медийной личностью"
NO_MEDIA_PERSONS_MESSAGE = "Медийные личности в видеоряде не найдены"
SEARCH_PENDING_MESSAGE = "Поиск медийных личностей по фото будет выполнен после сетевого поиска по лицам."
SEARCH_CACHE_NAME = "Media_Person_Search_Results.json"

NEGATIVE_FACE_RESULT_RE = re.compile(
    r"(?:не\s+установлен|не\s+найден|нет\s+совпад|публичная\s+личность\s+не|unknown|not\s+found|no\s+match)",
    flags=re.IGNORECASE,
)
CONTRACT_RE = re.compile(
    r"(?:договор|соглашени[ея]|контракт|лицензионн[а-я]*|согласие|разрешение|релиз|release|contract|agreement)",
    flags=re.IGNORECASE,
)
NAME_RE = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


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


def _cleanup_face_result(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", " ", text or "")
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"\b\d{1,3}(?:[.,]\d+)?\s*%", " ", cleaned)
    cleaned = re.sub(r"[_*`[\](){}<>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_media_person_names_from_text(text: str) -> list[str]:
    if not text or NEGATIVE_FACE_RESULT_RE.search(text):
        return []

    cleaned = _cleanup_face_result(text)
    candidates = NAME_RE.findall(cleaned)
    if not candidates and cleaned:
        first_sentence = re.split(r"[.;\n]", cleaned, maxsplit=1)[0].strip()
        words = re.findall(r"[A-Za-zА-ЯЁа-яё-]{2,}", first_sentence)
        if len(words) >= 2:
            candidates = [" ".join(words[:3])]

    names = []
    seen = set()
    for candidate in candidates:
        name = re.sub(r"\s+", " ", candidate).strip(" -:;,")
        key = name.lower().replace("ё", "е")
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def extract_media_persons(search_results: dict) -> list[dict]:
    persons: dict[str, dict] = {}
    for face_result in search_results.get("faces", []):
        face_file = str(face_result.get("file", "")).strip()
        raw_result = str(face_result.get("raw_result", "")).strip()
        names = face_result.get("names") or extract_media_person_names_from_text(raw_result)
        for name in names:
            key = str(name).lower().replace("ё", "е").strip()
            if not key:
                continue
            if key not in persons:
                persons[key] = {"name": str(name).strip(), "face_files": [], "source_lines": []}
            if face_file:
                persons[key]["face_files"].append(face_file)
            if raw_result:
                persons[key]["source_lines"].append(raw_result)

    for person in persons.values():
        person["face_files"] = sorted(set(person["face_files"]))
        person["source_lines"] = person["source_lines"][:3]
    return list(persons.values())


async def _search_faces_via_searchbyphoto_bot(face_paths: list[Path]) -> dict:
    from pyrogram import Client

    api_id_raw = os.getenv("TELEGRAM_API_ID", "")
    api_hash = os.getenv("TELEGRAM_API_HASH") or ""
    try:
        api_id = int(api_id_raw)
    except Exception:
        api_id = 0
    if not api_id or not api_hash:
        return {"ok": False, "pending": True, "error": "TELEGRAM_API_ID/TELEGRAM_API_HASH are not configured", "faces": []}

    client = Client("face_search_bot", api_id=api_id, api_hash=api_hash)
    await client.start()
    try:
        bot = await client.get_users("SearchByPhoto_Bot")
        await client.send_message(bot.id, "/start")
        await asyncio.sleep(2)

        results = []
        for face_path in face_paths:
            await client.send_photo(bot.id, str(face_path))
            await asyncio.sleep(6)

            raw_result = "Публичная личность не установлена"
            async for msg in client.get_chat_history(bot.id, limit=8):
                text = (getattr(msg, "caption", None) or getattr(msg, "text", None) or "").strip()
                if text:
                    raw_result = text
                    break

            results.append(
                {
                    "file": face_path.name,
                    "raw_result": raw_result,
                    "names": extract_media_person_names_from_text(raw_result),
                }
            )
            await asyncio.sleep(2)

        return {"ok": True, "pending": False, "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "faces": results}
    finally:
        await client.stop()


def search_media_persons(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    cache_path = base / SEARCH_CACHE_NAME
    cached = _read_json(cache_path)
    if cached and cached.get("ok"):
        return cached

    faces = _face_files(base / "faces")
    if not faces:
        result = {"ok": True, "pending": False, "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "faces": []}
        _write_json(cache_path, result)
        return result

    try:
        result = asyncio.run(_search_faces_via_searchbyphoto_bot(faces))
    except Exception as exc:
        result = {"ok": False, "pending": True, "error": str(exc), "faces": []}

    _write_json(cache_path, result)
    return result


def split_document_sections(documents_text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^Документ\s+\d+\.\s*(.+?)\s*$", documents_text or ""))
    if not matches:
        return [("", documents_text)] if (documents_text or "").strip() else []

    sections = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(documents_text)
        sections.append((match.group(1).strip(), documents_text[start:end]))
    return sections


def person_search_tokens(name: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-ЯЁа-яё-]{3,}", name or "")
    seen = set()
    out = []
    for token in reversed(tokens):
        normalized = token.lower().replace("ё", "е")
        if normalized not in seen:
            seen.add(normalized)
            out.append(token)
    return out


def find_contracts_for_person(name: str, documents_text: str) -> list[str]:
    tokens = person_search_tokens(name)
    if not tokens:
        return []

    found = []
    for title, text in split_document_sections(documents_text):
        haystack = f"{title}\n{text}"
        if not CONTRACT_RE.search(haystack):
            continue
        normalized = haystack.lower().replace("ё", "е")
        for token in tokens:
            normalized_token = token.lower().replace("ё", "е")
            pattern = rf"(?<![A-Za-zА-ЯЁа-яё]){re.escape(normalized_token)}(?![A-Za-zА-ЯЁа-яё])"
            if re.search(pattern, normalized):
                found.append(title or "документ без названия")
                break
    return sorted(set(found))


def _format_person_names_html(persons: list[dict]) -> str:
    return "; ".join(
        f'<strong class="media-person-name">{html.escape(person["name"])}</strong>'
        for person in persons
    )


def evaluate_media_person_contracts(result_dir: str | Path) -> dict:
    base = Path(result_dir)
    search_results = search_media_persons(base)
    if search_results.get("pending"):
        return {
            "status": "pending",
            "message": f"{SEARCH_PENDING_MESSAGE} Техническая причина: {search_results.get('error', '')}".strip(),
            "media_persons": [],
            "search_results": search_results,
        }

    persons = extract_media_persons(search_results)
    if not persons:
        return {
            "status": "pass",
            "message": NO_MEDIA_PERSONS_MESSAGE,
            "media_persons": [],
            "search_results": search_results,
        }

    documents_text = _read_text(base / "Documents_Texts.txt")
    missing = []
    matched = []
    for person in persons:
        contracts = find_contracts_for_person(person["name"], documents_text)
        person["contract_documents"] = contracts
        if contracts:
            matched.append(person)
        else:
            missing.append(person)

    names_text = ", ".join(person["name"] for person in persons)
    names_html = _format_person_names_html(persons)
    if missing:
        missing_names = ", ".join(person["name"] for person in missing)
        return {
            "status": "fail",
            "message": f"Найдены медийные личности: {names_text}. {CONTRACT_REQUIRED_MESSAGE}: {missing_names}.",
            "message_html": f"Найдены медийные личности: {names_html}. {html.escape(CONTRACT_REQUIRED_MESSAGE)}: {html.escape(missing_names)}.",
            "media_persons": persons,
            "missing_contracts": [person["name"] for person in missing],
            "matched_contracts": [person["name"] for person in matched],
            "search_results": search_results,
        }

    return {
        "status": "pass",
        "message": f"{CONTRACT_FOUND_MESSAGE}: {names_text}.",
        "message_html": f"{html.escape(CONTRACT_FOUND_MESSAGE)}: {names_html}.",
        "media_persons": persons,
        "missing_contracts": [],
        "matched_contracts": [person["name"] for person in matched],
        "search_results": search_results,
    }
