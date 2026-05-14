# -*- coding: utf-8 -*-
"""Recognize actors/media persons from extracted face images without Telegram."""

from __future__ import annotations

import html
import json
import os
import re
import time
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

try:
    import face_recognition
except Exception:  # pragma: no cover - optional runtime dependency
    face_recognition = None


NO_ACTORS_MESSAGE = "Актеров в ролике нет"
ACTOR_NOT_RECOGNIZED_MESSAGE = "Актер не распознан"
SEARCH_CACHE_NAME = "Actor_Web_Search_Results.json"
SEARCH_CACHE_VERSION = 2
KNOWN_FACES_CACHE_NAME = "Known_Faces_Encodings.json"
SEARCH_ENDPOINT_ENV = "MEDIA_PERSON_WEB_SEARCH_ENDPOINT"
SERPAPI_KEY_ENV = "SERPAPI_API_KEY"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
PUBLIC_BASE_URL_ENV = "MEDIA_PERSON_PUBLIC_FACE_BASE_URL"
PUBLIC_UPLOAD_ENDPOINT_ENV = "MEDIA_PERSON_FACE_UPLOAD_ENDPOINT"
SEARCH_PROVIDER_ENV = "MEDIA_PERSON_SEARCH_PROVIDER"
KNOWN_FACES_DIR_ENV = "MEDIA_PERSON_KNOWN_FACES_DIR"
DIRECT_WEB_PROVIDER = "direct_web"
LOCAL_KNOWN_FACES_PROVIDER = "local_known_faces"
SERPAPI_PROVIDER = "serpapi_google_lens"
DEFAULT_KNOWN_FACES_DIR = (
    r"C:\Users\PTambulatov\Desktop\PT\2.Расчеты\50. ХАКАТОН\7. face recognition\known_faces"
)

NAME_RE = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
)
NEGATIVE_RESULT_RE = re.compile(
    r"(?:не\s+установлен|не\s+найден|нет\s+совпад|unknown|not\s+found|no\s+match)",
    flags=re.IGNORECASE,
)
NON_PERSON_WORDS = {
    "google",
    "search",
    "english",
    "deutsch",
    "cymraeg",
    "dansk",
    "france",
    "gaeilge",
    "hrvatski",
    "indonesia",
    "italiano",
    "kiswahili",
    "melayu",
    "nederlands",
    "suomi",
    "svenska",
    "united",
    "kingdom",
    "states",
    "privacy",
    "terms",
    "policy",
    "конфиденциальность",
    "условия",
    "политика",
    "поиск",
    "картинки",
    "изображения",
    "википедия",
    "фото",
}
NON_PERSON_PHRASES = {
    "google search",
    "deutsch english",
    "english united",
    "united kingdom",
    "united states",
    "конфиденциальность условия",
    "условия политика",
}


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


def _known_faces_dir() -> Path:
    return Path(os.getenv(KNOWN_FACES_DIR_ENV, DEFAULT_KNOWN_FACES_DIR))


def _cache_matches_faces(cached: dict, faces: list[Path]) -> bool:
    if cached.get("cache_version") != SEARCH_CACHE_VERSION:
        return False
    cached_names = sorted(str(item.get("file", "")) for item in cached.get("faces", []))
    current_names = sorted(path.name for path in faces)
    return cached_names == current_names


def _configured_provider() -> str:
    configured = os.getenv(SEARCH_PROVIDER_ENV, "").strip().lower()
    if configured in {LOCAL_KNOWN_FACES_PROVIDER, DIRECT_WEB_PROVIDER, SERPAPI_PROVIDER, "endpoint", "none"}:
        if configured == "endpoint":
            return os.getenv(SEARCH_ENDPOINT_ENV, "").strip() or "endpoint"
        return configured
    if os.getenv("MEDIA_PERSON_ENABLE_SERPAPI", "").strip() == "1" and os.getenv(SERPAPI_KEY_ENV, "").strip():
        return SERPAPI_PROVIDER
    endpoint = os.getenv(SEARCH_ENDPOINT_ENV, "").strip()
    if endpoint:
        return endpoint
    return DIRECT_WEB_PROVIDER


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
        if not _looks_like_person_name(name):
            continue
        key = name.lower().replace("ё", "е")
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _looks_like_person_name(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(name).strip()).lower().replace("ё", "е")
    if not normalized:
        return False
    if normalized in NON_PERSON_PHRASES:
        return False

    tokens = re.findall(r"[a-zа-яё]+", normalized, flags=re.IGNORECASE)
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    if any(token in NON_PERSON_WORDS for token in tokens):
        return False
    return True


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


def _collect_serpapi_text(payload: dict) -> str:
    chunks = []

    def add(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            for key in (
                "title",
                "name",
                "source",
                "snippet",
                "description",
                "displayed_link",
                "link",
            ):
                add(value.get(key))
        elif isinstance(value, list):
            for item in value:
                add(item)

    for key in (
        "knowledge_graph",
        "visual_matches",
        "exact_matches",
        "image_results",
        "related_content",
        "organic_results",
    ):
        add(payload.get(key))
    return "\n".join(chunks)


def _html_to_search_text(html_text: str) -> str:
    if not html_text:
        return ""
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _direct_search_urls(image_url: str) -> list[str]:
    quoted_url = quote(image_url, safe="")
    return [
        f"https://lens.google.com/uploadbyurl?url={quoted_url}",
        f"https://www.google.com/searchbyimage?image_url={quoted_url}",
        f"https://yandex.com/images/search?rpt=imageview&url={quoted_url}",
        f"https://yandex.ru/images/search?rpt=imageview&url={quoted_url}",
    ]


def _direct_upload_search_pages(face_path: Path, headers: dict[str, str]) -> list[dict]:
    content_type = guess_type(face_path.name)[0] or "image/jpeg"
    urls = [
        "https://www.google.com/searchbyimage/upload",
        "https://www.google.ru/searchbyimage/upload",
    ]
    pages = []
    for url in urls:
        with face_path.open("rb") as fh:
            response = requests.post(
                url,
                headers=headers,
                files={"encoded_image": (face_path.name, fh, content_type)},
                data={"image_content": "", "filename": face_path.name},
                allow_redirects=True,
                timeout=float(os.getenv("MEDIA_PERSON_DIRECT_WEB_TIMEOUT", "45")),
            )
        response.raise_for_status()
        pages.append({"url": response.url, "text": response.text})
    return pages


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


def _name_from_reference_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0].replace("_", " ").strip()
    return path.stem.replace("_", " ").strip()


def _known_reference_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _reference_signature(files: list[Path]) -> list[dict]:
    return [
        {
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "size": path.stat().st_size,
        }
        for path in files
    ]


def _face_encoding_for_image(path: Path):
    if face_recognition is None:
        return None
    image = face_recognition.load_image_file(str(path))
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)
    if not encodings:
        encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return np.asarray(encodings[0], dtype=float)


def _load_known_face_encodings(root: Path) -> dict:
    files = _known_reference_files(root)
    signature = _reference_signature(files)
    cache_path = root / KNOWN_FACES_CACHE_NAME
    cached = _read_json(cache_path)
    if cached and cached.get("signature") == signature:
        return cached

    people = []
    errors = []
    for path in files:
        try:
            encoding = _face_encoding_for_image(path)
            if encoding is None:
                errors.append({"file": str(path), "error": "face was not detected"})
                continue
            people.append(
                {
                    "name": _name_from_reference_path(path, root),
                    "file": str(path),
                    "encoding": encoding.tolist(),
                }
            )
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})

    result = {
        "ok": True,
        "cache_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "signature": signature,
        "people": people,
        "errors": errors,
    }
    if root.is_dir():
        _write_json(cache_path, result)
    return result


def _search_face_via_known_faces(face_path: Path) -> dict:
    root = _known_faces_dir()
    if face_recognition is None:
        return {
            "file": face_path.name,
            "provider": LOCAL_KNOWN_FACES_PROVIDER,
            "raw_result": "",
            "names": [],
            "errors": [{"error": "face_recognition is not installed"}],
        }

    known = _load_known_face_encodings(root)
    people = known.get("people", [])
    if not people:
        return {
            "file": face_path.name,
            "provider": LOCAL_KNOWN_FACES_PROVIDER,
            "raw_result": "",
            "names": [],
            "known_faces_dir": str(root),
            "errors": [{"error": f"known faces database is empty: {root}"}],
        }

    try:
        face_encoding = _face_encoding_for_image(face_path)
        if face_encoding is None:
            return {
                "file": face_path.name,
                "provider": LOCAL_KNOWN_FACES_PROVIDER,
                "raw_result": "",
                "names": [],
                "known_faces_dir": str(root),
                "errors": [{"error": "face was not detected"}],
            }
    except Exception as exc:
        return {
            "file": face_path.name,
            "provider": LOCAL_KNOWN_FACES_PROVIDER,
            "raw_result": "",
            "names": [],
            "known_faces_dir": str(root),
            "errors": [{"error": str(exc)}],
        }

    known_encodings = np.asarray([item["encoding"] for item in people], dtype=float)
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    matches = []
    tolerance = float(os.getenv("MEDIA_PERSON_KNOWN_FACE_TOLERANCE", "0.5"))
    for person, distance in zip(people, distances):
        distance_value = float(distance)
        if distance_value <= tolerance:
            matches.append(
                {
                    "name": person["name"],
                    "distance": round(distance_value, 4),
                    "reference_file": person["file"],
                }
            )
    matches.sort(key=lambda item: item["distance"])
    best = matches[: int(os.getenv("MEDIA_PERSON_KNOWN_FACE_MAX_MATCHES", "3"))]
    return {
        "file": face_path.name,
        "provider": LOCAL_KNOWN_FACES_PROVIDER,
        "raw_result": "; ".join(f"{item['name']} ({item['distance']})" for item in best),
        "names": _dedupe_names([item["name"] for item in best]),
        "known_faces_dir": str(root),
        "matches": best,
        "errors": known.get("errors", []),
    }


def _public_url_from_base(face_path: Path, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(face_path.name)}"


def _upload_face_to_endpoint(face_path: Path, endpoint: str) -> str:
    content_type = guess_type(face_path.name)[0] or "application/octet-stream"
    with face_path.open("rb") as fh:
        response = requests.post(
            endpoint,
            files={"image": (face_path.name, fh, content_type)},
            timeout=float(os.getenv("MEDIA_PERSON_FACE_UPLOAD_TIMEOUT", "45")),
        )
    response.raise_for_status()
    payload = response.json()
    for key in ("url", "image_url", "public_url", "link"):
        value = payload.get(key)
        if value:
            return str(value)
    raise ValueError(f"{PUBLIC_UPLOAD_ENDPOINT_ENV} did not return public image url")


def _upload_face_to_yandex_object_storage(face_path: Path) -> str | None:
    bucket = os.getenv("YANDEX_BUCKET", "").strip()
    access_key = (
        os.getenv("YANDEX_S3_ACCESS_KEY_ID", "").strip()
        or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    )
    secret_key = (
        os.getenv("YANDEX_S3_SECRET_ACCESS_KEY", "").strip()
        or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    if not bucket or not access_key or not secret_key:
        return None

    try:
        import boto3  # type: ignore
    except Exception:
        return None

    key_prefix = os.getenv("MEDIA_PERSON_YANDEX_OBJECT_PREFIX", "clipchecker/faces").strip("/")
    object_key = f"{key_prefix}/{int(time.time())}_{face_path.name}"
    content_type = guess_type(face_path.name)[0] or "application/octet-stream"
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("YANDEX_S3_ENDPOINT", "https://storage.yandexcloud.net"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    extra_args = {"ContentType": content_type}
    if os.getenv("MEDIA_PERSON_YANDEX_PUBLIC_READ", "1") != "0":
        extra_args["ACL"] = "public-read"
    client.upload_file(str(face_path), bucket, object_key, ExtraArgs=extra_args)
    public_base = os.getenv("YANDEX_PUBLIC_BUCKET_URL", f"https://storage.yandexcloud.net/{bucket}")
    return f"{public_base.rstrip('/')}/{quote(object_key)}"


def _resolve_public_face_url(face_path: Path) -> str | None:
    base_url = os.getenv(PUBLIC_BASE_URL_ENV, "").strip()
    if base_url:
        return _public_url_from_base(face_path, base_url)

    upload_endpoint = os.getenv(PUBLIC_UPLOAD_ENDPOINT_ENV, "").strip()
    if upload_endpoint:
        return _upload_face_to_endpoint(face_path, upload_endpoint)

    return _upload_face_to_yandex_object_storage(face_path)


def _search_face_via_serpapi(face_path: Path, api_key: str) -> dict:
    image_url = _resolve_public_face_url(face_path)
    if not image_url:
        raise ValueError(
            f"{SERPAPI_KEY_ENV} is configured, but no public image URL source is configured. "
            f"Set {PUBLIC_BASE_URL_ENV}, {PUBLIC_UPLOAD_ENDPOINT_ENV}, or Yandex S3 static keys."
        )

    params = {
        "engine": "google_lens",
        "api_key": api_key,
        "url": image_url,
        "type": os.getenv("MEDIA_PERSON_SERPAPI_TYPE", "all"),
        "hl": os.getenv("MEDIA_PERSON_SERPAPI_HL", "ru"),
        "country": os.getenv("MEDIA_PERSON_SERPAPI_COUNTRY", "ru"),
        "safe": os.getenv("MEDIA_PERSON_SERPAPI_SAFE", "active"),
    }
    query = os.getenv("MEDIA_PERSON_SERPAPI_QUERY", "актёр знаменитость телеведущий")
    if query:
        params["q"] = query

    response = requests.get(
        SERPAPI_ENDPOINT,
        params=params,
        timeout=float(os.getenv("MEDIA_PERSON_SERPAPI_TIMEOUT", "60")),
    )
    response.raise_for_status()
    payload = response.json()
    raw_text = _collect_serpapi_text(payload)
    return {
        "file": face_path.name,
        "provider": SERPAPI_PROVIDER,
        "image_url": image_url,
        "raw_result": raw_text or json.dumps(payload, ensure_ascii=False),
        "names": _dedupe_names(
            _extract_names_from_response(payload) + extract_names_from_search_text(raw_text)
        ),
    }


def _search_face_via_direct_web(face_path: Path) -> dict:
    headers = {
        "User-Agent": os.getenv(
            "MEDIA_PERSON_WEB_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        ),
        "Accept-Language": os.getenv("MEDIA_PERSON_WEB_ACCEPT_LANGUAGE", "ru-RU,ru;q=0.9,en;q=0.6"),
    }
    chunks = []
    errors = []

    image_url = None
    try:
        for page in _direct_upload_search_pages(face_path, headers):
            chunks.append(f"{page['url']}\n{_html_to_search_text(page['text'])}")
    except Exception as exc:
        errors.append({"provider": "google_upload", "error": str(exc)})

    try:
        image_url = _resolve_public_face_url(face_path)
    except Exception as exc:
        errors.append({"provider": "public_url", "error": str(exc)})
        image_url = None

    for url in _direct_search_urls(image_url) if image_url else []:
        try:
            response = requests.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=float(os.getenv("MEDIA_PERSON_DIRECT_WEB_TIMEOUT", "45")),
            )
            response.raise_for_status()
            chunks.append(f"{response.url}\n{_html_to_search_text(response.text)}")
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    if not chunks and not image_url:
        raise ValueError(
            "Direct web search failed: Google image upload did not return searchable text and no public "
            f"image URL is available. Set {PUBLIC_BASE_URL_ENV}, {PUBLIC_UPLOAD_ENDPOINT_ENV}, "
            "or Yandex S3 static keys."
        )

    raw_text = "\n".join(chunks)
    return {
        "file": face_path.name,
        "provider": DIRECT_WEB_PROVIDER,
        "image_url": image_url,
        "raw_result": raw_text,
        "names": extract_names_from_search_text(raw_text),
        "errors": errors,
    }


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
    faces = _face_files(base / "faces")
    configured_provider = _configured_provider()
    cached = _read_json(cache_path)
    if (
        cached
        and cached.get("ok")
        and cached.get("provider") == configured_provider
        and _cache_matches_faces(cached, faces)
    ):
        return cached

    if not faces:
        result = {
            "ok": True,
            "cache_version": SEARCH_CACHE_VERSION,
            "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": "none",
            "faces": [],
        }
        _write_json(cache_path, result)
        return result

    endpoint = os.getenv(SEARCH_ENDPOINT_ENV, "").strip()
    serpapi_key = os.getenv(SERPAPI_KEY_ENV, "").strip()
    results = []
    errors = []
    provider = configured_provider
    if configured_provider == LOCAL_KNOWN_FACES_PROVIDER:
        provider = LOCAL_KNOWN_FACES_PROVIDER
        for face_path in faces:
            results.append(_search_face_via_known_faces(face_path))
    elif configured_provider == DIRECT_WEB_PROVIDER:
        provider = DIRECT_WEB_PROVIDER
        for face_path in faces:
            try:
                local_result = _search_face_via_known_faces(face_path)
                if local_result.get("names"):
                    results.append(local_result)
                    continue
                web_result = _search_face_via_direct_web(face_path)
                web_result["local_known_faces"] = local_result
                results.append(web_result)
            except Exception as exc:
                errors.append({"file": face_path.name, "provider": provider, "error": str(exc)})
                results.append({"file": face_path.name, "raw_result": "", "names": []})
    elif configured_provider == SERPAPI_PROVIDER and serpapi_key:
        provider = SERPAPI_PROVIDER
        for face_path in faces:
            try:
                results.append(_search_face_via_serpapi(face_path, serpapi_key))
            except Exception as exc:
                errors.append({"file": face_path.name, "provider": provider, "error": str(exc)})
                results.append({"file": face_path.name, "raw_result": "", "names": []})
    elif endpoint:
        provider = endpoint
        for face_path in faces:
            try:
                results.append(_search_face_via_endpoint(face_path, endpoint))
            except Exception as exc:
                errors.append({"file": face_path.name, "provider": "endpoint", "error": str(exc)})
                results.append({"file": face_path.name, "raw_result": "", "names": []})
    else:
        results = [{"file": face_path.name, "raw_result": "", "names": []} for face_path in faces]
        errors.append({"error": f"{DIRECT_WEB_PROVIDER} is not available and no fallback provider is configured"})

    result = {
        "ok": True,
        "cache_version": SEARCH_CACHE_VERSION,
        "searched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
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
