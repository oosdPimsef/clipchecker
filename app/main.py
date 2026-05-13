# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_from_directory, Response, stream_with_context, jsonify
import os
import sys
import json
import re
import time
import subprocess
from markupsafe import Markup
import html
from datetime import date, datetime
from pathlib import Path
from approval_checks import evaluate_approval_view_model
from approval_map import build_approval_view_model, load_approval_categories

app = Flask(__name__, static_url_path='/static', static_folder='.')

# === .env для Flask-процесса ===
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    # Явно укажем путь к .env рядом с main.py, чтобы не зависеть от текущей рабочей директории
    from pathlib import Path as _Path
    _BASE = _Path(__file__).resolve().parent
    load_dotenv(dotenv_path=str(_BASE / ".env"))
except Exception:
    # Если модуль не установлен – просто продолжаем (ожидаем, что переменные заданы в ОС)
    pass
# === / .env ===



# Абсолютные пути к статике (как у вас)
BRAND_STATIC_DIR = r"C:\Users\PTambulatov\Desktop\PT\2.Расчеты\50. ХАКАТОН\clip_checker\app\static"
IDENTITY_STATIC_DIR = r"C:\Users\PTambulatov\Desktop\PT\2.Расчеты\50. ХАКАТОН\13. Айдентика"

# ---------- PREVIEW_PATH с безопасным fallback ----------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREVIEW_PATH = os.environ.get("PREVIEW_PATH") or str(BASE_DIR / "Clipchecker_materials" / "preview")
os.makedirs(DEFAULT_PREVIEW_PATH, exist_ok=True)
os.makedirs(os.path.join(DEFAULT_PREVIEW_PATH, "Clipchecker_materials"), exist_ok=True)

def get_preview_path() -> str:
    path = os.environ.get("PREVIEW_PATH") or DEFAULT_PREVIEW_PATH
    try:
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "Clipchecker_materials"), exist_ok=True)
    except Exception:
        pass
    return path

# ---------------------- Утилиты чтения/форматирования ------------------------

def read_file(filepath):
    try:
        if not os.path.exists(filepath):
            return f"[Файл не найден] {os.path.basename(filepath)}"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            pass
        for enc in ("cp1251", "cp1252", "latin-1"):
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[Ошибка чтения {os.path.basename(filepath)}: {e}]"

def read_json(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _write_json(filepath, data):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ _write_json: {e}")

def _append_runtime_log(preview_path: str, text: str):
    try:
        result_dir = os.path.join(preview_path, "Clipchecker_materials")
        os.makedirs(result_dir, exist_ok=True)
        log_path = os.path.join(result_dir, "clipchecker.log")
        ts = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
    except Exception as e:
        print(f"⚠️ append log failed: {e}")

def _sanitize_model_text(raw: str, highlight_risk: bool = False) -> str:
    if not raw:
        return ""
    t = raw
    # Вытаскиваем «text='...'» из SDK-объектов, если попался сырой дамп
    if "GPTModelResult" in t or "Alternative(" in t or "Message(" in t:
        m = re.search(r"text='(.*?)'", t, flags=re.DOTALL)
        if not m:
            m = re.search(r'text="(.*?)"', t, flags=re.DOTALL)
        if m:
            t = m.group(1)

    # Нормализуем переводы строк/табуляции и убираем **жирный** из markdown
    t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("**", "")

    bullet_re  = re.compile(r'^\s*(?:—|\-|\•|\*|\d+[)\.])\s+')
    heading_re = re.compile(
        r'^\s*(?:(?:итог(?: по.*?)?|итого|вывод|заключение|сводка|требования|проблемы|нарушения|'
        r'рекомендац(?:ии|ия)|риск(?:.*)?|продукт(?:.*)?|документ(?:.*)?|маркировк(?:.*)?|'
        r'набивк(?:.*)?|элементы|рамк(?:.*)?|музыка|озвучка|текст|лица|кадры)(:)?\s*$)|.+:\s*$',
        re.IGNORECASE
    )

    # Ключевые индикаторы риска/проблем (минимальный, но практичный набор)
    risk_re = re.compile(
        r'(?:❌|⚠️|\[FAIL\]|\[RISK\]|'
        r'риск|проблем|нарушен|несоответ|ошибк|запрет|не\s+допускается|'
        r'нет\s+согласия|нет\s+прав|нет\s+лиценз|18\+|алкогол|табак|'
        r'экстрем|оскорб|дискрим|ненорматив|опасн|трудночит|мало\s*контраст|'
        r'слишком\s*мелк|контраст\s*<)'
        , re.IGNORECASE
    )

    lines_in  = t.split("\n")
    out_lines = []
    in_bullet_block = False
    for line in lines_in:
        raw_line = line.rstrip()
        is_bullet  = bool(bullet_re.match(raw_line))
        is_heading = bool(heading_re.match(raw_line)) and not is_bullet

        safe_line = html.escape(raw_line)

        if is_heading and safe_line.strip():
            safe_line = f"<strong>{safe_line}</strong>"

        # Подсветка рисков только если явно включено
        if highlight_risk and raw_line.strip() and risk_re.search(raw_line):
            safe_line = f"<span class=\"risk\">{safe_line}</span>"

        if is_bullet:
            in_bullet_block = True
            out_lines.append(safe_line)
            continue

        if in_bullet_block and not is_bullet:
            while out_lines and out_lines[-1] == "":
                out_lines.pop()
            out_lines.append("")
            in_bullet_block = False

        out_lines.append(safe_line)

    return "\n".join(out_lines)

# ---------- Прокси/Verify ----------
def _resolve_proxies_from_env():
    for k in ("OPENAI_PROXY_URL", "OPENAI_PROXY", "HTTPS_PROXY", "https_proxy",
              "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(k)
        if v:
            return v
    return ""

def _resolve_verify_from_env():
    v = os.environ.get("REQUESTS_VERIFY") or os.environ.get("OPENAI_VERIFY")
    if not v:
        return True
    val = str(v).strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if os.path.exists(v):
        return v
    return True

# ---------------------- OpenAI SDK ----------------------
_openai_client_cache = None
REASONING_MODELS = {"o3-mini", "o4-mini"}

def _build_openai_client():
    global _openai_client_cache
    if _openai_client_cache is not None:
        return _openai_client_cache

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY отсутствует в окружении")

    proxy_url = _resolve_proxies_from_env()
    verify    = _resolve_verify_from_env()

    try:
        import httpx
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"Не установлен пакет openai/httpx: {e}")

    httpx_kwargs = {"verify": verify, "timeout": 60.0}
    if proxy_url:
        httpx_kwargs["proxy"] = proxy_url
    http_client = httpx.Client(**httpx_kwargs)

    org     = os.environ.get("OPENAI_ORG")
    project = os.environ.get("OPENAI_PROJECT")

    _openai_client_cache = OpenAI(
        api_key=api_key,
        organization=org if org else None,
        project=project if project else None,
        http_client=http_client
    )
    return _openai_client_cache

# ---------- Нормализация под Responses (input=...) ----------
def _ensure_responses_input(input_messages: list) -> list:
    """
    Преобразует внутренний формат сообщений в допустимый для /v1/responses:
    role/user/assistant + content со списком input_* блоков.
    """
    out = []
    allowed_input_types = {"input_text", "input_image", "input_file", "computer_screenshot", "summary_text"}
    for m in input_messages or []:
        role = m.get("role", "user")
        content = m.get("content", [])
        parts = []
        if isinstance(content, str):
            parts = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    parts.append({"type": "input_text", "text": str(c)})
                    continue
                t = c.get("type")
                if t in ("text", "output_text", "input_text"):
                    parts.append({"type": "input_text", "text": c.get("text", "")})
                elif t in allowed_input_types:
                    if t == "input_image":
                        url = c.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if url:
                            parts.append({"type": "input_image", "image_url": url})
                    else:
                        parts.append(c)
                else:
                    parts.append({"type": "input_text", "text": c.get("text", "")})
        else:
            parts = [{"type": "input_text", "text": str(content)}]
        out.append({"role": role, "content": parts})
    return out

def _sdk_responses_call(input_messages: list, max_output_tokens: int = 400, temperature: float = 0.2):
    """
    Основной вызов: /v1/responses с параметром input (НЕ messages).
    """
    client = _build_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    normalized_input = _ensure_responses_input(input_messages)

    payload = {
        "model": model,
        "input": normalized_input,
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
    }
    if model in REASONING_MODELS:
        payload["reasoning"] = {"effort": os.environ.get("OPENAI_REASONING_EFFORT", "medium")}

    resp = client.responses.create(**payload)

    # Попробуем вытащить текст
    try:
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return {"output_text": text, "raw": resp}
    except Exception:
        pass

    try:
        out = getattr(resp, "output", None)
        if out and isinstance(out.content, list):
            for c in out.content:
                if getattr(c, "type", "") in ("output_text", "message"):
                    t = getattr(c, "text", "") or ""
                    if t.strip():
                        return {"output_text": t, "raw": resp}
    except Exception:
        pass

    return {"output_text": "", "raw": resp}

# ---------- Fallback: Chat Completions ----------
def _to_chat_completions_messages(input_messages: list) -> list:
    out = []
    for m in input_messages or []:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for c in content:
                if not isinstance(c, dict): 
                    continue
                t = c.get("type")
                if t in ("input_text", "output_text"):
                    parts.append({"type": "text", "text": c.get("text", "")})
                # input_image не добавляем — мы и так режем вложения на повторных сообщениях
            out.append({"role": role, "content": parts if parts else [{"type":"text","text":""}]})
        elif isinstance(content, str):
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
        else:
            out.append({"role": role, "content": [{"type": "text", "text": ""}]})
    return out

def _sdk_chat_completions_fallback(input_messages: list, max_output_tokens: int = 400, temperature: float = 0.2):
    client = _build_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    messages_cc = _to_chat_completions_messages(input_messages)
    resp = client.chat.completions.create(
        model=model,
        messages=messages_cc,
        temperature=float(temperature),
        max_tokens=int(max_output_tokens),
    )
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = json.dumps(resp.__dict__, ensure_ascii=False)[:5000]
    return {"output_text": text, "raw": resp}

# --------------------- Сбор данных для result.html ----------------------------
def _collect_usage_status(preview_path: str):
    def _first_day_of_month(d: date) -> date:
        return d.replace(day=1)

    def _fetch_openai_usage_month_and_today():
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None, None, "OPENAI_API_KEY is missing"
        try:
            import httpx
            verify = _resolve_verify_from_env()
            proxy  = _resolve_proxies_from_env() or None
            httpx_kwargs = {"verify": verify, "timeout": 15.0}
            if proxy:
                httpx_kwargs["proxy"] = proxy
            with httpx.Client(**httpx_kwargs) as s:
                end_dt = datetime.utcnow().date()
                start_dt = _first_day_of_month(end_dt)
                r = s.get(
                    "https://api.openai.com/v1/usage",
                    params={"start_date": start_dt.strftime("%Y-%m-%d"),
                            "end_date":   end_dt.strftime("%Y-%m-%d")},
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if r.status_code != 200:
                    return None, None, f"HTTP {r.status_code}"
                data = r.json()
                used_month = data.get("total_usage", None)
                used_day   = None
                return used_month, used_day, None
        except Exception as e:
            return None, None, str(e)

    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    openai_snap = read_json(os.path.join(result_dir, "OpenAI_Tokens_Status.json")) or {}
    yandex_snap = read_json(os.path.join(result_dir, "YandexGPT_Tokens_Status.json")) or {}

    openai_month_used = openai_snap.get("used_month")
    openai_day_used   = openai_snap.get("used_day")
    openai_error      = openai_snap.get("error")

    if openai_month_used is None and openai_day_used is None:
        m, d, err = _fetch_openai_usage_month_and_today()
        openai_month_used = m
        openai_day_used   = d
        openai_error      = err

    def _int_env(k):
        try: return int(os.environ.get(k)) if os.environ.get(k) else None
        except: return None

    return {
        "openai_usage": {
            "budget_month": _int_env("OPENAI_BUDGET_MONTH_TOKENS"),
            "budget_day": _int_env("OPENAI_BUDGET_DAY_TOKENS"),
            "used_month": openai_month_used,
            "used_day": openai_day_used,
            "error": openai_error
        },
        "yandex_usage": {
            "budget_month": _int_env("YANDEX_BUDGET_MONTH_TOKENS"),
            "budget_day": _int_env("YANDEX_BUDGET_DAY_TOKENS"),
            "used_month": (yandex_snap.get("used_month") if yandex_snap else None),
            "used_day":   (yandex_snap.get("used_day") if yandex_snap else None)
        }
    }

def _collect_results_for_template(preview_path: str):
    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    files_to_read = {
        "document_texts": "Documents_Texts.txt",
        "speech_fasterwhisper": "Speech.txt",
        "speech_yandex": "Speech_Yandex.txt",
        "music_info": "Music.txt",
        "music_shazam": "Music_Shazam.txt",
        "ocr_text": "All_Frames_Text.txt",
        "overlay_report": "Overlay_Report.txt",
        "face_recognition": "Face_Recognition_Results.txt",
        "broadcast_params": "Broadcast_video_param_with_description.txt",
        "buyer_experience": "Buyer_Experience.txt",
        "legal_analysis_full": "Comprehensive_Legal_Review.txt",
        "legal_analysis": "Legal_Analysis.txt",
        "tech_validation_json": "Tech_Validation_Result.json",
        "openai_review": "Comprehensive_Legal_Review_OpenAI.txt",
    }
    results = {}
    tech_json_path = os.path.join(result_dir, "Tech_Validation_Result.json")
    if os.path.exists(tech_json_path):
        try:
            with open(tech_json_path, encoding="utf-8") as f:
                results["tech_validation_data"] = json.load(f)
        except Exception:
            results["tech_validation_data"] = []
    else:
        results["tech_validation_data"] = []

    for key, filename in files_to_read.items():
        if key in ("tech_validation_json",):
            continue
        p = os.path.join(result_dir, filename)
        results[key] = read_file(p)

    log_path = os.path.join(result_dir, "clipchecker.log")
    results["runtime_log"] = read_file(log_path) if os.path.exists(log_path) else ""
    results["analysis_done"] = os.path.exists(os.path.join(result_dir, ".done"))

    if results.get("legal_analysis_full"):
        results["legal_analysis_full"] = Markup(_sanitize_model_text(results["legal_analysis_full"], highlight_risk=False))
    if results.get("openai_review"):
        results["openai_review"] = Markup(_sanitize_model_text(results["openai_review"], highlight_risk=True))

    approval_category = os.environ.get("APPROVAL_CATEGORY", "")
    results["approval"] = evaluate_approval_view_model(
        build_approval_view_model(approval_category),
        result_dir,
    )

    results.update(_collect_usage_status(preview_path))
    return results

# -------------------------- Запуск анализа -----------------------------------
def _launch_analysis_background():
    python_exe = sys.executable or "python"
    env = os.environ.copy()
    launch_time = time.time()
    process = subprocess.Popen([python_exe, "run_analysis.py"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, shell=False, cwd=BASE_DIR)
    return process, launch_time


def _wait_for_analysis_start(preview_path: str, process, launch_time: float, timeout_sec: float = 15.0):
    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    log_path = os.path.join(result_dir, "clipchecker.log")
    status_path = os.path.join(result_dir, "run_status.json")
    started_at = time.time()

    while time.time() - started_at < timeout_sec:
        if process.poll() is not None:
            break
        if os.path.exists(log_path) and os.path.getmtime(log_path) >= launch_time:
            return
        if os.path.exists(status_path) and os.path.getmtime(status_path) >= launch_time:
            return
        time.sleep(0.1)

# ------------------------------- Роуты ---------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            preview_path   = request.form['preview']
            docs_path      = request.form['docs']
            broadcast_path = request.form['broadcast']
            approval_category = request.form.get('approval_category', '').strip()
            openai_model   = request.form.get('openai_model', '').strip()

            os.environ['PREVIEW_PATH']   = preview_path
            os.environ['DOCUMENTS_PATH'] = docs_path
            os.environ['BROADCAST_PATH'] = broadcast_path
            os.environ['APPROVAL_CATEGORY'] = approval_category
            if openai_model:
                os.environ['OPENAI_MODEL'] = openai_model

            process, launch_time = _launch_analysis_background()
            _wait_for_analysis_start(preview_path, process, launch_time)
            results = _collect_results_for_template(get_preview_path())
            return render_template("result.html", **results)
        except Exception as e:
            return f"<h2>❌ Ошибка при обработке:</h2><pre>{e}</pre>"
    return render_template("form.html", approval_categories=load_approval_categories())

@app.route("/pdf")
def serve_pdf():
    preview_path = get_preview_path()
    return send_from_directory(os.path.join(preview_path, "Clipchecker_materials"), "Annotated_Frames.pdf")

@app.route("/faces.pdf")
def serve_faces_pdf():
    preview_path = get_preview_path()
    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    for fname in ["Faces_Thumbnails.pdf", "Faces_Strip.pdf"]:
        fpath = os.path.join(result_dir, fname)
        if os.path.exists(fpath):
            return send_from_directory(result_dir, fname)
    return "❌ Файл с лицами не найден (ожидались Faces_Thumbnails.pdf или Faces_Strip.pdf)", 404

@app.route("/brand/<path:fname>")
def serve_brand_file(fname):
    try:
        return send_from_directory(BRAND_STATIC_DIR, fname)
    except Exception:
        return "404", 404

@app.route("/identity/<path:fname>")
def serve_identity_file(fname):
    try:
        return send_from_directory(IDENTITY_STATIC_DIR, fname)
    except Exception:
        return "404", 404

@app.route("/stream-logs")
def stream_logs():
    preview_path = get_preview_path()
    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    log_path   = os.path.join(result_dir, "clipchecker.log")
    done_flag  = os.path.join(result_dir, ".done")

    def generate():
        while not os.path.exists(log_path) and not os.path.exists(done_flag):
            time.sleep(0.3)
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {line.rstrip()}\n\n"
                    else:
                        if os.path.exists(done_flag):
                            yield "event: done\ndata: done\n\n"
                            break
                        time.sleep(0.5)
        except GeneratorExit:
            return
        except Exception as e:
            yield f"data: [stream error] {e}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/status")
def status():
    preview_path = get_preview_path()
    done = os.path.exists(os.path.join(preview_path, "Clipchecker_materials", ".done"))
    return jsonify({"done": done})

@app.route("/log.txt")
def download_log():
    preview_path = get_preview_path()
    return send_from_directory(os.path.join(preview_path, "Clipchecker_materials"), "clipchecker.log", as_attachment=True)

# ---------------------- Кадры для чата ---------------------------------------
def _chat_frames_config():
    try:
        step = int(os.environ.get("OPENAI_CHAT_FRAMES_STEP", "3"))
    except Exception:
        step = 3
    try:
        # дефолт ещё ниже, чтобы исключить 429
        limit = int(os.environ.get("OPENAI_CHAT_FRAMES_MAX", "4"))
    except Exception:
        limit = 4
    step = max(1, min(step, 30))
    limit = max(1, min(limit, 200))
    return step, limit

def _gather_frames_for_chat(preview_path: str):
    import base64, mimetypes
    result_dir = os.path.join(preview_path, "Clipchecker_materials")
    frames_dir = os.path.join(result_dir, "frames")
    step, limit = _chat_frames_config()

    info = {"frames_dir": frames_dir, "found_total": 0, "selected_count": 0, "selected": [], "errors": []}
    image_blocks = []

    if not os.path.isdir(frames_dir):
        info["errors"].append({"stage": "check_dir", "error": "frames dir not found"})
        return image_blocks, info

    try:
        all_files = sorted([f for f in os.listdir(frames_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
        info["found_total"] = len(all_files)
        if not all_files:
            return image_blocks, info

        selected_paths = []
        for i in range(0, len(all_files), step):
            selected_paths.append(os.path.join(frames_dir, all_files[i]))
        if len(selected_paths) > limit:
            selected_paths = selected_paths[:limit]

        for p in selected_paths:
            try:
                mime = (mimetypes.guess_type(p)[0]) or "image/jpeg"
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                image_blocks.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
                info["selected"].append(os.path.basename(p))
            except Exception as e:
                info["errors"].append({"file": p, "error": f"{type(e).__name__}: {e}"})

        info["selected_count"] = len(info["selected"])
        return image_blocks, info
    except Exception as e:
        info["errors"].append({"stage": "listdir", "error": f"{type(e).__name__}: {e}"})
        return image_blocks, info

def _history_has_user_images(messages):
    try:
        for m in messages or []:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "input_image":
                        return True
    except Exception:
        pass
    return False

# -------------------------- ЧАТ С CHATGPT ------------------------------------
@app.route("/chat/history")
def chat_history():
    try:
        preview_path = get_preview_path()
        path = os.path.join(preview_path, "Clipchecker_materials", "OpenAI_Chat_History.json")
        data = read_json(path) or {}
        msgs = data.get("messages", [])
        return jsonify({"ok": True, "messages": msgs[-50:]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/chat/send", methods=["POST"])
def chat_send():
    try:
        preview_path = get_preview_path()

        path = os.path.join(preview_path, "Clipchecker_materials", "OpenAI_Chat_History.json")
        hist = read_json(path) or {"model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "messages": []}
        messages = hist.get("messages", [])

        user_text = (request.form.get("message") or "").strip()
        if not user_text and "files" not in request.files:
            return jsonify({"ok": False, "error": "Пустой запрос"}), 400

        # ---------- Разрешаем вложения и кадры ТОЛЬКО в первом сообщении ----------
        prior_user_msgs = sum(1 for m in messages if m.get("role") == "user")
        allow_attachments = (prior_user_msgs == 0)

        user_blocks = []
        if user_text:
            user_blocks.append({"type": "input_text", "text": user_text})

        uploads_dir = os.path.join(preview_path, "Clipchecker_materials", "chat_uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        attached_names = []
        if allow_attachments:
            files = request.files.getlist("files")
            for f in files:
                if not f or not getattr(f, "filename", ""):
                    continue
                fname = f.filename
                lower = fname.lower()
                save_path = os.path.join(uploads_dir, fname)
                f.save(save_path)
                attached_names.append(fname)

                if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    import base64, mimetypes
                    mime = (mimetypes.guess_type(save_path)[0]) or "image/jpeg"
                    with open(save_path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("ascii")
                    user_blocks.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
                    continue

                if lower.endswith((".txt", ".md", ".json", ".csv")):
                    try:
                        with open(save_path, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        if len(content) > 200000:
                            content = content[:200000] + "\n...[обрезано]"
                        user_blocks.append({"type": "input_text", "text": f"[Вложение: {fname}]\n{content}"})
                    except Exception as e:
                        user_blocks.append({"type": "input_text", "text": f"[Вложение: {fname}] (не удалось прочитать: {e})"})
                else:
                    user_blocks.append({"type": "input_text", "text": f"[Вложение: {fname}] (тип файла пока не поддержан для прямого анализа)"})

        # Кадры из frames — один раз за сессию и только в первом сообщении
        frames_sent_once = bool(hist.get("frames_sent_once")) or _history_has_user_images(messages)
        frames_info = {"found_total": 0, "selected_count": 0, "selected": [], "errors": []}
        if allow_attachments and not frames_sent_once:
            image_blocks, frames_info = _gather_frames_for_chat(preview_path)
            if frames_info["selected_count"] > 0:
                header = f"Встроенные кадры из папки frames (отобрано {frames_info['selected_count']} из {frames_info['found_total']})."
                user_blocks.append({"type": "input_text", "text": header})
                user_blocks.extend(image_blocks)
                # Помечаем сразу, чтобы при любой ошибке не слать их повторно
                hist["frames_sent_once"] = True
                _write_json(path, hist)

        # Системное сообщение (если истории ещё нет)
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": [{"type": "input_text", "text":
                "Ты делаешь предварительный скрининг рекламных материалов для ТВ. Отвечай по-русски, по чек-листу, "
                "без дисклеймеров и повторов. Если данных мало — перечисли, чего не хватает. "
                "Форматируй вывод компактно, с подзаголовками."
            }]})

        messages.append({"role": "user", "content": user_blocks})

        # 1) Responses API (input=...)
        try:
            resp = _sdk_responses_call(messages, max_output_tokens=400, temperature=0.2)
            reply_text = (resp.get("output_text") or "").strip() or "⚠️ Пустой ответ модели."
        except Exception as e1:
            # 2) Fallback: Chat Completions (только текст)
            try:
                resp2 = _sdk_chat_completions_fallback(messages, max_output_tokens=400, temperature=0.2)
                reply_text = (resp2.get("output_text") or "").strip() or "⚠️ Пустой ответ модели."
            except Exception as e2:
                _append_runtime_log(preview_path, f"⛔ Ошибка OpenAI (chat): {e1} | fallback: {e2}")
                return jsonify({"ok": False, "error": f"{e1}\nFallback error: {e2}"}), 502

        messages.append({"role": "assistant", "content": [{"type": "input_text", "text": reply_text}]})

        # Зафиксируем флаг отправки кадров
        hist["frames_sent_once"] = True if hist.get("frames_sent_once") else frames_info.get("selected_count", 0) > 0

        # Обрезаем историю
        if len(messages) > 60:
            messages = messages[-60:]

        hist["messages"] = messages
        _write_json(path, hist)

        if not allow_attachments and request.files:
            _append_runtime_log(preview_path, "ℹ️ Повторное сообщение: вложения и кадры проигнорированы (контекст сохранён).")
        else:
            _append_runtime_log(preview_path, "✅ ChatGPT: сообщение отправлено.")

        return jsonify({
            "ok": True,
            "reply": reply_text,
            "messages": messages,
            "attached": attached_names if allow_attachments else [],
            "frames_used": {
                "found_total": frames_info.get("found_total", 0),
                "selected_count": frames_info.get("selected_count", 0),
                "selected_sample": frames_info.get("selected", [])[:20],
                "reused": (not allow_attachments) or frames_sent_once or bool(hist.get("frames_sent_once"))
            },
            "attachments_ignored": (not allow_attachments) and bool(request.files)
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
