import json
import re
import time
import urllib.error
import urllib.request

from openai import OpenAI

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODELS,
)

try:
    from evaluation.config_eval import LLM_MAX_RETRIES, LLM_REQUEST_DELAY
except ImportError:
    LLM_MAX_RETRIES = 6
    LLM_REQUEST_DELAY = 0


def _is_embedding_model(name):
    lower = name.lower()
    return any(token in lower for token in ("embed", "embedding", "bge-m3", "bge_m3"))


def _normalize_model_name(name):
    return str(name or "").strip()


def _model_aliases(name):
    """Các biến thể tên model Ollama (có/không :latest)."""
    name = _normalize_model_name(name)
    if not name:
        return []
    aliases = {name}
    if ":" not in name:
        aliases.add(f"{name}:latest")
    elif name.endswith(":latest"):
        aliases.add(name.rsplit(":", 1)[0])
    return aliases


def _model_available(name, models):
    available = set(models)
    return any(alias in available for alias in _model_aliases(name))


def _pick_available(preferred, models):
    for name in preferred:
        for alias in _model_aliases(name):
            if alias in models:
                return alias
    return None


def _qwen_chat_rank(name):
    """Ưu tiên qwen3 lớn hơn; tránh embedding và qwen3:4b (hay content rỗng)."""
    lower = name.lower()
    if _is_embedding_model(lower) or "qwen" not in lower:
        return -1
    # qwen3:8b / 14b / 32b trước; 4b cuối vì thinking hay làm content=""
    size_score = 0
    for size in ("32b", "14b", "8b", "4b", "1.7b", "0.6b"):
        if size in lower:
            size_score = {
                "32b": 50,
                "14b": 40,
                "8b": 30,
                "4b": 5,
                "1.7b": 3,
                "0.6b": 1,
            }[size]
            break
    if "qwen3" in lower:
        return 100 + size_score
    return 50 + size_score


def is_ollama_available():
    if not OLLAMA_ENABLED:
        return False

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_ollama_models():
    with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [item["name"] for item in payload.get("models", [])]


def resolve_ollama_chat_model():
    """Chọn model chat Ollama: OLLAMA_MODEL → qwen3 lớn → model chat khác."""
    if not is_ollama_available():
        return None

    models = list_ollama_models()

    # 1) Đúng model cấu hình (ví dụ qwen3:8b)
    picked = _pick_available([OLLAMA_MODEL], models)
    if picked:
        return picked

    # 2) Qwen chat theo kích thước (8b trước 4b)
    qwen_models = sorted(
        (m for m in models if _qwen_chat_rank(m) >= 0),
        key=_qwen_chat_rank,
        reverse=True,
    )
    if qwen_models:
        return qwen_models[0]

    # 3) Fallback model chat bất kỳ
    for name in models:
        if not _is_embedding_model(name):
            return name

    return None


def list_ollama_chat_fallbacks(primary=None):
    """Danh sách model dự phòng nếu primary trả content rỗng."""
    if not is_ollama_available():
        return []

    models = list_ollama_models()
    primary = primary or resolve_ollama_chat_model()
    ordered = []

    for name in (OLLAMA_MODEL, "qwen3:8b", "llama3.1", "llama3.2:3b", "qwen3:4b"):
        picked = _pick_available([name], models)
        if picked and picked not in ordered:
            ordered.append(picked)

    for name in sorted(
        (m for m in models if _qwen_chat_rank(m) >= 0),
        key=_qwen_chat_rank,
        reverse=True,
    ):
        if name not in ordered:
            ordered.append(name)

    for name in models:
        if not _is_embedding_model(name) and name not in ordered:
            ordered.append(name)

    if primary and primary in ordered:
        ordered = [primary] + [m for m in ordered if m != primary]

    return ordered


def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY in config.py")

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


def get_ollama_client():
    return OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )


def _is_rate_limit_error(exc):
    text = str(exc).lower()
    return "429" in text or "rate" in text or "rate-limited" in text


def _is_non_retryable_error(exc):
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "resource_exhausted",
            "credits are depleted",
            "invalid api key",
            "authentication",
            "permission denied",
        )
    )


def _parse_retry_after(exc):
    text = str(exc)

    for pattern in (
        r"'retry_after_seconds(?:_raw)?':\s*([\d.]+)",
        r"'Retry-After':\s*'(\d+)'",
        r"retry after (\d+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return max(1, int(float(match.group(1))))

    return None


def _strip_think_tags(text):
    """Loại khối <think>...</think> nếu model vẫn nhét thinking vào content."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_message_text(message):
    """Lấy text trả lời; qwen3 đôi khi để trống content và chỉ có reasoning."""
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return _strip_think_tags(content)

    # OpenAI SDK / Ollama có thể gắn reasoning ngoài content
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            # reasoning thường là chuỗi suy nghĩ, không dùng làm câu trả lời cuối
            break

    try:
        dumped = message.model_dump()
    except Exception:
        dumped = {}

    for key in ("content",):
        value = dumped.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_think_tags(value)

    return ""


def _is_qwen3_model(model):
    return "qwen3" in str(model or "").lower()


def _ollama_native_chat(model, prompt, temperature, max_tokens, think=False):
    """Gọi /api/chat native — hỗ trợ think=false ổn định hơn OpenAI-compat."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if think is not None:
        body["think"] = think

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    message = payload.get("message") or {}
    return _strip_think_tags(message.get("content") or "")


def _ollama_once(client, model, prompt, temperature, max_tokens, disable_think=True):
    # Qwen3: ưu tiên native API để tắt thinking (OpenAI-compat hay bỏ qua)
    if _is_qwen3_model(model):
        try:
            return _ollama_native_chat(
                model,
                prompt,
                temperature,
                max_tokens,
                think=not disable_think if disable_think is not None else None,
            )
        except Exception as exc:
            print(f"[llm_utils] native /api/chat lỗi ({exc}), fallback OpenAI-compat")

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_think and _is_qwen3_model(model):
        kwargs["extra_body"] = {"think": False}

    response = client.chat.completions.create(**kwargs)
    return _extract_message_text(response.choices[0].message)


def _ollama_completion(prompt, temperature, max_tokens, request_delay):
    models = list_ollama_chat_fallbacks()
    if not models:
        raise RuntimeError(
            "Ollama không khả dụng hoặc không có model chat. "
            f"Chạy: ollama pull {OLLAMA_MODEL}"
        )

    client = get_ollama_client()

    if request_delay > 0:
        time.sleep(request_delay)

    last_empty = None
    for model in models:
        print(f"[llm_utils] Ollama local: {model}")

        # 1) Tắt thinking (nhanh, content đầy đủ)
        text = _ollama_once(
            client, model, prompt, temperature, max_tokens, disable_think=True
        )
        if text:
            return text

        # 2) Bật thinking + thêm budget token (qwen3:4b hay cần)
        text = _ollama_once(
            client,
            model,
            prompt,
            temperature,
            max(max_tokens * 2, 1000),
            disable_think=False,
        )
        if text:
            print(f"[llm_utils] {model}: dùng output với thinking")
            return text

        last_empty = model
        print(f"[llm_utils] {model}: content rỗng — thử model khác...")

    raise RuntimeError(
        f"Ollama trả content rỗng với các model: {models}. "
        f"Model cuối: {last_empty}. Thử: ollama pull qwen3:8b"
    )


def _openrouter_completion(prompt, temperature, max_tokens, request_delay):
    client = get_openrouter_client()
    last_exc = None

    for model in OPENROUTER_MODELS:
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                if request_delay > 0:
                    time.sleep(request_delay)

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as exc:
                last_exc = exc

                if _is_non_retryable_error(exc):
                    raise

                if _is_rate_limit_error(exc):
                    wait = _parse_retry_after(exc) or min(60, 5 * attempt)
                    print(
                        f"[llm_utils] 429 {model} — chờ {wait}s "
                        f"(lần {attempt}/{LLM_MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                print(f"[llm_utils] Model {model} lỗi: {exc}")
                break

    raise RuntimeError(
        f"Tất cả model OpenRouter đều lỗi sau {LLM_MAX_RETRIES} lần thử/model. "
        f"Lỗi cuối: {last_exc}"
    )


def _gemini_completion(prompt, temperature, max_tokens, request_delay):
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in config.py")

    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_exc = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            if request_delay > 0:
                time.sleep(request_delay)

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return response.text.strip()

        except Exception as exc:
            last_exc = exc

            if _is_non_retryable_error(exc):
                raise

            if _is_rate_limit_error(exc):
                wait = _parse_retry_after(exc) or min(60, 5 * attempt)
                print(
                    f"[llm_utils] Gemini 429 — chờ {wait}s "
                    f"(lần {attempt}/{LLM_MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            raise

    raise RuntimeError(f"Gemini lỗi sau {LLM_MAX_RETRIES} lần thử. Lỗi cuối: {last_exc}")


def chat_completion(
    prompt,
    temperature=0.2,
    max_tokens=1500,
    provider="auto",
    request_delay=None,
):
    """
    provider:
      - ollama     : chỉ dùng Ollama local
      - openrouter : chỉ dùng OpenRouter
      - gemini     : chỉ dùng Gemini
      - auto       : Ollama (nếu có) → OpenRouter → Gemini
    """
    delay = LLM_REQUEST_DELAY if request_delay is None else request_delay

    if provider == "ollama":
        return _ollama_completion(prompt, temperature, max_tokens, delay)

    if provider == "gemini":
        return _gemini_completion(prompt, temperature, max_tokens, delay)

    if provider == "openrouter":
        return _openrouter_completion(prompt, temperature, max_tokens, delay)

    if is_ollama_available() and resolve_ollama_chat_model():
        try:
            return _ollama_completion(prompt, temperature, max_tokens, 0)
        except Exception as ollama_exc:
            print(f"[llm_utils] Ollama lỗi: {ollama_exc}. Thử cloud...")

    try:
        return _openrouter_completion(prompt, temperature, max_tokens, delay or 8)
    except Exception as openrouter_exc:
        if not GEMINI_API_KEY:
            raise openrouter_exc

        print("[llm_utils] OpenRouter thất bại, chuyển sang Gemini...")
        return _gemini_completion(prompt, temperature, max_tokens, delay)


def extract_json_block(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def translate_to_vietnamese(text, field_name="text", provider="auto", request_delay=None):
    prompt = f"""
Dịch đoạn văn bản y khoa sau sang tiếng Việt.
Giữ nguyên nghĩa, dùng thuật ngữ y khoa phổ biến tại Việt Nam.
Chỉ trả về bản dịch, không giải thích thêm.

Trường: {field_name}
Nội dung:
{text}
"""
    return chat_completion(
        prompt,
        temperature=0.1,
        max_tokens=1200,
        provider=provider,
        request_delay=request_delay,
    )
