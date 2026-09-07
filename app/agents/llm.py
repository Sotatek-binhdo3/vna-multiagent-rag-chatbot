"""Factory LLM Gemini + tien ich parse JSON tra ve tu LLM."""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openrouter import OpenRouter

from app import config

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.2, model: str | None = None):
    """Tra ve LLM cho buoc chat.

    Neu USE_OPENROUTER=1 thi di qua OpenRouter (dung duoc dung model de bai chi
    dinh, khong dinh gioi han 20 request/ngay cua Gemini free tier).
    Nguoc lai goi thang Gemini. Embedding KHONG di qua day.
    """
    if config.USE_OPENROUTER:
        config.require("OPENROUTER_API_KEY")
        return OpenRouter(
            api_key=config.OPENROUTER_API_KEY,
            model=model or config.CHAT_MODEL,
            temperature=temperature,
            max_tokens=4096,
        )
    config.require("GEMINI_API_KEY")
    return GoogleGenAI(
        model=model or config.CHAT_MODEL,
        api_key=config.GEMINI_API_KEY,
        temperature=temperature,
    )


# Gemini free tier hay tra 503 (qua tai) hoac 429 (rate limit) theo tung dot.
# Khong retry thi dang demo bi dut giua chung.
_RETRYABLE = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand")
_MAX_ATTEMPTS = 4


def chat(system: str, user: str, temperature: float = 0.2, model: str | None = None) -> str:
    llm = get_llm(temperature, model)
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = llm.chat(messages)
            return (resp.message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - can phan loai theo noi dung loi
            last_exc = exc
            if not any(tok in str(exc) for tok in _RETRYABLE):
                raise
            if attempt < _MAX_ATTEMPTS - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                print(f"[llm] {type(exc).__name__} tam thoi, thu lai sau {wait}s "
                      f"({attempt + 2}/{_MAX_ATTEMPTS})", flush=True)
                time.sleep(wait)
    raise RuntimeError(
        f"LLM ({config.PROVIDER}) khong phan hoi sau {_MAX_ATTEMPTS} lan thu: {last_exc}"
    ) from last_exc


def chat_json(
    system: str, user: str, fallback: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """Goi LLM va ep ve dict. Neu parse hong thi tra fallback thay vi crash."""
    raw = chat(system + "\n\nCHI tra ve JSON hop le, khong kem giai thich.", user, 0.0, model)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return fallback
