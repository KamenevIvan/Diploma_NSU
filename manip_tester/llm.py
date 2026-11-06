"""Слой доступа к языковым моделям.

Все агенты общаются с моделью ТОЛЬКО через метод `chat(messages, ...) -> str`.
Чтобы подключить новую нейросеть, достаточно:
  (а) указать другой base_url/model в конфиге (любой OpenAI-совместимый сервер:
      Ollama, vLLM, llama.cpp server, LM Studio, облачные API), либо
  (б) написать свой класс с методом `chat` (см. MockBackend как минимальный пример).
"""
<<<<<<< HEAD
=======


>>>>>>> 26976c2 (First real prototype)
from __future__ import annotations

import itertools
import os
import re
import time
from typing import Iterable, Protocol

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLMBackend(Protocol):
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Убирает блоки рассуждений <think>...</think> (их выдают, например, модели Qwen3)."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text.lower():  # блок оборван по max_tokens
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


class OpenAICompatibleBackend:
    """Универсальный клиент для любого сервера с OpenAI-совместимым API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        use_json_mode: bool = True,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("Установите клиент: pip install openai") from e
        key = api_key or os.getenv(api_key_env) or "not-needed"  # локальным серверам ключ не нужен
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model
        self.use_json_mode = use_json_mode
        self.max_retries = max_retries

    def chat(self, messages, *, temperature=0.7, max_tokens=512, json_mode=False) -> str:
        # Если у конкретной модели другие имена параметров (напр. max_completion_tokens
        # у части облачных моделей) - правится только здесь.
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode and self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # сеть, rate limit и т.п.
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM-вызов не удался после ретраев: {last_err}") from last_err


class MockBackend:
    """Заглушка без нейросети: по кругу отдаёт заготовленные ответы. Для отладки пайплайна."""

    def __init__(self, responses: Iterable[str]):
        self._it = itertools.cycle(list(responses))
        self.calls: list[list[Message]] = []

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return next(self._it)


def make_backend(spec: dict) -> LLMBackend:
    """Строит бэкенд из словаря-конфига (удобно хранить конфиг по ролям)."""
    provider = spec.get("provider", "openai_compatible")
    if provider == "mock":
        return MockBackend(spec["responses"])
    if provider == "openai_compatible":
        return OpenAICompatibleBackend(
            model=spec["model"],
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key"),
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            use_json_mode=spec.get("use_json_mode", True),
        )
    raise ValueError(f"Неизвестный provider: {provider}")
