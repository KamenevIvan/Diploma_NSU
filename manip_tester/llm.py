"""Слой доступа к языковым моделям.

Все агенты общаются с моделью ТОЛЬКО через метод `chat(messages, ...) -> str`.
Чтобы подключить новую нейросеть, достаточно:
  (а) указать другой base_url/model в конфиге (любой OpenAI-совместимый сервер:
      Ollama, vLLM, llama.cpp server, LM Studio, облачные API), либо
  (б) написать свой класс с методом `chat` (см. MockBackend как минимальный пример).
"""
from __future__ import annotations

import itertools
import os
import re
import threading
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


class _Throttle:
    """Минимальный интервал между запросами. Общий для всех ролей с одним ключом/провайдером."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delta = self.min_interval - (time.monotonic() - self._last)
            if delta > 0:
                time.sleep(delta)
            self._last = time.monotonic()


_THROTTLES: dict[str, _Throttle] = {}
_THROTTLES_LOCK = threading.Lock()


def _get_throttle(key: str, min_interval: float) -> _Throttle:
    with _THROTTLES_LOCK:
        t = _THROTTLES.get(key)
        if t is None:
            t = _THROTTLES[key] = _Throttle(min_interval)
        else:
            t.min_interval = max(t.min_interval, min_interval)
        return t


_NO_RETRY_STATUS = {400, 401, 403, 404, 422}  # ретраить бессмысленно: неверный ключ/модель/запрос
_MAX_SLEEP_ON_429 = 300.0  # если сервер просит ждать дольше (дневной лимит) - не висим, а падаем


class OpenAICompatibleBackend:
    """Универсальный клиент для любого сервера с OpenAI-совместимым API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        use_json_mode: bool = True,
        max_retries: int = 5,
        timeout: float = 120.0,
        min_interval: float = 0.0,
        extra_body: dict | None = None,
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
        self.extra_body = extra_body
        self._throttle = _get_throttle(f"{base_url}|{api_key_env}", min_interval)
        # Накопленный расход: полезно следить за дневными лимитами бесплатных тарифов.
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _retry_delay(err: Exception, attempt: int) -> float:
        if getattr(err, "status_code", None) == 429:
            retry_after = 0.0
            headers = getattr(getattr(err, "response", None), "headers", None)
            if headers is not None:
                try:
                    retry_after = float(headers.get("retry-after", 0))
                except (TypeError, ValueError):
                    retry_after = 0.0
            return max(retry_after, 15.0 * (attempt + 1))
        return float(2**attempt)

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
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle.wait()
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as e:  # сеть, rate limit и т.п.
                last_err = e
                status = getattr(e, "status_code", None)
                if status in _NO_RETRY_STATUS:
                    raise RuntimeError(f"LLM-запрос отклонён сервером ({status}): {e}") from e
                delay = self._retry_delay(e, attempt)
                if status == 429 and delay > _MAX_SLEEP_ON_429:
                    raise RuntimeError(
                        f"Лимит исчерпан, сервер просит подождать {delay:.0f} с "
                        "(вероятно, дневной лимит токенов/запросов)."
                    ) from e
                time.sleep(delay)
                continue
            u = getattr(resp, "usage", None)
            self.usage["calls"] += 1
            if u is not None:
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            return resp.choices[0].message.content or ""
        raise RuntimeError(f"LLM-вызов не удался после {self.max_retries} попыток: {last_err}") from last_err


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
            max_retries=spec.get("max_retries", 5),
            min_interval=spec.get("min_interval", 0.0),
            extra_body=spec.get("extra_body"),
        )
    raise ValueError(f"Неизвестный provider: {provider}")
