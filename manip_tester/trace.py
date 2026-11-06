"""Логирование прогонов.

В папку прогона пишутся два файла:
  trace.jsonl     - машиночитаемые события (из них считаются метрики и делается разметка)
  transcript.log  - человекочитаемый журнал: "дата-время | эпизод | [роль] сообщение"
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TraceLogger:
    def __init__(self, run_dir: str | Path, echo: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._jsonl = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        self._text = open(self.dir / "transcript.log", "a", encoding="utf-8")

    def event(self, type_: str, **data) -> None:
        rec = {"ts": _now(), "type": type_, **data}
        self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def say(self, line: str) -> None:
        line = f"{_now()} | {line}"
        self._text.write(line + "\n")
        self._text.flush()
        if self.echo:
            print(line)

    def close(self) -> None:
        self._jsonl.close()
        self._text.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
