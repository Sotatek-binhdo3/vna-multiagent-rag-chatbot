"""Bo nho hoi thoai trong RAM.

Du cho POC: giu lich su chat va ket qua RAG gan nhat de xu ly follow-up.
Neu len production thi thay bang Redis/DB, interface giu nguyen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_TURNS = 20


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    # Ket qua RAG cua luot hoi tai lieu gan nhat -> dung lai khi user chi muon
    # dien dat lai (vd "tom tat ngan hon") ma khong can retrieve lai.
    last_rag: dict[str, Any] | None = None

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_TURNS * 2:
            self.messages = self.messages[-MAX_TURNS * 2 :]

    def history(self) -> list[dict[str, str]]:
        return list(self.messages)


_STORE: dict[str, Session] = {}


def get(session_id: str) -> Session:
    if session_id not in _STORE:
        _STORE[session_id] = Session(session_id=session_id)
    return _STORE[session_id]


def reset(session_id: str) -> None:
    _STORE.pop(session_id, None)
