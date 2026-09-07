"""Tao QdrantClient dung cho ca hai che do.

- Cloud  : co QDRANT_URL + QDRANT_API_KEY  -> ket noi toi cluster tren mang.
- Nhung  : co QDRANT_PATH                  -> Qdrant chay nhung trong tien trinh
           Python, luu ra thu muc tren dia. Khong can Docker, khong can tai khoan.

Cung mot thu vien qdrant-client, cung API, chi khac cho luu tru. Nho vay
build_index.py va rag_agent.py khong phai biet dang chay che do nao.

LUU Y che do nhung: thu muc du lieu bi khoa boi mot tien trinh duy nhat.
Khong chay build_index.py va uvicorn cung luc duoc.
"""
from __future__ import annotations

from qdrant_client import QdrantClient

from app import config


def is_local() -> bool:
    """Co dung Qdrant nhung khong."""
    return bool(config.QDRANT_PATH)


def describe() -> str:
    if is_local():
        return f"Qdrant nhung (local) tai {config.QDRANT_PATH}"
    return f"Qdrant Cloud tai {config.QDRANT_URL}"


def get_client(timeout: int = 60) -> QdrantClient:
    if is_local():
        config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(config.QDRANT_PATH))
    try:
        config.require("QDRANT_URL", "QDRANT_API_KEY")
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n\nHoac dung Qdrant nhung, khong can tai khoan: "
            "dat QDRANT_PATH=data/qdrant trong .env"
        ) from exc
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=timeout)
