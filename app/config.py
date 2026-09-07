"""Tap trung cau hinh doc tu .env cho toan bo POC."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
# Model nhe cho phan loai intent / query rewrite. Quota free tier tinh RIENG cho
# tung model, nen tach ra giup tang so cau hoi phuc vu duoc moi ngay.
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-flash-lite-latest")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = _int("EMBED_DIM", 1536)

# --- OpenRouter (gateway cho chat LLM) ---
# Bat USE_OPENROUTER=1 thi cac loi goi CHAT di qua OpenRouter.
# Embedding luon goi thang Gemini vi OpenRouter khong phuc vu embedding.
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "0").strip() in ("1", "true", "True", "yes")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "google/gemini-2.5-flash-lite")

# --- Model thuc te dung cho chat, tuy theo nha cung cap dang bat ---
CHAT_MODEL = OPENROUTER_MODEL if USE_OPENROUTER else GEMINI_CHAT_MODEL
FAST_MODEL = OPENROUTER_FAST_MODEL if USE_OPENROUTER else GEMINI_FAST_MODEL
PROVIDER = "openrouter" if USE_OPENROUTER else "gemini"

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vna_annual_report_2024")
# Dat QDRANT_PATH de chay Qdrant nhung trong tien trinh (khong can Docker/tai khoan).
# De trong thi dung Qdrant Cloud qua QDRANT_URL.
_qp = os.getenv("QDRANT_PATH", "").strip()
QDRANT_PATH = (ROOT / _qp).resolve() if _qp else None

# --- Cohere ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

# --- Source document ---
SOURCE_PDF = (ROOT / os.getenv("SOURCE_PDF", "../RAG-test/2024.pdf")).resolve()
SOURCE_NAME = os.getenv("SOURCE_NAME", SOURCE_PDF.name)

# --- Retrieval ---
RETRIEVE_TOP_K = _int("RETRIEVE_TOP_K", 20)
RERANK_TOP_N = _int("RERANK_TOP_N", 5)
RERANK_MIN_SCORE = _float("RERANK_MIN_SCORE", 0.05)

# --- Duong dan artifact cua buoc ingest ---
DATA_DIR = ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"        # anh render tung trang, dung cho preview citation
PARSED_DIR = DATA_DIR / "parsed"      # text + metadata da trich xuat
PARSED_JSONL = PARSED_DIR / "pages.jsonl"
WEB_DIR = ROOT / "web"


def require(*names: str) -> None:
    """Bao loi som va ro rang neu thieu key, thay vi de thu vien nem loi kho hieu."""
    missing = [
        n for n in names
        if not globals().get(n) or str(globals().get(n)).startswith("your_")
    ]
    if missing:
        raise RuntimeError(
            "Thieu bien moi truong: " + ", ".join(missing)
            + f"\nHay dien vao {ROOT / '.env'} (xem .env.example)."
        )
