"""Buoc 2 cua pipeline ingest.

pages.jsonl -> chunk -> embed bang Gemini -> ghi vao collection Qdrant.

Moi chunk mang theo metadata du de tra citation:
  source_file / page_label / page_index / section / chunk_id / image_file

Chay:  python -m app.ingest.build_index [--recreate] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from google.genai import types
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode, TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app import config, vector_store

# Namespace co dinh de uuid5(chunk_id) luon ra cung mot ket qua giua cac lan chay.
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "vna-annual-report-2024")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
# Tran cung theo ky tu: gemini-embedding-001 gioi han 2048 token/input,
# tieng Viet ~2-3 ky tu/token nen 3000 ky tu la nguong an toan.
MAX_CHUNK_CHARS = 3000
EMBED_BATCH = 16
# Free tier: 100 embedding/phut, dem theo tung van ban. Giu duoi nguong cho an toan.
EMBED_RPM = 90


def load_documents(limit: int | None = None) -> list[Document]:
    if not config.PARSED_JSONL.exists():
        raise SystemExit(
            f"Chua co {config.PARSED_JSONL}. Chay truoc: python -m app.ingest.parse_pdf"
        )
    docs: list[Document] = []
    with config.PARSED_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec["text"].strip():
                continue  # trang thuan anh, khong co text de embed
            metadata = {
                "source_file": config.SOURCE_NAME,
                "page_index": rec["page_index"],
                "page_label": rec["page_label"],
                "section": rec["section"] or "(khong xac dinh)",
                "image_file": rec["image_file"],
                # trang nhieu anh/vector graphic => nhieu kha nang co bieu do, dashboard
                "has_figure": bool(rec["n_images"] > 0 or rec["n_drawings"] > 40),
            }
            docs.append(
                Document(
                    text=rec["text"],
                    metadata=metadata,
                    # image_file chi de UI dung, khong dua vao embedding/LLM
                    excluded_embed_metadata_keys=["image_file", "has_figure"],
                    excluded_llm_metadata_keys=["image_file"],
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Cat cung mot doan qua dai, uu tien cat o ranh gioi dong."""
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.split("\n"):
        # Ban than mot dong cung co the dai hon max_chars -> cat tho theo do dai
        while len(line) > max_chars:
            if buf:
                parts.append("\n".join(buf))
                buf, size = [], 0
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if size + len(line) + 1 > max_chars and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append("\n".join(buf))
    return [p for p in parts if p.strip()]


def build_nodes(docs: list[Document]):
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs, show_progress=False)

    # SentenceSplitter cat theo cau. Cac trang bang bieu/infographic gan nhu khong co
    # dau cham cau nen no khong tim duoc cho cat, tao ra chunk dai hang chuc nghin
    # ky tu -> vuot gioi han 2048 token cua gemini-embedding-001. Cat cung lai o day.
    sized: list[TextNode] = []
    for node in nodes:
        content = node.get_content()
        if len(content) <= MAX_CHUNK_CHARS:
            sized.append(node)
            continue
        for piece in _hard_split(content, MAX_CHUNK_CHARS):
            sized.append(TextNode(text=piece, metadata=dict(node.metadata)))
    nodes = sized

    # chunk_id on dinh, doc duoc: p041-c02 = trang PDF 41, chunk thu 2
    counter: dict[int, int] = {}
    for node in nodes:
        page = node.metadata["page_index"]
        counter[page] = counter.get(page, 0) + 1
        chunk_id = f"p{page:03d}-c{counter[page]:02d}"
        node.metadata["chunk_id"] = chunk_id
        node.excluded_embed_metadata_keys = ["image_file", "has_figure", "chunk_id"]
        node.excluded_llm_metadata_keys = ["image_file"]
        # Qdrant chi nhan point id dang UUID hoac so nguyen, khong nhan "p041-c02".
        # Dung uuid5 de id van xac dinh duoc tu chunk_id => ingest lai se ghi de
        # dung point cu thay vi tao ban sao. chunk_id doc duoc van nam o metadata.
        node.id_ = str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))
    return nodes


def embed_nodes_paced(nodes, embed_model) -> None:
    """Sinh embedding co dieu tiet toc do, gan thang vao node.

    Free tier gioi han 100 embedding moi PHUT va dem theo TUNG VAN BAN, khong
    phai tung loi goi HTTP - nen gop lo khong giup lach tran. Ham nay giu nhip
    duoi nguong va tu cho khi bi 429, thay vi de ca lan ingest chet giua chung.

    Gan san node.embedding de VectorStoreIndex khong embed lai lan nua.
    """
    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    total = len(texts)
    window_start = time.time()
    in_window = 0
    done = 0

    for i in range(0, total, EMBED_BATCH):
        group_nodes = nodes[i : i + EMBED_BATCH]
        group_texts = texts[i : i + EMBED_BATCH]

        # Sap vuot han muc trong cua so 60s hien tai -> cho het cua so
        if in_window + len(group_texts) > EMBED_RPM:
            wait = 61 - (time.time() - window_start)
            if wait > 0:
                print(f"  ...cham tran {EMBED_RPM}/phut, cho {wait:.0f}s", flush=True)
                time.sleep(wait)
            window_start, in_window = time.time(), 0

        for attempt in range(5):
            try:
                vectors = embed_model.get_text_embedding_batch(group_texts, show_progress=False)
                break
            except Exception as exc:  # noqa: BLE001
                if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                    raise
                print(f"  ...bi 429, cho 60s roi thu lai (lan {attempt + 2}/5)", flush=True)
                time.sleep(60)
                window_start, in_window = time.time(), 0
        else:
            raise RuntimeError("Van bi 429 sau 5 lan thu. Hay chay lai sau vai phut.")

        for node, vec in zip(group_nodes, vectors):
            node.embedding = vec
        in_window += len(group_texts)
        done += len(group_texts)
        print(f"  embed {done}/{total} chunk", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Xoa collection cu roi tao lai")
    parser.add_argument("--limit", type=int, default=None, help="Chi ingest N trang dau (de test)")
    args = parser.parse_args()

    try:
        config.require("GEMINI_API_KEY")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(vector_store.describe())

    docs = load_documents(args.limit)
    nodes = build_nodes(docs)
    print(f"{len(docs)} trang -> {len(nodes)} chunk")

    embed_model = GoogleGenAIEmbedding(
        model_name=config.GEMINI_EMBED_MODEL,
        api_key=config.GEMINI_API_KEY,
        embed_batch_size=EMBED_BATCH,
        embedding_config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
    )
    Settings.embed_model = embed_model
    Settings.llm = None  # buoc ingest khong can LLM

    try:
        client = vector_store.get_client(timeout=120)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.recreate and client.collection_exists(config.QDRANT_COLLECTION):
        print(f"Xoa collection cu: {config.QDRANT_COLLECTION}")
        client.delete_collection(config.QDRANT_COLLECTION)

    # Khong dat ten bien la vector_store: se che module vector_store da import
    # va lam ca ham nem UnboundLocalError.
    store = QdrantVectorStore(
        collection_name=config.QDRANT_COLLECTION,
        client=client,
        batch_size=32,
    )
    storage_context = StorageContext.from_defaults(vector_store=store)

    t0 = time.time()
    print(f"Sinh embedding cho {len(nodes)} chunk (gioi han {EMBED_RPM}/phut)...")
    embed_nodes_paced(nodes, embed_model)
    print("Day len Qdrant...")
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    info = client.get_collection(config.QDRANT_COLLECTION)
    print(
        f"\nXong sau {time.time() - t0:.0f}s. "
        f"Collection '{config.QDRANT_COLLECTION}' co {info.points_count} points."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
