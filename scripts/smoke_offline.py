"""Kiem tra pipeline ingest + retrieval OFFLINE.

Dung Qdrant in-memory va embedding gia, KHONG goi Gemini/Cohere/Qdrant Cloud
=> khong ton quota, chay duoc bat ky luc nao.

Muc dich: bat loi ve chunking, metadata, id, va wiring retrieval truoc khi
dot quota that.

Chay:  python scripts/smoke_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app import config
from app.ingest.build_index import build_nodes, load_documents

FAKE_DIM = 64
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))


def main() -> int:
    print("=" * 74)
    print("1. Load documents tu pages.jsonl")
    print("=" * 74)
    docs = load_documents(limit=25)
    check("doc duoc file parsed", len(docs) > 0, f"{len(docs)} trang")
    md = docs[0].metadata
    for key in ("source_file", "page_index", "page_label", "section", "image_file", "has_figure"):
        check(f"metadata co '{key}'", key in md)
    check("image_file bi loai khoi embedding",
          "image_file" in docs[0].excluded_embed_metadata_keys)

    print()
    print("=" * 74)
    print("2. Chunking + sinh chunk_id")
    print("=" * 74)
    nodes = build_nodes(docs)
    check("tao duoc node", len(nodes) > 0, f"{len(nodes)} chunk tu {len(docs)} trang")
    ids = [n.metadata["chunk_id"] for n in nodes]
    check("chunk_id khong trung nhau", len(ids) == len(set(ids)),
          f"{len(ids)} id, {len(set(ids))} unique")
    check("chunk_id dung dinh dang pXXX-cYY",
          all(i[0] == "p" and "-c" in i for i in ids), ids[0])
    check("moi node giu duoc page_index", all("page_index" in n.metadata for n in nodes))
    check("anh trang ton tai tren dia",
          all((config.PAGES_DIR / n.metadata["image_file"]).exists() for n in nodes[:20]))

    print()
    print("=" * 74)
    print("3. Ingest vao Qdrant in-memory + retrieve")
    print("=" * 74)
    embed = MockEmbedding(embed_dim=FAKE_DIM)
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(collection_name="smoke", client=client)
    try:
        index = VectorStoreIndex(
            nodes,
            storage_context=StorageContext.from_defaults(vector_store=store),
            embed_model=embed,
            show_progress=False,
        )
        check("ghi duoc vao Qdrant", True,
              f"{client.get_collection('smoke').points_count} points")
    except Exception as exc:
        check("ghi duoc vao Qdrant", False, f"{type(exc).__name__}: {str(exc)[:150]}")
        print("\n=> Dung lai, khong retrieve duoc.")
        return 1

    retriever = index.as_retriever(similarity_top_k=5)
    hits = retriever.retrieve(QueryBundle(query_str="thi phan noi dia va quoc te"))
    check("retrieve tra ve ket qua", len(hits) > 0, f"{len(hits)} hit")
    if hits:
        hmd = hits[0].node.metadata
        for key in ("source_file", "page_label", "page_index", "section", "chunk_id", "image_file"):
            check(f"metadata '{key}' song sot qua Qdrant", key in hmd,
                  str(hmd.get(key))[:45])
        check("noi dung chunk khong rong", len(hits[0].node.get_content().strip()) > 0)

    print()
    print("=" * 74)
    print("KET QUA:", "TAT CA PASS" if ok else "CO LOI CAN SUA")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
