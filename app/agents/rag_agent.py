"""AGENT 2 - VNA Internal Document RAG Agent.

Trach nhiem duy nhat: tra loi dua tren tai lieu noi bo VNA.
Luong: query rewrite -> retrieve (Qdrant) -> rerank (Cohere) -> chon evidence
       -> sinh answer draft kem citation.

Agent nay KHONG xu ly chao hoi hay cau ngoai pham vi. Do la viec cua orchestrator.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from google.genai import types
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app import config, vector_store
from app.agents import llm as llm_util

# Sentinel thuan ASCII, khong phai tu tieng Viet -> model khong "sua lai co dau".
NOT_FOUND_TOKEN = "<<INSUFFICIENT_EVIDENCE>>"
# Van chap nhan cac bien the cu / co dau, phong khi model tu dien dat lai.
_NOT_FOUND_VARIANTS = ("<<INSUFFICIENT_EVIDENCE>>", "INSUFFICIENT_EVIDENCE",
                       "KHONG_DU_BANG_CHUNG", "KHONG DU BANG CHUNG")


def _strip_accents(text: str) -> str:
    # 'd'/'D' la chu cai rieng trong bang chu cai tieng Viet, KHONG phai 'd' co dau,
    # nen normalize("NFD") khong tach duoc. Phai thay tay truoc.
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def says_not_found(draft: str) -> bool:
    """Model bao thieu bang chung hay khong.

    Phai chiu duoc viec model tra token co dau tieng Viet (da tung gap
    "KHONG_DU_BANG_CHUNG" bi viet thanh "KHONG_DU..." co dau O mu), neu khong
    thi co che chong bia se that bai theo huong MO: he thong tuong cau tra loi
    co can cu va gan citation vao mot cau "toi khong biet".
    """
    norm = _strip_accents(draft).upper()
    return any(v in norm for v in _NOT_FOUND_VARIANTS)


def _clean_not_found(draft: str) -> str:
    """Bo sentinel khoi cau tra loi truoc khi hien cho nguoi dung."""
    out = re.sub(r"<<\s*INSUFFICIENT_EVIDENCE\s*>>", "", draft, flags=re.IGNORECASE)
    out = re.sub(r"INSUFFICIENT_EVIDENCE", "", out, flags=re.IGNORECASE)
    # ban tieng Viet, co hoac khong dau, ngan cach bang gach duoi hoac khoang trang
    out = re.sub(
        r"KH[OÔ]NG[ _]+D[UỦ][ _]+B[AẰ]NG[ _]+CH[UỨ]NG",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return out.strip(" .:\n")


_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    """Mot nguon trich dan. Du thong tin de user mo lai dung cho trong tai lieu."""

    marker: int          # so [1], [2] xuat hien trong cau tra loi
    source_file: str
    page_label: str      # so trang in tren giay, vd "78-79"
    page_index: int      # so trang trong file PDF, dung de mo anh preview
    section: str
    chunk_id: str        # anchor noi bo, vd "p041-c02"
    image_file: str
    score: float         # diem rerank cua Cohere
    snippet: str
    has_figure: bool


@dataclass
class RagResult:
    found: bool
    answer_draft: str
    citations: list[Citation] = field(default_factory=list)
    rewritten_query: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "answer_draft": self.answer_draft,
            "citations": [asdict(c) for c in self.citations],
            "rewritten_query": self.rewritten_query,
            "trace": self.trace,
        }


REWRITE_SYSTEM = """Ban la bo phan query rewrite cua he thong RAG tra cuu tai lieu noi bo
Vietnam Airlines (Bao cao thuong nien 2024).

Nhiem vu: bien cau hoi cua nguoi dung thanh MOT cau truy van doc lap, day du ngu canh,
giau tu khoa de tim kiem vector tieng Viet.

Quy tac:
- Neu cau hoi phu thuoc vao hoi thoai truoc (vd "con cai kia thi sao"), phai thay dai tu
  bang doi tuong cu the da nhac den truoc do.
- Giu nguyen thuat ngu chuyen nganh, ten viet tat, moc thoi gian.
- Khong tu them dieu kien ma nguoi dung khong hoi.
- Tra ve JSON: {"query": "...", "keywords": ["...", "..."]}"""


ANSWER_SYSTEM = f"""Ban la RAG agent tra loi dua tren tai lieu noi bo Vietnam Airlines.

QUY TAC BAT BUOC:
1. CHI dung thong tin co trong phan BANG CHUNG ben duoi. Tuyet doi khong dung kien thuc
   ben ngoai, khong suy dien, khong lam tron hay uoc luong so lieu.
2. Moi y lay tu bang chung phai gan marker [n] tuong ung ngay sau y do.
3. Neu bang chung KHONG du de tra loi, BAT DAU cau tra loi bang dung chuoi ASCII
   {NOT_FOUND_TOKEN} (giu nguyen tung ky tu, KHONG dich, KHONG them dau tieng Viet),
   roi viet mot cau ngan noi ro thieu thong tin gi. Khong duoc doan.
4. Neu bang chung chi tra loi duoc mot phan, tra loi phan do va noi ro phan nao chua co
   trong tai lieu.
5. Neu bang chung co bang bieu/so lieu, trich dung dung con so, khong tu tinh toan them.
5b. Khi user yeu cau SO SANH: neu bang chung co du so lieu cua cac ben can so sanh, hay
   trinh bay chung canh nhau va neu ro khac biet. KHONG doi hoi tai lieu phai co san mot
   muc "so sanh" - dat cac con so co that canh nhau la du. Van cam tu tinh toan con so moi
   (hieu, ty le, phan tram tang giam) neu tai lieu khong ghi san.
6. Tra loi bang tieng Viet, ngan gon, co cau truc.
7. Khong bia them marker cho nguon khong ton tai trong danh sach bang chung."""


@lru_cache(maxsize=1)
def _embed_model() -> GoogleGenAIEmbedding:
    return GoogleGenAIEmbedding(
        model_name=config.GEMINI_EMBED_MODEL,
        api_key=config.GEMINI_API_KEY,
        embed_batch_size=8,
        embedding_config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
    )


@lru_cache(maxsize=1)
def _retriever():
    """Mo lai collection Qdrant da ingest. Cache de khong ket noi lai moi request."""
    config.require("GEMINI_API_KEY")
    client = vector_store.get_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        raise RuntimeError(
            f"Collection '{config.QDRANT_COLLECTION}' chua ton tai. "
            "Chay truoc: python -m app.ingest.build_index --recreate"
        )
    # Khong dat ten bien la vector_store: se che module vector_store da import.
    store = QdrantVectorStore(collection_name=config.QDRANT_COLLECTION, client=client)
    embed_model = _embed_model()
    Settings.embed_model = embed_model
    Settings.llm = llm_util.get_llm()
    index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
    return index.as_retriever(similarity_top_k=config.RETRIEVE_TOP_K)


@lru_cache(maxsize=1)
def _reranker() -> CohereRerank:
    config.require("COHERE_API_KEY")
    return CohereRerank(
        api_key=config.COHERE_API_KEY,
        model=config.COHERE_RERANK_MODEL,
        top_n=config.RERANK_TOP_N,
    )


class RagAgent:
    name = "vna_document_rag_agent"

    def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        """Bien cau hoi thanh truy van doc lap. Quan trong cho test follow-up."""
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:])
        data = llm_util.chat_json(
            REWRITE_SYSTEM,
            f"Hoi thoai truoc do:\n{convo}\n\nCau hoi moi: {question}",
            fallback={"query": question},
            model=config.FAST_MODEL,
        )
        rewritten = (data.get("query") or question).strip()
        return rewritten or question

    def _format_evidence(self, nodes: list[NodeWithScore]) -> str:
        blocks = []
        for i, nws in enumerate(nodes, start=1):
            md = nws.node.metadata
            header = (
                f"[{i}] file={md.get('source_file')} | trang={md.get('page_label')} "
                f"| section={md.get('section')} | chunk={md.get('chunk_id')}"
            )
            blocks.append(f"{header}\n{nws.node.get_content()}")
        return "\n\n---\n\n".join(blocks)

    def _to_citations(self, nodes: list[NodeWithScore]) -> list[Citation]:
        out = []
        for i, nws in enumerate(nodes, start=1):
            md = nws.node.metadata
            text = nws.node.get_content().strip()
            out.append(
                Citation(
                    marker=i,
                    source_file=md.get("source_file", config.SOURCE_NAME),
                    page_label=md.get("page_label", "?"),
                    page_index=int(md.get("page_index", 0)),
                    section=md.get("section", ""),
                    chunk_id=md.get("chunk_id", nws.node.node_id),
                    image_file=md.get("image_file", ""),
                    score=round(float(nws.score or 0.0), 4),
                    snippet=re.sub(r"\s+", " ", text)[:320],
                    has_figure=bool(md.get("has_figure", False)),
                )
            )
        return out

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        already_standalone: bool = False,
    ) -> RagResult:
        history = history or []
        trace: list[dict[str, Any]] = []

        # --- B1: query rewrite ---
        # Khi orchestrator da chuan hoa cau hoi (giai quyet dai tu, bo sung ngu canh)
        # thi bo qua buoc nay de khong ton them 1 luot goi LLM.
        if already_standalone:
            query = question
            trace.append({"step": "query_rewrite", "input": question, "output": query,
                          "note": "bo qua - orchestrator da chuan hoa san"})
        else:
            query = self.rewrite(question, history)
            trace.append({"step": "query_rewrite", "input": question, "output": query})

        # --- B2: retrieve tu Qdrant ---
        retrieved = _retriever().retrieve(QueryBundle(query_str=query))
        trace.append(
            {
                "step": "vector_retrieve",
                "store": f"Qdrant/{config.QDRANT_COLLECTION}",
                "top_k": config.RETRIEVE_TOP_K,
                "hits": len(retrieved),
                "chunks": [n.node.metadata.get("chunk_id") for n in retrieved],
            }
        )
        if not retrieved:
            return RagResult(False, "Khong tim thay chunk nao trong tai lieu.", [], query, trace)

        # --- B3: rerank bang Cohere ---
        reranked = _reranker().postprocess_nodes(retrieved, QueryBundle(query_str=query))
        kept = [n for n in reranked if (n.score or 0) >= config.RERANK_MIN_SCORE]
        trace.append(
            {
                "step": "cohere_rerank",
                "model": config.COHERE_RERANK_MODEL,
                "top_n": config.RERANK_TOP_N,
                "min_score": config.RERANK_MIN_SCORE,
                "kept": len(kept),
                "scores": [
                    {"chunk": n.node.metadata.get("chunk_id"), "score": round(n.score or 0, 4)}
                    for n in reranked
                ],
            }
        )
        if not kept:
            trace.append(
                {"step": "decision", "output": "duoi nguong tin cay -> khong du bang chung"}
            )
            return RagResult(
                False,
                "Toi khong tim thay noi dung nao trong tai lieu du lien quan de tra loi.",
                [],
                query,
                trace,
            )

        # --- B4: sinh cau tra loi rang buoc theo bang chung ---
        evidence = self._format_evidence(kept)
        draft = llm_util.chat(
            ANSWER_SYSTEM,
            f"CAU HOI: {question}\nTRUY VAN DA CHUAN HOA: {query}\n\nBANG CHUNG:\n{evidence}",
            # Tac vu bam bang chung: de tat dinh cho ket qua on dinh giua cac lan chay.
            # O 0.1 da gap cung cau hoi, cung evidence ma luc tra loi luc tu choi.
            temperature=0.0,
        )
        found = not says_not_found(draft)
        draft = _clean_not_found(draft)

        # --- B5: chi giu citation thuc su duoc trich dan ---
        all_citations = self._to_citations(kept)
        if found:
            used = {int(m) for m in _MARKER_RE.findall(draft)}
            citations = [c for c in all_citations if c.marker in used] or all_citations[:1]
        else:
            citations = []
        trace.append(
            {
                "step": "synthesize",
                "model": config.CHAT_MODEL,
                "grounded": found,
                "citations_used": [c.chunk_id for c in citations],
            }
        )
        return RagResult(found, draft, citations, query, trace)
