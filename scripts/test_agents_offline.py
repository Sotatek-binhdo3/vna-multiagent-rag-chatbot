"""Test logic dinh tuyen cua 2 agent OFFLINE.

Thay LLM / retriever / reranker bang ban gia => khong goi Gemini, Cohere, Qdrant,
khong ton quota. Muc dich la kiem tra phan LOGIC ma de bai cham:

  - small_talk / out_of_scope co goi retrieval khong
  - follow-up kieu "tom tat ngan hon" co dung lai evidence cu khong
  - thieu bang chung co bao dung khong, co bia citation khong
  - chi tra ve citation thuc su duoc trich dan trong cau tra loi

Chay:  python scripts/test_agents_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from app.agents import llm as llm_util
from app.agents import orchestrator as orch_mod
from app.agents import rag_agent as rag_mod
from app.session import Session

ok = True
llm_calls: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))


def fake_node(chunk_id: str, page: int, score: float) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=f"Noi dung gia cua {chunk_id}",
            metadata={
                "source_file": "test.pdf", "page_index": page,
                "page_label": f"{page*2-1}-{page*2}", "section": f"Section {page}",
                "image_file": f"page_{page:03d}.jpg", "chunk_id": chunk_id,
                "has_figure": False,
            },
        ),
        score=score,
    )


class FakeRetriever:
    def __init__(self, nodes): self.nodes = nodes
    def retrieve(self, _q): return self.nodes


class FakeReranker:
    def __init__(self, nodes): self.nodes = nodes
    def postprocess_nodes(self, _n, _q): return self.nodes


def install(classify: dict, draft: str, reranked: list[NodeWithScore]) -> None:
    """Cai dat LLM va retrieval gia cho mot kich ban."""
    llm_calls.clear()

    def fake_chat_json(system, user, fallback, model=None):
        llm_calls.append("classify" if "orchestrator" in system[:60].lower() else "json")
        return classify

    def fake_chat(system, user, temperature=0.2, model=None):
        if "RAG agent tra loi" in system:
            llm_calls.append("synthesize"); return draft
        if "viet lai thanh cau" in system:
            llm_calls.append("compose"); return user.split("Ban nhap tu RAG agent:\n")[-1].split("\n\nDanh sach")[0]
        if "dien dat lai" in system:
            llm_calls.append("refine"); return "Ban rut gon: " + draft
        llm_calls.append("direct"); return "Cau tra loi xa giao gia."

    llm_util.chat_json = fake_chat_json
    llm_util.chat = fake_chat
    orch_mod.llm_util = llm_util
    rag_mod.llm_util = llm_util
    rag_mod._retriever = lambda: FakeRetriever(reranked)
    rag_mod._reranker = lambda: FakeReranker(reranked)


def check_real_signatures() -> None:
    """Cac test duoi day thay llm_util bang ban gia, nen KHONG phat hien duoc
    truong hop agent goi ham that voi tham so ma ham that khong nhan.
    Kiem tra chu ky ham that o day de bit lo hong do."""
    import inspect

    from app.agents import llm as real_llm

    for fn, kwargs in [
        (real_llm.chat, {"system": "s", "user": "u", "temperature": 0.2, "model": "m"}),
        (real_llm.chat_json, {"system": "s", "user": "u", "fallback": {}, "model": "m"}),
        (real_llm.get_llm, {"temperature": 0.2, "model": "m"}),
    ]:
        try:
            inspect.signature(fn).bind(**kwargs)
            check(f"chu ky that cua {fn.__name__}() nhan du tham so", True)
        except TypeError as exc:
            check(f"chu ky that cua {fn.__name__}() nhan du tham so", False, str(exc))


def main() -> int:
    nodes = [fake_node("p041-c02", 41, 0.91), fake_node("p027-c05", 27, 0.62),
             fake_node("p010-c01", 10, 0.31)]

    print("=" * 74)
    print("0. Chu ky ham that (truoc khi thay bang ban gia)")
    print("=" * 74)
    check_real_signatures()

    print()
    print("=" * 74)
    print("1. small_talk -> KHONG duoc goi retrieval, KHONG duoc co citation")
    print("=" * 74)
    install({"intent": "small_talk", "needs_retrieval": False, "standalone_question": "hi"},
            "", nodes)
    r = orch_mod.Orchestrator().handle("Xin chao", Session("s1"))
    steps = [t.get("step") for t in r["trace"]]
    check("intent = small_talk", r["intent"] == "small_talk")
    check("khong co buoc vector_retrieve", "vector_retrieve" not in steps, str(steps))
    check("khong co citation", len(r["citations"]) == 0)

    print()
    print("=" * 74)
    print("2. out_of_scope -> tu choi, khong dung tai lieu")
    print("=" * 74)
    install({"intent": "out_of_scope", "needs_retrieval": False, "standalone_question": "x"},
            "", nodes)
    r = orch_mod.Orchestrator().handle("Thoi tiet Ha Noi?", Session("s2"))
    check("intent = out_of_scope", r["intent"] == "out_of_scope")
    check("khong co citation", len(r["citations"]) == 0)
    check("khong goi retrieval", "vector_retrieve" not in [t.get("step") for t in r["trace"]])

    print()
    print("=" * 74)
    print("3. document_qa -> chi tra citation THUC SU duoc trich dan")
    print("=" * 74)
    # draft chi nhac [1] va [3], KHONG nhac [2]
    install({"intent": "document_qa", "needs_retrieval": True, "standalone_question": "q"},
            "Y thu nhat [1]. Y thu hai [3].", nodes)
    r = orch_mod.Orchestrator().handle("Cau hoi tai lieu", Session("s3"))
    markers = sorted(c["marker"] for c in r["citations"])
    check("grounded = True", r["grounded"])
    check("chi giu marker duoc trich dan", markers == [1, 3], f"markers={markers}")
    check("khong tra ve marker [2] khong dung den", 2 not in markers)
    check("citation co du field cho UI",
          all(k in r["citations"][0] for k in
              ("page_label", "page_index", "chunk_id", "image_file", "section", "score")))
    check("bo qua query_rewrite (tiet kiem quota)",
          any(t.get("note", "").startswith("bo qua") for t in r["trace"]))

    print()
    print("=" * 74)
    print("4. Thieu bang chung -> bao ro, KHONG bia citation")
    print("=" * 74)
    install({"intent": "document_qa", "needs_retrieval": True, "standalone_question": "q"},
            f"{rag_mod.NOT_FOUND_TOKEN} Tai lieu khong de cap noi dung nay.", nodes)
    r = orch_mod.Orchestrator().handle("Co noi ve sao Hoa khong?", Session("s4"))
    check("grounded = False", not r["grounded"])
    check("khong co citation nao", len(r["citations"]) == 0, f"{len(r['citations'])}")
    check("cau tra loi noi ro khong tim thay",
          "khong tim thay" in r["answer"].lower().replace("ô", "o").replace("ì", "i"))

    print()
    print("=" * 74)
    print("5. Tat ca chunk duoi nguong -> khong goi LLM sinh cau tra loi")
    print("=" * 74)
    low = [fake_node("p001-c01", 1, 0.001)]
    install({"intent": "document_qa", "needs_retrieval": True, "standalone_question": "q"},
            "khong nen duoc goi", low)
    r = orch_mod.Orchestrator().handle("Cau hoi la", Session("s5"))
    check("grounded = False", not r["grounded"])
    check("khong co citation", len(r["citations"]) == 0)
    check("KHONG goi LLM synthesize", "synthesize" not in llm_calls, str(llm_calls))

    print()
    print("=" * 74)
    print("6. follow_up 'tom tat ngan hon' -> dung lai evidence, KHONG retrieve lai")
    print("=" * 74)
    sess = Session("s6")
    install({"intent": "document_qa", "needs_retrieval": True, "standalone_question": "q"},
            "Noi dung dai [1].", nodes)
    orch_mod.Orchestrator().handle("Cau hoi goc", sess)
    check("luot dau co luu last_rag", sess.last_rag is not None)

    install({"intent": "follow_up", "needs_retrieval": False, "standalone_question": "ngan hon"},
            "Noi dung dai [1].", nodes)
    r = orch_mod.Orchestrator().handle("Tom tat ngan hon giup toi.", sess)
    steps = [t.get("step") for t in r["trace"]]
    check("intent = follow_up", r["intent"] == "follow_up")
    check("KHONG retrieve lai", "vector_retrieve" not in steps, str(steps))
    check("co buoc reuse_previous_evidence", "reuse_previous_evidence" in steps)
    check("giu lai citation cua luot truoc", len(r["citations"]) > 0)

    print()
    print("=" * 74)
    print("KET QUA:", "TAT CA PASS" if ok else "CO LOI CAN SUA")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
