"""AGENT 1 - Chit-chat / Orchestrator Agent.

Trach nhiem:
  - Nhan input tu user.
  - Phan loai intent: small_talk | document_qa | follow_up | out_of_scope.
  - Chi goi RAG agent khi that su can tra cuu tai lieu.
  - Tong hop ket qua tu RAG agent thanh cau tra loi cuoi cung, de doc.

Agent nay KHONG tu truy cap Qdrant/Cohere. Muon co bang chung thi phai hoi RAG agent.
"""
from __future__ import annotations

from typing import Any

from app import config
from app.agents import llm as llm_util
from app.agents.rag_agent import RagAgent
from app.session import Session

VALID_INTENTS = {"small_talk", "document_qa", "follow_up", "out_of_scope"}

CLASSIFY_SYSTEM = """Ban la orchestrator cua mot chatbot noi bo Vietnam Airlines.
Chatbot chi co MOT nguon kien thuc: file "Bao cao thuong nien 2024 - Vietnam Airlines"
(ket qua kinh doanh, doi tau bay, thi phan, dich vu, nhan su, cong nghe thong tin,
quan tri rui ro, phat trien ben vung, bao cao tai chinh).

Phan loai tin nhan moi nhat cua user vao DUNG MOT intent:

- "small_talk": chao hoi, cam on, hoi chatbot lam duoc gi, tan gau. KHONG can tra tai lieu.
- "document_qa": cau hoi moi can tra cuu noi dung tai lieu VNA.
- "follow_up": cau noi tiep phu thuoc vao luot truoc (vd "tom tat ngan hon", "con muc kia
  thi sao", "giai thich ro hon"). Neu la follow_up, xac dinh them:
    needs_retrieval = true  neu can them thong tin MOI tu tai lieu
    needs_retrieval = false neu chi can dien dat lai / rut gon / mo rong cau tra loi truoc
- "out_of_scope": cau hoi khong lien quan den tai lieu VNA (thoi tiet, ty gia, tin tuc,
  kien thuc chung, hang hang khong khac...). KHONG duoc goi tra cuu tai lieu.

Luu y quan trong:
- Cau hoi ve Vietnam Airlines nhung ro rang khong nam trong pham vi mot bao cao thuong nien
  (vd gia ve hom nay, lich bay) van la "out_of_scope".
- Neu phan van giua document_qa va out_of_scope, uu tien "document_qa" de he thong tu kiem
  chung bang bang chung.

Tra ve JSON:
{"intent": "...", "needs_retrieval": true/false, "standalone_question": "...", "reason": "..."}
"""

SMALLTALK_SYSTEM = """Ban la tro ly noi bo cua Vietnam Airlines, chi tra cuu duoc file
"Bao cao thuong nien 2024 - Vietnam Airlines".

Dang o che do tro chuyen xa giao. Hay:
- Tra loi ngan gon, than thien, tu nhien bang tieng Viet.
- Neu user hoi ban lam duoc gi, neu ro pham vi: tra cuu noi dung bao cao thuong nien 2024
  cua VNA va luon dan nguon trang/section cu the.
- TUYET DOI khong bia so lieu, khong tao trich dan, khong nhac den [1] [2].
"""

OUT_OF_SCOPE_SYSTEM = """Ban la tro ly noi bo cua Vietnam Airlines, chi co kien thuc tu file
"Bao cao thuong nien 2024 - Vietnam Airlines".

Cau hoi cua user nam ngoai pham vi tai lieu. Hay:
- Noi ro va lich su rang cau hoi nay ngoai pham vi tai lieu ban duoc cung cap.
- Goi y user hoi ve nhung chu de co trong bao cao (ket qua kinh doanh, doi tau bay,
  thi phan, dich vu, nhan su, phat trien ben vung...).
- TUYET DOI khong doan cau tra loi, khong bia so lieu, khong tao trich dan gia.
"""

COMPOSE_SYSTEM = """Ban la orchestrator, nhan ban nhap tu RAG agent va viet lai thanh cau
tra loi cuoi cung gui cho user.

Quy tac:
- GIU NGUYEN moi so lieu va moi marker trich dan dang [1], [2]. Khong doi so, khong bo,
  khong them marker moi.
- Khong them bat ky thong tin nao khong co trong ban nhap.
- Viet tieng Viet tu nhien, mach lac; dung gach dau dong khi liet ke nhieu y.
- Khong mo dau bang "Dua tren tai lieu..." lap di lap lai; vao thang y chinh.
- Khong tu them muc "Nguon" o cuoi, giao dien da hien thi citation rieng.
"""

REFINE_SYSTEM = """Ban la orchestrator. User muon dien dat lai cau tra loi TRUOC DO
(rut gon, mo rong, doi cach trinh bay) chu khong hoi thong tin moi.

Quy tac:
- Chi dung lai noi dung da co trong cau tra loi truoc. Khong them du kien moi.
- GIU NGUYEN cac marker [1], [2] gan voi y tuong ung.
- Neu user yeu cau ngan hon, hay that su ngan hon.
"""


class Orchestrator:
    name = "chitchat_orchestrator_agent"

    def __init__(self) -> None:
        self.rag_agent = RagAgent()

    def classify(self, message: str, session: Session) -> dict[str, Any]:
        history = session.history()
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
        payload = f"Hoi thoai truoc do:\n{convo or '(chua co)'}\n\nTin nhan moi: {message}"
        data = llm_util.chat_json(
            CLASSIFY_SYSTEM,
            payload,
            fallback={"intent": "document_qa", "needs_retrieval": True,
                      "standalone_question": message, "reason": "fallback khi parse loi"},
            model=config.FAST_MODEL,
        )
        intent = str(data.get("intent", "")).strip()
        if intent not in VALID_INTENTS:
            intent = "document_qa"
        # follow_up ma khong co lich su thi thuc chat la cau hoi moi
        if intent == "follow_up" and not history:
            intent = "document_qa"
        data["intent"] = intent
        data.setdefault("standalone_question", message)
        data["needs_retrieval"] = bool(data.get("needs_retrieval", intent != "small_talk"))
        return data

    def _answer_without_documents(self, system: str, message: str, session: Session) -> str:
        convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in session.history()[-4:])
        return llm_util.chat(system, f"Hoi thoai:\n{convo}\n\nUser: {message}", temperature=0.6)

    def _refine_previous(self, message: str, session: Session) -> dict[str, Any]:
        """Follow-up kieu 'tom tat ngan hon' - dung lai bang chung cu, khong retrieve lai."""
        last = session.last_rag
        if not last or not last.get("answer_draft"):
            return {"handled": False}
        final = llm_util.chat(
            REFINE_SYSTEM,
            f"Cau tra loi truoc do:\n{last['answer_draft']}\n\nYeu cau cua user: {message}",
            temperature=0.3,
        )
        return {
            "handled": True,
            "answer": final,
            "citations": last.get("citations", []),
            "trace": [{"step": "reuse_previous_evidence",
                       "output": "dung lai evidence cua luot truoc, khong goi retrieval"}],
        }

    def handle(self, message: str, session: Session) -> dict[str, Any]:
        decision = self.classify(message, session)
        intent = decision["intent"]
        trace: list[dict[str, Any]] = [
            {
                "agent": self.name,
                "step": "intent_classification",
                "intent": intent,
                "needs_retrieval": decision["needs_retrieval"],
                "reason": decision.get("reason", ""),
            }
        ]

        # --- Nhanh 1: khong dung tai lieu, khong citation ---
        if intent in ("small_talk", "out_of_scope"):
            system = SMALLTALK_SYSTEM if intent == "small_talk" else OUT_OF_SCOPE_SYSTEM
            answer = self._answer_without_documents(system, message, session)
            trace.append({"agent": self.name, "step": "direct_answer",
                          "output": "khong goi RAG agent, khong sinh citation"})
            session.add("user", message)
            session.add("assistant", answer)
            return {"intent": intent, "answer": answer, "citations": [],
                    "grounded": False, "trace": trace}

        # --- Nhanh 2: follow-up chi can dien dat lai ---
        if intent == "follow_up" and not decision["needs_retrieval"]:
            refined = self._refine_previous(message, session)
            if refined["handled"]:
                trace.extend({"agent": self.name, **t} for t in refined["trace"])
                session.add("user", message)
                session.add("assistant", refined["answer"])
                return {"intent": intent, "answer": refined["answer"],
                        "citations": refined["citations"], "grounded": True, "trace": trace}
            # khong co ket qua cu de dua vao -> chuyen sang retrieve binh thuong

        # --- Nhanh 3: goi RAG agent ---
        question = decision.get("standalone_question") or message
        result = self.rag_agent.answer(question, session.history(), already_standalone=True)
        trace.extend({"agent": self.rag_agent.name, **t} for t in result.trace)

        if not result.found:
            answer = (
                "Toi khong tim thay thong tin nay trong "
                f"{config.SOURCE_NAME}. "
                + (result.answer_draft or "")
                + "\n\nBan thu dien dat lai cau hoi hoac hoi ve mot muc khac trong bao cao nhe."
            ).strip()
            trace.append({"agent": self.name, "step": "compose",
                          "output": "bao thieu bang chung, khong gan citation"})
            session.add("user", message)
            session.add("assistant", answer)
            session.last_rag = None
            return {"intent": intent, "answer": answer, "citations": [],
                    "grounded": False, "trace": trace}

        citations = [c.__dict__ for c in result.citations]
        cite_lines = "\n".join(
            f"[{c['marker']}] trang {c['page_label']} - {c['section']}" for c in citations
        )
        answer = llm_util.chat(
            COMPOSE_SYSTEM,
            f"Cau hoi cua user: {message}\n\nBan nhap tu RAG agent:\n{result.answer_draft}"
            f"\n\nDanh sach marker hop le:\n{cite_lines}",
            temperature=0.3,
        )
        trace.append({"agent": self.name, "step": "compose",
                      "output": f"tong hop cau tra loi cuoi voi {len(citations)} citation"})

        session.add("user", message)
        session.add("assistant", answer)
        session.last_rag = {"answer_draft": answer, "citations": citations}
        return {"intent": intent, "answer": answer, "citations": citations,
                "grounded": True, "trace": trace}
