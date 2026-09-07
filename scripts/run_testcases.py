"""Chay bo test pass/fail cua de bai.

Hai che do:
  --inprocess  chay thang trong tien trinh, KHONG can uvicorn (khuyen dung)
  mac dinh     goi qua HTTP, yeu cau server dang chay o --url

Ca hai deu can da ingest xong len Qdrant.

Chay:  python scripts/run_testcases.py --inprocess

Script KHONG tu cham pass/fail cho cac cau hoi factual - no in ra day du intent,
citation va trace de nguoi cham doi chieu voi tai lieu. Rieng cac tieu chi may kiem
duoc (co goi retrieval khong, co citation gia khong) thi co check tu dong.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# cho phep import package app khi chay tu thu muc scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (ma test, cau hoi, ky vong may kiem duoc)
#   expect_intent: intent mong doi, None = khong rang buoc
#   expect_citations: True = phai co, False = tuyet doi khong duoc co, None = tuy
CASES = [
    ("A1", "Xin chào, bạn làm được gì?", "small_talk", False),
    ("A2", "Trong tài liệu VNA, hoạt động công nghệ thông tin năm 2024 gồm những gì?",
     "document_qa", True),
    ("A3", "Tóm tắt ngắn hơn giúp tôi.", "follow_up", None),
    ("A4", "Hôm nay thời tiết ở Hà Nội thế nào?", "out_of_scope", False),
    ("A5", "Trong tài liệu có nói gì về kế hoạch mở đường bay tới sao Hỏa không?",
     None, False),
    ("A6", "So sánh kết quả vận tải hành khách nội địa và quốc tế năm 2024 trong tài liệu.",
     None, True),
    ("B1", "Báo cáo kết quả phát triển bền vững năm 2024 của Vietnam Airlines.",
     "document_qa", True),
    ("B2", "Thị phần năm 2024 của Vietnam Airlines ở thị trường nội địa và quốc tế là bao nhiêu?",
     "document_qa", True),
]


def make_asker(args):
    """Tra ve ham hoi, chay qua HTTP hoac chay thang trong tien trinh.

    Che do in-process khong can uvicorn -> tien khi may/moi truong khong giu
    duoc server song, va loai bo mot bien so khi go loi.
    """
    if not args.inprocess:
        return lambda msg, sid: post(f"{args.url}/api/chat", {"message": msg, "session_id": sid})

    from app.agents.orchestrator import Orchestrator
    from app.session import Session

    orc = Orchestrator()
    sessions: dict[str, Session] = {}

    def ask(msg: str, sid: str) -> dict:
        if sid not in sessions:
            sessions[sid] = Session(sid)
        return orc.handle(msg, sessions[sid])

    return ask


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--inprocess", action="store_true",
                    help="Chay thang trong tien trinh, khong can uvicorn dang chay")
    args = ap.parse_args()

    ask = make_asker(args)
    session_id = "testcases"
    failures = 0

    for code, question, want_intent, want_cites in CASES:
        print("=" * 78)
        print(f"{code}  {question}")
        print("-" * 78)
        # A3 la cau noi tiep -> phai dung chung session voi A2.
        sid = session_id if code in ("A1", "A2", "A3", "A4") else f"{session_id}-{code}"
        try:
            data = ask(question, sid)
        except urllib.error.HTTPError as e:
            print(f"  LOI HTTP {e.code}: {e.read().decode()[:300]}")
            failures += 1
            continue
        except Exception as exc:
            print(f"  LOI: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        intent = data.get("intent")
        cites = data.get("citations", [])
        print(f"  intent    : {intent}")
        print(f"  grounded  : {data.get('grounded')}")
        print(f"  citations : {len(cites)}")
        for c in cites:
            print(f"     [{c['marker']}] trang {c['page_label']} (PDF {c['page_index']}) "
                  f"| {c['chunk_id']} | rerank={c['score']} | {c['section'][:55]}")
        print(f"  answer    : {data.get('answer', '')[:600]}")

        # --- check tu dong ---
        problems = []
        if want_intent and intent != want_intent:
            problems.append(f"intent={intent}, mong doi {want_intent}")
        if want_cites is True and not cites:
            problems.append("khong co citation nao")
        if want_cites is False and cites:
            problems.append(f"CITATION GIA: co {len(cites)} citation cho cau khong can tai lieu")
        # retrieval khong duoc goi voi small_talk / out_of_scope
        steps = [t.get("step") for t in data.get("trace", [])]
        if intent in ("small_talk", "out_of_scope") and "vector_retrieve" in steps:
            problems.append("da goi retrieval cho cau khong can tai lieu")

        if problems:
            failures += 1
            print("  => CAN XEM LAI: " + "; ".join(problems))
        else:
            print("  => check tu dong: OK")

    print("=" * 78)
    print(f"Xong. {len(CASES) - failures}/{len(CASES)} case qua duoc check tu dong.")
    print("Cac case factual (B1, B2, A6) van can nguoi doi chieu voi tai lieu.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
