/* UI chat cho POC VNA multi-agent RAG.
   Nhiem vu chinh: hien thi cau tra loi, bien marker [n] thanh nut bam duoc,
   va mo dung anh trang PDF trong panel ben phai khi user bam citation. */

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("sendBtn");
const sourcePane = $("sourcePane");

let sessionId = null;
let busy = false;

const INTENT_LABEL = {
  small_talk: "Xã giao",
  document_qa: "Tra tài liệu",
  follow_up: "Câu nối tiếp",
  out_of_scope: "Ngoài phạm vi",
};

const SUGGESTIONS = [
  "Xin chào, bạn làm được gì?",
  "Báo cáo kết quả phát triển bền vững năm 2024 của Vietnam Airlines.",
  "Thị phần năm 2024 của Vietnam Airlines ở thị trường nội địa và quốc tế là bao nhiêu?",
  "Tóm tắt ngắn hơn giúp tôi.",
  "Hôm nay thời tiết ở Hà Nội thế nào?",
];

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* Gemini tra loi bang markdown (**dam**, gach dau dong, tieu de).
   Render toi thieu de UI khong hien thi tho cac dau ** va -. */
function inlineMarkdown(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let list = null; // "ul" | "ol" | null

  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);

    if (bullet) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
    } else if (numbered) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
    } else if (heading) {
      closeList();
      out.push(`<div class="md-h">${inlineMarkdown(heading[2])}</div>`);
    } else if (!line.trim()) {
      closeList();
      out.push('<div class="md-gap"></div>');
    } else {
      closeList();
      out.push(`<div>${inlineMarkdown(line)}</div>`);
    }
  }
  closeList();
  return out.join("");
}

/* Bien [1] [2] trong text thanh nut bam duoc, chi khi marker do co citation that. */
function renderAnswer(text, citations) {
  const valid = new Set(citations.map((c) => c.marker));
  return renderMarkdown(text).replace(/\[(\d{1,2})\]/g, (full, n) => {
    const num = Number(n);
    if (!valid.has(num)) return full;
    return `<span class="cite-marker" data-marker="${num}" title="Xem nguồn">${num}</span>`;
  });
}

function citationCard(c) {
  return `<button class="cite-card" data-marker="${c.marker}">
    <div class="line1">[${c.marker}] Trang ${escapeHtml(c.page_label)} · ${escapeHtml(c.section)}</div>
    <div class="line2">PDF tr.${c.page_index} · <code>${escapeHtml(c.chunk_id)}</code> · độ khớp ${Number(c.score).toFixed(2)}</div>
    <div class="snip">${escapeHtml(c.snippet)}</div>
  </button>`;
}

function traceBlock(trace) {
  if (!trace || !trace.length) return "";
  const steps = trace
    .map((t) => {
      const { agent, step, ...rest } = t;
      const detail = Object.entries(rest)
        .filter(([, v]) => v !== "" && v !== null && v !== undefined)
        .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
        .join("\n");
      return `<div class="trace-step">
        <div class="agent">${escapeHtml(agent || "system")}</div>
        <b>${escapeHtml(step || "")}</b>
        ${detail ? `<pre>${escapeHtml(detail)}</pre>` : ""}
      </div>`;
    })
    .join("");
  return `<details class="trace"><summary>Luồng xử lý · ${trace.length} bước</summary>${steps}</details>`;
}

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="who">B</div><div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addPlaceholder() {
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = `<div class="who">VNA</div><div class="bubble typing">Đang trả lời…</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function fillBotMessage(node, data) {
  const citations = data.citations || [];
  const intentBadge = `<span class="badge ${data.intent}">${INTENT_LABEL[data.intent] || data.intent}</span>`;
  const groundBadge = citations.length
    ? `<span class="badge grounded">${citations.length} nguồn</span>`
    : data.intent === "document_qa" || data.intent === "follow_up"
    ? `<span class="badge ungrounded">Không tìm thấy trong tài liệu</span>`
    : "";

  const citeHtml = citations.length
    ? `<div class="citations">${citations.map(citationCard).join("")}</div>`
    : "";

  node.querySelector(".bubble").outerHTML = `<div class="bubble">
    <div class="meta-row">${intentBadge}${groundBadge}</div>
    <div class="answer">${renderAnswer(data.answer || "", citations)}</div>
    ${citeHtml}
    ${traceBlock(data.trace)}
  </div>`;

  node._citations = citations;
  node.querySelectorAll("[data-marker]").forEach((el) => {
    el.addEventListener("click", () => {
      const c = citations.find((x) => x.marker === Number(el.dataset.marker));
      if (c) openSource(c);
    });
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function fillError(node, msg) {
  node.querySelector(".bubble").outerHTML =
    `<div class="bubble error">Lỗi: ${escapeHtml(msg)}</div>`;
}

/* Mo anh trang PDF tuong ung voi citation. */
function openSource(c) {
  sourcePane.classList.remove("hidden");
  $("sourceTitle").textContent = `Trang ${c.page_label}`;
  $("sourceMeta").innerHTML = `
    <div>${escapeHtml(c.section)}</div>
    <div><span class="kv">Trang giấy</span> ${escapeHtml(c.page_label)} &nbsp;·&nbsp;
         <span class="kv">Trang PDF</span> ${c.page_index}</div>
    <div><span class="kv">Đoạn</span> <code>${escapeHtml(c.chunk_id)}</code> &nbsp;·&nbsp;
         <span class="kv">Độ khớp</span> ${Number(c.score).toFixed(2)}</div>`;
  $("sourceBody").innerHTML =
    `<img src="/api/page/${c.page_index}" alt="Trang ${c.page_label}" />`;
  $("sourceBody").scrollTop = 0;
}

async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  addUserMessage(text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  const node = addPlaceholder();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    sessionId = data.session_id;
    fillBotMessage(node, data);
  } catch (err) {
    fillError(node, err.message);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

/* ---------- wiring ---------- */
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  send(inputEl.value);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send(inputEl.value);
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
});

$("closeSource").addEventListener("click", () => sourcePane.classList.add("hidden"));

SUGGESTIONS.forEach((s) => {
  const b = document.createElement("button");
  b.textContent = s;
  b.addEventListener("click", () => send(s));
  $("suggestions").appendChild(b);
});

sourcePane.classList.add("hidden");

/* Chao mo dau + canh bao som neu backend chua san sang. */
(async () => {
  const node = addPlaceholder();
  try {
    const h = await (await fetch("/api/health")).json();
    const missing = Object.entries(h.keys || {})
      .filter(([, ok]) => !ok)
      .map(([k]) => k);
    let warn = "";
    if (missing.length) warn += `\n\nChưa cấu hình khoá: ${missing.join(", ")}. Kiểm tra file .env.`;
    if (!h.qdrant_points) warn += `\n\nChưa có dữ liệu trong Qdrant. Chạy lệnh ingest trước khi hỏi.`;
    node.querySelector(".bubble").outerHTML = `<div class="bubble"><div class="answer">${renderMarkdown(
      `Tôi trả lời dựa trên Báo cáo thường niên 2024 của Vietnam Airlines, và luôn dẫn nguồn về đúng trang.${warn}`
    )}</div></div>`;
    if (h.qdrant_points) {
      // Giu ten tai lieu co dau san trong HTML, chi bo sung so doan da nap.
      $("docName").textContent += ` · ${h.qdrant_points} đoạn`;
    }
  } catch {
    fillError(node, "Không kết nối được backend.");
  }
})();
