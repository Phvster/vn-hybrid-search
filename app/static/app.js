/* Hybrid Research — front-end controller.
   Security: corpus text and model output are only ever inserted with textContent (never innerHTML). */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const STORE = { sessions: "hr.sessions.v1", theme: "hr.theme", sidebar: "hr.sidebar" };
const MAX_SESSIONS = 30;
const CONFIG_INFO = {
  r0: { name: "BM25 · âm tiết", desc: "Tách theo khoảng trắng: mỗi âm tiết là một token." },
  r1: { name: "BM25 · tách từ", desc: "pyvi ghép âm tiết thành từ trước khi đánh chỉ mục." },
  r2: { name: "Dense · bge-m3", desc: "Mã hóa ngữ nghĩa, tìm lân cận bằng FAISS." },
  r3: { name: "Hybrid · RRF", desc: "Hợp nhất R1 và R2 bằng thứ hạng. Cấu hình đề xuất." },
  r4: { name: "Hybrid + Reranker", desc: "R3 rồi xếp lại top-20 bằng cross-encoder." },
};
const STAGES = [
  { id: "r1", kind: "lex", icon: "i-lexical", label: "BM25 · tách từ (R1)", desc: "Khớp từ khóa sau khi pyvi ghép âm tiết thành từ." },
  { id: "r2", kind: "sem", icon: "i-semantic", label: "Dense · bge-m3 (R2)", desc: "Mã hóa câu hỏi thành vector, tìm điều luật gần nghĩa nhất." },
  { id: "fusion", kind: "fuse", icon: "i-fusion", label: "Hợp nhất RRF (R3)", desc: "Gộp hai bảng xếp hạng top-100: điểm = Σ 1 / (k + hạng)." },
  { id: "gen", kind: "gen", icon: "i-spark", label: "Claude Haiku · trả lời có trích dẫn", desc: "Chỉ dựa trên 5 điều luật được truy xuất." },
];

const state = { info: null, sessions: [], currentId: null, busy: false, compare: null, metrics: null, metric: "ndcg@10" };

/* ---------- helpers ---------- */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
function icon(id, cls = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  if (cls) svg.setAttribute("class", cls);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
}
const fmt = (v, digits = 3) => v.toLocaleString("vi-VN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const fmtMs = (ms) => (ms >= 1000 ? `${fmt(ms / 1000, 2)} s` : ms < 10 ? `${fmt(ms, 1)} ms` : `${Math.round(ms)} ms`);
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
function load(key, fallback) {
  try { const raw = localStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw); } catch { return fallback; }
}
function save(key, value) {
  // Keys starting with "_" hold live DOM references and are never persisted.
  const replacer = (k, v) => (k.startsWith("_") ? undefined : v);
  try { localStorage.setItem(key, JSON.stringify(value, replacer)); return true; } catch { return false; }
}
let toastTimer;
function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
async function api(path, body) {
  let res;
  try {
    res = await fetch(path, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Không kết nối được máy chủ. Hãy kiểm tra server demo còn chạy không.");
  }
  if (!res.ok) {
    let detail = `Máy chủ trả lỗi ${res.status}.`;
    try { const data = await res.json(); if (typeof data.detail === "string") detail = data.detail; } catch { /* keep default */ }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

/* ---------- theme + sidebar ---------- */
const THEMES = [
  { id: "system", label: "Theo hệ thống", icon: "i-auto" },
  { id: "light", label: "Giao diện sáng", icon: "i-sun" },
  { id: "dark", label: "Giao diện tối", icon: "i-moon" },
];
function applyTheme(id) {
  const theme = THEMES.find((t) => t.id === id) || THEMES[0];
  if (theme.id === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme.id;
  $("#theme-label").textContent = theme.label;
  $("#theme-btn use").setAttribute("href", `#${theme.icon}`);
  $("#theme-btn").setAttribute("aria-label", `Giao diện: ${theme.label}. Bấm để đổi`);
  save(STORE.theme, theme.id);
}
function cycleTheme() {
  const current = load(STORE.theme, "system");
  const next = THEMES[(THEMES.findIndex((t) => t.id === current) + 1) % THEMES.length];
  applyTheme(next.id);
}
function setCollapsed(collapsed) {
  $("#shell").classList.toggle("collapsed", collapsed);
  const btn = $("#collapse-btn");
  btn.setAttribute("aria-expanded", String(!collapsed));
  btn.setAttribute("aria-label", collapsed ? "Mở rộng thanh bên" : "Thu gọn thanh bên");
  save(STORE.sidebar, collapsed ? "collapsed" : "expanded");
}
function openDrawer() {
  $("#shell").classList.add("drawer-open");
  $("#scrim").hidden = false;
  $("#menu-btn").setAttribute("aria-expanded", "true");
  $("#new-session").focus();
}
function closeDrawer(returnFocus = true) {
  if (!$("#shell").classList.contains("drawer-open")) return;
  $("#shell").classList.remove("drawer-open");
  $("#scrim").hidden = true;
  $("#menu-btn").setAttribute("aria-expanded", "false");
  if (returnFocus) $("#menu-btn").focus();
}

/* ---------- routing ---------- */
const ROUTES = { "#/": "ask", "#/compare": "compare", "#/benchmark": "benchmark" };
const TITLES = { ask: "Hỏi đáp", compare: "So sánh cấu hình", benchmark: "Benchmark" };
function route() {
  const view = ROUTES[location.hash] || "ask";
  for (const section of document.querySelectorAll(".view")) section.hidden = section.dataset.view !== view;
  for (const link of document.querySelectorAll(".nav-item")) {
    if (link.dataset.route === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("#page-title").textContent = TITLES[view];
  document.title = view === "ask" ? "Hybrid Research" : `${TITLES[view]} · Hybrid Research`;
  closeDrawer(false);
  if (view === "benchmark") loadMetrics();
  if (view === "compare") renderCompare();
}

/* ---------- sessions + history ---------- */
function persistSessions() {
  state.sessions.sort((a, b) => b.updated - a.updated);
  state.sessions = state.sessions.slice(0, MAX_SESSIONS);
  while (!save(STORE.sessions, state.sessions) && state.sessions.length > 1) state.sessions.pop();
  renderHistory();
}
function currentSession() { return state.sessions.find((s) => s.id === state.currentId) || null; }
function renderHistory() {
  const list = $("#history-list");
  list.replaceChildren();
  for (const session of state.sessions) {
    const isCurrent = session.id === state.currentId;
    list.append(el("li", { class: "history-item", "aria-current": isCurrent ? "true" : null },
      el("button", { class: "history-open", type: "button", title: session.title, text: session.title, onclick: () => openSession(session.id) }),
      el("button", { class: "icon-btn history-del", type: "button", "aria-label": `Xóa phiên: ${session.title}`, onclick: () => deleteSession(session.id) }, icon("i-trash")),
    ));
  }
  const empty = state.sessions.length === 0;
  $("#history-empty").hidden = !empty;
  $("#clear-history").hidden = empty;
}
function newSession() {
  state.currentId = null;
  renderHistory();
  renderThread();
  if (location.hash !== "#/" && location.hash !== "") location.hash = "#/";
  closeDrawer(false);
  $("#prompt").focus();
}
function openSession(id) {
  state.currentId = id;
  renderHistory();
  renderThread();
  if (location.hash !== "#/" && location.hash !== "") location.hash = "#/";
  closeDrawer(false);
}
function deleteSession(id) {
  state.sessions = state.sessions.filter((s) => s.id !== id);
  if (state.currentId === id) { state.currentId = null; renderThread(); }
  persistSessions();
  toast("Đã xóa phiên tra cứu.");
}

/* ---------- welcome ---------- */
function greeting() {
  const h = new Date().getHours();
  return h < 11 ? "Chào buổi sáng" : h < 18 ? "Chào buổi chiều" : "Chào buổi tối";
}
function renderWelcome() {
  const size = state.info ? state.info.corpus_size.toLocaleString("vi-VN") : "61.425";
  const examples = state.info ? state.info.examples : [];
  return el("div", { class: "welcome" },
    el("div", { class: "hello" },
      icon("i-logo", "hello-mark"),
      el("h2", {}, `${greeting()}. `, el("span", { class: "accent", text: "Bạn cần tra cứu quy định nào?" })),
      el("p", { text: `Hybrid Research tìm trong ${size} điều luật Việt Nam bằng hai cách cùng lúc, khớp từ khóa và hiểu ngữ nghĩa, rồi tổng hợp câu trả lời kèm trích dẫn nguyên văn để bạn kiểm chứng.` }),
    ),
    el("div", { class: "capabilities" },
      el("div", { class: "capability cap-lex" }, icon("i-lexical"), el("div", {}, el("strong", { text: "Khớp từ khóa chính xác" }), el("span", { text: "BM25 trên từ đã tách: mạnh với số hiệu, mức phạt, thuật ngữ." }))),
      el("div", { class: "capability cap-sem" }, icon("i-semantic"), el("div", {}, el("strong", { text: "Hiểu ngữ nghĩa" }), el("span", { text: "bge-m3 tìm được điều luật diễn đạt khác câu hỏi." }))),
      el("div", { class: "capability cap-fuse" }, icon("i-quote"), el("div", {}, el("strong", { text: "Trích dẫn kiểm chứng được" }), el("span", { text: "Mỗi ý trong câu trả lời trỏ về đúng đoạn văn bản gốc." }))),
    ),
    examples.length ? el("section", { "aria-labelledby": "suggest-title" },
      el("h3", { class: "section-label", id: "suggest-title", text: "Thử một câu hỏi" }),
      el("div", { class: "suggestions" }, examples.map((ex) =>
        el("button", { class: "suggestion", type: "button", onclick: () => ask(ex.query) },
          el("span", { class: "tag", text: ex.topic }), el("span", { class: "q", text: ex.query })))),
    ) : null,
    el("div", { class: "quick-actions" },
      el("a", { class: "btn btn-ghost btn-sm", href: "#/compare" }, icon("i-columns"), "So sánh R0–R3 trên một câu hỏi"),
      el("a", { class: "btn btn-ghost btn-sm", href: "#/benchmark" }, icon("i-chart"), "Xem kết quả benchmark"),
    ),
  );
}

/* ---------- thread + turns ---------- */
function renderThread() {
  const thread = $("#thread");
  thread.replaceChildren();
  const session = currentSession();
  if (!session) { thread.append(renderWelcome()); return; }
  for (const turn of session.turns) thread.append(buildTurn(turn).root);
  // Start at the latest question rather than the bottom of its sources.
  const last = thread.lastElementChild;
  if (last) thread.scrollTop = last.offsetTop - thread.offsetTop - 16;
}

function buildTurn(turn) {
  const pipe = el("details", { class: "pipeline", open: true });
  const answerSlot = el("div", { class: "answer-slot" });
  const sourcesSlot = el("section", { class: "sources-slot", "aria-label": "Nguồn tham khảo" });
  const actions = el("div", { class: "turn-actions" });
  const root = el("article", { class: "turn", "aria-label": `Câu hỏi: ${turn.query}` },
    el("div", { class: "turn-query", text: turn.query }), pipe, answerSlot, sourcesSlot, actions);
  const view = { root, pipe, answerSlot, sourcesSlot, actions };
  turn._view = view;
  paintTurn(turn);
  return view;
}

function stageState(turn, id) {
  if (id === "gen") {
    if (turn.mode !== "answer") return "skipped";
    if (turn.answer) return "done";
    if (turn.answerError) return "error";
    return turn.search ? "running" : "pending";
  }
  if (turn.search) return "done";
  return turn.searchError ? "error" : "running";
}

function paintPipeline(turn) {
  const { pipe } = turn._view;
  const running = !turn.search && !turn.searchError || (turn.mode === "answer" && turn.search && !turn.answer && !turn.answerError);
  const measured = turn.search ? turn.search.stages.reduce((sum, s) => sum + s.ms, 0) + (turn.answer ? turn.answer.ms : 0) : 0;
  const summary = el("summary", {},
    el("span", { class: running ? "pipe-icon spin" : "pipe-icon" }, icon(running ? "i-retry" : "i-check")),
    el("span", { text: running ? "Đang truy xuất…" : "Quy trình truy xuất" }),
    measured ? el("span", { class: "total", text: `· ${fmtMs(measured)}` }) : null,
    icon("i-chevron", "chev"),
  );
  const items = STAGES.map((stage) => {
    const st = stageState(turn, stage.id);
    const server = turn.search ? turn.search.stages.find((s) => s.id === stage.id) : null;
    let ms = "";
    if (server) ms = fmtMs(server.ms);
    else if (stage.id === "gen" && turn.answer) ms = fmtMs(turn.answer.ms);
    else if (st === "running") ms = "đang chạy";
    else if (st === "skipped") ms = "bỏ qua";
    else if (st === "error") ms = "lỗi";
    return el("li", { class: "stage", "data-kind": stage.kind, "data-state": st },
      el("span", { class: "stage-dot" }, icon(stage.icon)),
      el("span", { class: "stage-text" }, el("strong", { text: server ? server.label : stage.label }), el("span", { text: stage.desc })),
      el("span", { class: "stage-ms", text: ms }),
    );
  });
  pipe.replaceChildren(summary, el("ol", { class: "stages", "aria-label": "Các bước truy xuất" }, items));
}

function skeleton(lines = 4) {
  return el("div", { class: "skeleton", "aria-hidden": "true" }, Array.from({ length: lines }, () => el("div", { class: "sk-line" })));
}
function notice(kind, iconId, message, action) {
  return el("div", { class: `notice ${kind}`, role: kind === "error" ? "alert" : null },
    icon(iconId), el("span", { text: message }), action || null);
}

/* Inline **bold** without innerHTML. */
function inline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((p) => (p.startsWith("**") && p.endsWith("**") ? el("strong", { text: p.slice(2, -2) }) : document.createTextNode(p)));
}
/* Model text arrives as segments with citation refs; lay it out as paragraphs/lists with chips. */
function renderAnswerBody(answer, onCite) {
  const blocks = [[]];
  for (const seg of answer.segments) {
    const lines = seg.text.split("\n");
    lines.forEach((line, i) => {
      if (i > 0) blocks.push([]);
      if (line) blocks[blocks.length - 1].push(line);
    });
    for (const n of seg.refs) blocks[blocks.length - 1].push({ cite: n });
  }
  const body = el("div", { class: "answer-body" });
  let list = null;
  for (const block of blocks) {
    if (!block.length) { list = null; continue; }
    const first = typeof block[0] === "string" ? block[0] : "";
    const bullet = first.match(/^\s*[-*•]\s+/);
    const numbered = first.match(/^\s*\d+[.)]\s+/);
    const heading = first.match(/^\s*#{1,4}\s+/);
    const marker = bullet || numbered || heading;
    const content = block.flatMap((piece, i) => {
      if (typeof piece !== "string") {
        return [el("button", { class: "cite", type: "button", "aria-label": `Xem nguồn ${piece.cite}`, onclick: () => onCite(piece.cite), text: String(piece.cite) })];
      }
      return inline(i === 0 && marker ? piece.slice(marker[0].length) : piece);
    });
    if (bullet || numbered) {
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) { list = el(tag); body.append(list); }
      list.append(el("li", {}, content));
    } else {
      list = null;
      body.append(heading ? el("p", {}, el("strong", {}, content)) : el("p", {}, content));
    }
  }
  return body;
}

function answerText(answer) {
  return answer.segments.map((s) => s.text + s.refs.map((n) => ` [${n}]`).join("")).join("").trim();
}

function paintAnswer(turn) {
  const { answerSlot } = turn._view;
  answerSlot.replaceChildren();
  if (turn.mode !== "answer") return;
  if (turn.searchError) return;
  if (state.info && !state.info.has_key && !turn.answer) {
    answerSlot.append(notice("", "i-key", "Chức năng trả lời đang tắt vì máy chủ chưa cấu hình ANTHROPIC_API_KEY. Các nguồn truy xuất vẫn hiển thị bên dưới."));
    return;
  }
  if (turn.answerError) {
    const retry = turn.answerError.status === 429 ? null
      : el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => runAnswer(turn) }, icon("i-retry"), "Thử lại");
    answerSlot.append(notice("error", "i-alert", turn.answerError.message, retry));
    return;
  }
  const head = el("div", { class: "answer-head" },
    el("span", { class: "ai-badge" }, icon("i-spark"), "AI"),
    el("span", { text: "Câu trả lời tổng hợp" }),
    el("span", { class: "model", text: "· Claude Haiku" }),
  );
  if (!turn.answer) {
    answerSlot.append(el("section", { class: "answer", "aria-busy": "true", "aria-label": "Đang tạo câu trả lời" }, head, skeleton(4)));
    return;
  }
  const a = turn.answer;
  if (a.refused) {
    answerSlot.append(notice("", "i-alert", "Mô hình từ chối trả lời câu hỏi này."));
    return;
  }
  const meta = el("div", { class: "answer-meta" },
    el("span", { text: `${a.input_tokens.toLocaleString("vi-VN")} + ${a.output_tokens.toLocaleString("vi-VN")} token` }),
    el("span", { text: `≈ $${a.cost_usd.toFixed(4)}` }),
    el("span", { text: fmtMs(a.ms) }),
    a.session_remaining !== undefined ? el("span", { text: `Còn ${a.session_remaining} lượt trả lời trong phiên` }) : null,
  );
  answerSlot.append(el("section", { class: "answer", "aria-label": "Câu trả lời do AI tạo" },
    head, renderAnswerBody(a, (n) => focusSource(turn, n)), meta, el("p", { class: "disclaimer", text: a.disclaimer })));
}

function rankPill(kind, iconId, name, rank, depth) {
  if (!rank) return el("span", { class: "pill pill-miss", title: `Không nằm trong top-${depth} của ${name}` }, icon(iconId), `${name} ngoài top-${depth}`);
  return el("span", { class: `pill pill-${kind}`, title: `${name} xếp điều luật này ở hạng ${rank}` }, icon(iconId), `${name} #${rank}`);
}
/* Wrap every cited quote inside the full text in <mark>, using text nodes only. */
function highlighted(text, quotes) {
  const ranges = [];
  for (const q of quotes) {
    const needle = q.trim();
    if (!needle) continue;
    const at = text.indexOf(needle);
    if (at >= 0) ranges.push([at, at + needle.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const out = [];
  let pos = 0;
  for (const [start, end] of ranges) {
    if (start < pos) continue;
    out.push(document.createTextNode(text.slice(pos, start)), el("mark", { text: text.slice(start, end) }));
    pos = end;
  }
  out.push(document.createTextNode(text.slice(pos)));
  return out;
}

function paintSources(turn) {
  const { sourcesSlot } = turn._view;
  sourcesSlot.replaceChildren();
  if (turn.searchError) {
    sourcesSlot.append(notice("error", "i-alert", turn.searchError.message,
      el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => runTurn(turn) }, icon("i-retry"), "Thử lại")));
    return;
  }
  const head = el("div", { class: "sources-head" },
    el("h3", { text: "Nguồn tham khảo" }),
    el("span", { class: "sub", text: "Top-5 của R3, hợp nhất từ BM25 và Dense" }),
  );
  if (!turn.search) {
    sourcesSlot.append(head, el("div", { class: "sources" }, [0, 1, 2].map(() => el("div", { class: "source" }, skeleton(3)))));
    return;
  }
  const { sources, depth } = turn.search;
  const quotes = (turn.answer && turn.answer.quotes) || {};
  if (!sources.length) {
    sourcesSlot.append(head, el("div", { class: "empty-state" }, icon("i-doc"), "Không tìm thấy điều luật nào khớp với câu hỏi."));
    return;
  }
  const cards = sources.map((src, i) => {
    const cited = quotes[String(src.n)] || [];
    const panelId = `full-${turn.id}-${src.n}`;
    const wrap = el("div", { class: "fulltext", id: panelId, tabindex: "0", hidden: true, "aria-label": `Toàn văn ${src.title}` },
      highlighted(src.text, cited), src.truncated ? "\n… (chỉ hiển thị 6.000 ký tự đầu)" : "");
    const toggle = el("button", { class: "expand-btn", type: "button", "aria-expanded": "false", "aria-controls": panelId,
      onclick: (e) => {
        wrap.hidden = !wrap.hidden;
        const open = !wrap.hidden;
        e.currentTarget.setAttribute("aria-expanded", String(open));
        e.currentTarget.firstChild.textContent = open ? "Thu gọn" : "Xem toàn văn";
      } }, "Xem toàn văn", icon("i-chevron"));
    const card = el("article", { class: cited.length ? "source cited" : "source", "data-n": src.n, tabindex: "-1" },
      el("span", { class: "source-n", "aria-hidden": "true", text: String(src.n) }),
      el("div", { class: "source-top" },
        el("span", { class: "law", text: src.law_id }),
        el("div", { class: "pills" },
          rankPill("lex", "i-lexical", "BM25", src.r1_rank, depth),
          rankPill("sem", "i-semantic", "Dense", src.r2_rank, depth),
          el("span", { class: "pill pill-fuse", title: "Điểm RRF" }, icon("i-fusion"), `RRF ${fmt(src.rrf, 4)}`),
        ),
      ),
      el("h4", { class: "source-title", text: `[${src.n}] ${src.title}` }),
      cited.length
        ? el("div", { class: "quotes" }, el("span", { class: "quote-label", text: "Đoạn được trích dẫn" }),
          cited.map((q) => el("blockquote", { class: "quote" }, icon("i-quote"), el("span", { text: q }))))
        : el("p", { class: "snippet", text: src.snippet }),
      toggle, wrap,
    );
    card.style.setProperty("--i", i);
    return card;
  });
  sourcesSlot.append(head, el("div", { class: "sources" }, cards));
}

function paintActions(turn) {
  const { actions } = turn._view;
  actions.replaceChildren();
  if (!turn.search) return;
  if (turn.answer && !turn.answer.refused) {
    actions.append(el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: async () => {
      try { await navigator.clipboard.writeText(answerText(turn.answer)); toast("Đã sao chép câu trả lời."); }
      catch { toast("Trình duyệt không cho phép sao chép."); }
    } }, icon("i-copy"), "Sao chép câu trả lời"));
  }
  actions.append(el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => {
    state.compare = { query: turn.query, data: turn.search };
    location.hash = "#/compare";
  } }, icon("i-columns"), "So sánh R0–R3 cho câu này"));
}

function paintTurn(turn) {
  if (!turn._view) return;
  paintPipeline(turn);
  paintAnswer(turn);
  paintSources(turn);
  paintActions(turn);
}

function focusSource(turn, n) {
  const card = turn._view.sourcesSlot.querySelector(`[data-n="${n}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("flash");
  void card.offsetWidth; // restart the highlight animation
  card.classList.add("flash");
  card.focus({ preventScroll: true });
  setTimeout(() => card.classList.remove("flash"), 1600);
}

/* ---------- ask flow ---------- */
function storeTurn(session) {
  session.updated = Date.now();
  persistSessions();
}
/* A reload in the middle of a request leaves a turn half-finished; mark it instead of spinning forever. */
function settleInterrupted(sessions) {
  const interrupted = { status: 0, message: "Yêu cầu bị gián đoạn trước khi hoàn tất. Bấm Thử lại." };
  for (const s of sessions) {
    for (const t of s.turns || []) {
      if (!t.search && !t.searchError) t.searchError = interrupted;
      else if (t.mode === "answer" && t.search && !t.answer && !t.answerError && t.search.sources.length) t.answerError = interrupted;
    }
  }
  return sessions;
}

async function runAnswer(turn) {
  turn.answerError = null;
  turn.answer = null;
  paintTurn(turn);
  try {
    turn.answer = await api("/api/answer", { query: turn.query, doc_ids: turn.search.sources.map((s) => s.doc_id) });
  } catch (err) {
    turn.answerError = { status: err.status, message: err.message };
  }
  paintTurn(turn);
}

async function runTurn(turn) {
  turn.searchError = null;
  turn.search = null;
  turn.answer = null;
  turn.answerError = null;
  paintTurn(turn);
  try {
    turn.search = await api("/api/search", { query: turn.query });
  } catch (err) {
    turn.searchError = { status: err.status, message: err.message };
  }
  paintTurn(turn);
  if (turn.search && turn.mode === "answer" && turn.search.sources.length && state.info && state.info.has_key) {
    await runAnswer(turn);
  }
}

async function ask(query) {
  query = query.trim();
  if (!query || state.busy) return;
  if (location.hash !== "#/" && location.hash !== "") location.hash = "#/";
  const mode = document.querySelector('input[name="mode"]:checked').value;
  let session = currentSession();
  if (!session) {
    session = { id: uid(), title: query.slice(0, 80), updated: Date.now(), turns: [] };
    state.sessions.unshift(session);
    state.currentId = session.id;
    $("#thread").replaceChildren();
  }
  const turn = { id: uid(), query, mode, search: null, answer: null, searchError: null, answerError: null };
  session.turns.push(turn);
  renderHistory();
  const view = buildTurn(turn);
  $("#thread").append(view.root);
  view.root.scrollIntoView({ behavior: "smooth", block: "start" });
  const prompt = $("#prompt");
  prompt.value = "";
  syncComposer();
  setBusy(true);
  try {
    await runTurn(turn);
  } finally {
    setBusy(false);
    storeTurn(session);
  }
}

/* ---------- composer ---------- */
function syncComposer() {
  const prompt = $("#prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 200)}px`;
  const len = prompt.value.length;
  const max = state.info ? state.info.max_query_chars : 500;
  const counter = $("#counter");
  counter.textContent = `${len}/${max}`;
  counter.classList.toggle("near", len > max * 0.9);
  $("#send-btn").disabled = state.busy || !prompt.value.trim();
}
function setBusy(busy) {
  state.busy = busy;
  $("#send-btn").classList.toggle("busy", busy);
  $("#send-btn").setAttribute("aria-label", busy ? "Đang xử lý" : "Gửi câu hỏi");
  syncComposer();
}

/* ---------- compare ---------- */
function renderCompare() {
  const grid = $("#compare-grid");
  const legend = $("#compare-legend");
  if (!state.compare) {
    legend.hidden = true;
    grid.replaceChildren(el("div", { class: "empty-state" }, icon("i-columns"),
      el("span", { text: "Nhập một câu hỏi ở trên, hoặc bấm “So sánh R0–R3” bên dưới một câu trả lời." })));
    grid.style.gridTemplateColumns = "1fr";
    return;
  }
  grid.style.gridTemplateColumns = "";
  $("#compare-input").value = state.compare.query;
  if (state.compare.loading) {
    legend.hidden = true;
    grid.replaceChildren(...["r0", "r1", "r2", "r3"].map(() => el("div", { class: "config-col" }, skeleton(6))));
    return;
  }
  if (state.compare.error) {
    legend.hidden = true;
    grid.replaceChildren(notice("error", "i-alert", state.compare.error));
    grid.style.gridTemplateColumns = "1fr";
    return;
  }
  const data = state.compare.data.compare;
  const counts = {};
  for (const name of Object.keys(data)) for (const h of data[name].hits) counts[h.doc_id] = (counts[h.doc_id] || 0) + 1;
  const total = Object.keys(data).length;
  legend.hidden = false;
  const cols = Object.entries(data).map(([name, col], i) => {
    const info = CONFIG_INFO[name];
    const hits = col.hits.map((h, rank) => {
      const shared = counts[h.doc_id] > 1;
      const item = el("li", { class: shared ? "hit shared" : "hit", tabindex: "0", "data-doc": h.doc_id,
        "aria-label": `${name.toUpperCase()} hạng ${rank + 1}: ${h.title}. Có ở ${counts[h.doc_id]} trên ${total} cấu hình`,
        onmouseenter: () => matchDoc(h.doc_id), onmouseleave: () => matchDoc(null), onfocus: () => matchDoc(h.doc_id), onblur: () => matchDoc(null) },
        el("span", { class: "hit-rank", text: String(rank + 1) }),
        el("span", {}, el("span", { class: "hit-title", text: h.title }), el("span", { class: "hit-law", text: h.law_id }),
          shared ? el("span", { class: "hit-shared", text: `Có ở ${counts[h.doc_id]}/${total} cấu hình` }) : null),
      );
      return item;
    });
    const column = el("section", { class: name === "r3" ? "config-col best" : "config-col", "aria-label": info.name },
      el("header", { class: "config-head" },
        el("div", { class: "row" }, el("span", { class: "config-id", text: name.toUpperCase() }), el("span", { class: "config-name", text: info.name }),
          el("span", { class: "config-ms", text: fmtMs(col.ms) })),
        el("span", { class: "config-desc", text: info.desc }),
      ),
      hits.length ? el("ol", { class: "hit-list" }, hits) : el("p", { class: "config-desc", text: "Không có kết quả." }),
    );
    column.style.setProperty("--i", i);
    return column;
  });
  grid.replaceChildren(...cols);
}
function matchDoc(docId) {
  for (const node of document.querySelectorAll("#compare-grid .hit")) node.classList.toggle("match", docId !== null && node.dataset.doc === docId);
}
async function runCompare(query) {
  query = query.trim();
  if (!query) { $("#compare-input").focus(); return; }
  state.compare = { query, loading: true };
  renderCompare();
  try {
    state.compare = { query, data: await api("/api/search", { query }) };
  } catch (err) {
    state.compare = { query, error: err.message };
  }
  renderCompare();
}

/* ---------- benchmark ---------- */
const METRICS = [
  { id: "ndcg@10", label: "nDCG@10", help: "Chất lượng xếp hạng của cả top-10, thưởng khi điều luật đúng nằm cao." },
  { id: "recall@10", label: "Recall@10", help: "Tỉ lệ điều luật đúng tìm được trong top-10." },
  { id: "hit_rate@10", label: "Hit@10", help: "Tỉ lệ câu hỏi có ít nhất một điều luật đúng trong top-10." },
  { id: "mrr@10", label: "MRR@10", help: "Nghịch đảo hạng của điều luật đúng đầu tiên." },
  { id: "recall@1", label: "Recall@1", help: "Điều luật đúng đứng ngay vị trí số 1." },
];
async function loadMetrics() {
  if (state.metrics) return;
  $("#kpis").replaceChildren(...[0, 1, 2, 3].map(() => el("div", { class: "kpi" }, skeleton(2))));
  try {
    state.metrics = await api("/api/metrics");
  } catch (err) {
    $("#kpis").replaceChildren(notice("error", "i-alert", err.message));
    return;
  }
  renderBenchmark();
}
function renderBenchmark() {
  const rows = state.metrics;
  if (!rows.length) {
    $("#kpis").replaceChildren(notice("", "i-alert", "Chưa có file results/metrics_test.csv trên máy chủ."));
    return;
  }
  const by = Object.fromEntries(rows.map((r) => [r.config, r]));
  const best = rows.reduce((a, b) => (b["ndcg@10"] > a["ndcg@10"] ? b : a));
  const kpis = [
    { label: "nDCG@10 cao nhất", value: fmt(best["ndcg@10"]), sub: `${best.config.toUpperCase()} · ${CONFIG_INFO[best.config].name}` },
  ];
  if (by.r3 && by.r1) kpis.push({ label: "Hybrid so với BM25 tách từ", value: `+${fmt(by.r3["ndcg@10"] - by.r1["ndcg@10"])}`, sub: "nDCG@10, R3 so với R1" });
  if (by.r3) {
    kpis.push({ label: "Hit@10 của R3", value: `${fmt(by.r3["hit_rate@10"] * 100, 1)}%`, sub: "câu hỏi có ≥ 1 điều luật đúng trong top-10" });
    kpis.push({ label: "Độ trễ R3 (p50)", value: fmtMs(by.r3.p50_ms), sub: "CPU, gồm cả mã hóa câu hỏi" });
  }
  $("#kpis").replaceChildren(...kpis.map((k, i) => {
    const node = el("div", { class: "kpi" }, el("span", { class: "kpi-label", text: k.label }), el("span", { class: "kpi-value", text: k.value }), el("span", { class: "kpi-sub", text: k.sub }));
    node.style.setProperty("--i", i);
    return node;
  }));
  renderTabs();
  renderBars();
  renderTable(rows);
}
function renderTabs() {
  const tabs = METRICS.map((m) => el("button", { class: "tab", type: "button", role: "tab", id: `tab-${m.id}`,
    "aria-selected": String(m.id === state.metric), tabindex: m.id === state.metric ? "0" : "-1", "aria-controls": "bar-chart",
    onclick: () => { state.metric = m.id; renderTabs(); renderBars(); $(`#tab-${CSS.escape(m.id)}`).focus(); },
    text: m.label }));
  const list = $("#metric-tabs");
  list.replaceChildren(...tabs);
  list.onkeydown = (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = METRICS.findIndex((m) => m.id === state.metric);
    const next = METRICS[(i + (e.key === "ArrowRight" ? 1 : METRICS.length - 1)) % METRICS.length];
    state.metric = next.id;
    renderTabs();
    renderBars();
    $(`#tab-${CSS.escape(next.id)}`).focus();
  };
}
function renderBars() {
  const metric = METRICS.find((m) => m.id === state.metric);
  const chart = $("#bar-chart");
  chart.setAttribute("aria-labelledby", `tab-${metric.id}`);
  chart.setAttribute("role", "list");
  const tooltip = el("div", { class: "tooltip", hidden: true, role: "presentation" });
  const show = (row, node) => {
    tooltip.replaceChildren(
      el("strong", { text: `${row.config.toUpperCase()} · ${CONFIG_INFO[row.config].name}` }),
      ...METRICS.map((m) => el("div", { class: "tt-row" }, el("span", { text: m.label }), el("b", { text: fmt(row[m.id]) }))),
      el("div", { class: "tt-row" }, el("span", { text: "Độ trễ p50" }), el("b", { text: fmtMs(row.p50_ms) })),
    );
    tooltip.hidden = false;
    const top = node.offsetTop + node.offsetHeight + 6;
    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${Math.min(node.querySelector(".bar-track").offsetLeft + 8, chart.clientWidth - 220)}px`;
  };
  const hide = () => { tooltip.hidden = true; };
  const rows = state.metrics.map((row) => {
    const bar = el("div", { class: "bar" });
    bar.style.setProperty("--v", row[metric.id]);
    return el("div", { class: "bar-row", role: "listitem", tabindex: "0",
      "aria-label": `${row.config.toUpperCase()} ${CONFIG_INFO[row.config].name}: ${metric.label} ${fmt(row[metric.id])}`,
      onmouseenter: (e) => show(row, e.currentTarget), onmouseleave: hide, onfocus: (e) => show(row, e.currentTarget), onblur: hide },
      el("span", { class: "bar-label" }, el("span", { class: "id", text: row.config.toUpperCase() }), el("span", { class: "name", text: CONFIG_INFO[row.config].name })),
      el("span", { class: "bar-track" }, bar),
      el("span", { class: "bar-value", text: fmt(row[metric.id]) }),
    );
  });
  const axis = el("div", { class: "chart-axis", "aria-hidden": "true" }, el("span"),
    el("span", { class: "ticks" }, ["0", "0,25", "0,5", "0,75", "1"].map((t) => el("span", { text: t }))), el("span"));
  chart.replaceChildren(...rows, axis, tooltip);
  $("#chart-note").textContent = `${metric.label}: ${metric.help} Thang đo từ 0 đến 1, cao hơn là tốt hơn. Di chuột hoặc focus vào một hàng để xem mọi chỉ số.`;
}
function renderTable(rows) {
  const cols = [...METRICS.map((m) => ({ id: m.id, label: m.label, better: "max" })), { id: "p50_ms", label: "p50", better: "min" }];
  const bestOf = Object.fromEntries(cols.map((c) => [c.id, (c.better === "max" ? Math.max : Math.min)(...rows.map((r) => r[c.id]))]));
  $("#metrics-table").replaceChildren(
    el("caption", { class: "sr-only", text: "Chỉ số truy xuất trên tập test theo từng cấu hình" }),
    el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "Cấu hình" }), cols.map((c) => el("th", { scope: "col", text: c.label })))),
    el("tbody", {}, rows.map((r) => el("tr", {},
      el("th", { scope: "row", text: `${r.config.toUpperCase()} · ${CONFIG_INFO[r.config].name}` }),
      cols.map((c) => el("td", { class: r[c.id] === bestOf[c.id] ? "best" : null, text: c.id === "p50_ms" ? fmtMs(r[c.id]) : fmt(r[c.id]) })),
    ))),
  );
}

/* ---------- boot ---------- */
async function loadInfo() {
  const dot = $("#status-dot");
  const text = $("#status-text");
  try {
    state.info = await api("/api/info");
  } catch {
    dot.className = "dot err";
    text.textContent = "Mất kết nối máy chủ";
    return;
  }
  const size = state.info.corpus_size.toLocaleString("vi-VN");
  dot.className = state.info.has_key ? "dot ok" : "dot warn";
  text.textContent = state.info.has_key ? `${size} điều luật · Trả lời: bật` : `${size} điều luật · Trả lời: tắt`;
  $("#status-row").title = state.info.has_key ? `Mô hình: ${state.info.model}` : "Máy chủ chưa cấu hình ANTHROPIC_API_KEY";
  $("#prompt").maxLength = state.info.max_query_chars;
  if (!state.info.has_key) {
    const answerOpt = document.querySelector('input[name="mode"][value="answer"]');
    answerOpt.disabled = true;
    answerOpt.closest("label").title = "Cần ANTHROPIC_API_KEY trên máy chủ";
    document.querySelector('input[name="mode"][value="search"]').checked = true;
  }
  if (!currentSession()) renderThread();
  syncComposer();
}

function init() {
  applyTheme(load(STORE.theme, "system"));
  if (load(STORE.sidebar, "expanded") === "collapsed") setCollapsed(true);
  const stored = load(STORE.sessions, []);
  state.sessions = settleInterrupted(Array.isArray(stored) ? stored : []);
  renderHistory();
  renderThread();

  $("#collapse-btn").addEventListener("click", () => setCollapsed(!$("#shell").classList.contains("collapsed")));
  $("#menu-btn").addEventListener("click", openDrawer);
  $("#close-drawer").addEventListener("click", () => closeDrawer());
  $("#scrim").addEventListener("click", () => closeDrawer());
  $("#theme-btn").addEventListener("click", cycleTheme);
  $("#new-session").addEventListener("click", newSession);
  $("#topbar-new").addEventListener("click", newSession);
  $("#clear-history").addEventListener("click", () => {
    if (!window.confirm("Xóa toàn bộ lịch sử tra cứu trên trình duyệt này?")) return;
    state.sessions = [];
    state.currentId = null;
    persistSessions();
    renderThread();
  });

  const prompt = $("#prompt");
  prompt.addEventListener("input", syncComposer);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(prompt.value); }
  });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(prompt.value); });
  $("#compare-form").addEventListener("submit", (e) => { e.preventDefault(); runCompare($("#compare-input").value); });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (e.key === "/" && !typing && (ROUTES[location.hash] || "ask") === "ask") { e.preventDefault(); prompt.focus(); }
  });
  window.addEventListener("hashchange", route);
  window.matchMedia("(max-width: 860px)").addEventListener("change", () => closeDrawer(false));

  route();
  syncComposer();
  loadInfo();
}

document.addEventListener("DOMContentLoaded", init);
