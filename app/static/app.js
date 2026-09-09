const fileInput = document.getElementById("file-input");
const pickFile = document.getElementById("pick-file");
const fileLabel = document.getElementById("file-label");
const uploadForm = document.getElementById("upload-form");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const docEmpty = document.getElementById("doc-empty");
const refreshDocs = document.getElementById("refresh-docs");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const clearChat = document.getElementById("clear-chat");
const healthPill = document.getElementById("health-pill");
const activeDocEl = document.getElementById("active-doc");

let selectedFile = null;
let sessionId = crypto.randomUUID();
let activeDocumentId = null;
let pollTimer = null;

function setStatus(text, kind) {
  uploadStatus.textContent = text;
  uploadStatus.className = `status-line ${kind || ""}`;
}

function renderEmptyChat() {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <h3>No questions yet</h3>
      <p>Upload a document, wait until processing completes, then ask a question grounded in retrieved chunks.</p>
    </div>`;
}

function appendMessage(role, html) {
  if (messagesEl.querySelector(".empty-state") || messagesEl.querySelector(".error-state")) {
    messagesEl.innerHTML = "";
  }
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function refreshHealth() {
  try {
    const health = await fetchJson("/health");
    healthPill.textContent = health.status;
    healthPill.className = `pill ${health.status === "healthy" ? "pill-ok" : "pill-warn"}`;
  } catch {
    healthPill.textContent = "offline";
    healthPill.className = "pill pill-bad";
  }
}

function statusClass(status) {
  if (status === "completed") return "pill-ok";
  if (status === "failed") return "pill-bad";
  return "pill-warn";
}

function renderDocs(docs) {
  docList.innerHTML = "";
  docEmpty.style.display = docs.length ? "none" : "block";
  docs.forEach((doc) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `doc-item ${doc.document_id === activeDocumentId ? "active" : ""}`;
    btn.innerHTML = `
      <span class="row">
        <span class="name">${escapeHtml(doc.file_name)}</span>
        <span class="pill ${statusClass(doc.status)}">${escapeHtml(doc.status)}</span>
      </span>
      <span class="meta">${doc.chunk_count ? `${doc.chunk_count} chunks` : "awaiting index"} · ${doc.document_id.slice(0, 8)}</span>
    `;
    btn.addEventListener("click", () => {
      activeDocumentId = doc.document_id;
      activeDocEl.textContent = `Scoped to ${doc.file_name}`;
      renderDocs(docs);
    });
    const del = document.createElement("button");
    del.type = "button";
    del.className = "delete-doc";
    del.textContent = "Remove";
    del.addEventListener("click", async (event) => {
      event.stopPropagation();
      await fetch(`/documents/${doc.document_id}`, { method: "DELETE" });
      if (activeDocumentId === doc.document_id) {
        activeDocumentId = null;
        activeDocEl.textContent = "All indexed documents";
      }
      await loadDocuments();
    });
    btn.appendChild(del);
    li.appendChild(btn);
    docList.appendChild(li);
  });
}

async function loadDocuments() {
  const docs = await fetchJson("/documents");
  renderDocs(docs);
  const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
  if (busy && !pollTimer) {
    pollTimer = setInterval(loadDocuments, 2500);
  }
  if (!busy && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

pickFile.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  fileLabel.textContent = selectedFile ? selectedFile.name : "No file selected";
  uploadBtn.disabled = !selectedFile;
  setStatus("");
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  setStatus("Uploading file…");
  const body = new FormData();
  body.append("file", selectedFile);
  try {
    const result = await fetchJson("/documents", { method: "POST", body });
    setStatus(`Queued ${result.filename || selectedFile.name} (${result.document_id.slice(0, 8)})`, "ok");
    selectedFile = null;
    fileInput.value = "";
    fileLabel.textContent = "No file selected";
    await loadDocuments();
  } catch (error) {
    setStatus(error.message, "error");
    uploadBtn.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = questionEl.value.trim();
  if (!query) return;
  appendMessage("user", `<p class="role">You</p><div>${escapeHtml(query)}</div>`);
  questionEl.value = "";
  askBtn.disabled = true;
  const loading = appendMessage(
    "assistant",
    `<p class="role">Assistant</p><div class="loading"><span class="dot"></span><span class="dot"></span><span class="dot"></span> Retrieving context and generating an answer</div>`
  );
  try {
    const payload = { query, session_id: sessionId, top_k: 5 };
    if (activeDocumentId) payload.document_id = activeDocumentId;
    const result = await fetchJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const sources = (result.citations || [])
      .map(
        (c) => `
        <article class="source-card">
          <div class="src-meta">${escapeHtml(c.document_name)} · page ${c.page_number ?? "n/a"} · relevance ${Number(c.relevance_score).toFixed(3)}</div>
          <div>${escapeHtml(c.chunk_text)}</div>
        </article>`
      )
      .join("");
    loading.innerHTML = `
      <p class="role">Assistant · ${escapeHtml(result.model_used)} · ${(result.processing_time_ms / 1000).toFixed(1)}s</p>
      <div>${escapeHtml(result.answer)}</div>
      ${sources ? `<div class="sources"><h4>Retrieved chunks</h4>${sources}</div>` : ""}
    `;
  } catch (error) {
    loading.className = "msg assistant error-state";
    loading.innerHTML = `<h3>Request failed</h3><p>${escapeHtml(error.message)}</p>`;
  } finally {
    askBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
});

clearChat.addEventListener("click", async () => {
  await fetch(`/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
  sessionId = crypto.randomUUID();
  activeDocumentId = null;
  activeDocEl.textContent = "All indexed documents";
  renderEmptyChat();
  loadDocuments().catch(() => {});
});

refreshDocs.addEventListener("click", () => loadDocuments().catch((e) => setStatus(e.message, "error")));

renderEmptyChat();
refreshHealth();
loadDocuments().catch((e) => setStatus(e.message, "error"));
setInterval(refreshHealth, 15000);
