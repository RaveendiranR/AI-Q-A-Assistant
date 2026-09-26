const API_BASE = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");
const modelName = document.getElementById("modelName");

const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const clearChatBtn = document.getElementById("clearChatBtn");


function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }

  return data;
}


function renderDocuments(documents) {
  if (!documents.length) {
    documentList.innerHTML = '<div class="empty-docs">No documents indexed.</div>';
    return;
  }

  documentList.innerHTML = documents.map(doc => `
    <div class="document">
      <div class="document-main">
        <div class="document-name" title="${escapeHtml(doc.filename)}">
          ${escapeHtml(doc.filename)}
        </div>
        <div class="document-meta">
          ${doc.pages} pages • ${doc.chunks} chunks
        </div>
      </div>
      <button
        class="delete-btn"
        title="Delete document"
        data-delete="${escapeHtml(doc.filename)}"
      >×</button>
    </div>
  `).join("");

  document.querySelectorAll("[data-delete]").forEach(button => {
    button.addEventListener("click", () => deleteDocument(button.dataset.delete));
  });
}


async function refreshDocuments() {
  try {
    const data = await api("/documents");
    renderDocuments(data.documents);
  } catch (error) {
    documentList.innerHTML = `<div class="empty-docs">${escapeHtml(error.message)}</div>`;
  }
}


async function checkHealth() {
  try {
    const data = await api("/health");
    healthDot.className = "dot online";
    healthText.textContent = "API online";
    modelName.textContent = data.llm;
  } catch {
    healthDot.className = "dot offline";
    healthText.textContent = "API offline";
  }
}


fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (file.type !== "application/pdf") {
    uploadStatus.textContent = "Please select a PDF file.";
    uploadStatus.className = "status error";
    fileInput.value = "";
    return;
  }

  uploadStatus.className = "status";
  uploadStatus.textContent = `Uploading and indexing ${file.name}...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await api("/upload", {
      method: "POST",
      body: formData
    });

    uploadStatus.textContent =
      `Indexed ${data.chunks_indexed} chunks from ${data.filename}.`;

    await refreshDocuments();
    await checkHealth();
  } catch (error) {
    uploadStatus.className = "status error";
    uploadStatus.textContent = error.message;
  } finally {
    fileInput.value = "";
  }
});


async function deleteDocument(filename) {
  const confirmed = window.confirm(`Delete "${filename}"?`);
  if (!confirmed) return;

  try {
    await api(`/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE"
    });
    uploadStatus.className = "status";
    uploadStatus.textContent = `Deleted ${filename}.`;
    await refreshDocuments();
    await checkHealth();
  } catch (error) {
    uploadStatus.className = "status error";
    uploadStatus.textContent = error.message;
  }
}


function addUserMessage(question) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  wrapper.innerHTML = `
    <div class="user-bubble">${escapeHtml(question)}</div>
  `;
  chat.appendChild(wrapper);
}


function addLoadingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message";
  wrapper.innerHTML = `
    <div class="assistant-bubble">
      <div class="answer-label">Assistant</div>
      <div class="typing">Reading documents and generating an answer...</div>
    </div>
  `;
  chat.appendChild(wrapper);
  chat.scrollTop = chat.scrollHeight;
  return wrapper;
}


// Recognizes the various "couldn't find it" phrasings the backend can
// return: the LLM's own refusal (per the rag_query.py prompt), the
// no-relevant-chunks-retrieved message, and the word-count "not found"
// message. Used to decide whether to show fallback suggestions.
function looksLikeNoAnswer(answerText) {
  if (!answerText) return false;

  return /\b(i\s*couldn'?t find|i\s*could not find|couldn'?t find that information|wasn'?t found|was not found|no relevant information|does(?:n'?t| not) (?:mention|contain|cover)|doesn'?t have (?:that|this) information)\b/i
    .test(answerText);
}


function buildFallbackHtml(question) {
  const q = encodeURIComponent(question);

  return `
    <div class="fallback-actions">
      <div class="fallback-label">Not in your uploaded documents. You could:</div>
      <div class="fallback-buttons">
        <a class="fallback-btn" href="https://www.google.com/search?q=${q}" target="_blank" rel="noopener">
          🔍 Search Google
        </a>
        <a class="fallback-btn" href="https://duckduckgo.com/?q=${q}" target="_blank" rel="noopener">
          🦆 Search DuckDuckGo
        </a>
        <button type="button" class="fallback-btn" data-rephrase="${escapeHtml(question)}">
          ✏️ Rephrase &amp; retry
        </button>
      </div>
    </div>
  `;
}


function addAssistantMessage(data, loadingElement) {
  const sources = data.sources || [];

  const sourceHtml = sources.length
    ? `
      <details class="sources">
        <summary>${sources.length} source${sources.length > 1 ? "s" : ""}</summary>
        ${sources.map(source => `
          <div class="source-item">
            <div class="source-title">
              ${escapeHtml(source.source)}
              • Page ${escapeHtml(source.page)}
              ${source.ocr ? " • OCR" : ""}
              ${source.count !== undefined
                ? ` • ${source.count} occurrence${source.count !== 1 ? "s" : ""}`
                : ""}
            </div>
            ${source.preview
              ? `<div class="source-preview">${escapeHtml(source.preview)}</div>`
              : ""}
          </div>
        `).join("")}
      </details>
    `
    : "";

  const fallbackHtml = looksLikeNoAnswer(data.answer)
    ? buildFallbackHtml(data.question || questionInput.value)
    : "";

  loadingElement.innerHTML = `
    <div class="assistant-bubble">
      <div class="answer-label">Assistant</div>
      <div class="answer-text">${escapeHtml(data.answer)}</div>
      ${sourceHtml}
      ${fallbackHtml}
    </div>
  `;

  const rephraseBtn = loadingElement.querySelector("[data-rephrase]");
  if (rephraseBtn) {
    rephraseBtn.addEventListener("click", () => {
      questionInput.value = rephraseBtn.dataset.rephrase;
      questionInput.focus();
      questionInput.setSelectionRange(0, questionInput.value.length);
    });
  }

  chat.scrollTop = chat.scrollHeight;
}


askForm.addEventListener("submit", async event => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  welcome?.remove();
  addUserMessage(question);
  const loadingElement = addLoadingMessage();

  questionInput.value = "";
  askButton.disabled = true;

  try {
    const data = await api("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        k: 4
      })
    });

    addAssistantMessage(data, loadingElement);
  } catch (error) {
    loadingElement.innerHTML = `
      <div class="assistant-bubble">
        <div class="answer-label">Error</div>
        <div class="answer-text">${escapeHtml(error.message)}</div>
      </div>
    `;
  } finally {
    askButton.disabled = false;
    questionInput.focus();
  }
});


document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});


clearChatBtn.addEventListener("click", () => {
  chat.innerHTML = `
    <div id="welcome" class="welcome">
      <div class="welcome-icon">✦</div>
      <h3>Ask questions grounded in your PDFs.</h3>
      <p>Upload one or more documents, then ask a question.</p>
    </div>
  `;
});


refreshDocuments();
checkHealth();