/* JEV DataOps — dependency-free, same-origin workspace. */
"use strict";

const $ = (id) => document.getElementById(id);
const emptyRunList = $("run-list").firstElementChild.cloneNode(true);
const state = {
  datasets: [], runs: [], datasetId: null, runId: null, health: null,
  token: sessionStorage.getItem("jev_api_token") || "", loading: false,
  uploading: false, submitting: false, polling: false, timer: null,
  toastTimer: null, logsSignature: "", artifactsSignature: "", actionPending: false, settlePending: false, lossChartWidth: null,
  details: new Map(),
  records: { key: "", decision: "review", reason: "", offset: 0, hasMore: false, loading: false },
};
// Measured on typesafe/jev-1.13 with the bundled rubrics: $0.0023 for 81 rows.
const COST_PER_ROW = 0.0000285;
const reasonNames = {
  exact_duplicate: "Exact duplicate", content_length_outside_bounds: "Length outside the allowed range",
  invalid_content: "Unreadable content", invalid_response: "Malformed JEV answer", missing_supported_content: "No supported text fields",
  demo_email_pattern: "Email address (demo rule)", cancelled: "Cancelled before evaluation",
  keep_probability_below_threshold: "low probability on keep", reject_probability_above_threshold: "reject probability too high",
  confidence_below_threshold: "JEV confidence below threshold",
};
function humanReason(label) {
  return String(label).split(", ").map((part) => {
    const [dimension, cause] = part.includes(":") ? part.split(":") : [null, part];
    const text = reasonNames[cause] || cause.replaceAll("_", " ");
    return dimension ? `${dimension} · ${text}` : text;
  }).join(" + ");
}
const statuses = { queued: "Queued", running: "Running", completed: "Completed", failed: "Failed", cancelled: "Canceled" };
const rubricNames = { general: "General", finance: "Finance", code: "Code" };
const stages = [
  { key: "upload", label: "Upload" }, { key: "screening", label: "Screen" },
  { key: "data_evaluation", label: "Data eval" }, { key: "training", label: "Train" },
  { key: "model_evaluation", label: "Model eval" },
];
const number = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString("en-US") : "—";
const formatMetric = (value) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
const finiteNonnegative = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const isActive = (run) => run && ["running", "queued"].includes(run.status);
function el(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}
function displayDate(value, timeOnly = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", timeOnly ? { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false } : { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(date);
}
function fileSize(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let size = Number(value), unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit++; }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}
function showToast(message, error = false) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("toast-error", error);
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 6500 : 3500);
}
function errorMessage(error) {
  return error instanceof TypeError ? "Cannot connect to the server. Check that it is running and try again." : error.message || "Something went wrong. Please try again.";
}
function showError(error) {
  const message = errorMessage(error);
  $("global-error").textContent = message;
  $("global-error").hidden = false;
  showToast(message, true);
}
function setConnection(online) {
  $("connection").classList.toggle("offline", !online);
  $("connection-label").textContent = online ? "Connected" : "Disconnected";
}
async function request(path, { method = "GET", body, raw = false } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body); }
  const response = await fetch(path, { method, body, headers, credentials: "same-origin" });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      const detail = payload.detail || payload.message;
      if (typeof detail === "string") message = detail;
      else if (Array.isArray(detail)) message = detail.map((item) => `${(item.loc || []).filter((key) => key !== "body").join(".")}: ${item.msg}`).join("; ");
    } catch (_) { /* Keep the HTTP error when the response is not JSON. */ }
    if (response.status === 401 || response.status === 403) message = "The access token is missing or invalid. Enter your server token in Connection settings.";
    throw new Error(message);
  }
  return raw ? response : response.json();
}
function updateStartState() {
  $("start-button").disabled = !state.datasetId || state.loading || state.uploading || state.submitting;
  $("start-button").querySelector("span").textContent = state.submitting ? "Creating workflow…" : "Start workflow";
  $("start-hint").textContent = state.datasetId ? "Runs and artifacts are saved on the server. Come back anytime." : "Add a dataset to get started";
}
function applyHealth(health) {
  state.health = health;
  $("version").textContent = `v${health.version || "0.1"}`;
  for (const [value, name] of [["openrouter", "OpenRouter"], ["typesafe", "TypeSafe"]]) {
    const option = $("provider").querySelector(`option[value="${value}"]`);
    option.disabled = !health.providers?.[value];
    option.textContent = `JEV · ${name}${option.disabled ? " (not configured)" : ""}`;
  }
  const trainer = $("trainer").querySelector('option[value="huggingface"]');
  trainer.disabled = !health.training?.huggingface;
  trainer.textContent = `Hugging Face · LoRA${trainer.disabled ? " (not installed)" : ""}`;
  if ($( "provider").selectedOptions[0]?.disabled) $("provider").value = "demo";
  if ($( "trainer").selectedOptions[0]?.disabled) $("trainer").value = "demo";
  if (health.max_upload_mb) $("upload-subtitle").textContent = `JSONL / CSV · Up to ${number(health.max_upload_mb)} MB · Streamed upload`;
  updateModeNotice();
}
function updateModeNotice() {
  const provider = $("provider").value;
  const trainer = $("trainer").value;
  const autoTrain = $("auto-train").checked;
  const title = $("mode-notice").querySelector("strong");
  const description = $("mode-notice").querySelector("p > span");
  $("mode-notice").classList.toggle("real-mode", provider !== "demo");
  $("trainer").disabled = !autoTrain;
  for (const id of ["confidence", "concurrency", "max-requests"]) {
    $(id).disabled = provider === "demo";
    $(id).closest(".confidence-field, .field").classList.toggle("inactive", provider === "demo");
  }
  $("demo-controls-note").hidden = provider !== "demo";
  updateBudget();
  if (provider === "demo") {
    title.textContent = autoTrain && trainer === "huggingface" ? "Local screening + LLM training" : "Demo mode";
    description.textContent = autoTrain && trainer === "huggingface"
      ? `Local rules screen the data without JEV. Retained records train a LoRA adapter for ${state.health?.training?.base_model || "the configured model"}.`
      : autoTrain ? "Local rules and a byte-bigram model validate the pipeline. No JEV calls or LLM training." : "Local screening and data evaluation only. No JEV calls or model training.";
  } else {
    title.textContent = "JEV screening enabled";
    description.textContent = !autoTrain ? "Data is sent to your selected JEV provider. Only submit data you can share. The request limit caps API usage."
      : trainer === "demo" ? "Data is sent to your selected JEV provider. Only submit data you can share. A byte-bigram model validates training; no LLM is trained."
      : `Data is sent to your JEV provider. Only submit data you can share. Retained records train a LoRA adapter for ${state.health?.training?.base_model || "the configured model"}.`;
  }
  updateRubricNotice();
}
// Every unique row costs at least one request, and a malformed answer is asked for
// again, so a limit below the row count ends the run incomplete and blocks training.
function updateBudget() {
  const note = $("budget-note"), raise = $("budget-raise");
  const dataset = state.datasets.find((item) => item.id === state.datasetId);
  const rows = Number(dataset?.rows);
  if ($("provider").value === "demo" || !finiteNonnegative(rows) || !rows) { note.hidden = true; return; }
  const limit = Number($("max-requests").value) || 0;
  const cost = rows * COST_PER_ROW;
  const estimate = `About $${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)} for ${number(rows)} rows at the measured rate; the actual cost is recorded in the run.`;
  const suggested = Math.min(1000000, Math.ceil(rows * 1.1));
  const short = limit < rows;
  note.classList.toggle("warning", short);
  $("budget-text").textContent = short
    ? `The request limit (${number(limit)}) is below the ${number(rows)} rows in this dataset. Screening will stop after about ${number(limit)} rows, the run will be incomplete, and training will not start. ${estimate}`
    : estimate;
  raise.hidden = !short;
  raise.textContent = `Raise limit to ${number(suggested)}`;
  raise.dataset.value = String(suggested);
  note.hidden = false;
}
function updateRubricNotice() {
  const rubric = $("rubric").value;
  const name = rubricNames[rubric] || rubricNames.general;
  if ($("provider").value === "demo") {
    $("rubric-note").textContent = `${name} selected. Demo uses local rules without domain understanding. Connect JEV to apply basic domain screening.`;
    return;
  }
  const limitation = rubric === "code" ? "It does not execute code or verify domain facts." : rubric === "finance" ? "It does not verify financial facts or replace expert review." : "It does not verify domain facts or execute code.";
  $("rubric-note").textContent = `${name} screening checks content quality, privacy, and trainability. ${limitation}`;
}
function renderOverview() {
  $("overview-datasets").textContent = number(state.datasets.length);
  const rows = state.datasets.map((dataset) => dataset.rows);
  $("overview-records").textContent = rows.every(finiteNonnegative) ? number(rows.reduce((sum, value) => sum + value, 0)) : "—";
  $("overview-completed").textContent = number(state.runs.filter((run) => run.status === "completed").length);
  const active = state.runs.filter(isActive).length;
  $("overview-run-note").textContent = `${number(state.runs.length)} recent run${state.runs.length === 1 ? "" : "s"} loaded${active ? ` · ${number(active)} active` : ""}`;
}
function renderDatasets() {
  const select = $("dataset-select");
  select.replaceChildren();
  if (!state.datasets.length) select.append(el("option", "", "Upload a dataset to get started"));
  for (const dataset of state.datasets) {
    const option = el("option", "", `${dataset.name} · ${number(dataset.rows)} rows`);
    option.value = dataset.id;
    select.append(option);
  }
  select.disabled = !state.datasets.length;
  if (!state.datasets.some((dataset) => dataset.id === state.datasetId)) state.datasetId = state.datasets[0]?.id || null;
  select.value = state.datasetId || "";
  $("nav-dataset-count").textContent = number(state.datasets.length);
  renderOverview();
  renderDatasetPreview();
  updateStartState();
  updateBudget();
}
function renderDatasetPreview() {
  const dataset = state.datasets.find((item) => item.id === state.datasetId);
  $("dataset-summary").hidden = !dataset;
  $("empty-preview").hidden = Boolean(dataset);
  $("preview-wrap").hidden = !dataset || !dataset.preview?.length;
  if (!dataset) return;
  $("dataset-name").textContent = dataset.name;
  $("dataset-name").title = dataset.name;
  $("dataset-meta").textContent = `${number(dataset.rows)} rows · ${fileSize(dataset.size)}`;
  const table = $("preview-table");
  const head = table.querySelector("thead"), body = table.querySelector("tbody");
  head.replaceChildren(); body.replaceChildren();
  const rows = (dataset.preview || []).slice(0, 5);
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row || {})))].slice(0, 5);
  const header = el("tr");
  const rowHeader = el("th", "", "#"); rowHeader.scope = "col"; header.append(rowHeader);
  for (const key of keys) { const cell = el("th", "", key); cell.scope = "col"; header.append(cell); }
  head.append(header);
  rows.forEach((row, index) => {
    const tr = el("tr"); tr.append(el("td", "", String(index + 1).padStart(2, "0")));
    for (const key of keys) {
      const value = row[key] === undefined ? "—" : typeof row[key] === "object" ? JSON.stringify(row[key]) : String(row[key]);
      const td = el("td", "", value); td.title = value; tr.append(td);
    }
    body.append(tr);
  });
}
function renderRunList() {
  const list = $("run-list");
  const focusedRun = list.contains(document.activeElement) ? document.activeElement.dataset.runId : null;
  $("history-count").textContent = number(state.runs.length);
  $("nav-run-count").textContent = number(state.runs.length);
  $("run-total").textContent = `${number(state.runs.length)} RUN${state.runs.length === 1 ? "" : "S"}`;
  renderOverview();
  if (!state.runs.length) { list.replaceChildren(emptyRunList.cloneNode(true)); return; }
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const button = el("button", `run-list-button${run.id === state.runId ? " selected" : ""}`);
    button.type = "button"; button.dataset.runId = run.id;
    button.setAttribute("aria-pressed", String(run.id === state.runId));
    button.append(el("div", "run-list-name", run.name || `Run ${run.id.slice(0, 8)}`));
    const meta = el("div", "run-list-meta");
    meta.append(el("span", `status-badge status-${Object.hasOwn(statuses, run.status) ? run.status : "queued"}`, statuses[run.status] || run.status), el("span", "", displayDate(run.created_at)));
    button.append(meta, el("div", "run-list-provider", `${run.config?.provider === "demo" ? "DEMO · Local validation" : `JEV · ${run.config?.provider || "—"}`} / ${run.id.slice(0, 8)}`));
    button.addEventListener("click", () => selectRun(run.id));
    fragment.append(button);
  }
  list.replaceChildren(fragment);
  if (focusedRun) [...list.querySelectorAll("button")].find((button) => button.dataset.runId === focusedRun)?.focus({ preventScroll: true });
}
function renderStages(run) {
  const track = $("stage-track"); track.replaceChildren();
  const complete = run.status === "completed";
  const noTraining = run.config?.auto_train === false;
  const index = complete ? (noTraining ? 2 : 4) : Math.max(0, stages.findIndex((stage) => stage.key === run.stage));
  stages.forEach((stage, position) => {
    const skipped = noTraining && position > 2;
    const done = !skipped && (position < index || complete && position <= index);
    const active = !skipped && !complete && position === index;
    const failed = active && ["failed", "cancelled"].includes(run.status);
    const item = el("div", `run-stage${done ? " done" : ""}${active ? " active" : ""}${failed ? " failed" : ""}`);
    item.append(el("span", "run-stage-symbol", skipped ? "–" : done ? "✓" : failed ? "!" : String(position + 1).padStart(2, "0")), el("span", "", skipped ? `${stage.label} · Skipped` : stage.label));
    if (active) item.setAttribute("aria-current", "step");
    track.append(item);
  });
}
function renderProgress(run) {
  const progress = run.progress || {};
  const dataset = state.datasets.find((item) => item.id === run.dataset_id);
  let processed = Number(progress.processed ?? run.data_report?.processed ?? 0);
  let total = Number(progress.total ?? run.data_report?.counts?.total ?? dataset?.rows ?? 0);
  let label = stages.find((stage) => stage.key === run.stage)?.label || "Workflow";
  if (["training", "model_evaluation"].includes(run.stage) && progress.max_steps) {
    processed = Number(progress.step || 0); total = Number(progress.max_steps); label = "Training steps";
  }
  if (progress.stage === "loading_model") label = "Loading the base model";
  if (progress.stage === "splitting") label = "Creating held-out data splits";
  if (["evaluating", "evaluating_baseline"].includes(progress.stage)) label = progress.stage === "evaluating_baseline" ? "Evaluating the baseline model" : "Evaluating the trained model";
  if (run.status === "queued") label = "Queued for local execution";
  if (run.status === "completed") { label = "Workflow completed"; processed = total; }
  if (run.status === "cancelled") label += " · Canceled";
  if (run.status === "failed") label += " · Failed";
  $("progress-label").textContent = label;
  $("progress-numbers").textContent = total ? `${number(processed)} / ${number(total)}` : isActive(run) ? "Processing…" : "—";
  if (!total && isActive(run)) $("run-progress").removeAttribute("value");
  else $("run-progress").value = run.status === "completed" ? 100 : total > 0 ? Math.min(100, Math.max(0, processed / total * 100)) : 0;
  $("run-progress").setAttribute("aria-label", label);
}
function renderScreeningSummary(run) {
  const report = run.data_report;
  const summary = $("screening-summary"); const notice = $("screening-notice");
  summary.hidden = true; notice.hidden = true;
  if (!report) return;
  const parts = [];
  if (report.mode === "jev_api") {
    parts.push(`${number(report.api_requests || 0)} JEV requests`);
    if (report.cache_hits) parts.push(`${number(report.cache_hits)} from cache`);
    const cost = report.usage?.cost;
    if (finiteNonnegative(cost)) parts.push(`$${cost.toFixed(4)}`);
    const models = Object.keys(report.models || {});
    if (models.length) parts.push(models.join(", "));
    if (report.unevaluated) parts.push(`${number(report.unevaluated)} not evaluated`);
  } else if (report.mode) {
    parts.push("Demo rules only, no JEV model");
  }
  if (report.counts?.duplicates) parts.push(`${number(report.counts.duplicates)} duplicates (${report.dedupe || "whitespace"} match)`);
  if (parts.length) { summary.textContent = parts.join(" · "); summary.hidden = false; }
  if (report.notice && report.complete && report.training_ready === false) { notice.textContent = report.notice; notice.hidden = false; }
}
function renderDimensionReport(report) {
  const container = $("dimension-report"); container.replaceChildren();
  const dimensions = Object.entries(report?.dimensions || {});
  if (!dimensions.length) {
    container.append(el("p", "chart-note", report?.mode === "jev_api" ? "No rows reached JEV." : "Demo rules do not score dimensions."));
    return;
  }
  const table = el("table", "dimension-table");
  table.append(el("caption", "sr-only", "Decisions per screening dimension"));
  const head = el("tr");
  for (const label of ["Dimension", "Keep", "Review", "Reject", "Mean P(answer)", "Mean confidence", "Gate", "Turned to review"]) {
    const cell = el("th", "", label); cell.scope = "col"; head.append(cell);
  }
  const thead = el("thead"); thead.append(head);
  const tbody = el("tbody");
  for (const [name, stats] of dimensions) {
    const gate = report.thresholds?.[name];
    const gateText = gate ? [`P ≥ ${gate.min_probability}`, `reject ≤ ${gate.max_reject_probability}`, gate.use_confidence ? `confidence ≥ ${gate.min_confidence}` : null].filter(Boolean).join(" · ") : "—";
    const triggered = Object.entries(stats.gates || {}).map(([key, count]) => `${reasonNames[key] || key}: ${number(count)}`).join("; ") || "—";
    const row = el("tr");
    const title = el("th", "", name); title.scope = "row"; row.append(title);
    for (const value of [number(stats.keep), number(stats.review), number(stats.reject),
      finiteNonnegative(stats.mean_probability) ? stats.mean_probability.toFixed(2) : "—",
      finiteNonnegative(stats.mean_confidence) ? stats.mean_confidence.toFixed(2) : "—", gateText, triggered]) row.append(el("td", "", value));
    tbody.append(row);
  }
  table.append(thead, tbody);
  const wrap = el("div", "table-scroll"); wrap.append(table);
  container.append(wrap);
}
function recordText(record) {
  let messages = record?.messages;
  if (typeof messages === "string") { try { messages = JSON.parse(messages); } catch (_) { messages = null; } }
  if (Array.isArray(messages)) return messages.map((message) => `${message?.role ?? "?"}: ${message?.content ?? ""}`).join("\n");
  if (typeof record?.text === "string") return record.text;
  if (record?.instruction !== undefined) return [record.instruction, record.input, record.output].filter((part) => typeof part === "string" && part).join("\n→ ");
  if (record?.prompt !== undefined) return `${record.prompt}\n→ ${record.response ?? ""}`;
  return JSON.stringify(record);
}
function renderRecordsSection(run) {
  const section = $("records-section");
  const report = run.data_report;
  const held = (run.counts?.review || 0) + (run.counts?.reject || 0);
  section.hidden = !report || !held;
  if (section.hidden) return;
  const records = state.records;
  const reasons = report.decision_reasons?.[records.decision] || {};
  for (const button of section.querySelectorAll(".segment")) {
    const decision = button.dataset.decision;
    button.setAttribute("aria-pressed", String(decision === records.decision));
    button.textContent = `${decision === "review" ? "Review" : "Reject"} · ${number(run.counts?.[decision] || 0)}`;
  }
  const select = $("records-reason");
  const options = [["", "All reasons"], ...Object.entries(reasons).sort((a, b) => b[1] - a[1]).map(([key, count]) => [key, `${humanReason(key)} (${number(count)})`])];
  if (select.dataset.signature !== JSON.stringify(options)) {
    select.dataset.signature = JSON.stringify(options);
    select.replaceChildren(...options.map(([value, label]) => { const option = el("option", "", label); option.value = value; return option; }));
  }
  if (!options.some(([value]) => value === records.reason)) records.reason = "";
  select.value = records.reason;
  const list = $("reason-list"); list.replaceChildren();
  for (const [key, count] of Object.entries(reasons).sort((a, b) => b[1] - a[1]).slice(0, 5)) {
    const item = el("li", "reason-item"); item.append(el("span", "", humanReason(key)), el("strong", "", number(count))); list.append(item);
  }
  const key = `${run.id}:${run.attempt ?? 0}:${records.decision}:${records.reason}`;
  if (key !== records.key) { records.key = key; loadRecords(run.id, true); }
}
async function loadRecords(runId, reset) {
  const records = state.records;
  if (reset) { records.offset = 0; $("records-list").replaceChildren(el("p", "chart-note", "Loading records…")); }
  const key = records.key;
  records.loading = true; $("records-more").disabled = true;
  try {
    const query = new URLSearchParams({ decision: records.decision, limit: "20", offset: String(records.offset) });
    if (records.reason) query.set("reason", records.reason);
    const page = await request(`/api/runs/${encodeURIComponent(runId)}/records?${query}`);
    if (key !== records.key) return;
    const list = $("records-list");
    if (reset) list.replaceChildren();
    for (const entry of page.records) list.append(recordCard(entry));
    if (!list.children.length) list.append(el("p", "chart-note", "No records in this partition."));
    records.offset += page.records.length; records.hasMore = page.has_more;
    $("records-count").textContent = `Showing ${number(records.offset)}${page.has_more ? "+" : ""}`;
  } catch (error) { if (key === records.key) $("records-list").replaceChildren(el("p", "chart-note", errorMessage(error))); }
  finally { records.loading = false; $("records-more").hidden = !records.hasMore; $("records-more").disabled = false; }
}
function recordCard(entry) {
  const card = el("article", "record-card");
  const header = el("div", "record-header");
  header.append(el("span", "record-line", `Line ${number(entry.line)}`), el("span", "record-reason", humanReason(entry.reason)));
  card.append(header);
  const text = recordText(entry.record);
  const body = el("p", `record-text${text.trim() ? "" : " empty"}`, text.trim() ? (text.length > 600 ? `${text.slice(0, 600)}…` : text) : "(empty content)");
  card.append(body);
  if (text.length > 600) {
    const toggle = el("button", "text-button record-toggle", "Show all");
    toggle.type = "button"; toggle.setAttribute("aria-expanded", "false");
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") !== "true";
      body.textContent = open ? text : `${text.slice(0, 600)}…`;
      toggle.textContent = open ? "Show less" : "Show all"; toggle.setAttribute("aria-expanded", String(open));
    });
    card.append(toggle);
  }
  const dimensions = Object.entries(entry.dimensions || {});
  if (dimensions.length) {
    const pills = el("div", "record-dimensions");
    for (const [name, dimension] of dimensions) {
      const probability = finiteNonnegative(dimension.probability) ? ` · P ${dimension.probability.toFixed(2)}` : "";
      const confidence = finiteNonnegative(dimension.confidence) ? ` · conf ${dimension.confidence.toFixed(2)}` : "";
      pills.append(el("span", `dimension-pill ${dimension.decision}`, `${name}: ${dimension.value}${probability}${confidence}`));
    }
    card.append(pills);
  }
  if (entry.detail) card.append(el("p", "record-detail", entry.detail));
  return card;
}
function renderDistribution(run) {
  const categories = [["keep", "Keep"], ["review", "Review"], ["reject", "Reject"]];
  const counts = run.counts || {};
  const measured = categories.every(([key]) => finiteNonnegative(counts[key]));
  const total = measured ? categories.reduce((sum, [key]) => sum + counts[key], 0) : 0;
  $("distribution-section").hidden = !total;
  const chart = $("distribution-chart"); chart.replaceChildren();
  if (!total) return;
  const bar = el("div", "distribution-bar");
  // The adjacent legend carries the same values in accessible text.
  bar.setAttribute("aria-hidden", "true");
  const legend = el("ul", "distribution-legend");
  for (const [key, label] of categories) {
    const percentage = counts[key] / total * 100;
    const segment = el("span", `distribution-segment ${key}`);
    segment.style.width = `${percentage}%`;
    segment.title = `${label}: ${number(counts[key])} records · ${percentage.toFixed(1)}%`;
    bar.append(segment);
    const item = el("li", "distribution-item");
    const dot = el("span", `distribution-dot ${key}`); dot.setAttribute("aria-hidden", "true");
    item.append(dot, el("span", "distribution-label", label), el("span", "distribution-value", `${number(counts[key])} · ${percentage.toFixed(1)}%`));
    legend.append(item);
  }
  chart.append(bar, legend);
  $("distribution-note").textContent = `Based on ${number(total)} classified records${isActive(run) ? " · Updates as screening progresses" : ""}`;
}
function renderLossChart(report, demo) {
  const byStep = new Map();
  for (const point of Array.isArray(report?.loss_history) ? report.loss_history : []) {
    if (point && finiteNonnegative(point.step) && finiteNonnegative(point.loss)) byStep.set(point.step, point.loss);
  }
  const points = [...byStep].map(([step, loss]) => ({ step, loss })).sort((a, b) => a.step - b.step);
  $("loss-chart-section").hidden = points.length < 2;
  const chart = $("loss-chart"); chart.replaceChildren();
  if (points.length < 2) return;
  const svg = (tag, attributes = {}, text) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    if (text !== undefined) element.textContent = String(text);
    return element;
  };
  const width = Math.min(680, Math.max(280, chart.clientWidth || 680));
  state.lossChartWidth = width;
  const height = 206, left = 54, right = 20, top = 20, bottom = 36;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const first = points[0], last = points[points.length - 1];
  let minLoss = Infinity, maxLoss = -Infinity;
  for (const point of points) { minLoss = Math.min(minLoss, point.loss); maxLoss = Math.max(maxLoss, point.loss); }
  const padding = Math.max((maxLoss - minLoss) * 0.16, maxLoss * 0.04, 0.05);
  const yMin = Math.max(0, minLoss - padding), yMax = maxLoss + padding;
  const x = (step) => left + (step - first.step) / (last.step - first.step) * plotWidth;
  const y = (loss) => top + (yMax - loss) / (yMax - yMin) * plotHeight;
  const root = svg("svg", { class: "loss-svg", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": "loss-svg-title loss-svg-description" });
  root.append(svg("title", { id: "loss-svg-title" }, `${demo ? "Demo byte-bigram model" : "LLM"} training loss`));
  root.append(svg("desc", { id: "loss-svg-description" }, `${points.length} recorded measurements. Step ${first.step}: ${formatMetric(first.loss)}; step ${last.step}: ${formatMetric(last.loss)}. Minimum ${formatMetric(minLoss)}, maximum ${formatMetric(maxLoss)}. Vertical axis: ${formatMetric(yMin)} to ${formatMetric(yMax)}.`));
  for (const value of [yMax, (yMax + yMin) / 2, yMin]) {
    root.append(svg("line", { class: "loss-grid", x1: left, x2: width - right, y1: y(value), y2: y(value) }));
    root.append(svg("text", { class: "loss-axis-label", x: left - 10, y: y(value) + 4, "text-anchor": "end" }, value.toFixed(2)));
  }
  const line = points.map((point, index) => `${index ? "L" : "M"}${x(point.step).toFixed(2)},${y(point.loss).toFixed(2)}`).join(" ");
  root.append(svg("path", { class: "loss-area", d: `${line} L${x(last.step)},${top + plotHeight} L${x(first.step)},${top + plotHeight} Z`, "aria-hidden": "true" }));
  root.append(svg("path", { class: "loss-line", d: line, fill: "none", "aria-hidden": "true" }));
  for (const point of [first, last]) {
    const marker = svg("circle", { class: "loss-point", cx: x(point.step), cy: y(point.loss), r: 4 });
    marker.append(svg("title", {}, `Step ${number(point.step)} · NLL ${formatMetric(point.loss)}`));
    root.append(marker);
  }
  root.append(svg("text", { class: "loss-axis-label", x: left, y: height - 9 }, `Step ${number(first.step)}`));
  root.append(svg("text", { class: "loss-axis-label", x: width - right, y: height - 9, "text-anchor": "end" }, `Step ${number(last.step)}`));
  chart.append(root);
  $("loss-chart-note").textContent = `${number(points.length)} recorded steps · NLL / ${demo ? "UTF-8 byte (including EOS)" : "model token"} · Training-batch loss, separate from held-out evaluation.`;
}
function renderModelReport(report) {
  $("model-report-section").hidden = !report;
  if (!report) { renderLossChart(null, false); return; }
  const demo = report.is_llm === false || report.trainer === "demo" || report.mode === "demo";
  $("evaluation-kind").textContent = demo ? "DEMO · BYTE BIGRAM" : "LLM · HELD-OUT TEST";
  const metrics = $("model-metrics"); metrics.replaceChildren();
  const comparable = [report.baseline_loss, report.trained_loss].every(finiteNonnegative);
  const ceiling = comparable ? Math.max(report.baseline_loss, report.trained_loss) : 0;
  for (const [kind, label, value, perplexity] of [["baseline", "Before · Test NLL ↓", report.baseline_loss, report.baseline_perplexity], ["trained", "After · Test NLL ↓", report.trained_loss, report.trained_perplexity]]) {
    const metric = el("div", "model-metric"); metric.append(el("span", "", label), el("strong", "", finiteNonnegative(value) ? formatMetric(value) : "—"));
    metric.append(el("small", "model-metric-unit", demo ? "nats / UTF-8 byte + EOS" : "nats / model token"));
    if (comparable) {
      const meter = el("div", "model-meter"); meter.setAttribute("aria-hidden", "true");
      const fill = el("span", `model-meter-fill ${kind}`); fill.style.width = `${ceiling ? value / ceiling * 100 : 0}%`;
      meter.append(fill); metric.append(meter);
    }
    if (finiteNonnegative(perplexity)) metric.append(el("small", "", `Perplexity ${formatMetric(perplexity)}`));
    if (kind === "trained" && typeof report.delta_loss === "number" && Number.isFinite(report.delta_loss)) {
      const delta = report.delta_loss;
      metric.append(el("span", `metric-delta ${delta < 0 ? "improved" : delta > 0 ? "worsened" : "unchanged"}`, `${delta < 0 ? "↓" : delta > 0 ? "↑" : "="} ${Math.abs(delta).toFixed(4)} NLL ${delta < 0 ? "decrease" : delta > 0 ? "increase" : "unchanged"}`));
    }
    metrics.append(metric);
  }
  const notes = [];
  if (typeof report.delta_loss === "number" && Number.isFinite(report.delta_loss)) notes.push(`Δ loss ${report.delta_loss > 0 ? "+" : ""}${report.delta_loss.toFixed(4)} (negative means lower loss)`);
  if (report.split_counts) notes.push(`Train / validation / test: ${["train", "validation", "test"].map((key) => number(report.split_counts[key] || 0)).join(" / ")}`);
  notes.push(demo ? "Demo uses a UTF-8 byte-bigram model, not an LLM. Its metrics are not directly comparable to LLM token loss." : "Loss is compared on the same held-out test set. It does not measure task accuracy or production performance.");
  $("evaluation-note").textContent = notes.join(" · ");
  renderLossChart(report, demo);
}
function renderLogs(run) {
  const logs = run.logs || [];
  const signature = `${run.id}:${JSON.stringify(logs)}`;
  if (signature === state.logsSignature) return;
  state.logsSignature = signature;
  const container = $("run-logs");
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 35;
  const previousTop = container.scrollTop;
  const fragment = document.createDocumentFragment();
  if (!logs.length) fragment.append(el("span", "", "Waiting for output…"));
  for (const log of logs) {
    const row = el("div", "log-row"); row.append(el("span", "log-time", displayDate(log.time, true)), el("span", "log-text", log.message ?? "")); fragment.append(row);
  }
  container.replaceChildren(fragment);
  container.scrollTop = nearBottom ? container.scrollHeight : previousTop;
  $("log-count").textContent = `${logs.length} entries`;
}
function renderArtifacts(run) {
  const artifacts = run.artifacts || [];
  $("artifact-section").hidden = !artifacts.length;
  const signature = `${run.id}:${JSON.stringify(artifacts)}`;
  if (signature === state.artifactsSignature) return;
  state.artifactsSignature = signature;
  $("artifacts").replaceChildren();
  for (const artifact of artifacts) {
    const button = el("button", "artifact-button"); button.type = "button";
    button.append(el("span", "", "↓"), el("span", "", artifact.name));
    button.title = `${artifact.name}${artifact.size !== undefined ? ` · ${fileSize(artifact.size)}` : ""}`;
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const url = new URL(artifact.url, location.origin);
        if (url.origin !== location.origin || !url.pathname.startsWith("/api/runs/")) throw new Error("Invalid artifact download URL.");
        if (!state.token) {
          // Let the browser stream large local downloads directly to disk.
          const link = el("a"); link.href = url.pathname + url.search; link.download = artifact.name.split("/").pop() || "artifact";
          document.body.append(link); link.click(); link.remove(); return;
        }
        const response = await request(url.pathname + url.search, { raw: true });
        const blob = await response.blob();
        const objectURL = URL.createObjectURL(blob);
        const link = el("a"); link.href = objectURL; link.download = artifact.name.split("/").pop() || "artifact";
        document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(objectURL), 10000);
      } catch (error) { showError(error); }
      finally { button.disabled = false; }
    });
    $("artifacts").append(button);
  }
}
function renderRun() {
  const run = state.runs.find((item) => item.id === state.runId);
  $("run-empty").hidden = Boolean(run);
  $("run-content").hidden = !run;
  if (!run) return;
  $("run-name").textContent = run.name || `Run ${run.id.slice(0, 8)}`;
  $("run-status").textContent = statuses[run.status] || run.status;
  $("run-status").className = `status-badge status-${Object.hasOwn(statuses, run.status) ? run.status : "queued"}`;
  const mode = run.config?.provider === "demo" ? "Demo rule screening" : `JEV ${run.config?.provider || ""}`;
  $("run-meta").textContent = `${run.id.slice(0, 8)} · ${displayDate(run.created_at)} · ${mode} · Domain: ${rubricNames[run.config?.rubric] || rubricNames.general}`;
  $("cancel-button").hidden = !isActive(run); $("cancel-button").disabled = state.actionPending;
  $("retry-button").hidden = !["failed", "cancelled"].includes(run.status); $("retry-button").disabled = state.actionPending;
  $("run-error").hidden = !run.error; $("run-error").textContent = run.error || "";
  renderStages(run); renderProgress(run);
  for (const key of ["keep", "review", "reject"]) $( `count-${key}`).textContent = run.counts?.[key] !== undefined ? number(run.counts[key]) : "—";
  renderScreeningSummary(run);
  renderDistribution(run);
  renderRecordsSection(run);
  renderModelReport(run.model_report);
  $("data-report-section").hidden = !run.data_report;
  $("data-report").textContent = run.data_report ? JSON.stringify(run.data_report, null, 2) : "";
  if (run.data_report) renderDimensionReport(run.data_report);
  renderLogs(run); renderArtifacts(run);
}
function upsertRun(run) {
  state.details.set(run.id, run);
  const position = state.runs.findIndex((item) => item.id === run.id);
  if (position < 0) state.runs.unshift(run); else state.runs[position] = run;
}
// The list endpoint returns summaries; the selected run's logs and reports come
// from its own endpoint, fetched again only when the run has changed.
function mergeSummaries(summaries) {
  state.runs = summaries.map((summary) => {
    const detail = state.details.get(summary.id);
    return detail && detail.updated_at === summary.updated_at ? detail : summary;
  });
}
async function refreshSelected() {
  const id = state.runId;
  const summary = state.runs.find((run) => run.id === id);
  if (!summary || state.details.get(id)?.updated_at === summary.updated_at) return;
  upsertRun(await request(`/api/runs/${encodeURIComponent(id)}`));
}
async function selectRun(id) {
  state.runId = id;
  renderRunList(); renderRun();
  try {
    upsertRun(await request(`/api/runs/${encodeURIComponent(id)}`));
    if (state.runId === id) { renderRunList(); renderRun(); }
    schedulePoll();
  } catch (error) { showError(error); }
}
async function loadWorkspace() {
  if (state.loading) return;
  state.loading = true; $("refresh-button").disabled = true; updateStartState();
  try {
    const [health, datasets, runs] = await Promise.all([request("/api/health"), request("/api/datasets"), request("/api/runs?view=summary")]);
    applyHealth(health);
    state.datasets = datasets;
    state.details.clear(); state.records.key = "";
    mergeSummaries(runs);
    if (!state.runs.some((run) => run.id === state.runId)) state.runId = state.runs[0]?.id || null;
    await refreshSelected();
    renderDatasets(); renderRunList(); renderRun();
    $("global-error").hidden = true; setConnection(true);
  } catch (error) { setConnection(false); showError(error); }
  finally { state.loading = false; $("refresh-button").disabled = false; updateStartState(); schedulePoll(); }
}
function schedulePoll(delay = 1500) {
  clearTimeout(state.timer);
  if (state.runs.some(isActive) || state.settlePending) state.timer = setTimeout(pollRuns, delay);
}
async function pollRuns() {
  if (state.polling) return schedulePoll();
  if (document.hidden) return schedulePoll(5000);
  state.polling = true;
  let failed = false;
  try {
    const runs = await request("/api/runs?view=summary");
    const priorSelected = state.runs.find((run) => run.id === state.runId);
    // One follow-up fetch also picks up artifacts finalized after terminal status.
    state.settlePending = state.runs.some((previous) => isActive(previous) && runs.some((current) => current.id === previous.id && !isActive(current)));
    mergeSummaries(runs);
    await refreshSelected();
    renderRunList(); renderRun(); setConnection(true);
    const current = state.runs.find((run) => run.id === state.runId);
    if (isActive(priorSelected) && current && !isActive(current)) showToast(`Workflow ${String(statuses[current.status] || current.status).toLowerCase()}`, current.status === "failed");
  } catch (_) { failed = true; setConnection(false); }
  finally { state.polling = false; schedulePoll(failed ? 5000 : 1500); }
}
function setUploading(uploading) {
  state.uploading = uploading;
  $("drop-zone").classList.toggle("uploading", uploading);
  $("drop-zone").setAttribute("aria-busy", String(uploading));
  $("file-input").disabled = uploading; $("example-button").disabled = uploading;
  $("upload-progress").hidden = !uploading;
  $("upload-title").textContent = uploading ? "Uploading and checking your data…" : "Drop your dataset here";
  updateStartState();
}
async function uploadDataset(file, example = false) {
  if (state.uploading) return;
  if (!example) {
    if (!file) return;
    if (!/\.(jsonl|csv)$/i.test(file.name)) return showToast("Choose a UTF-8 encoded .jsonl or .csv file.", true);
    if (state.health?.max_upload_mb && file.size > state.health.max_upload_mb * 1024 * 1024) return showToast(`The file exceeds the server limit of ${number(state.health.max_upload_mb)} MB.`, true);
    if (!file.size) return showToast("The file is empty. Choose a dataset with at least one record.", true);
  }
  setUploading(true);
  try {
    const form = new FormData(); if (file) form.append("file", file);
    const dataset = await request(example ? "/api/datasets/example" : "/api/datasets", { method: "POST", body: example ? undefined : form });
    state.datasets.unshift(dataset); state.datasetId = dataset.id;
    renderDatasets(); $("global-error").hidden = true;
    showToast(`${example ? "Example added" : "Upload complete"} · ${number(dataset.rows)} records`);
  } catch (error) { showError(error); }
  finally { setUploading(false); $("file-input").value = ""; }
}
$("pipeline-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.datasetId || state.submitting || !event.currentTarget.reportValidity()) return;
  state.submitting = true; updateStartState();
  try {
    const run = await request("/api/runs", { method: "POST", body: {
      dataset_id: state.datasetId, provider: $("provider").value, trainer: $("trainer").value, rubric: $("rubric").value,
      confidence: Number($("confidence").value), concurrency: Number($("concurrency").value),
      max_requests: Number($("max-requests").value), auto_train: $("auto-train").checked,
    } });
    upsertRun(run); state.runId = run.id; renderRunList(); renderRun(); schedulePoll();
    $("global-error").hidden = true; showToast("Workflow created. Processing will begin shortly.");
    $("run-section").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (error) { showError(error); }
  finally { state.submitting = false; updateStartState(); }
});
async function runAction(action) {
  if (!state.runId || state.actionPending) return;
  const runId = state.runId;
  state.actionPending = true; renderRun();
  try {
    const run = await request(`/api/runs/${encodeURIComponent(runId)}/${action}`, { method: "POST" });
    upsertRun(run); if (state.runId === runId) state.runId = run.id;
    renderRunList(); renderRun(); schedulePoll();
    showToast(action === "cancel" ? "Stop requested. The run will exit after the current step." : "Workflow restarted.");
  } catch (error) { showError(error); }
  finally { state.actionPending = false; renderRun(); }
}
$("cancel-button").addEventListener("click", () => runAction("cancel"));
$("retry-button").addEventListener("click", () => runAction("retry"));
$("confidence").addEventListener("input", () => { $("confidence-value").textContent = Number($("confidence").value).toFixed(2); });
for (const id of ["provider", "trainer", "auto-train"]) $(id).addEventListener("change", updateModeNotice);
$("rubric").addEventListener("change", updateRubricNotice);
$("dataset-select").addEventListener("change", (event) => { state.datasetId = event.target.value; renderDatasetPreview(); updateStartState(); updateBudget(); });
$("max-requests").addEventListener("input", updateBudget);
for (const button of document.querySelectorAll("#records-section .segment")) {
  button.addEventListener("click", () => { state.records.decision = button.dataset.decision; state.records.reason = ""; renderRun(); });
}
$("records-reason").addEventListener("change", (event) => { state.records.reason = event.target.value; renderRun(); });
$("records-more").addEventListener("click", () => { if (state.runId && !state.records.loading) loadRecords(state.runId, false); });
$("budget-raise").addEventListener("click", () => { $("max-requests").value = $("budget-raise").dataset.value; updateBudget(); });
$("file-input").addEventListener("change", (event) => uploadDataset(event.target.files[0]));
$("example-button").addEventListener("click", () => uploadDataset(null, true));
$("refresh-button").addEventListener("click", loadWorkspace);
let dragDepth = 0;
$("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth++; if (!state.uploading) $("drop-zone").classList.add("drag-over"); });
$("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
$("drop-zone").addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-zone").classList.remove("drag-over"); });
$("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-zone").classList.remove("drag-over"); if (event.dataTransfer?.files.length > 1) showToast("Upload one dataset at a time. The first file was selected."); uploadDataset(event.dataTransfer?.files[0]); });
$("settings-button").addEventListener("click", () => { $("api-token").value = state.token; $("settings-dialog").showModal(); });
$("close-settings").addEventListener("click", () => $("settings-dialog").close());
$("settings-dialog").addEventListener("click", (event) => { if (event.target === $("settings-dialog")) { const bounds = event.target.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close(); } });
$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault(); state.token = $("api-token").value.trim();
  if (state.token) sessionStorage.setItem("jev_api_token", state.token); else sessionStorage.removeItem("jev_api_token");
  $("settings-dialog").close(); loadWorkspace();
});
const navigation = [...document.querySelectorAll('nav a[href^="#"]')]
  .map((link) => ({ link, section: $(link.hash.slice(1)) })).filter((item) => item.section);
function highlightNavigation(active) {
  for (const { link } of navigation) {
    link.classList.toggle("active", link === active);
    if (link === active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
for (const { link } of navigation) link.addEventListener("click", () => highlightNavigation(link));
if ("IntersectionObserver" in window && navigation.length) {
  const observer = new IntersectionObserver(() => {
    let active = navigation[0].link;
    for (const item of navigation) if (item.section.getBoundingClientRect().top <= window.innerHeight * 0.3) active = item.link;
    highlightNavigation(active);
  }, { rootMargin: "-10% 0px -60% 0px", threshold: [0, 1] });
  for (const { section } of navigation) observer.observe(section);
}
if ("ResizeObserver" in window) {
  const chart = $("loss-chart");
  const observer = new ResizeObserver(() => {
    if (!chart.clientWidth || $("loss-chart-section").hidden) return;
    const width = Math.min(680, Math.max(280, chart.clientWidth));
    if (width === state.lossChartWidth) return;
    const report = state.runs.find((run) => run.id === state.runId)?.model_report;
    if (report) renderLossChart(report, report.is_llm === false || report.trainer === "demo" || report.mode === "demo");
  });
  observer.observe(chart);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && state.runs.some(isActive)) schedulePoll(0); });
loadWorkspace();
