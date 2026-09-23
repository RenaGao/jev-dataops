/* Browser-only public demo. No uploaded records leave this tab. */
"use strict";
(function (root) {
  const MAX_BYTES = 2 * 1024 * 1024, MAX_ROWS = 1000;
  const datasets = [], runs = [], payloads = new Map(), files = new Map(), screened = new Map();
  const RUN_DETAIL_FIELDS = ["logs", "data_report", "model_report", "artifacts"];
  const encoder = new TextEncoder();
  const now = () => new Date().toISOString();
  const pause = () => new Promise(resolve => setTimeout(resolve, 0));
  const id = () => crypto.randomUUID().replaceAll("-", "");
  const copy = value => JSON.parse(JSON.stringify(value));
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const jsonl = rows => rows.map(row => JSON.stringify(row)).join("\n") + "\n";
  async function hash(value) {
    return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(value))), n => n.toString(16).padStart(2, "0")).join("");
  }
  function canonical(value) {
    if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
    if (value && typeof value === "object") return "{" + Object.keys(value).sort().map(k => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}";
    return JSON.stringify(value);
  }
  function validUnicode(text) {
    for (let i = 0; i < text.length; i++) {
      const n = text.charCodeAt(i);
      if (n >= 0xd800 && n <= 0xdbff) {
        const next = text.charCodeAt(++i);
        if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
      } else if (n >= 0xdc00 && n <= 0xdfff) return false;
    }
    return true;
  }
  function normalize(row) {
    assert(row && typeof row === "object" && !Array.isArray(row), "record_must_be_object");
    let state;
    if (row.messages != null && row.messages !== "") {
      const messages = typeof row.messages === "string" ? JSON.parse(row.messages) : row.messages;
      assert(Array.isArray(messages) && messages.length, "messages_must_be_nonempty_array");
      state = { messages: messages.map(m => {
        assert(m && ["system", "user", "assistant", "tool"].includes(m.role) && typeof m.content === "string", "messages_require_string_role_and_content");
        return { role: m.role, content: m.content };
      }) };
    } else if (typeof row.text === "string" && row.text.trim()) state = { text: row.text };
    else if ("instruction" in row || "output" in row) {
      state = Object.fromEntries(["instruction", "input", "output"].filter(k => row[k] != null).map(k => [k, row[k]]));
      assert(Object.values(state).every(v => typeof v === "string") && state.instruction?.trim() && state.output?.trim(), "instruction_and_output_required");
    } else if ("prompt" in row || "response" in row) {
      assert(typeof row.prompt === "string" && row.prompt.trim() && typeof row.response === "string" && row.response.trim(), "prompt_and_response_required");
      state = { prompt: row.prompt, response: row.response };
    } else throw new Error("missing_supported_content");
    const text = state.messages ? state.messages.map(m => m.content).join("\n") : Object.values(state).join("\n");
    assert(text.trim() && validUnicode(text), "empty_or_invalid_unicode_content");
    const trainingText = state.messages ? state.messages.map(m => m.role + ": " + m.content).join("\n").trim() : text.trim();
    return { state, text, trainingText };
  }
  function parseCSV(text) {
    const rows = []; let row = [], field = "", mode = "plain", rowStarted = false;
    function finishField() { row.push(field); field = ""; mode = "plain"; }
    function finishRow() {
      finishField();
      if (rowStarted) rows.push(row);
      row = []; rowStarted = false;
      assert(rows.length <= MAX_ROWS + 1, "The public demo supports at most 1,000 records.");
    }
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char !== "\n" && char !== "\r") rowStarted = true;
      if (mode === "quoted") {
        if (char === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else mode = "closed";
        } else field += char;
      } else if (char === ",") finishField();
      else if (char === "\n" || char === "\r") {
        if (char === "\r" && text[i + 1] === "\n") i++;
        finishRow();
      } else if (char === '"' && mode === "plain" && !field) mode = "quoted";
      else {
        assert(mode !== "closed" && char !== '"', "Invalid CSV quoting. Check the file and try again.");
        field += char;
      }
    }
    assert(mode !== "quoted", "Unterminated quoted CSV field.");
    if (field || row.length || mode === "closed") finishRow();
    const headers = rows.shift() || [];
    assert(headers.length && headers.every(h => h.trim()) && new Set(headers).size === headers.length, "CSV headers must be unique and nonempty.");
    return rows.map((cells, i) => cells.length === headers.length ? Object.fromEntries(headers.map((h, j) => [h, cells[j]])) : { _invalid_input: { line: i + 2, reason: "csv_column_count" } });
  }
  function parseRecords(text, name) {
    text = text.replace(/^\uFEFF/, "");
    let records;
    if (/\.csv$/i.test(name)) records = parseCSV(text);
    else {
      const lines = text.split(/\r?\n/).filter(line => line.trim());
      assert(lines.length <= MAX_ROWS, "The public demo supports at most 1,000 records.");
      records = lines.map((line, i) => {
        try { return JSON.parse(line, (_key, value) => {
          assert(typeof value !== "number" || Number.isFinite(value), "unsupported_number");
          return value;
        }); }
        catch (_) { return { _invalid_input: { line: i + 1, reason: "invalid_json" } }; }
      });
    }
    assert(records.length, "The dataset is empty.");
    assert(records.length <= MAX_ROWS, "The public demo supports at most 1,000 records.");
    return records;
  }
  async function upload(file) {
    assert(datasets.length < 5, "This tab already has 5 datasets. Download any results you need, then reload to start a new workspace.");
    assert(file && /\.(jsonl|csv)$/i.test(file.name), "Choose a UTF-8 JSONL or CSV file.");
    assert(file.size > 0 && file.size <= MAX_BYTES, "The public demo accepts files up to 2 MiB.");
    let text;
    try { text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer()); }
    catch (_) { throw new Error("Use a valid UTF-8 file."); }
    const records = parseRecords(text, file.name);
    const item = { id: id(), name: file.name, size: file.size, rows: records.length, created_at: now(), preview: records.slice(0, 8).map(row => {
      if (!row || typeof row !== "object" || Array.isArray(row)) return { _error: "Invalid record; will be reviewed" };
      return Object.fromEntries(Object.entries(row).slice(0, 20).map(([k, v]) => [k, typeof v === "string" ? v.slice(0, 800) : typeof v === "number" || typeof v === "boolean" ? v : JSON.stringify(v).slice(0, 800)]));
    }) };
    payloads.set(item.id, records); datasets.unshift(item); return copy(item);
  }
  async function screen(records, run) {
    const seen = new Set(), parts = { keep: [], review: [], reject: [] }, audit = [], entries = [];
    const counts = { total: 0, keep: 0, review: 0, reject: 0, duplicates: 0 };
    const decision_reasons = { review: {}, reject: {} };
    for (let i = 0; i < records.length; i++) {
      checkCancel(run);
      const record = records[i]; let decision = "review", reason = "invalid_content", stateHash = null;
      try {
        const normalized = normalize(record), key = canonical(normalized.state);
        stateHash = await hash(key);
        if (seen.has(key)) { decision = "reject"; reason = "exact_duplicate"; counts.duplicates++; }
        else {
          seen.add(key);
          const length = [...normalized.text].length;
          if (length < 8 || length > 32000) { decision = "reject"; reason = "content_length_outside_bounds"; }
          else if (/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/.test(normalized.text)) reason = "demo_email_pattern";
          else { decision = "keep"; reason = "demo_rules_passed"; }
        }
      } catch (_) { /* Invalid records remain in the review partition. */ }
      parts[decision].push(record); counts[decision]++; counts.total++;
      audit.push({ line: i + 1, state_hash: stateHash, decision, reason, mode: "browser_demo_rules" });
      entries.push({ line: i + 1, decision, reason, record });
      if (decision !== "keep") decision_reasons[decision][reason] = (decision_reasons[decision][reason] || 0) + 1;
      run.counts = { ...counts }; run.progress = { processed: i + 1, total: records.length };
      if (i % 20 === 0) await pause();
    }
    return { parts, audit, counts, entries, decision_reasons };
  }
  async function splitRecords(records, run) {
    const parents = new Map(), items = [];
    function find(key) {
      if (!parents.has(key)) parents.set(key, key);
      let r = key;
      while (parents.get(r) !== r) r = parents.get(r);
      while (parents.get(key) !== key) { const next = parents.get(key); parents.set(key, r); key = next; }
      return r;
    }
    function join(a, b) {
      const roots = [find(a), find(b)].sort(); parents.set(roots[1], roots[0]);
    }
    for (const row of records) {
      checkCancel(run);
      const text = normalize(row).trainingText;
      const key = "text:" + await hash(text.trim().split(/\s+/u).join(" "));
      find(key);
      for (const field of ["group_id", "conversation_id"]) {
        const value = row[field];
        if (value == null) continue;
        assert(typeof value === "string" || (typeof value === "number" && Number.isSafeInteger(value)), field + " must be a string or safe integer.");
        if (String(value).trim()) join(key, field + ":" + await hash(String(value)));
      }
      items.push({ row, text, key });
    }
    const groups = [...new Set(items.map(item => find(item.key)))];
    assert(groups.length >= 6, "Screening finished, but training requires at least 6 independent groups. Add more unrelated examples or turn off auto-training.");
    const ordered = await Promise.all(groups.map(async key => ({ key, order: await hash("42:" + key) })));
    ordered.sort((a, b) => a.order < b.order ? -1 : a.order > b.order ? 1 : a.key.localeCompare(b.key));
    const n = Math.max(1, Math.floor(groups.length * .15)), assignments = new Map();
    ordered.forEach((item, i) => assignments.set(item.key, i < n ? "validation" : i < 2 * n ? "test" : "train"));
    const splits = { train: [], validation: [], test: [] };
    for (const item of items) splits[assignments.get(find(item.key))].push({ ...item.row, text: item.text, _split_group: find(item.key) });
    return { splits, group_counts: { train: groups.length - n * 2, validation: n, test: n } };
  }
  async function train(records, run) {
    run.progress = { stage: "splitting" };
    const { splits, group_counts } = await splitRecords(records, run);
    const counts = new Float64Array(258 * 257), totals = new Float64Array(258);
    const tokens = row => [...encoder.encode(row.text).slice(0, 256), 256];
    function measure(rows, learn = false) {
      let loss = 0, n = 0;
      for (const row of rows) {
        checkCancel(run); let prev = 257;
        for (const token of tokens(row)) {
          loss -= Math.log((counts[prev * 257 + token] + .5) / (totals[prev] + .5 * 257));
          if (learn) { counts[prev * 257 + token]++; totals[prev]++; }
          prev = token; n++;
        }
      }
      assert(n, "The evaluation split contains no usable bytes.");
      return loss / n;
    }
    const baseline_loss = measure(splits.test), baseline_validation_loss = measure(splits.validation);
    const history = []; let visits = 0;
    for (let offset = 0; offset < splits.train.length && history.length < 20; offset += 4) {
      checkCancel(run);
      const batch = splits.train.slice(offset, offset + 4), loss = measure(batch, true);
      visits += batch.length; history.push({ step: history.length + 1, epoch: 1, loss });
      run.progress = { stage: "training", step: history.length, max_steps: Math.min(20, Math.ceil(splits.train.length / 4)) };
      await pause();
    }
    run.stage = "model_evaluation";
    const trained_loss = measure(splits.test), trained_validation_loss = measure(splits.validation);
    const report = {
      trainer: "demo", mode: "demo", execution: "browser", is_llm: false,
      model_type: "utf8_byte_bigram", baseline_loss, trained_loss, baseline_validation_loss, trained_validation_loss,
      baseline_perplexity: Math.exp(baseline_loss), trained_perplexity: Math.exp(trained_loss),
      delta_loss: trained_loss - baseline_loss, loss_history: history, steps: history.length, training_record_visits: visits,
      split_counts: Object.fromEntries(Object.entries(splits).map(([k, rows]) => [k, rows.length])), group_counts,
      config: { epochs: 1, max_steps: 20, batch_size: 4, max_seq_length: 256, seed: 42 },
      metric_unit: "natural-log NLL per UTF-8 byte/EOS",
      limitations: ["Local byte-bigram demonstration, not an LLM or JEV.", "No semantic quality or domain fact evaluation.", "Group isolation does not detect paraphrases or temporal leakage.", "Data and results live only in this tab; reload clears them."]
    };
    for (const [key, rows] of Object.entries(splits)) artifact(run, "training/" + key + ".jsonl", jsonl(rows));
    artifact(run, "training/model_report.json", JSON.stringify(report, null, 2));
    artifact(run, "training/byte_bigram.json", JSON.stringify({ model_type: "utf8_byte_bigram", vocabulary_size: 257, bos: 257, eos: 256, smoothing_alpha: .5, max_seq_bytes: 256, counts: Array.from({ length: 258 }, (_, i) => Array.from(counts.slice(i * 257, (i + 1) * 257))), context_totals: Array.from(totals) }));
    return report;
  }
  function artifact(run, name, content) {
    const url = "/api/runs/" + run.id + "/artifacts/" + name;
    const blob = new Blob([content], { type: name.endsWith(".jsonl") ? "application/x-ndjson" : "application/json" });
    files.set(url, blob); run.artifacts.push({ name, url, size: blob.size });
  }
  function checkCancel(run) { if (run.cancelRequested) throw new Error("Canceled by user."); }
  function log(run, message) { run.logs.push({ time: now(), message }); }
  async function execute(run) {
    run.status = "running"; run.stage = "screening";
    try {
      log(run, "Browser Demo: local schema, length, duplicate, and email checks. No JEV request.");
      const result = await screen(payloads.get(run.dataset_id), run);
      run.stage = "data_evaluation";
      screened.set(run.id, result.entries);
      run.data_report = { mode: "browser_demo_rules", complete: true, status: "complete", counts: result.counts, processed: result.counts.total, decision_reasons: result.decision_reasons, limitations: "Local rules only; domain and confidence settings do not change these decisions." };
      for (const [key, rows] of Object.entries(result.parts)) artifact(run, "screening/" + key + ".jsonl", rows.length ? jsonl(rows) : "");
      artifact(run, "screening/audit.jsonl", jsonl(result.audit));
      artifact(run, "screening/data_report.json", JSON.stringify(run.data_report, null, 2));
      log(run, "Screening complete. Only retained rows can enter demo training.");
      await pause(); checkCancel(run);
      if (run.config.auto_train) {
        run.stage = "training"; log(run, "Training a byte-bigram model in this browser using group-isolated splits.");
        run.model_report = await train(result.parts.keep, run);
        log(run, "Measured the frozen baseline and trained model on the same held-out records.");
      }
      checkCancel(run);
      run.status = "completed"; run.stage = "completed";
      run.progress = { processed: run.data_report.processed, total: run.data_report.processed };
      log(run, "Demo completed. Download results before reloading this tab.");
    } catch (error) {
      run.status = run.cancelRequested ? "cancelled" : "failed";
      run.error = error.message; log(run, error.message);
    }
    run.updated_at = now();
  }
  function start(body, previous) {
    assert(!runs.some(run => ["queued", "running"].includes(run.status)), "Wait for the current browser run to finish, or stop it first.");
    assert(previous || runs.length < 10, "This tab has 10 runs. Download your results, then reload to start a new workspace.");
    const dataset = datasets.find(item => item.id === body.dataset_id);
    assert(dataset, "Choose a dataset first.");
    assert(body.provider === "demo" && body.trainer === "demo", "The public demo supports local rules and byte-bigram training only. Self-host for JEV and LoRA.");
    const run = { id: previous?.id || id(), dataset_id: dataset.id, name: dataset.name, status: "queued", stage: "upload", config: { ...body }, created_at: previous?.created_at || now(), updated_at: now(), counts: {}, progress: { processed: 0, total: dataset.rows }, logs: [], artifacts: [], model_report: null, data_report: null, error: null, cancelRequested: false };
    if (previous) {
      runs.splice(runs.indexOf(previous), 1);
      screened.delete(previous.id);
      for (const artifact of previous.artifacts) files.delete(artifact.url);
    }
    runs.unshift(run); setTimeout(() => execute(run), 0); return copy(run);
  }
  function summary(run) {
    const light = copy(run);
    for (const field of RUN_DETAIL_FIELDS) delete light[field];
    return light;
  }
  function records(run, query) {
    const decision = query.get("decision") || "review", reason = query.get("reason");
    assert(["keep", "review", "reject"].includes(decision), "decision must be keep, review, or reject.");
    const limit = Math.min(200, Math.max(1, Number(query.get("limit")) || 50)), offset = Math.max(0, Number(query.get("offset")) || 0);
    const matches = (screened.get(run.id) || []).filter(entry => entry.decision === decision && (!reason || entry.reason === reason));
    const page = matches.slice(offset, offset + limit).map(entry => ({ line: entry.line, reason: entry.reason, error: null, detail: null, dimensions: {}, record: entry.record }));
    return copy({ decision, reason, offset, records: page, has_more: matches.length > offset + limit });
  }
  async function request(fullPath, options = {}) {
    const method = options.method || "GET", body = options.body;
    const [path, search = ""] = fullPath.split("?"), query = new URLSearchParams(search);
    if (path === "/api/health") return { status: "ok", version: "0.1.0-browser-demo", max_upload_mb: 2, providers: { demo: true, openrouter: false, typesafe: false }, training: { huggingface: false, base_model: "Browser byte-bigram demo" } };
    if (path === "/api/datasets/example" && method === "POST") {
      const response = await fetch("/static/example.jsonl");
      assert(response.ok, "Could not load the bundled example. Please refresh and try again.");
      return upload(new File([await response.blob()], "synthetic-dialogues.jsonl"));
    }
    if (path === "/api/datasets") return method === "POST" ? upload(body.get("file")) : copy(datasets);
    if (path === "/api/runs") return method === "POST" ? start(body) : query.get("view") === "summary" ? runs.map(summary) : copy(runs);
    if (files.has(path)) return new Response(files.get(path));
    const match = path.match(/^\/api\/runs\/([a-f0-9]+)(?:\/(cancel|retry|records))?$/);
    if (match) {
      const run = runs.find(item => item.id === match[1]); assert(run, "Run not found.");
      if (match[2] === "records") return records(run, query);
      if (match[2] === "cancel") {
        assert(["running", "queued"].includes(run.status), "This run has already finished.");
        run.cancelRequested = true; return copy(run);
      }
      if (match[2] === "retry") {
        assert(["failed", "cancelled"].includes(run.status), "Only failed or canceled runs can be restarted.");
        return start(run.config, run);
      }
      return copy(run);
    }
    throw new Error("This action is not available in the browser demo.");
  }
  root.jevDemo = { request };
})(typeof window === "undefined" ? globalThis : window);
