/* Real browser-runtime checks using only Node's built-in Web APIs. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { webcrypto } from "node:crypto";
import vm from "node:vm";

const code = readFileSync(new URL("../public_demo/runtime.js", import.meta.url), "utf8");
function runtime() {
  const context = { crypto: webcrypto, TextEncoder, TextDecoder, File, Blob, Response, URLSearchParams, setTimeout, fetch: () => { throw new Error("Unexpected network request"); } };
  vm.runInNewContext(code, context);
  return context.jevDemo;
}
async function upload(api, text, name = "test.jsonl") {
  const data = new FormData(); data.append("file", new File([text], name));
  return api.request("/api/datasets", { method: "POST", body: data });
}
async function run(api, dataset, autoTrain = true) {
  const started = await api.request("/api/runs", { method: "POST", body: { dataset_id: dataset.id, provider: "demo", trainer: "demo", auto_train: autoTrain, rubric: "general" } });
  for (let attempt = 0; attempt < 500; attempt++) {
    const state = await api.request("/api/runs/" + started.id);
    if (!["queued", "running"].includes(state.status)) return JSON.parse(JSON.stringify(state));
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  throw new Error("Browser run timed out");
}
async function artifact(api, result, name) {
  return (await api.request(result.artifacts.find(item => item.name === name).url)).text();
}
{
  const api = runtime(), sample = readFileSync(new URL("../examples/dialogues.jsonl", import.meta.url), "utf8");
  const result = await run(api, await upload(api, sample));
  assert.equal(result.status, "completed", result.error);
  assert.deepEqual(result.counts, { total: 84, keep: 80, review: 2, reject: 2, duplicates: 1 });
  assert.deepEqual(result.model_report.split_counts, { train: 56, validation: 12, test: 12 });
  assert.equal(result.model_report.is_llm, false);
  assert.equal(result.model_report.loss_history.length, 14);
  // Independent reference: the Python demo on this same English fixture.
  assert.ok(Math.abs(result.model_report.trained_loss - 2.4415996131142346) < 1e-10);
  const groups = [];
  for (const key of ["train", "validation", "test"]) {
    const rows = (await artifact(api, result, "training/" + key + ".jsonl")).trim().split("\n").map(JSON.parse);
    groups.push(new Set(rows.map(row => row._split_group)));
  }
  for (let a = 0; a < groups.length; a++) for (let b = a + 1; b < groups.length; b++)
    assert.ok([...groups[a]].every(group => !groups[b].has(group)), "No group may cross splits.");
  assert.deepEqual(result.data_report.decision_reasons,
                   { review: { demo_email_pattern: 1, invalid_content: 1 }, reject: { exact_duplicate: 1, content_length_outside_bounds: 1 } });
  const [light] = JSON.parse(JSON.stringify(await api.request("/api/runs?view=summary")));
  assert.deepEqual(light.counts, result.counts);
  assert.ok(["logs", "data_report", "model_report", "artifacts"].every(field => !(field in light)));
  const rejected = JSON.parse(JSON.stringify(await api.request("/api/runs/" + result.id + "/records?decision=reject&limit=20&offset=0")));
  assert.deepEqual(rejected.records.map(entry => [entry.line, entry.reason]), [[81, "exact_duplicate"], [82, "content_length_outside_bounds"]]);
  assert.equal(rejected.records[1].record.text, "hi"); assert.equal(rejected.has_more, false);
  const filtered = JSON.parse(JSON.stringify(await api.request("/api/runs/" + result.id + "/records?decision=review&reason=demo_email_pattern&limit=1")));
  assert.deepEqual(filtered.records.map(entry => entry.line), [83]);
  await assert.rejects(api.request("/api/runs/" + result.id + "/records?decision=maybe"), /decision must be/);
}
{
  const api = runtime();
  const dataset = await upload(api, 'text,__proto__\r\n"A useful text, with a comma\nand a newline",harmless\r\n"A second useful text",inert\r\n', "quoted.csv");
  assert.equal(dataset.rows, 2);
  const result = await run(api, dataset, false);
  assert.equal(result.status, "completed"); assert.equal(result.counts.keep, 2);
  const rows = (await artifact(api, result, "screening/keep.jsonl")).trim().split("\n").map(JSON.parse);
  assert.equal(rows[0].__proto__, "harmless");
  assert.equal(rows[0].text, "A useful text, with a comma\nand a newline");
  assert.equal(result.model_report, null);
  await assert.rejects(upload(api, 'text\n"unfinished', "broken.csv"), /Unterminated/);
  const emptyCells = await run(api, await upload(api, 'text,group_id\n,\nA valid training row,one\n', 'empty-cells.csv'), false);
  assert.equal(emptyCells.counts.total, 2); assert.equal(emptyCells.counts.review, 1);
  const quotedEmpty = await run(api, await upload(api, 'text\n""\n', 'empty-quoted.csv'), false);
  assert.equal(quotedEmpty.counts.total, 1); assert.equal(quotedEmpty.counts.review, 1);
}
{
  const api = runtime();
  const source = '{broken}\n{"text":""}\n{"text":"\\ud800"}\n{"instruction":"Valid question","output":""}\n{"text":"Contact demo@example.com for a demonstration."}\n{"text":"hi"}\n';
  const result = await run(api, await upload(api, source), false);
  assert.equal(result.counts.review, 5); assert.equal(result.counts.reject, 1); assert.equal(result.counts.keep, 0);
  const overflow = await run(api, await upload(api, '{"text":"A valid training record","metadata":{"amount":1e400}}'), false);
  assert.equal(overflow.counts.keep, 0); assert.equal(overflow.counts.review, 1);
}
{
  const api = runtime();
  const rows = Array.from({ length: 12 }, (_, i) => ({ text: "A different useful record " + i + " in one shared conversation.", group_id: "same" }));
  const result = await run(api, await upload(api, rows.map(JSON.stringify).join("\n")));
  assert.equal(result.status, "failed"); assert.match(result.error, /6 independent groups/);
  assert.equal(result.counts.keep, 12); assert.equal(result.model_report, null);
  assert.ok(result.artifacts.some(item => item.name === "screening/data_report.json"));
}
{
  const api = runtime();
  await assert.rejects(upload(api, '{"text":"valid input"}\n'.repeat(1001)), /1,000/);
  await assert.rejects(upload(api, "a".repeat(2 * 1024 * 1024 + 1)), /2 MiB/);
  const dataset = await upload(api, '{"text":"A valid synthetic record."}');
  const first = await api.request("/api/runs", { method: "POST", body: { dataset_id: dataset.id, provider: "demo", trainer: "demo", auto_train: false } });
  await api.request("/api/runs/" + first.id + "/cancel", { method: "POST" });
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal((await api.request("/api/runs/" + first.id)).status, "cancelled");
}
console.log("Public demo: sample parity, isolated evaluation, CSV parsing, malformed records, limits, and cancellation passed.");
