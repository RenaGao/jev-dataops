"""Build the browser-only public demo without changing the Python workbench."""
import argparse
from pathlib import Path
import shutil


def build(destination):
    root = Path(__file__).resolve().parents[1]
    static = root / "jev_dataops/static"
    destination = Path(destination).resolve()
    (destination / "static").mkdir(parents=True, exist_ok=True)
    html = (static / "index.html").read_text()
    html = html.replace('<script src="/static/app.js" defer></script>',
                        '<script src="/static/demo-runtime.js" defer></script>\n  <script src="/static/app.js" defer></script>')
    html = html.replace("JEV DataOps · Data &amp; Model Workspace", "JEV DataOps · Public Browser Demo")
    html = html.replace("JEV DataOps · Data & Model Workspace", "JEV DataOps · Public Browser Demo")
    html = html.replace("Local workspace", "Public demo").replace("Personal workspace", "Runs in your browser")
    html = html.replace("DOMAIN AI WORKFLOWS", "PUBLIC BROWSER DEMO")
    html = html.replace("Demo / LoRA", "Byte-bigram")
    html = html.replace("Better data. <span>Better domain models.</span>", "JEV DataOps <span>Browser demo.</span>")
    html = html.replace('id="settings-button"', 'id="settings-button" hidden')
    html = html.replace('id="rubric" name="rubric"', 'id="rubric" name="rubric" disabled')
    html = html.replace('class="confidence-field"', 'class="confidence-field" hidden')
    html = html.replace('<div class="form-grid"><div class="field"><label for="concurrency">',
                        '<div class="form-grid" hidden><div class="field"><label for="concurrency">')
    banner = """<section class="demo-banner" aria-label="Public demo information">
      <div><strong>Try the pipeline in your browser</strong>
      <p>Up to 1,000 rows / 2 MiB. Files stay in this tab. Reloading clears your data and results.
      Local rules and byte-bigram training only — no JEV calls or LLM training.</p></div>
      <a href="https://github.com/RenaGao/jev-dataops#quickstart" target="_blank" rel="noopener noreferrer">Self-host the full version ↗</a>
    </section>"""
    html = html.replace('<main id="main-content">', '<main id="main-content">\n' + banner)
    html = html.replace('An open-source workspace for domain-specific data screening, automated training, and model evaluation.',
                        'Try JEV DataOps in your browser: local data screening, byte-bigram demo training, and measured evaluation. No cloud LLM training.')
    app = (static / "app.js").read_text()
    app = app.replace('token: sessionStorage.getItem("jev_api_token") || ""', 'token: ""')
    signature = 'async function request(path, { method = "GET", body, raw = false } = {}) {'
    assert signature in app
    start, end = app.index(signature), app.index("function updateStartState()")
    app = app[:start] + signature + '\n  return window.jevDemo.request(path, { method, body, raw });\n}\n' + app[end:]
    app = app.replace("Runs and artifacts are saved on the server. Come back anytime.",
                      "Download your results before reloading or closing this tab.")
    app = app.replace('online ? "Connected" : "Disconnected"', 'online ? "Browser ready" : "Unavailable"')
    app = app.replace("Streamed upload", "Local to this tab")
    app = app.replace('if (!state.token) {', 'if (!state.token && !window.jevDemo) {')
    start, end = app.index("function updateRubricNotice()"), app.index("function renderOverview()")
    app = app[:start] + 'function updateRubricNotice() {\n  $("rubric-note").textContent = "This public demo uses general local rules. Domain-specific JEV screening and LoRA training are available in the self-hosted version.";\n}\n' + app[end:]
    app = app.replace('$("trainer").disabled = !autoTrain;', '$("trainer").disabled = true;')
    note = '$("demo-controls-note").hidden = provider !== "demo";'
    assert note in app
    app = app.replace(note, '$("demo-controls-note").hidden = true;')
    app = app.replace("Uploading and checking your data…", "Reading and checking your data…")
    app = app.replace("Upload complete", "File loaded")
    app = app.replace("Queued for local execution", "Queued in this browser")
    app = app.replace(" (not configured)", " (self-host only)").replace(" (not installed)", " (self-host only)")
    app = app.replace("Cancellation requested. The task will stop after the current step.",
                      "Stop requested. The browser run will exit at the next processing boundary.")
    (destination / "index.html").write_text(html)
    (destination / "static/app.js").write_text(app)
    css = (static / "styles.css").read_text()
    css += """
/* The public browser demo preserves the full workbench's visual language. */
.demo-banner { display: flex; gap: 24px; align-items: center; justify-content: space-between; margin-bottom: 26px; padding: 18px 22px; border: 1px solid #bfd6c5; border-radius: 12px; background: #edf6ef; color: #214d3c; }
.demo-banner strong { display: block; font-size: 16px; }
.demo-banner p { margin: 6px 0 0; max-width: 720px; font-size: 14px; line-height: 1.6; }
.demo-banner a { flex-shrink: 0; color: #214d3c; font-weight: 650; font-size: 14px; text-decoration: underline; text-underline-offset: 4px; }
.page-heading { min-height: 0; margin-bottom: 20px; padding: 0; }
.page-heading .hero-visual, .page-heading .product-badge, .page-heading .hero-copy > p, .page-heading .domain-guide-link { display: none; }
.page-heading h1 { margin: 0; font-size: 28px; }
.page-heading h1 span { display: inline; }
@media (max-width: 900px) { .demo-banner { flex-direction: column; align-items: flex-start; gap: 12px; } }
"""
    (destination / "static/styles.css").write_text(css)
    shutil.copy2(root / "public_demo/runtime.js", destination / "static/demo-runtime.js")
    shutil.copy2(root / "jev_dataops/examples/dialogues.jsonl", destination / "static/example.jsonl")
    shutil.copy2(root / "LICENSE", destination / "LICENSE")
    print("Built public browser demo at", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path, help="Static output directory")
    build(parser.parse_args().destination)
