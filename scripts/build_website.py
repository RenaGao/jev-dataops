"""Build the dependency-free public website and its isolated browser demo.

Only explicit static inputs enter the output. The Python API, uploaded datasets,
credentials, and local training artifacts are never copied into the website.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MARKER = ".jev-website-build.json"
SITE_FILES = (
    "index.html",
    "guide/index.html",
    "metrics/index.html",
    "assets/site.css",
    "assets/site.js",
    "assets/metrics.js",
    "assets/metrics.css",
    "assets/favicon.svg",
)
METRIC_DOMAINS = ("general", "finance", "code", "enterprise", "legal", "medical")
ASSET_REFERENCE = re.compile(r'(src|href)="(/(?:static|assets)/[^"?#]+\.(?:js|css))"')


def _fingerprint_assets(site: Path) -> None:
    # Static hosts cache scripts heuristically; a stale runtime paired with a new app.js breaks the demo.
    def versioned(match: re.Match) -> str:
        asset = site / match[2].lstrip("/")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:10]
        return f'{match[1]}="{match[2]}?v={digest}"'

    for page in site.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        page.write_text(ASSET_REFERENCE.sub(versioned, html), encoding="utf-8")


def _build_demo(destination: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "jev_public_demo_builder", ROOT / "scripts" / "build_public_demo.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the public demo builder.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(destination)
    html = (destination / "index.html").read_text(encoding="utf-8")
    html = html.replace('class="brand" href="/"', 'class="brand" href="/demo/"')
    for anchor in ("main-content", "dataset-section", "run-section"):
        html = html.replace(f'href="#{anchor}"', f'href="/demo/#{anchor}"')
    home_link = (
        '<a class="nav-item website-home" href="/" aria-label="Website home">'
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="m3 10 9-7 9 7v10H3zM9 20v-7h6v7"/></svg>'
        '<span>Website home</span></a>'
    )
    assert '<div class="sidebar-bottom">' in html, "Demo sidebar contract changed."
    html = html.replace('<div class="sidebar-bottom">', '<div class="sidebar-bottom">' + home_link, 1)
    html = html.replace(
        'href="https://github.com/RenaGao/jev-dataops#readme" target="_blank" rel="noopener noreferrer"',
        'href="/guide/"',
    )
    html = html.replace(
        'href="https://github.com/RenaGao/jev-dataops#quickstart" target="_blank" rel="noopener noreferrer"',
        'href="/guide/#install"',
    )
    (destination / "demo").mkdir()
    (destination / "demo" / "index.html").write_text(html, encoding="utf-8")
    (destination / "index.html").unlink()


def build(destination: str | Path) -> Path:
    requested = Path(destination).expanduser()
    if requested.is_symlink():
        raise ValueError("Choose a real output directory, not a symlink.")
    target = requested.resolve()
    if target == ROOT or target in ROOT.parents:
        raise ValueError("The output cannot replace the source repository or an ancestor.")
    if target.exists() and not target.is_dir():
        raise ValueError("The output exists and is not a directory.")
    if target.exists() and any(target.iterdir()) and not (target / MARKER).is_file():
        raise ValueError(
            f"Refusing to replace unrelated files in {target}. "
            "Choose an empty directory or a previous website build."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jev-website-", dir=target.parent) as temporary:
        staging = Path(temporary) / "output"
        _build_demo(staging)
        for relative in SITE_FILES:
            source = ROOT / "website" / relative
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        for domain in METRIC_DOMAINS:
            for source_dir, destination_dir, suffix in (
                ("metric_packs", "metric-packs", ".json"),
                ("metric_examples", "metric-examples", ".jsonl"),
            ):
                source = ROOT / "jev_dataops" / source_dir / (domain + suffix)
                output = staging / "assets" / destination_dir / source.name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output)
        _fingerprint_assets(staging)
        (staging / MARKER).write_text(
            json.dumps({"project": "jev-dataops-public-website", "schema_version": 1}) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    print(f"Built JEV DataOps website at {target}")
    print("Routes: /, /guide/, /metrics/, /demo/. No API or training server is deployed.")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Empty directory or previous website output")
    args = parser.parse_args()
    try:
        build(args.destination)
    except ValueError as error:
        parser.exit(2, f"Build error: {error}\n")
