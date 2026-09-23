"""Upload API and same-origin workbench. Run exactly one server worker."""
from __future__ import annotations

from contextlib import asynccontextmanager
import csv
import fcntl
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import uuid
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .runner import Runner
from .screening import decision_records
from .store import Store, now

PACKAGE = Path(__file__).parent
MAX_LINE = 1024 * 1024
RUN_DETAIL_FIELDS = frozenset({"logs", "data_report", "model_report", "artifacts"})


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    provider: Literal["demo", "openrouter", "typesafe"] = "demo"
    # A pinned Jev version such as "typesafe/jev-1.13-20260917"; the provider's
    # floating alias when unset. Pinning keeps a run's cache to one model.
    model: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._~/:-]+$")
    confidence: float = Field(default=0.85, ge=0, le=1, allow_inf_nan=False)
    concurrency: int = Field(default=4, ge=1, le=16)
    max_requests: int = Field(default=1000, ge=1, le=1000000)
    rubric: Literal["general", "finance", "code"] = "general"
    min_chars: int = Field(default=8, ge=1, le=MAX_LINE)
    max_chars: int = Field(default=32000, ge=1, le=MAX_LINE)
    trainer: Literal["demo", "huggingface"] = "demo"
    auto_train: bool = True
    epochs: int = Field(default=1, ge=1, le=10)
    max_steps: int = Field(default=20, ge=1, le=10000)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=0.0002, ge=0.0000001, le=0.01, allow_inf_nan=False)
    max_seq_length: int = Field(default=256, ge=32, le=4096)
    seed: int = Field(default=42, ge=0, le=2147483647)
    lora_r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    # "answer" masks the prompt so only the response is learned; "full" is plain causal SFT.
    loss_mask: Literal["answer", "full"] = "answer"


def inspect_dataset(path: Path):
    """Inspect with the same byte-bounded parser used by the screening engine."""
    from .screening import iter_records

    preview, rows, invalid = [], 0, 0
    for _, record, error, fatal in iter_records(path):
        if fatal or error in {"invalid_utf8", "row_exceeds_1_mib", "csv_row_exceeds_1_mib"}:
            message = "Use valid UTF-8 data" if error == "invalid_utf8" else "A row exceeds 1 MiB. Split oversized records before uploading." if "exceeds_1_mib" in error else "Invalid CSV structure: " + error
            raise ValueError(message)
        rows += 1
        if error:
            invalid += 1
            record = {"_error": "Invalid input row; will be sent to review", "reason": error}
        if len(preview) < 8:
            preview.append({str(k): (v if isinstance(v, (int, float, bool)) else str(v)[:800])
                            for k, v in list(record.items())[:20]})
    if not rows:
        raise ValueError("Dataset is empty")
    return {"rows": rows, "invalid_rows": invalid, "preview": preview}


class BodyLimitMiddleware:
    """Enforce request bytes before multipart parsing or disk spooling."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.limit + MAX_LINE if scope["path"] == "/api/datasets" else MAX_LINE
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = limit + 1
        if length > limit:
            return await JSONResponse({"detail": "Upload exceeds configured size limit"}, status_code=413)(scope, receive, send)
        count = 0
        async def bounded_receive():
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > limit:
                raise HTTPException(413, "Upload exceeds configured size limit")
            return message
        await self.app(scope, bounded_receive, send)


def create_app(data_dir=None):
    root = Path(data_dir or os.environ.get("JEV_DATA_DIR", ".jev-dataops"))
    store = Store(root)
    max_upload = int(os.environ.get("JEV_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
    token = os.environ.get("JEV_API_TOKEN", "")
    allowed_hosts = [s.strip() for s in os.environ.get("JEV_ALLOWED_HOSTS", "localhost,127.0.0.1,::1,testserver").split(",")]

    @asynccontextmanager
    async def lifespan(app):
        lock = (store.root / "server.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError("This data directory already has a server. Use one uvicorn worker.") from None
        store.recover()
        app.state.runner = Runner(store)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.runner.close)
            lock.close()

    app = FastAPI(title="JEV DataOps", version=__version__, lifespan=lifespan, docs_url="/docs")
    app.state.store = store
    app.add_middleware(BodyLimitMiddleware, limit=max_upload)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if token:
                authorization = request.headers.get("Authorization", "")
                if not hmac.compare_digest(authorization, "Bearer " + token):
                    return JSONResponse({"detail": "API token required"}, status_code=401)
            elif request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
                return JSONResponse({"detail": "Remote access requires JEV_API_TOKEN"}, status_code=403)
            elif any(header in request.headers for header in ("x-forwarded-for", "x-real-ip", "forwarded")):
                # Loopback traffic that a reverse proxy relayed from elsewhere is remote
                # access; the loopback exemption above is only for the operator's own browser.
                return JSONResponse({"detail": "Proxied access requires JEV_API_TOKEN"}, status_code=403)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                origin = request.headers.get("origin")
                if origin and urlsplit(origin).netloc != request.headers.get("host"):
                    return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/":
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "max_upload_mb": max_upload // 1048576, "providers": {"demo": True, "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")), "typesafe": bool(os.environ.get("TYPESAFE_API_KEY"))}, "training": {"huggingface": all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft")), "base_model": os.environ.get("JEV_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")}}

    @app.get("/api/datasets")
    def datasets(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        return store.list("dataset", limit, offset)

    async def save_upload(file):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".jsonl", ".csv"}:
            raise HTTPException(422, "Use UTF-8 .jsonl or .csv files")
        dataset_id = uuid.uuid4().hex
        directory = store.root / "datasets"
        directory.mkdir(exist_ok=True, mode=0o700)
        path = directory / (dataset_id + suffix)
        size, digest = 0, hashlib.sha256()
        try:
            with path.open("xb") as target:
                while chunk := await file.read(MAX_LINE):
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(413, "Upload exceeds configured size limit")
                    target.write(chunk)
                    digest.update(chunk)
            info = await run_in_threadpool(inspect_dataset, path)
        except (ValueError, UnicodeError, csv.Error) as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "Invalid dataset: " + (str(exc)[:200] if isinstance(exc, ValueError) else "UTF-8 / CSV format error")) from None
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        item = {"id": dataset_id, "name": Path(file.filename.replace("\\", "/")).name[:200], "suffix": suffix, "size": size, "sha256": digest.hexdigest(), "created_at": now(), **info}
        return store.put("dataset", item)

    @app.post("/api/datasets", status_code=201)
    async def upload(file: UploadFile = File(...)):
        return await save_upload(file)

    @app.post("/api/datasets/example", status_code=201)
    async def example():
        with (PACKAGE / "examples" / "dialogues.jsonl").open("rb") as stream:
            return await save_upload(UploadFile(filename="synthetic-dialogues.jsonl", file=stream))

    @app.get("/api/runs")
    def runs(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
             view: Literal["full", "summary"] = "full"):
        items = store.list("run", limit, offset)
        if view == "summary":
            # Logs, reports and artifact inventories are fetched per run; a polling
            # client only needs status and counts for the list.
            items = [{key: value for key, value in item.items() if key not in RUN_DETAIL_FIELDS} for item in items]
        return items

    def get_run(run_id):
        item = store.get("run", run_id)
        if not item:
            raise HTTPException(404, "Run not found")
        return item

    def ensure_runtime(config):
        state = health()
        if not state["providers"][config["provider"]]:
            raise HTTPException(422, "Configure the selected provider's API key on the server")
        if config["auto_train"] and config["trainer"] == "huggingface" and not state["training"]["huggingface"]:
            raise HTTPException(422, "Install jev-dataops[train] on the training host first")

    @app.post("/api/runs", status_code=201)
    def start(config: RunConfig):
        dataset = store.get("dataset", config.dataset_id)
        if not dataset:
            raise HTTPException(404, "Dataset not found")
        values = config.model_dump(exclude={"dataset_id"}, exclude_none=True)
        values["base_model"] = os.environ.get("JEV_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M")
        ensure_runtime(values)
        try:
            return app.state.runner.submit(dataset, values)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        return get_run(run_id)

    @app.get("/api/runs/{run_id}/records")
    def records(run_id: str, decision: Literal["keep", "review", "reject"] = "review",
                reason: str | None = Query(None, max_length=300), limit: int = Query(50, ge=1, le=200),
                offset: int = Query(0, ge=0, le=10_000_000)):
        get_run(run_id)
        page = decision_records(store.root / "runs" / run_id / "screening", decision, reason, limit, offset)
        return {"decision": decision, "reason": reason, "offset": offset, **page}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        get_run(run_id)
        return app.state.runner.cancel(run_id)

    @app.post("/api/runs/{run_id}/retry")
    def retry(run_id: str):
        item = get_run(run_id)
        ensure_runtime(item["config"])
        try:
            return app.state.runner.submit(store.get("dataset", item["dataset_id"]), item["config"], run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/runs/{run_id}/artifacts/{artifact:path}")
    def artifact(run_id: str, artifact: str):
        item = get_run(run_id)
        if artifact not in {a["name"] for a in item["artifacts"]}:
            raise HTTPException(404, "Artifact not found")
        root = (store.root / "runs" / run_id).resolve()
        path = (root / artifact).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/")
    def index():
        return FileResponse(PACKAGE / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=PACKAGE / "static"), name="static")
    return app
