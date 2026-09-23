# Deployment and operation

## Local workstation

`jev-dataops serve` binds to `127.0.0.1:8000`. Raw data, SQLite metadata, cache and artifacts are stored in `.jev-dataops/`, which is ignored by Git. Set `--data-dir` to a dedicated data volume. The process runs one workflow at a time and queues up to 32 unfinished runs; this protects a single training device from overlapping model loads.

Use **one process and one uvicorn worker**. A filesystem lock rejects a second server sharing the data directory. On startup, interrupted jobs become failed, with a retry instruction. Screening retries replay validated cached results and rebuild partitions. Every training attempt has its own directory (`training-1`, `training-2`, …); training does not yet resume optimizer state.

Cancellation is cooperative at records, batches and network timeout boundaries. Model downloads and some device operations are not immediately interruptible. Avoid unbounded shutdown deadlines when terminating an active model download.

## Team deployment

Set `JEV_API_TOKEN` before binding a remote interface. Use a long random shared token, HTTPS reverse proxy, an explicit `JEV_ALLOWED_HOSTS` list and an appropriately sized request-body limit. The API rejects off-host access without a token and cross-origin writes. Loopback requests that carry `X-Forwarded-For`, `X-Real-IP` or `Forwarded` are treated as off-host too, so a reverse proxy in front of an unconfigured server is refused rather than silently trusted. Do not enable wildcard hosts or permissive CORS on a public deployment.

The shared token is an operator access control, **not multi-tenant account isolation**. All authorized users share datasets and results. For distinct trust domains, use separate service instances/data volumes or add an identity provider and per-user access controls. TLS, durable backups, disk quotas, ingress rate limits and tenant scheduling belong to the deployment layer.

```bash
export JEV_API_TOKEN='replace-with-a-random-secret'
export JEV_ALLOWED_HOSTS='dataops.example.org,localhost,127.0.0.1'
export JEV_MAX_UPLOAD_MB=1024
jev-dataops serve --host 0.0.0.0 --data-dir /srv/jev-data
```

Environment `.env` files are not loaded automatically by the Python CLI. Docker Compose reads its project `.env`. Keep secrets outside Git and pass them through your secret manager or process environment.

## Resource boundaries

- Upload: 1 GiB default; streamed to disk, request capped before multipart parsing.
- Record: 1 MiB. Preview: at most 8 records with bounded field text.
- Selection: at most twice the configured concurrency in flight, disk-backed seen/cache tables.
- Queue: up to 32 active/queued workflows, one execution thread per instance.
- API lists: 100 results by default; `limit` up to 500 and `offset` pagination.
- Training: streamed batches, disk-backed group index; model weights/optimizer still occupy RAM/VRAM.
- Exact duplicate isolation does not detect paraphrases; an independent benchmark is still needed for release decisions.

SQLite files, raw uploads and outputs may contain sensitive data. Protect the data directory, configure retention externally and back it up consistently while the service is stopped or using SQLite's backup mechanism. The app does not execute shell commands supplied in data and does not deserialize arbitrary uploaded model files.

## API

Interactive schema: `/docs`. Authentication: `Authorization: Bearer <JEV_API_TOKEN>` when configured.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime/provider availability, no keys |
| GET / POST | `/api/datasets` | List / multipart upload (`file`) |
| POST | `/api/datasets/example` | Load bundled synthetic records |
| GET / POST | `/api/runs` | List / launch configured workflow; `?view=summary` omits logs, reports, and artifacts |
| GET | `/api/runs/{id}` | Status, reports, bounded logs |
| GET | `/api/runs/{id}/records` | Screened records with their reasons: `decision=keep\|review\|reject`, optional `reason`, `limit` (≤200), `offset` |
| POST | `/api/runs/{id}/cancel` | Cooperative cancellation |
| POST | `/api/runs/{id}/retry` | Reuse screening cache, fresh training attempt |
| GET | `/api/runs/{id}/artifacts/{path}` | Download enumerated run artifact |

Required run field: `dataset_id`. Defaults: `provider=demo`, `trainer=demo`, `auto_train=true`, `confidence=0.85`, `concurrency=4`, `max_requests=1000`, `epochs=1`, `max_steps=20`, `max_seq_length=256`, `learning_rate=0.0002`, `seed=42`. API accepts `rubric=general|finance|code`. Model selection is server-controlled with `JEV_BASE_MODEL`.
