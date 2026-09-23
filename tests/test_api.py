"""HTTP integration checks: real demo workflow plus failure and access boundaries."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from jev_dataops.server import create_app


def sample_jsonl(count=48):
    return "".join(json.dumps({"text": f"Synthetic educational record {i}: explain how careful data evaluation supports reproducible model training.",
                               "group_id": f"synthetic-{i}", "metadata": {"source": "unit test"}}) + "\n" for i in range(count)).encode()


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.environment = patch.dict(os.environ, {"JEV_API_TOKEN": "", "JEV_MAX_UPLOAD_MB": "2",
                                                   "JEV_ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
                                                   "OPENROUTER_API_KEY": "", "TYPESAFE_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.app = create_app(self.root)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def upload(self, body=None, name="training.jsonl"):
        response = self.client.post("/api/datasets", files={"file": (name, body if body is not None else sample_jsonl(), "application/octet-stream")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def start(self, dataset, **config):
        response = self.client.post("/api/runs", json={"dataset_id": dataset["id"], **config})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def wait_run(self, run_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(response.status_code, 200, response.text)
            run = response.json()
            # The durable terminal status precedes the final artifact inventory.
            if run["status"] in {"completed", "failed", "cancelled"} and run_id not in self.app.state.runner.events:
                return self.client.get(f"/api/runs/{run_id}").json()
            time.sleep(0.01)
        self.fail("Background run did not finish within 10 seconds")

    def test_real_upload_screen_train_evaluate_and_download(self):
        dataset = self.upload()
        self.assertEqual(dataset["rows"], 48)
        self.assertEqual(len(dataset["preview"]), 8)
        self.assertEqual(dataset["invalid_rows"], 0)
        run = self.wait_run(self.start(dataset)["id"])
        self.assertEqual(run["status"], "completed", run.get("error"))
        self.assertEqual(run["counts"]["keep"], 48)
        self.assertEqual(run["data_report"]["mode"], "demo_rule_based")
        model = run["model_report"]
        self.assertFalse(model["is_llm"])
        self.assertTrue(math.isfinite(model["baseline_loss"]))
        self.assertTrue(math.isfinite(model["trained_loss"]))
        self.assertGreater(model["steps"], 0)
        self.assertEqual(sum(model["split_counts"].values()), 48)
        names = {artifact["name"] for artifact in run["artifacts"]}
        self.assertIn("screening/data_report.json", names)
        model_report_name = next(name for name in names if name.endswith("/model_report.json"))
        for name in ("screening/data_report.json", model_report_name):
            response = self.client.get(f"/api/runs/{run['id']}/artifacts/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-disposition"].startswith("attachment"))
            self.assertIsInstance(response.json(), dict)
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}/artifacts/screening/cache.sqlite3").status_code, 404)
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}/artifacts/%2e%2e/metadata.sqlite3").status_code, 404)
        self.assertEqual(self.client.post(f"/api/runs/{run['id']}/retry").status_code, 409)

    def test_held_back_records_are_listed_with_reasons_and_summary_view_is_light(self):
        rows = [{"text": f"Useful support record number {i}"} for i in range(6)]
        rows += [{"text": "Useful support record number 0"}, {"text": "tiny"}, {"text": "Write to jane.doe@example.com for help"}]
        body = "".join(json.dumps(row) + "\n" for row in rows).encode()
        run = self.wait_run(self.start(self.upload(body), auto_train=False)["id"])
        self.assertEqual(run["counts"], {"total": 9, "keep": 6, "review": 1, "reject": 2, "duplicates": 1})
        self.assertEqual(run["data_report"]["decision_reasons"],
                         {"review": {"demo_email_pattern": 1}, "reject": {"exact_duplicate": 1, "content_length_outside_bounds": 1}})
        rejected = self.client.get(f"/api/runs/{run['id']}/records", params={"decision": "reject"}).json()
        self.assertEqual([(r["line"], r["reason"], r["record"]["text"]) for r in rejected["records"]],
                         [(7, "exact_duplicate", "Useful support record number 0"), (8, "content_length_outside_bounds", "tiny")])
        self.assertFalse(rejected["has_more"])
        page = self.client.get(f"/api/runs/{run['id']}/records", params={"decision": "reject", "limit": 1, "offset": 1}).json()
        self.assertEqual([r["line"] for r in page["records"]], [8])
        filtered = self.client.get(f"/api/runs/{run['id']}/records", params={"decision": "reject", "reason": "exact_duplicate"}).json()
        self.assertEqual([r["line"] for r in filtered["records"]], [7])
        review = self.client.get(f"/api/runs/{run['id']}/records").json()
        self.assertEqual(review["records"][0]["reason"], "demo_email_pattern")
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}/records", params={"decision": "maybe"}).status_code, 422)
        self.assertEqual(self.client.get("/api/runs/missing/records").status_code, 404)
        summary = self.client.get("/api/runs", params={"view": "summary"}).json()[0]
        self.assertEqual(summary["counts"], run["counts"])
        self.assertFalse({"logs", "data_report", "model_report", "artifacts"} & set(summary))
        self.assertIn("logs", self.client.get("/api/runs").json()[0])

    def test_csv_upload_preserves_quoted_newlines_and_metadata(self):
        dataset = self.upload(b'text,source\n"A useful text\nwith two lines",synthetic\nAnother useful text,synthetic\n', "table.csv")
        self.assertEqual(dataset["rows"], 2)
        run = self.wait_run(self.start(dataset, auto_train=False)["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["counts"]["keep"], 2)
        kept = self.client.get(f"/api/runs/{run['id']}/artifacts/screening/keep.jsonl").text
        self.assertEqual(json.loads(kept.splitlines()[0])["source"], "synthetic")
        self.assertIsNone(run["model_report"])

    def test_bundled_example_runs_without_external_models(self):
        response = self.client.post("/api/datasets/example")
        self.assertEqual(response.status_code, 201, response.text)
        dataset = response.json()
        self.assertGreaterEqual(dataset["rows"], 40)
        run = self.wait_run(self.start(dataset)["id"])
        self.assertEqual(run["status"], "completed", run.get("error"))
        self.assertEqual(run["model_report"]["trainer"], "demo")

    def test_invalid_rows_go_to_review_and_never_training(self):
        dataset = self.upload(b'{broken}\n{"text": NaN}\n{"text":"\\ud800"}\n' + sample_jsonl(1))
        self.assertEqual(dataset["invalid_rows"], 3)
        run = self.wait_run(self.start(dataset, auto_train=False)["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["counts"]["review"], 3)
        self.assertEqual(run["counts"]["keep"], 1)
        self.assertNotIn("NaN", json.dumps(dataset))

    def test_invalid_uploads_clean_up_payload_files(self):
        for name, body, expected in [
            ("data.txt", b"unsupported", 422),
            ("empty.jsonl", b" \n", 422),
            ("binary.jsonl", b'{"text":"\xff"}\n', 422),
            ("oversized-row.jsonl", b"x" * (1048576 + 1), 422),
            ("too-big.jsonl", b"x" * (3 * 1048576 + 1), 413),
            ("bad.csv", b'text,text\nfirst,second\n', 422),
            ("unfinished.csv", b'text\n"unclosed\n', 422),
        ]:
            with self.subTest(name=name):
                response = self.client.post("/api/datasets", files={"file": (name, body)})
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(self.client.get("/api/datasets").json(), [])
                self.assertEqual(list((self.root / "datasets").glob("*")), [])

    def test_filename_is_sanitized_and_live_keys_are_required(self):
        dataset = self.upload(sample_jsonl(1), "../../outside.jsonl")
        self.assertEqual(dataset["name"], "outside.jsonl")
        self.assertFalse((self.root.parent / "outside.jsonl").exists())
        response = self.client.post("/api/runs", json={"dataset_id": dataset["id"], "provider": "openrouter"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/api/runs").json(), [])

    def test_cross_origin_write_and_untrusted_host_rejected(self):
        response = self.client.post("/api/datasets", files={"file": ("a.jsonl", sample_jsonl())}, headers={"Origin": "https://attacker.invalid"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"Host": "attacker.invalid"}).status_code, 400)
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertFalse(response.json()["providers"]["openrouter"])

    def test_incomplete_screening_prevents_training(self):
        dataset = self.upload()
        partial = {"counts": {"total": 1, "keep": 1, "review": 0, "reject": 0, "duplicates": 0},
                   "status": "incomplete", "complete": False}
        with patch("jev_dataops.screening.screen_dataset", return_value=partial), patch("jev_dataops.training.train_and_evaluate") as training:
            run = self.wait_run(self.start(dataset)["id"])
            self.assertEqual(run["status"], "failed")
            self.assertIn("incomplete", run["error"])
            training.assert_not_called()

    def test_cancellation_and_retry_reuses_run(self):
        dataset = self.upload()
        entered = threading.Event()
        def waiting_screen(input_path, output_dir, config, progress=None, cancelled=None):
            entered.set()
            deadline = time.monotonic() + 5
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(0.01)
            return {"status": "cancelled", "complete": False,
                    "counts": {"total": 0, "keep": 0, "review": 0, "reject": 0, "duplicates": 0}}
        with patch("jev_dataops.screening.screen_dataset", side_effect=waiting_screen):
            first = self.start(dataset, auto_train=False)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.client.post(f"/api/runs/{first['id']}/cancel").status_code, 200)
            cancelled = self.wait_run(first["id"])
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertIsNone(cancelled["model_report"])
        response = self.client.post(f"/api/runs/{first['id']}/retry")
        self.assertEqual(response.status_code, 200, response.text)
        second = self.wait_run(response.json()["id"])
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["status"], "completed", second.get("error"))
        self.assertEqual(second["counts"]["keep"], 48)

    def test_unknown_and_invalid_configuration_cannot_start(self):
        self.assertEqual(self.client.post("/api/runs", json={"dataset_id": "0" * 32}).status_code, 404)
        dataset = self.upload()
        for values in ({"provider": "other"}, {"concurrency": 999}, {"max_steps": 0}, {"endpoint": "https://attacker.invalid"}):
            response = self.client.post("/api/runs", json={"dataset_id": dataset["id"], **values})
            self.assertEqual(response.status_code, 422, response.text)


class APIAccessTests(unittest.TestCase):
    def test_explicit_token_protects_upload_read_and_artifact_routes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"JEV_API_TOKEN": "integration-test-token"}):
            with TestClient(create_app(directory)) as client:
                self.assertEqual(client.get("/api/health").status_code, 401)
                self.assertEqual(client.get("/api/datasets").status_code, 401)
                self.assertEqual(client.post("/api/datasets/example").status_code, 401)
                self.assertEqual(client.get("/api/runs/anything/artifacts/secret").status_code, 401)
                self.assertEqual(client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code, 401)
                response = client.get("/api/health", headers={"Authorization": "Bearer integration-test-token"})
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("integration-test-token", response.text)

    def test_remote_client_without_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"JEV_API_TOKEN": ""}):
            with TestClient(create_app(directory), client=("198.51.100.9", 54321)) as client:
                response = client.get("/api/health")
                self.assertEqual(response.status_code, 403)

    def test_proxied_loopback_traffic_without_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"JEV_API_TOKEN": ""}):
            with TestClient(create_app(directory), client=("127.0.0.1", 54321)) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                for header in ("X-Forwarded-For", "X-Real-IP", "Forwarded"):
                    response = client.get("/api/health", headers={header: "198.51.100.9"})
                    self.assertEqual(response.status_code, 403, header)
                    self.assertIn("JEV_API_TOKEN", response.text)


class CLITests(unittest.TestCase):
    def test_env_file_fills_missing_variables_only(self):
        from jev_dataops.cli import load_env_file

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"KEEP_ME": "original"}):
            os.environ.pop("JEV_TEST_KEY", None)
            path = os.path.join(directory, ".env")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("# comment\nexport JEV_TEST_KEY='sk-test-value'\nKEEP_ME=overwritten\n\nMALFORMED LINE\n")
            self.assertEqual(load_env_file(path), 1)
            self.assertEqual(os.environ["JEV_TEST_KEY"], "sk-test-value")
            self.assertEqual(os.environ["KEEP_ME"], "original")
            os.environ.pop("JEV_TEST_KEY", None)

    def test_summary_mentions_cost_and_gaps(self):
        from jev_dataops.cli import summarize

        report = {"status": "complete", "mode": "jev_api", "counts": {"total": 10, "keep": 6, "review": 3, "reject": 1, "duplicates": 0},
                  "api_requests": 12, "cache_hits": 0, "usage": {"cost": 0.00123}, "models": {"jev-1.13": 10},
                  "unevaluated": 1, "unevaluated_fraction": 0.1, "error_count": 1, "errors": [{"line": 3, "error": "invalid_response"}],
                  "notice": "1 of 10 rows were not evaluated by Jev."}
        text = summarize(report)
        for fragment in ("keep 6", "12 requests", "$0.0012", "jev-1.13", "1 rows not evaluated", "invalid_response", "not evaluated by Jev"):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
