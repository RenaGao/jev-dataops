import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_dataops.jev import JevAPIError, JevClient, validate_response
from jev_dataops.screening import MAX_ROW_BYTES, normalize_record, screen_dataset


def valid_response(questions, confidence=0.95):
    answers = {}
    for name, question in questions.items():
        choice = next(iter(question["criteria"]))
        answers[name] = {"type": "choice", "choice": choice, "confidence": confidence,
                         "probabilities": {key: 1 if key == choice else 0 for key in question["criteria"]}}
    return {"model": "test-jev", "answers": answers}


class FakeClient:
    calls = []
    behavior = None

    def __init__(self, provider, max_requests, timeout, attempts):
        self.requests = 0

    def __call__(self, payload, cancelled=None):
        self.requests += 1
        type(self).calls.append(payload)
        if type(self).behavior:
            return type(self).behavior(payload)
        return valid_response(payload["questions"])

    def close(self):
        pass


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "dataset.jsonl"
        self.out = self.root / "result"
        FakeClient.calls = []
        FakeClient.behavior = None

    def write(self, rows):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def live(self, **config):
        with patch("jev_dataops.screening.JevClient", FakeClient):
            return screen_dataset(self.source, self.out, {"provider": "openrouter", **config})

    def records(self, partition):
        return [json.loads(line) for line in (self.out / f"{partition}.jsonl").read_text().splitlines()]

    def test_normalizes_layouts_without_metadata_leak(self):
        for row, expected in [
            ({"text": "Training data", "secret_metadata": "local"}, {"text": "Training data"}),
            ({"messages": [{"role": "user", "content": "Hello there", "private": "local"}], "metadata": {"a": 1}}, {"messages": [{"role": "user", "content": "Hello there"}]}),
            ({"instruction": "Explain", "input": "Gravity", "output": "Mass attracts mass", "owner": "local"}, {"instruction": "Explain", "input": "Gravity", "output": "Mass attracts mass"}),
            ({"prompt": "Explain", "response": "A complete answer", "id": 1}, {"prompt": "Explain", "response": "A complete answer"}),
        ]:
            self.assertEqual(normalize_record(row), expected)

    def test_metadata_preserved_locally_and_not_sent(self):
        row = {"text": "A sufficiently useful training example.", "private_metadata": {"customer": "private"}}
        self.write([row])
        report = self.live()
        self.assertTrue(report["complete"])
        self.assertEqual(self.records("keep"), [row])
        self.assertEqual(FakeClient.calls[0]["state"], {"text": row["text"]})
        self.assertNotIn("private_metadata", json.dumps(FakeClient.calls))

    def test_malformed_message_roles_are_reviewed_without_aborting(self):
        roles = [[], {}, 123, True, None]
        malformed = [{"messages": [{"role": role, "content": "A useful training example"}]} for role in roles]
        self.write(malformed + [{"text": "A valid row after malformed message roles"}])
        report = screen_dataset(self.source, self.out, {})
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["review"], len(roles))
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertEqual(self.records("review"), malformed)

    def test_cache_resume_does_not_turn_first_row_into_duplicate(self):
        self.write([{"text": "A useful training example", "id": 1}, {"text": "A useful training example", "id": 2}])
        first = self.live(concurrency=1)
        second = self.live(concurrency=2, max_requests=3)
        self.assertEqual(first["counts"], {"total": 2, "keep": 1, "review": 0, "reject": 1, "duplicates": 1})
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(len(FakeClient.calls), 1)
        self.assertEqual(second["cache_hits"], 1)
        self.assertEqual(second["api_requests"], 0)
        self.assertEqual(self.records("keep")[0]["id"], 1)

    def test_semantic_config_change_invalidates_cache(self):
        self.write([{"text": "A useful training example"}])
        self.live(confidence=0.85)
        report = self.live(confidence=0.99)
        self.assertEqual(len(FakeClient.calls), 2)
        self.assertEqual(report["counts"]["review"], 1)

    def test_invalid_response_never_kept_or_cached(self):
        self.write([{"text": "A useful training example"}])
        FakeClient.behavior = lambda payload: {"model": "test", "answers": {}}
        first = self.live()
        self.assertEqual(first["counts"]["keep"], 0)
        self.assertEqual(first["counts"]["review"], 1)
        # A malformed answer is asked for once more, and the reason is on the record.
        self.assertEqual(len(FakeClient.calls), 2)
        self.assertEqual(first["unevaluated"], 1)
        self.assertFalse(first["training_ready"])
        self.assertEqual(first["errors"][0]["detail"], "unexpected answer dimensions")
        FakeClient.behavior = None
        second = self.live()
        self.assertEqual(second["counts"]["keep"], 1)
        self.assertTrue(second["training_ready"])
        self.assertEqual(len(FakeClient.calls), 3)

    def test_one_bad_answer_then_a_good_one_is_kept(self):
        self.write([{"text": "A useful training example"}])
        answers = iter([{"model": "test", "answers": {}}, None])
        FakeClient.behavior = lambda payload: next(answers) or valid_response(payload["questions"])
        report = self.live()
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertEqual(report["unevaluated"], 0)
        self.assertEqual(len(FakeClient.calls), 2)

    def test_whitespace_variants_are_one_row(self):
        self.write([{"text": "A useful   training example"}, {"text": " A useful training example \n"}])
        report = self.live()
        self.assertEqual(report["counts"], {"total": 2, "keep": 1, "review": 0, "reject": 1, "duplicates": 1})
        self.assertEqual(report["dedupe"], "whitespace")
        self.assertEqual(len(FakeClient.calls), 1)

    def test_code_rubric_keeps_whitespace_variants_apart(self):
        self.write([{"text": "def f():\n    return 1"}, {"text": "def f():\n  return 1"}])
        report = self.live(rubric="code")
        self.assertEqual(report["dedupe"], "exact")
        self.assertEqual(report["counts"]["duplicates"], 0)
        self.assertEqual(len(FakeClient.calls), 2)

    def test_jev_reasons_name_the_deciding_dimensions(self):
        from jev_dataops.screening import decision_records, reason_label

        self.write([{"text": "A useful training example"}, {"text": "Another useful training example"}])
        def low_quality(payload):
            response = valid_response(payload["questions"])
            response["answers"]["quality"].update(choice="good", probabilities={"good": 0.5, "uncertain": 0.3, "bad": 0.2})
            return response
        FakeClient.behavior = low_quality
        report = self.live()
        self.assertEqual(report["decision_reasons"]["review"], {"quality:keep_probability_below_threshold": 2})
        page = decision_records(self.out, "review", "quality:keep_probability_below_threshold", limit=1)
        self.assertEqual(page["records"][0]["record"]["text"], "A useful training example")
        self.assertTrue(page["has_more"])
        self.assertEqual(reason_label({"reason": "jev_decision", "decision": "reject", "dimensions": {
            "privacy": {"decision": "reject", "value": "sensitive"}, "quality": {"decision": "reject", "value": "bad"},
            "trainability": {"decision": "keep", "value": "suitable"}}}), "privacy:sensitive, quality:bad")

    def test_usage_and_models_are_aggregated_but_not_cached(self):
        self.write([{"text": f"Useful unique record number {i}"} for i in range(3)])
        def priced(payload):
            response = valid_response(payload["questions"])
            response.update(model="jev-1.13", usage={"input_tokens": 100, "output_tokens": 10, "cost": 0.001}, id="gen-1")
            return response
        FakeClient.behavior = priced
        first = self.live()
        self.assertEqual(first["usage"], {"input_tokens": 300, "output_tokens": 30, "cost": 0.003})
        self.assertEqual(first["models"], {"jev-1.13": 3})
        audit = [json.loads(line) for line in (self.out / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(audit[0]["provider_id"], "gen-1")
        second = self.live()
        self.assertEqual(second["cache_hits"], 3)
        self.assertEqual(second["usage"]["cost"], 0)
        self.assertEqual(second["models"], {"jev-1.13": 3})

    def test_thresholds_are_reported(self):
        self.write([{"text": "A useful training example"}])
        report = self.live(confidence=0.9)
        self.assertEqual(report["thresholds"]["privacy"]["min_confidence"], 0.9)
        self.assertIsNone(report["thresholds"]["quality"]["min_confidence"])
        self.assertEqual(report["thresholds"]["quality"]["min_probability"], 0.6)

    def test_authentication_failure_marks_incomplete(self):
        self.write([{"text": f"Useful unique record number {i}"} for i in range(20)])
        def fail(payload):
            raise JevAPIError("authentication", 401)
        FakeClient.behavior = fail
        report = self.live(concurrency=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["keep"], 0)
        self.assertFalse(report["input_exhausted"])
        self.assertLessEqual(report["processed"], 2)

    def test_invalid_rows_and_utf8_are_reviewed(self):
        self.source.write_bytes(b'{broken}\n[1,2]\n{"text": "\xff"}\n{"text":"A valid long enough example"}\n')
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"]["review"], 3)
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertEqual(report["error_count"], 3)
        self.assertEqual(report["mode"], "demo_rule_based")

    def test_oversized_jsonl_row_recovers_next_row(self):
        with self.source.open("wb") as stream:
            stream.write(b'x' * (MAX_ROW_BYTES + 1) + b'\n')
            stream.write(b'{"text":"The next record remains usable"}\n')
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"]["review"], 1)
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertTrue(report["complete"])

    def test_csv_quoted_newlines_and_metadata(self):
        self.source = self.root / "dataset.csv"
        self.source.write_text('text,owner\n"A multiline\nuseful passage",local\n"Another useful passage",other\n', encoding="utf-8")
        report = self.live()
        self.assertEqual(report["counts"]["keep"], 2)
        self.assertEqual(self.records("keep")[0]["owner"], "local")
        self.assertNotIn("owner", json.dumps(FakeClient.calls))

    def test_corrupt_csv_is_incomplete(self):
        self.source = self.root / "dataset.csv"
        self.source.write_text('text,owner\n"unterminated\n', encoding="utf-8")
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["keep"], 0)

    def test_cancellation_does_not_report_complete(self):
        self.write([{"text": "A useful training example"}])
        report = screen_dataset(self.source, self.out, {}, cancelled=lambda: True)
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(report["complete"])
        self.assertEqual(report["processed"], 0)

    def test_demo_only_checks_rules_and_has_no_semantic_dimensions(self):
        self.write([{"text": "Just enough"}, {"text": "a"}, {"text": "Email example@example.com for details."}])
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"], {"total": 3, "keep": 1, "review": 1, "reject": 1, "duplicates": 0})
        self.assertEqual(report["dimensions"], {})
        self.assertIn("no Jev", report["notice"])

    def test_nonfinite_or_boolean_configuration_rejected(self):
        self.write([])
        for config in ({"confidence": float("nan")}, {"confidence": True}, {"confidence": 10 ** 1000}, {"concurrency": True}, {"max_chars": 0}, {"rubric": "../../secret"}, {"rubric": []}, {"provider": {}}):
            with self.assertRaises(ValueError):
                screen_dataset(self.source, self.out, config)

    def test_provider_key_only_from_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                JevClient("openrouter")

    def test_request_budget_is_enforced_before_network(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-secret"}):
            client = JevClient("openrouter", max_requests=0)
            with patch.object(JevClient, "_post") as post:
                with self.assertRaisesRegex(JevAPIError, "request_budget_exhausted"):
                    client({"state": {"text": "demo"}})
                post.assert_not_called()

    def test_connection_is_reused_across_requests(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-secret"}):
            client = JevClient("openrouter", max_requests=10)
            body = json.dumps({"model": "m", "answers": {}}).encode()
            with patch("jev_dataops.jev.http.client.HTTPSConnection") as factory:
                connection = factory.return_value
                connection.getresponse.return_value.status = 200
                connection.getresponse.return_value.read.return_value = body
                connection.getresponse.return_value.getheader.return_value = "keep-alive"
                client({"state": {"text": "one"}})
                client({"state": {"text": "two"}})
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(connection.request.call_count, 2)

    def test_https_proxy_is_used_as_a_connect_tunnel(self):
        env = {"OPENROUTER_API_KEY": "test-not-secret", "HTTPS_PROXY": "http://user:p%40ss@proxy.corp.example:8080", "NO_PROXY": "internal.example"}
        with patch.dict(os.environ, env, clear=False):
            client = JevClient("openrouter", max_requests=10)
            self.assertEqual(client._proxy[:2], ("proxy.corp.example", 8080))
            self.assertEqual(client._proxy[2]["Proxy-Authorization"], "Basic dXNlcjpwQHNz")
            with patch("jev_dataops.jev.http.client.HTTPSConnection") as factory:
                response = factory.return_value.getresponse.return_value
                response.status, response.read.return_value, response.getheader.return_value = 200, b'{"model": "m", "answers": {}}', ""
                client({"state": {"text": "one"}})
                self.assertEqual(factory.call_args.args[:2], ("proxy.corp.example", 8080))
                factory.return_value.set_tunnel.assert_called_once_with("openrouter.ai", 443, headers=client._proxy[2])

    def test_no_proxy_without_environment(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-secret"}, clear=False):
            for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
                os.environ.pop(name, None)
            self.assertIsNone(JevClient("openrouter")._proxy)

    def test_redirects_are_not_followed(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-secret"}):
            client = JevClient("openrouter", max_requests=10, attempts=1)
            with patch("jev_dataops.jev.http.client.HTTPSConnection") as factory:
                response = factory.return_value.getresponse.return_value
                response.status, response.headers = 302, {"Location": "https://elsewhere.example/"}
                response.read.return_value = b""
                response.getheader.return_value = ""
                with self.assertRaisesRegex(JevAPIError, "provider_http_error"):
                    client({"state": {"text": "one"}})


class ResponseValidationTests(unittest.TestCase):
    def setUp(self):
        self.rubric = json.loads((Path(__file__).parents[1] / "jev_dataops/rubrics/general.json").read_text())
        self.valid = valid_response(self.rubric["questions"])

    def test_valid_and_low_confidence(self):
        self.assertEqual(validate_response(self.valid, self.rubric, 0.85)["decision"], "keep")
        self.assertEqual(validate_response(self.valid, self.rubric, 0.99)["decision"], "review")

    def answer(self, choice, probabilities, confidence):
        return {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence}

    def test_rounded_probabilities_from_the_live_api_are_accepted(self):
        response = copy.deepcopy(self.valid)
        response["answers"]["quality"] = self.answer("good", {"good": 0.81, "uncertain": 0.17, "bad": 0.02}, 0.71)
        response["answers"]["trainability"] = self.answer("suitable", {"suitable": 0.33, "uncertain": 0.33, "unsuitable": 0.33}, 0.9)
        result = validate_response(response, self.rubric, 0.85)
        self.assertEqual(result["dimensions"]["quality"]["decision"], "keep")
        self.assertEqual(result["dimensions"]["quality"]["probability"], 0.81)
        self.assertEqual(result["dimensions"]["trainability"]["gate"], "keep_probability_below_threshold")

    def test_quality_gates_on_probability_mass_not_confidence(self):
        response = copy.deepcopy(self.valid)
        # A clearly good row with Jev's confidence field at 0.41: the shape live data has.
        response["answers"]["quality"] = self.answer("good", {"good": 0.81, "uncertain": 0.17, "bad": 0.02}, 0.41)
        self.assertEqual(validate_response(response, self.rubric, 0.85)["decision"], "keep")
        # Half the mass on good and a real chance of bad goes to review.
        response["answers"]["quality"] = self.answer("good", {"good": 0.5, "uncertain": 0.25, "bad": 0.25}, 0.95)
        result = validate_response(response, self.rubric, 0.85)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["dimensions"]["quality"]["gate"], "keep_probability_below_threshold")
        response["answers"]["quality"] = self.answer("good", {"good": 0.7, "uncertain": 0.05, "bad": 0.25}, 0.95)
        result = validate_response(response, self.rubric, 0.85)
        self.assertEqual(result["dimensions"]["quality"]["gate"], "reject_probability_above_threshold")

    def test_privacy_still_honours_the_run_confidence(self):
        response = copy.deepcopy(self.valid)
        response["answers"]["privacy"] = self.answer("clear", {"clear": 1, "uncertain": 0, "sensitive": 0}, 0.6)
        result = validate_response(response, self.rubric, 0.85)
        self.assertEqual(result["dimensions"]["privacy"]["gate"], "confidence_below_threshold")
        self.assertEqual(validate_response(response, self.rubric, 0.5)["decision"], "keep")

    def test_usage_and_id_are_carried(self):
        response = copy.deepcopy(self.valid)
        response.update(usage={"input_tokens": 652, "output_tokens": 124, "cost": 2.7e-05, "junk": "x"}, id="gen-dec-1")
        result = validate_response(response, self.rubric, 0.85)
        self.assertEqual(result["usage"], {"input_tokens": 652, "output_tokens": 124, "cost": 2.7e-05})
        self.assertEqual(result["provider_id"], "gen-dec-1")

    def test_invalid_distributions_types_and_missing_dimensions(self):
        mutations = [
            lambda r: r["answers"].pop("quality"),
            lambda r: r["answers"]["quality"].update(confidence=True),
            lambda r: r["answers"]["quality"].update(confidence=float("nan")),
            lambda r: r["answers"]["quality"].update(type="score"),
            lambda r: r["answers"]["quality"]["probabilities"].update(good=0.2),
            lambda r: r["answers"]["quality"].update(choice="bad"),
            lambda r: r["answers"]["quality"].update(choice=[]),
            lambda r: r["answers"]["quality"].update(probabilities=[]),
            lambda r: r["answers"]["quality"].update(confidence={}),
            lambda r: r.update(model="invalid-surrogate-\ud800"),
            lambda r: r.pop("model"),
        ]
        for mutate in mutations:
            response = copy.deepcopy(self.valid)
            mutate(response)
            with self.assertRaises(ValueError):
                validate_response(response, self.rubric, 0.85)


if __name__ == "__main__":
    unittest.main()
