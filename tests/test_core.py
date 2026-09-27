import types
import unittest
from unittest.mock import patch

import jev_like.runtime as runtime
from demo.live import transcript
from jev_like import parse_schema
from jev_like.benchmark import DEFAULT_DATA, expected_calibration_error, load_cases


class CoreTest(unittest.TestCase):
    def test_schema_and_calibration_metric(self):
        schema = parse_schema({"urgent": {"type": "boolean", "description": "urgent"}})
        self.assertEqual(schema["urgent"].choices, ("true", "false"))
        self.assertAlmostEqual(expected_calibration_error([0.8, 0.8], [True, False]), 0.3)

    def test_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            parse_schema({"x": {"type": "enum", "choices": []}})

    def test_benchmark_has_20_valid_cases(self):
        cases = load_cases(DEFAULT_DATA)
        self.assertEqual(len(cases), 20)
        for case in cases:
            schema = parse_schema(case["schema"])
            self.assertEqual(set(schema), set(case["expected"]))
            for field, expected in case["expected"].items():
                self.assertIn(expected, schema[field].choices)

    def test_live_transcript_scores_the_parallel_path(self):
        case = {"context": "x", "schema": {"urgent": {"type": "boolean"}}, "expected": {"urgent": "true"}}
        auto = {"value": {"urgent": "false"}, "forward_passes": 3, "elapsed_ms": 300.0, "tokens_per_s": 10.0}
        par = {"value": {"urgent": {"value": "true", "score": 0.9}}, "elapsed_ms": 50.0, "tokens_per_s": 100.0}
        text = "\n".join(line for line, _ in transcript("model", case, auto, par))
        self.assertIn("1/1 fields match expected", text)
        self.assertIn("6.0x faster", text)


class BackendDetectionTest(unittest.TestCase):
    def test_reports_available_backend(self):
        stubs = [
            (types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)), "cuda"),
            (types.SimpleNamespace(metal=types.SimpleNamespace(is_available=lambda: True)), "metal"),
            (types.SimpleNamespace(), "cpu"),
        ]
        for stub, expected in stubs:
            with self.subTest(expected=expected), patch.object(runtime, "mx", stub):
                self.assertEqual(runtime.detect_backend(), expected)

    def test_missing_backend_raises_with_instructions(self):
        with patch.object(runtime, "mx", None), self.assertRaises(ImportError) as caught:
            runtime.detect_backend()
        self.assertIn("mlx[cuda12]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
