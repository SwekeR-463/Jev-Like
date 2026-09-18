import importlib.util
import unittest
from pathlib import Path

from benchmark import load_cases
from jev_like import expected_calibration_error, parse_schema


class CoreTest(unittest.TestCase):
    def test_schema_and_calibration_metric(self):
        schema = parse_schema({"urgent": {"type": "boolean", "description": "urgent"}})
        self.assertEqual(schema["urgent"].choices, ("true", "false"))
        self.assertAlmostEqual(expected_calibration_error([0.8, 0.8], [True, False]), 0.3)

    def test_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            parse_schema({"x": {"type": "enum", "choices": []}})

    def test_benchmark_has_20_valid_cases(self):
        cases = load_cases("data/benchmark.jsonl")
        self.assertEqual(len(cases), 20)
        for case in cases:
            schema = parse_schema(case["schema"])
            self.assertEqual(set(schema), set(case["expected"]))
            for field, expected in case["expected"].items():
                self.assertIn(expected, schema[field].choices)

    def test_live_transcript_scores_the_parallel_path(self):
        spec = importlib.util.spec_from_file_location("live", Path(__file__).parent / "demo/live.py")
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        case = {"context": "x", "schema": {"urgent": {"type": "boolean"}}, "expected": {"urgent": "true"}}
        auto = {"value": {"urgent": "false"}, "forward_passes": 3, "elapsed_ms": 300.0, "tokens_per_s": 10.0}
        par = {"value": {"urgent": {"value": "true", "score": 0.9}}, "elapsed_ms": 50.0, "tokens_per_s": 100.0}
        text = "\n".join(line for line, _ in live.transcript("model", case, auto, par))
        self.assertIn("1/1 fields match expected", text)
        self.assertIn("6.0x faster", text)


if __name__ == "__main__":
    unittest.main()
