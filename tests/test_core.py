import unittest

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


if __name__ == "__main__":
    unittest.main()
