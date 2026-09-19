"""Adapter-level checks for the JevBench bridge.

The letter-fallback branch is the one that bit us: scoring ids the prompt never
listed reads out at chance with no error, so it gets a guard.
"""

import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.path.join(ROOT, "bench", "jev_like_direct.py")
PUBLIC = os.path.join(ROOT, "jevbench", "datasets", "public", "original.jsonl")

sys.path.insert(0, os.path.join(ROOT, "jevbench"))
try:
    from jevbench.tasks import load_jsonl
    HAVE_JEZBENCH = True
except ImportError:  # a fresh clone needs `git clone ... jevbench` first
    HAVE_JEZBENCH = False


def load_adapter():
    spec = importlib.util.spec_from_file_location("jev_like_direct", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER_MODULE = load_adapter() if HAVE_JEZBENCH else None


class FakeTokenizer:
    """Single-token iff the string is one letter or a short known word."""

    def encode(self, text, add_special_tokens=False):
        stripped = text.strip()
        known = {"no", "yes", "true", "false", "0", "1", "2"}
        return [ord(stripped[0])] if stripped.lower() in known or len(stripped) == 1 else [7, 7, 7]


@unittest.skipUnless(HAVE_JEZBENCH, "needs a jevbench clone at ./jevbench")
class AdapterTest(unittest.TestCase):
    def test_single_token_labels_are_shown_and_scored_as_themselves(self):
        display, ids = ADAPTER_MODULE.option_display(FakeTokenizer(), ["yes", "no"])
        self.assertEqual(display, ["yes", "no"])
        self.assertEqual(len(set(ids)), 2)

    def test_long_labels_fall_back_to_letters(self):
        display, ids = ADAPTER_MODULE.option_display(
            FakeTokenizer(), ["track_order", "cancel_order"])
        self.assertEqual(display, ["A", "B"])
        self.assertEqual(len(set(ids)), 2)

    def test_labels_follow_the_canonical_record_shape(self):
        seen = set()
        for task in load_jsonl(PUBLIC):
            labels, rubric = ADAPTER_MODULE._labels_and_rubric(task)
            self.assertEqual(set(labels), set(rubric))
            qtype, crit = task.question["type"], task.question["criteria"]
            seen.add(qtype)
            if qtype == "noul":
                # Order-sensitive family; yes-first is the canonical order.
                self.assertEqual(labels, ["yes", "no"])
            elif qtype == "score":
                self.assertEqual(labels, [str(i) for i in range(len(crit))])
            else:
                self.assertEqual(labels, list(crit))
        self.assertEqual(seen, {"noul", "choice", "score"})


if __name__ == "__main__":
    unittest.main()
