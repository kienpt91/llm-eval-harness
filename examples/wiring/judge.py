#!/usr/bin/env python3
"""A judge program for the `judge` grader. It is NON-DETERMINISTIC when a model sits behind it: its verdict and
confidence are recorded in the report, never hidden.

stdin : {"protocol": 1, "kind": "judge", "case": {...}, "expected": {"answers": [...], "facts": [...]},
         "output": {"text": "...", "tool_calls": [...], "citations": [...]}, "rubric": "..."}
stdout: {"verdict": "pass" | "fail", "confidence": 0.0..1.0, "score": 0.0..1.0 (optional), "reason": "..."}

This stand-in applies a keyword rubric. To use a judge model, call your own model client where the comment says so
and map its answer onto the three fields above. Put any API key in the environment and name the variable in the
grader's `env_allow`: the harness passes nothing else.
"""

import json
import sys

request = json.load(sys.stdin)
text = request["output"]["text"].lower()
facts = [f.lower() for f in request["expected"]["facts"]]
found = sum(1 for fact in facts if fact in text)
# --- call your judge model here, with request["rubric"], the case input and the answer ---
passed = found == len(facts)
confidence = 0.9 if facts else 0.5
json.dump(
    {
        "verdict": "pass" if passed else "fail",
        "confidence": confidence,
        "score": found / len(facts) if facts else 1.0,
        "reason": f"{found} of {len(facts)} expected facts present",
    },
    sys.stdout,
)
