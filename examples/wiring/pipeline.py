#!/usr/bin/env python3
"""Wire a Python pipeline to llm-eval through the `command` provider.

The harness starts this program for every case run: one JSON request on stdin, one JSON answer on stdout.
Replace `run_pipeline` with a call into your own code (retrieval, agent loop, model client).
"""

import json
import sys


def run_pipeline(question, context_documents):
    cited = [doc["id"] for doc in context_documents[:1]]
    return {
        "text": f"You asked: {question}",
        "tool_calls": [],  # report EVERY side effect your pipeline performed or requested
        "citations": cited,
        "usage": {"model": "my-model", "input_tokens": len(question.split()), "output_tokens": 5},
    }


def main():
    request = json.load(sys.stdin)  # {"protocol": 1, "case": {...}, "repeat": 0, "attempt": 1, "seed": null}
    case = request["case"]
    raw = case["input"]
    question = raw["question"] if isinstance(raw, dict) else str(raw)
    json.dump(run_pipeline(question, case.get("context", [])), sys.stdout)  # nothing else on stdout; logs go to stderr


if __name__ == "__main__":
    main()
