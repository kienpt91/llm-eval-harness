# Wiring your own pipeline

The harness never calls a hosted model and ships no vendor SDK. It evaluates **a program you already have**, in
any language, through a small JSON protocol. Your retrieval, prompts, agent loop and model client stay in your
code; the harness only needs your program to answer one case at a time.

Three providers, chosen under `provider` in the configuration:

| `type` | what it does |
| --- | --- |
| `command` | starts your program for every run: JSON on stdin, JSON on stdout. Any language. |
| `http` | POSTs the same JSON to your endpoint. Loopback, or a host you list explicitly. |
| `fake` | a deterministic built-in used by the tests and the demo; see [configuration.md](configuration.md#the-fake-provider). |

## The protocol

**Request** (stdin, or the POST body): one JSON object. It holds the case's `input` and `context` and nothing that
reveals the answer key: **expectations and graders are never sent**.

```json
{"protocol": 1,
 "case": {"id": "refund-window", "category": "quality", "tags": ["returns"],
          "input": {"question": "How many days do I have to return headphones?"},
          "context": [{"id": "returns", "text": "...", "metadata": {}}],
          "metadata": {}},
 "repeat": 0, "attempt": 1, "seed": 7}
```

`repeat` counts the repeats of `--repeat`, `attempt` the infrastructure retries of one run, `seed` is `--seed`.

**Response**: one JSON object on stdout (or in the 200 response). Unknown keys are an error, so a typo is caught.

| key | required | meaning |
| --- | --- | --- |
| `text` | yes | the answer, a string |
| `structured` | no | the answer as JSON, for `json_schema` and `set_overlap` |
| `tool_calls` | no | `[{"name": "...", "arguments": {...}}]`: **every** side effect your pipeline performed or requested. Safety grading is only as good as this list. |
| `citations` | no | ids of the documents the answer cites |
| `retrieved` | no | ids of the documents your retrieval step returned |
| `usage` | no | `{"model": "name", "input_tokens": 12, "output_tokens": 30}`: cost is computed from it and your [price table](configuration.md#price-table) |
| `metadata` | no | ignored |

Your program must write nothing else to stdout (logs go to stderr; the tail of stderr is shown if the program
fails). Keys such as `__proto__` are plain data everywhere in the harness.

## Python

`examples/wiring/pipeline.py`:

```python
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
```

```json
{"dataset": "cases.jsonl", "provider": {"type": "command", "argv": ["python3", "pipeline.py"]}}
```

## Node

`examples/wiring/pipeline.mjs` (Node 18 or newer, no packages). The answer uses the protocol's snake_case keys.

```js
// Wire a Node pipeline to llm-eval through the `command` provider.
// One JSON request on stdin, one JSON answer on stdout. Replace the body with a call into your own code.
import { stdin, stdout } from "node:process";

let text = "";
for await (const chunk of stdin) text += chunk;
const { case: testCase } = JSON.parse(text); // {"protocol": 1, "case": {...}, "repeat": 0, "attempt": 1, "seed": null}

const raw = testCase.input;
const question = typeof raw === "object" && raw !== null ? raw.question : String(raw);
stdout.write(JSON.stringify({
  text: `You asked: ${question}`,
  tool_calls: [], // report EVERY side effect your pipeline performed or requested
  citations: (testCase.context ?? []).slice(0, 1).map((doc) => doc.id),
  usage: { model: "my-model", input_tokens: question.split(/\s+/).length, output_tokens: 5 },
}));
```

```json
{"provider": {"type": "command", "argv": ["node", "pipeline.mjs"]}}
```

## What the `command` provider guarantees

- **Fixed argv, no shell.** `argv` is a list, used as given. Arguments containing `;` or `$(...)` are just text.
- **The environment is an allow-list.** The program gets `PATH`, `LANG`, `LC_ALL`, the variables named in
  `env_allow`, the fixed values in `env_set`, and `LLM_EVAL=1`. Nothing else from the harness's environment, so a
  program cannot read your CI secrets by accident. If it needs an API key, list the variable *by name*.
- **The working directory** is the project directory, or `cwd` (which must be inside it).
- **Caps and deadline.** Stdout is capped at `max_output_bytes` (default 1 MiB; the program is killed at the cap,
  so memory stays bounded however much it prints), stderr at 64 KiB, the run at `--timeout` seconds.
- **Process hygiene.** The program runs in its own session. A timeout, an overflow or a normal exit all end with
  the whole process group killed and reaped: no orphans, no zombies.
- **Failures are classified.** Timeout, crash and non-zero exit are infrastructure errors (retried, then exit 3);
  invalid JSON, binary output, a response with the wrong shape or too much output are protocol errors (not retried;
  the message says what is wrong and never echoes the output). Latency is measured around the run.

The program is *your* code and runs with your privileges: the harness is not a sandbox. Run untrusted pipelines in
a container (see [SECURITY.md](../SECURITY.md)).

## The `http` provider

```json
{"provider": {"type": "http", "url": "http://127.0.0.1:8080/run",
              "headers": {"X-Team": "evals"}, "headers_env": {"Authorization": "EVAL_ENDPOINT_AUTH"}}}
```

Same request and response. The URL must be on **loopback** (`127.0.0.0/8`, `::1`, `localhost`), or its host must be
listed in `allow_hosts` (exact names, or `host:port`) **and** use `https`. Credentials in the URL are refused,
redirects are never followed, the whole exchange has one deadline (`--timeout`), the body is capped at
`max_output_bytes`. Header values from `headers_env` are read from the environment by name, sent, and never
printed or stored: reports list header *names* only. HTTP 408, 425, 429 and 5xx and connection problems are
retried; other statuses are protocol errors.

## A judge model through the `judge` grader

LLM-as-judge needs a model, and the harness calls none. A [`judge`](graders.md#judge) grader runs *your* program
instead: it receives the case, the answer, the expected facts and a rubric, and answers
`{"verdict": "pass" | "fail", "confidence": 0.0..1.0, "reason": "..."}`. Inside that program you may call any model
with any client. `examples/wiring/judge.py` is the skeleton (with a keyword rubric standing in for the model):

```json
{"type": "judge", "argv": ["python3", "judge.py"], "rubric": "The answer must be polite and correct.",
 "env_allow": ["JUDGE_API_KEY"], "min_confidence": 0.7}
```

The judge runs with the same guarantees as a provider command, including the environment allow-list: an API key
reaches it only if you name the variable. Its verdict and confidence are recorded in the report with
`deterministic: false`. Do not make a judge the only guard of a safety case.

## Tips

- Report tool calls for *intended* actions even when a dry-run flag stops the real effect: the question is what the
  pipeline decided to do.
- Make the pipeline deterministic where you can (temperature 0, fixed seed from the request) and pass
  `--deterministic` to the gate; otherwise use `--repeat 5` or more.
- Keep the dataset's `context` for documents you *supply*. If your pipeline retrieves its own, report their ids in
  `retrieved` and let `grounding` accept them.
- Start with `llm-eval validate`, then `llm-eval run --sample 3`, then the gate.
