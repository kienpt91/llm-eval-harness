# Dataset format

A dataset is a set of test cases. The format is JSON only (no YAML), read with the standard library, so a dataset
can be produced or reviewed by any tool.

## Files

- **`.jsonl`** (recommended): one case per line, blank lines allowed, no comments. Diffs are one line per case.
- **`.json`**: an array of cases, or an object `{"cases": [...]}`.

Limits: 8 MiB per file, 10,000 cases, 256 KiB per case, JSON nested at most 32 levels deep. A UTF-8 byte order
mark is ignored. Duplicate keys in one object, `NaN`, `Infinity` and numbers outside the float range are errors
(other readers would disagree about what they mean).

## A case

```json
{"id": "refund-window", "category": "quality", "tags": ["policy", "returns"], "input": {"question": "How many days do I have to return headphones?"}, "context": [{"id": "returns", "text": "Electronics can be returned within 30 days."}], "expected": {"facts": ["30 days"], "forbidden": ["CANARY-7f3a91"], "tool_calls": [{"name": "lookup_policy", "args": {"topic": "returns"}}]}, "graders": [{"type": "grounding", "min_citations": 1, "must_cite": ["returns"]}], "threshold": 1.0, "metadata": {"owner": "support"}}
```

| field | required | meaning |
| --- | --- | --- |
| `id` | yes | Unique name. Letters, digits and `. _ : / -`, starting with a letter or digit, at most 128 characters. Reports and baselines are keyed by it. |
| `input` | yes | Anything JSON: a string, or an object such as `{"question": "...", "user": {"id": "u1"}}`. It is sent to the provider as is. |
| `category` | no | `quality` (default) or `safety`. A failing safety case is always fatal: see [gate.md](gate.md). |
| `tags` | no | Up to 32 labels for filtering with `--tag`. Same characters as ids, no `/`. |
| `context` | no | Up to 64 documents `{"id", "text", "metadata"}` sent to the provider, and the set of documents a `grounding` grader accepts as citations. |
| `expected` | one of `expected` or `graders` | What a good answer looks like. See below. |
| `graders` | one of `expected` or `graders` | Extra graders with weights and `must`. See below. |
| `threshold` | no | Score a run needs to pass, 0 to 1, default 1 (every grader must pass). A `safety` case cannot lower it. |
| `metadata` | no | Free-form object, sent to the provider, otherwise ignored. |

Unknown keys are errors, so a typo (`"expect"`) is caught instead of silently grading nothing.

### `expected`

The expectations are turned into graders, so `expected` is shorthand and `graders` is the full form.

| key | becomes | strictness |
| --- | --- | --- |
| `answers`: `["Paris"]` | [`exact`](graders.md#exact), any of the answers | counts in the score |
| `facts`: `["30 days"]` | [`contains`](graders.md#contains), all of them | counts in the score (fraction found) |
| `forbidden`: `["CANARY-7f3a91"]` | [`not_contains`](graders.md#not_contains) | **must**: a failure fails the run |
| `forbidden_patterns`: `["ssn\\s*\\d+"]` | `not_contains` with patterns | **must** |
| `tool_calls`: `[{"name": ..., "args": {...}}]` | [`tool_calls`](graders.md#tool_calls) | counts in the score |
| `forbidden_tools`: `["issue_refund"]` | [`forbidden_tools`](graders.md#forbidden_tools) | **must** |
| `tool_order`: `"any"` or `"ordered"` | order of `tool_calls` | |
| `allow_extra_tools`: `true` (default) or `false` | whether calls beyond `tool_calls` are allowed | |

`"tool_calls": []` together with `"allow_extra_tools": false` means *no tool may be called at all*.

Tool call arguments are constraints, one per argument name. A plain value means equality. An object means an
operator: `{"equals": v}`, `{"one_of": [...]}`, `{"regex": "^ORD-\\d+$"}`, `{"contains": "text"}`,
`{"type": "string"}`, `{"min": 1, "max": 5}` (these two combine), `{"present": true}`, `{"absent": true}`. To require
an argument that is itself an object use `{"equals": {...}}`.

### `graders`

Each entry is `{"type": ..., "weight": 1, "must": false, "name": "...", ...parameters}`. `weight` (0 to 1000,
default 1) sets a grader's share of the score, `must: true` makes a failure of that grader fail the run whatever
the score is, `name` labels it in reports (default: the type; repeated names get `#2`, `#3`). The parameters of
each type are in [graders.md](graders.md). Graders from the configuration file's `graders` list are added to every
case after the case's own.

A **safety** case also gets a `secret_leak` grader as a `must` unless it lists one itself: a leaked key fails
every safety case, whatever else the case checks. A safety case cannot lower its `threshold` and cannot carry a grader
with weight 0 that is not a `must` (it could never make the case fail): the dataset is refused with the grader named.

### How a run is graded

Score = weighted mean of the graders' scores. A run passes when no `must` grader failed **and** the score is at
least `threshold`. A case with several runs (`--repeat`) has a pass rate: the share of its graded runs that passed.

## Validation and error messages

`llm-eval validate --dataset FILE` checks a file and prints its hash. Every problem is reported, with the line and
the path inside the case:

```
datasets/support.jsonl: 3 problem(s)
  line 1.input: is required
  line 2.expected.tool_calls[0].args.limit.min: must be a number (got string)
  line 3, column 1: invalid JSON: Expecting value
```

Regular expressions in a dataset run on the harness's own linear-time engine with a restricted syntax (no back-references
or look-around), refused here with the position; see [graders.md](graders.md#patterns) and
[SECURITY.md](../SECURITY.md#regular-expressions).

## Hashing

Each case has a hash: SHA-256 of its canonical JSON (sorted keys, no whitespace, ASCII). The dataset hash is the
SHA-256 of the lines `id NUL case-hash`, sorted by id. It does not depend on the order of the cases, changes when
any case changes, and is printed by `validate` and recorded in every report and baseline
(`sha256:6a92802f1e17...`). A baseline records each case's hash too, so `compare` knows which cases still are the
same test.
