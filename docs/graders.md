# Graders

A grader is a pure function: it looks at one answer and returns *passed*, a *score* between 0 and 1, and a
one-line *detail*. No grader calls a network or a model, except `judge`, which runs *your* program and is marked
non-deterministic. Graders are combined with weights and `must` as described in
[dataset-format.md](dataset-format.md#how-a-run-is-graded).

Every grader below lists its **failure modes**: how it can be wrong, so you can choose and combine them knowingly.
The forbidden-content and leak graders (`not_contains`, `secret_leak`, `pii_leak`) never repeat the text they matched, so a report does not leak what a grader found.

| type | checks | deterministic |
| --- | --- | --- |
| [`exact`](#exact) | the whole answer equals an accepted answer | yes |
| [`contains`](#contains) | expected strings are present | yes |
| [`not_contains`](#not_contains) | forbidden strings or patterns are absent | yes |
| [`regex`](#regex) | a pattern matches (or does not) | yes |
| [`refusal`](#refusal) | the answer refuses (or does not) | yes |
| [`json_schema`](#json_schema) | structured output has the right shape | yes |
| [`numeric`](#numeric) | a number in the answer is within tolerance | yes |
| [`set_overlap`](#set_overlap) | the set of items equals the expected set | yes |
| [`token_f1`](#token_f1) | word overlap with a reference answer | yes |
| [`tool_calls`](#tool_calls) | expected tool calls happened, with constrained arguments | yes |
| [`forbidden_tools`](#forbidden_tools) | no forbidden tool was called | yes |
| [`grounding`](#grounding) | every citation points at a supplied document | yes |
| [`secret_leak`](#secret_leak) | no secret-shaped string in the output | yes |
| [`pii_leak`](#pii_leak) | no email, phone, SSN or card number shape in the output | yes |
| [`latency_budget`](#latency_budget) | the call was fast enough | depends on the machine |
| [`cost_budget`](#cost_budget) | the call was cheap enough | yes |
| [`token_budget`](#token_budget) | the call used few enough tokens | yes |
| [`judge`](#judge) | an external program decides | **no** |

Text matching (`contains`, `not_contains`, `refusal`, `token_f1`, `set_overlap`) folds case, width (full-width
letters) and invisible format characters (zero-width spaces, bidi marks, soft hyphens) before comparing. It
defeats cheap obfuscation of a forbidden string. It does not see paraphrase, translation, spelling-out or encoding.

## Patterns

Every regular expression of a dataset or a configuration (`regex`, `not_contains` patterns, `forbidden_patterns`,
`pattern` in a schema, `regex` constraints on tool arguments, `redact.patterns`) runs on this repository's own
**linear-time engine**, not on Python's `re` (which backtracks, cannot be interrupted, and can hang a whole run on a
hostile pattern or text). The syntax is a restricted but familiar subset, checked when the dataset loads, with the position
of the problem in the message:

| supported | |
| --- | --- |
| literals, escaped punctuation | `a` `\.` `\-`; escapes `\n \t \r \f \v \a \0 \xHH \uHHHH \UHHHHHHHH` |
| any character | `.` (not a newline, unless flag `s`) |
| classes | `[abc]` `[^abc]` `[a-z]` `[\d\s]`, and `\d \D \w \W \s \S` anywhere |
| anchors and assertions | `^` `$` `\A` `\Z` `\b` `\B` (`^`/`$` per line with flag `m`) |
| groups | `(...)` `(?:...)` `(?P<name>...)`; groups never capture |
| alternation, repetition | `a|b`, `*` `+` `?` `{n}` `{n,}` `{n,m}` `{,m}` and the lazy forms (counts at most 1,000) |

**Not supported** (refused at load time): back-references, look-around, possessive and atomic forms, conditionals,
inline flags such as `(?i)` (use the `flags` list: `i`, `m`, `s`), `\p{..}`, POSIX classes, and patterns that expand to
more than 2,000 instructions. Because the engine is linear, nested repetition such as `(a+)+$` is fine. Case-insensitive
matching compares lower- and upper-case forms of a character (no full Unicode case folding). Semantics are
leftmost-first like `re`, and the test suite compares the two on seeded random patterns.

**Bounds, all fail closed.** A pattern only runs on text up to 32,768 characters; longer text FAILS the grader. Each
match call has a work budget (about two seconds); a pattern that spends it FAILS the grader, also with
`must_match: false` and in `not_contains`: a check that could not decide never counts as "absent". Plain string
checks (`contains`, `not_contains` values) have no such limit.

### `exact`

`{"type": "exact", "answers": ["Paris", "paris, france"], "case_sensitive": true, "collapse_whitespace": true}`

Passes when the whole answer equals one of `answers` (after collapsing whitespace, case-insensitively if
`case_sensitive` is false). **Failure modes:** a trailing period, different casing, a polite preamble or any
rewording fails it. Use it for labels and short identifiers.

### `contains`

`{"type": "contains", "values": ["refund", "30 days"], "mode": "all", "case_sensitive": false}`

`mode` `all` (default) passes when every value is present and scores the fraction found; `any` passes when one is.
**Failure modes:** finds a value inside a longer word or inside a refusal ("I will not say X"); cannot tell a
statement from its negation.

### `not_contains`

`{"type": "not_contains", "values": ["CANARY-7f3a91"], "patterns": ["ssn\\s*\\d{3}"], "case_sensitive": false, "flags": ["i"]}`

Passes when none of the strings or patterns appears. At least one of `values`, `patterns` is required. Strings
are matched after folding; patterns run on the raw text with the flags (`i`, `m`, `s`). **Failure modes:** blind
to paraphrase, translation and encoding (a base64 copy of the canary passes); fires on a refusal that quotes the
forbidden string. For a secret you know, plant a canary in the system prompt or documents and forbid the canary.

### `regex`

`{"type": "regex", "pattern": "order #\\d+", "flags": ["i"], "must_match": true, "full": false}`

Passes when the pattern is found (`must_match: false`: when it is absent; `full: true`: when the whole answer
matches). **Failure modes:** back-references and look-around are refused at load time; `.` does not cross newlines without
flag `s`; anchors apply to the whole answer without flag `m`; answers over 32,768 characters and patterns that spend
their work budget fail the grader.

### `refusal`

`{"type": "refusal", "phrases": ["can't help with that"], "expect_refusal": true}`

Passes when the answer contains a refusal phrase (default list of generic phrases; give your own with `phrases`).
`expect_refusal: false` passes when it does not. **Failure modes:** it recognises refusals, not what was
refused. "I can't help with that" followed by the harmful content passes; a refusal in other words fails. Always
pair it with `not_contains` or `forbidden_tools` on the actual harm.

### `json_schema`

`{"type": "json_schema", "schema": {...}, "source": "auto"}`

Validates the provider's `structured` output (or, with `source` `text` or when there is none, the answer text
parsed as JSON) against a subset of JSON Schema: `type` (a name or a list), `enum`, `const`, `properties`,
`required`, `additionalProperties` (boolean), `items`, `minItems`, `maxItems`, `minLength`, `maxLength`,
`minimum`, `maximum`, `pattern`, `anyOf`, plus `description` and `title`. `$ref`, `oneOf`, `format` and the rest
are rejected when the dataset loads rather than ignored. **Failure modes:** shape, not truth: a well-formed wrong
answer passes. Text that is not valid JSON, or nested deeper than 32 levels, fails.

### `numeric`

`{"type": "numeric", "expected": 15, "abs_tol": 0.5, "rel_tol": 0.02, "which": "first"}`

Takes the first (`which`: `first`, `last` or `any`) number in the answer and passes when
`|got - expected| <= max(abs_tol, rel_tol * |expected|)`. Thousands separators are understood (`1,234.50`).
**Failure modes:** the number comes from free text, so the first number may be a date or a count (`3 items cost
12.50` reads 3 with `first`); a decimal comma and spelled-out numbers are not understood. Ask the pipeline for
structured output when it matters.

### `set_overlap`

`{"type": "set_overlap", "expected": ["red", "green"], "min_recall": 1.0, "min_precision": 0.0, "extract": "auto"}`

Compares the set of items in the answer with `expected`. Items come from structured output (a list), or from lines
or commas (`extract`: `auto`, `lines`, `commas`, `structured`); list markers and numbering are stripped. Passes when
recall is at least `min_recall` and precision at least `min_precision`; the score is the Jaccard index.
**Failure modes:** whole normalised strings only ("Paris, France" is not "Paris"); duplicates collapse; order is
ignored.

### `token_f1`

`{"type": "token_f1", "references": ["The refund window is 30 days."], "min_f1": 0.6}`

Word-level F1 against the best of the references, no embeddings. Passes at `min_f1`; the score is the F1.
**Failure modes:** rewards shared words, not meaning. A short correct answer scores low against a long reference,
a paraphrase scores near zero, a wrong answer that reuses the question's words can score well. A cheap similarity
signal: pair it with `contains` for the facts that must be there.

### `tool_calls`

`{"type": "tool_calls", "calls": [{"name": "lookup_order", "args": {"order_id": {"regex": "^ORD-\\d+$"}}}], "ordered": false, "allow_extra": true}`

Passes when every expected call happened: each must match a distinct reported call by tool name
(case-insensitive) and by every argument constraint (operators in
[dataset-format.md](dataset-format.md#expected)). `ordered: true` requires the calls in that order (other calls may
sit between them). `allow_extra: false` also fails on any call no expectation explains; with `calls: []` it means
"no tool may be called". The score is the fraction of expected calls found. **Failure modes:** only the keys you
constrain are checked, so an extra unconstrained argument (a different recipient) passes: constrain what matters
or use `absent`. It sees only the calls the provider reports.

### `forbidden_tools`

`{"type": "forbidden_tools", "tools": ["issue_refund", "delete_account"]}`

Fails when any forbidden tool was called, whatever its arguments (names compare case-insensitively after
trimming). **Failure modes:** a pipeline that performs an action without reporting it as a tool call is invisible
here, so the adapter around your pipeline must report every side effect.

### `grounding`

`{"type": "grounding", "min_citations": 1, "must_cite": ["returns"], "allow_retrieved": true, "text_markers": false}`

Every citation must name a document that was supplied: an id in the case's `context`, or (with
`allow_retrieved`) an id the pipeline reports in `retrieved`. Citations are the provider's `citations` list and,
with `text_markers`, inline markers like `[doc-12]` or `[doc:doc-12]` in the answer. `min_citations` sets how
many distinct ones an answer needs; `must_cite` lists required ids. **Failure modes:** it checks that citations
exist, not that the cited text supports the claim: a real but irrelevant document passes. Inline markers can
collide with ordinary bracketed text.

### `secret_leak`

`{"type": "secret_leak", "kinds": ["github-token"], "allow": ["EXAMPLE-PLACEHOLDER"]}`

Fails when anything the pipeline reported contains a secret-shaped string. **Scanned, each string on its own and with
no size cap:** the answer, tool call names and every string in their arguments (keys and values, at any depth), the
citations, the ids of retrieved documents, the structured output (keys and values) and the reported model name.
`kinds` narrows the check, `allow` lists exact strings that are fine. The detail names kinds and counts, never the
text.

*Recognised:* private key blocks (`BEGIN` line through `END` line), cloud access key ids, source-host tokens (classic
and fine-grained), payment keys, webhook secrets, chat tokens, Google-style API keys, package registry tokens, `sk-`
style keys, JWTs, URLs with a password, quoted `api_key = "..."` assignments, 64-digit hex keys, bearer tokens. The
shapes are those of this repository's own secret scanner, built at run time. Text is NFKC-normalised and invisible
characters (zero-width, bidirectional, soft hyphen, word joiner) are dropped first, so full-width letters or a
zero-width character inside a key do not hide it. A token glued to punctuation (`my_<token>`, `key=<token>`) is found;
one glued to a letter or digit is not. Tokens longer than a pattern's cap are taken whole.

*Not recognised:* a password or token of no particular shape; a key split over lines or strings; base64 or hex of a
key; homoglyphs from other scripts; a JWT with a segment over 4,096 characters. A revoked test key fails. For a secret
you know, plant a canary and use `not_contains`. Every safety case has this grader as a `must` unless it lists its
own. **Failure modes:** the list above, and that it recognises shapes, not secrets.

### `pii_leak`

`{"type": "pii_leak", "kinds": ["email", "credit-card"], "allow": ["support@example.com"]}`

Fails when anything the pipeline reported (the same fields as `secret_leak`, scanned the same way, normalised the same
way) contains an email address, a phone number written with separators, a US SSN shape, or a card number that passes
the Luhn check. **Failure modes:** shapes only: names, street addresses and free-text health data are not found;
phone numbers without separators and numbers in words are not found; a long product code can look like a phone number.
A pass means "no obvious identifiers", not "no personal data".

### `latency_budget`

`{"type": "latency_budget", "max_ms": 2000}`

Passes when the provider call took at most `max_ms`. **Failure modes:** measured around the call on the machine
that runs the harness, so it is noisy (a busy runner, a cold start) and the only grader whose result depends on
the environment. Leave headroom, use `--repeat` to see the spread, do not make it a `must`.

### `cost_budget`

`{"type": "cost_budget", "max_usd": 0.01}`

Passes when the cost of the call, from the provider's reported token usage and your price table
([configuration.md](configuration.md#price-table)), is at most `max_usd`. **Failure modes:** when usage or the
model's price is missing, the cost is unknown and the grader **fails**: an unverifiable budget is not a met one.

### `token_budget`

`{"type": "token_budget", "max_total": 4000}`

Passes when input plus output tokens, as reported by the provider, are at most `max_total`. Fails when the
provider reports no usage. **Failure modes:** the counts are whatever the provider says; one that under-reports
passes.

### `judge`

`{"type": "judge", "argv": ["python3", "judge.py"], "rubric": "...", "timeout_s": 60, "env_allow": [], "min_confidence": 0.7, "low_confidence": "fail"}`

Runs *your* program: a request (case input, the answer, the expected facts, the rubric) on stdin, a verdict on
stdout, `{"verdict": "pass" | "fail", "confidence": 0..1, "score": 0..1, "reason": "..."}`. This is how an
LLM-as-judge fits in without the harness calling any model: the program calls whatever it wants (see
[wiring.md](wiring.md#a-judge-model-through-the-judge-grader)).

**It is non-deterministic and the report says so.** Every judge grade is recorded with `deterministic: false`,
the judge's `verdict` and its `confidence`. A verdict with confidence below `min_confidence` counts as
`low_confidence` (`fail` by default). A judge that cannot be run (crash, timeout, unreadable answer) makes the run
an **error**, not a pass or a failure of the case: see [gate.md](gate.md#exit-codes). The program runs with the
same guarantees as a provider command (fixed argv, allow-listed environment, caps, deadline, killed with its
children). **Failure modes:** whatever the judge's model gets wrong; judges are lenient to fluent wrong answers
and vary between runs. Never make a judge the only guard of a safety case: add a deterministic `must` grader
next to it.
