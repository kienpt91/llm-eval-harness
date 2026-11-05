# 5. Handling untrusted output

Status: accepted

## Context

The harness prints and stores what a model produced, and a prompt injection can target the human reading the
result as easily as the pipeline: escape sequences that rewrite a terminal, bidirectional marks that reorder a
line, Markdown that forges a verdict in a pull request comment, a megabyte of text, secrets copied into a report that
is uploaded as a CI artifact.

## Decision

- **Sanitise on output.** Everything printed or rendered passes `domain.text.sanitize`: complete CSI/OSC/DCS/APC
  sequences are removed; every other control, format, private-use, unassigned, surrogate or separator character is
  shown as a visible escape. In Markdown, **no untrusted value is ever plain text**: it is an inline code span (outputs:
  a fenced block) with a delimiter longer than any backtick run inside it. A renderer makes no links, images, HTML,
  mentions or autolinks inside a code span; the report has no HTML or links of its own. (The first version escaped
  characters in plain text and let Markdown links and HTML image tags through grader details and error messages.)
- **Store ASCII JSON.** Reports and baselines are written with `\uXXXX` escapes, so no raw control or bidi
  character exists in a file.
- **Redact on store, at every sink.** Secret-shaped strings (the shapes of the repository's own secret scanner) are
  replaced by `[REDACTED:kind]` before anything is stored: outputs, grader details, tool names and arguments,
  citations, the reported model name, inputs, error messages, judge reasons. Text is normalised (NFKC, invisible
  characters dropped) before matching and spans map back to the original; PEM blocks are removed whole; long tokens
  are extended over their tail. Only the first 32,768 characters of a string are examined and kept. Grading runs on
  the original. Redaction is on by default and loud when off. Baselines hold no text.
- **Bound everything.** Output caps are enforced while reading (the program is killed at the cap), parsers have
  depth and size limits, user regular expressions run on a linear-time engine with a work budget on bounded text
  ([ADR 8](0008-a-linear-time-pattern-engine.md)), and exceeding a bound fails closed.
- **Never execute.** Output is data. No `eval`, no shell, no deserialiser that runs code; enforced by the AST test.

## Consequences

- A report is safe to open, paste and upload; a hostile run cannot make a failure look like a pass or hang the gate.
- Some legitimate long outputs fail regex graders (32 KiB limit); `contains`-style graders are unaffected.
