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
  shown as a visible escape. Markdown puts untrusted text in fences longer than any backtick run and escapes cells.
- **Store ASCII JSON.** Reports and baselines are written with `\uXXXX` escapes, so no raw control or bidi
  character exists in a file.
- **Redact on store.** Secret-shaped strings (the shapes of the repository's own secret scanner) are replaced by
  `[REDACTED:kind]` before anything is stored; grading runs on the original. Redaction is on by default and loud
  when off. Baselines hold no text.
- **Bound everything.** Output caps are enforced while reading (the program is killed at the cap), parsers have
  depth and size limits, user regular expressions are screened and only run on bounded text, and exceeding a bound
  fails closed.
- **Never execute.** Output is data. No `eval`, no shell, no deserialiser that runs code; enforced by the AST test.

## Consequences

- A report is safe to open, paste and upload; a hostile run cannot make a failure look like a pass or hang the gate.
- Some legitimate long outputs fail regex graders (32 KiB limit); `contains`-style graders are unaffected.
