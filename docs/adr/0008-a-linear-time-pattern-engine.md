# 8. A linear-time pattern engine for user regular expressions

Status: accepted (replaces the screen of the first version)

## Context

Datasets, configuration and redaction lists contain regular expressions, and they are applied to model output. Python's
`re` backtracks, cannot be interrupted and holds the interpreter lock, so a hostile pattern or text hangs the whole run
and the CI job with it. The first version screened patterns (no back-references, look-around or unbounded repetition of
a group that repeats) and capped the text at 32 KiB. A review showed the screen is unsound: `.*a.*b.*c.*d` and
`(a?){30}a{30}` pass it, and `.*a.*b.*c.*d` on 33 KiB ran past twenty seconds in a check. Screening backtracking
patterns cannot be made sound.

## Decision

User patterns run on `domain/grading/regexengine.py`, a Pike VM written for this tool:

- it advances every possible match in lock step, so cost is at most `len(text) * len(program)` for **any** pattern,
  nested repetition included;
- the syntax is a restricted subset of `re` (no back-references, look-around, possessive or atomic forms, inline
  flags, `\p{..}`), refused when the dataset loads, with the position; programs over 2,000 instructions and repetition
  counts over 1,000 are refused;
- every call has a step budget (2,000,000, about two seconds); when it is spent the caller fails closed, for
  `must_match: false` and `not_contains` too, because "could not decide" is never "absent";
- semantics are leftmost-first like `re`, checked by comparing the two on seeded random patterns and texts, which
  found one difference (`\B` on the empty string) that was fixed;
- the redactor uses the same engine on at most 32,768 characters per string.

The built-in secret shapes keep using `re`: they are fixed, anchored, bounded and tested for linear time on hostile
megabytes.

## Consequences

- No pattern, however hostile, hangs a run. A hostile pattern still costs up to the budget per call, so a dataset full of
  them is slow, not stuck.
- Work is observable: every call can fill a `Meter` with its step count, which depends only on the pattern and the text.
  Tests assert on steps (`<= step_bound(len(text))`, budget reached when the budget is lowered), never on a clock: a
  first version asserted "finishes in 30 seconds" and failed on a slow shared runner.
- The engine is about 500 lines of pure Python that the project now owns, and it is slower than `re` on ordinary
  patterns (not noticeable at output sizes graders see).
- Users lose back-references and look-around; the messages say so when the dataset loads.
