# 3. Safety cases are never averaged away

Status: accepted

## Context

An accuracy number can rise while an agent starts doing something it must never do: a prompt change that makes
answers better can also make a poisoned document obeyable. An average, a threshold on pass rate, or a baseline
comparison can hide one such case among many passes. Agents that call tools and write SQL fail in exactly this way.

## Decision

- Cases have a `category`; `safety` is a first-class value next to `quality`, with its own starter dataset
  (`init-safety`) covering prompt injection, data exfiltration and unauthorised tool calls.
- Any failed run of any safety case makes the gate exit **4**, the strongest code. No threshold, no pass-rate option,
  no baseline and no quality improvement changes that. Under `--repeat`, one failed repeat is a failure.
- A safety case that was **not run** is not a passing one: a `gate` whose selection (`--category`, `--tag`, `--case`,
  `--sample`) leaves out a safety case of the dataset, or whose baseline has a safety case that is gone or no longer
  a safety case, exits 4 too. The one escape hatch is explicit and only for what did not run:
  `--allow-partial-safety`. It never accepts a failure.
- A safety case cannot lower its threshold or carry a grader that could never fail it; the dataset is refused.
- Safety cases get a hard `must` secret-leak grader automatically. `forbidden`, `forbidden_patterns` and
  `forbidden_tools` expectations are `must` by default: a failed `must` fails the run whatever the score.
- A safety case that could not be run (provider error) is exit 3, never a pass.

## Consequences

- The gate is deliberately blunt for safety and statistical for quality ([ADR 4](0004-statistical-comparison.md)).
- A review found the first version of this ADR overstated the guarantee: a gate configured without the safety suite
  passed silently. Coverage is now part of the verdict.
- Safety graders are heuristics and say so ([docs/safety-suite.md](../safety-suite.md)): a pass is evidence, not a
  proof.
