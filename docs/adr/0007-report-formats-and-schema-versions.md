# 7. Report and baseline formats

Status: accepted

## Context

CI, dashboards and reviewers consume the output, and a baseline lives in git for years.

## Decision

- Three JSON documents with `schema_version` and `kind`: the **report** (everything about a run), the
  **baseline** (what is committed) and the **gate decision**. Readers validate field by field and refuse another
  kind or version with a message naming the field. A change that is not backward compatible bumps the version.
- The baseline holds counts, hashes and aggregates only: no inputs, outputs or timestamps, sorted keys, trailing
  newline. The same run always yields the same bytes, so a diff shows only real movement and cannot leak text.
- Cases are matched by id and carry a content hash; a changed case is not compared with its old self.
- The report is built as a JSON document first and read back through the same validating reader that `compare`
  and `gate` use, so what is written is what is read.

## Consequences

- Reports are larger than baselines (outputs are kept, redacted and cut); baselines stay reviewable.
- A new field is an additive change within version 1; renaming one is a version 2.
