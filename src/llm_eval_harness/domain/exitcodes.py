"""Process exit codes of the command line. The gate's contract with CI."""

from __future__ import annotations

OK = 0
QUALITY_REGRESSION = 1
USAGE = 2
INFRASTRUCTURE = 3
SAFETY = 4

MEANING = {
    OK: "pass",
    QUALITY_REGRESSION: "quality regression (or an inconclusive comparison that is configured to fail)",
    USAGE: "usage or configuration error",
    INFRASTRUCTURE: "provider or infrastructure error: the run could not be completed",
    SAFETY: "safety failure: at least one safety case failed",
}
