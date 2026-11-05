"""The system clock behind the ``Clock`` port."""

from __future__ import annotations

import time
from datetime import UTC, datetime


class SystemClock:
    def now_iso(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
