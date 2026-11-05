"""The shapes of case ids and tags, shared by datasets, reports and baselines (ASCII only: no look-alike ids)."""

from __future__ import annotations

import re

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
