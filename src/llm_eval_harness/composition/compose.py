"""Wiring: builds the adapters from the configuration. The only place that knows every concrete class."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from llm_eval_harness.application.config import (
    CommandProviderConfig,
    FakeProviderConfig,
    HttpProviderConfig,
    ProviderConfig,
)
from llm_eval_harness.application.ports import JudgeRunner, Provider
from llm_eval_harness.infrastructure.providers.command import CommandProvider
from llm_eval_harness.infrastructure.providers.fake import FakeProvider
from llm_eval_harness.infrastructure.providers.http import HttpProvider
from llm_eval_harness.infrastructure.providers.judge import CommandJudge


def build_provider(config: ProviderConfig, root: Path, environ: Mapping[str, str]) -> Provider:
    if isinstance(config, CommandProviderConfig):
        return CommandProvider(config, root, environ)
    if isinstance(config, HttpProviderConfig):
        return HttpProvider(config, environ)
    assert isinstance(config, FakeProviderConfig)
    return FakeProvider(config)


def build_judge(root: Path, environ: Mapping[str, str]) -> JudgeRunner:
    return CommandJudge(root, environ)
