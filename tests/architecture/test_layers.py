"""The architecture, enforced mechanically over the real source tree and checked against hostile snippets."""

from __future__ import annotations

from pathlib import Path

import pytest

from support.arch import LAYERS, layer_of, violations

SRC = Path(__file__).resolve().parents[2] / "src" / "llm_eval_harness"


def _files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def test_every_layer_has_source_files() -> None:
    assert {layer_of(p.relative_to(SRC).as_posix()) for p in _files()} >= set(LAYERS)


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.relative_to(SRC).as_posix())
def test_source_file_obeys_the_dependency_rules(path: Path) -> None:
    rel = path.relative_to(SRC).as_posix()
    assert violations(rel, path.read_text(encoding="utf-8")) == []


def test_every_source_directory_is_a_package() -> None:
    for directory in {p.parent for p in _files()}:
        assert (directory / "__init__.py").exists(), directory


def test_package_has_no_runtime_dependencies_declared() -> None:
    text = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ndependencies = []\n" in text


# --- the checker itself: each rule must fire on a snippet that breaks it, and stay quiet on a clean one ---

BAD: list[tuple[str, str, str]] = [
    ("domain/x.py", "import os\n", "domain must be pure"),
    ("domain/x.py", "import time\n", "domain must be pure"),
    ("domain/x.py", "import random\n", "domain must be pure"),
    ("domain/x.py", "from datetime import datetime\n", "domain must be pure"),
    ("domain/x.py", "import pathlib\n", "domain must be pure"),
    ("domain/x.py", "import threading\n", "domain must be pure"),
    ("domain/x.py", "import subprocess\n", "may only be imported by"),
    ("domain/x.py", "import socket\n", "may only be imported by"),
    ("domain/x.py", "import requests\n", "not in the standard library"),
    ("domain/x.py", "from llm_eval_harness.application import ports\n", "domain must not import application"),
    (
        "domain/x.py",
        "from llm_eval_harness.infrastructure.process import run\n",
        "domain must not import infrastructure",
    ),
    ("domain/x.py", "from llm_eval_harness import composition\n", "domain must not import composition"),
    ("domain/x.py", "x = open('f')\n", "open() in the domain layer"),
    ("domain/x.py", "print(1)\n", "print() in the domain layer"),
    ("domain/x.py", "x = os.environ\n", "only the composition layer reads the environment"),
    ("application/x.py", "import os\n", "reaches the outside world only through ports"),
    ("application/x.py", "import time\n", "reaches the outside world only through ports"),
    (
        "application/x.py",
        "from llm_eval_harness.infrastructure import fs\n",
        "application must not import infrastructure",
    ),
    ("application/x.py", "open('f')\n", "open() in the application layer"),
    (
        "infrastructure/x.py",
        "from llm_eval_harness.composition import main\n",
        "infrastructure must not import composition",
    ),
    ("infrastructure/x.py", "import subprocess\n", "may only be imported by infrastructure/process.py"),
    ("infrastructure/x.py", "import http.client\n", "may only be imported by infrastructure/providers/http.py"),
    ("infrastructure/x.py", "import urllib.request\n", "may only be imported by infrastructure/providers/http.py"),
    ("infrastructure/x.py", "import pickle\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "import yaml\n", "not in the standard library"),
    ("infrastructure/x.py", "x = os.environ['A']\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "import os\nos.getenv('A')\n", "only the composition layer reads the environment"),
    (
        "infrastructure/process.py",
        "import subprocess\nsubprocess.run(['a'], shell=True)\n",
        "shell= must be the literal False",
    ),
    ("infrastructure/process.py", "import os\nos.system('a')\n", "os.system runs a shell"),
    ("infrastructure/x.py", "eval('1')\n", "call to eval()"),
    ("infrastructure/x.py", "exec('1')\n", "call to exec()"),
    ("composition/x.py", "exec('1')\n", "call to exec()"),
    ("composition/x.py", "from os import *\n", "wildcard import"),
    ("composition/x.py", "import importlib\n", "can execute or load arbitrary code"),
    ("__main__.py", "x = 1\nexec('1')\n", "call to exec()"),
    # evasions of the first rules
    ("infrastructure/x.py", "import builtins\nbuiltins.eval('1')\n", "builtins"),
    ("composition/x.py", "ev = eval\nev('1')\n", "reference to eval"),
    ("composition/x.py", "x = [exec, compile]\n", "reference to exec"),
    ("infrastructure/x.py", "f = __import__\n", "reference to __import__"),
    ("infrastructure/x.py", "import os\ngetattr(os, 'sys' + 'tem')('x')\n", "getattr with a computed name"),
    ("domain/x.py", "name = 'a'\nsetattr(object(), name, 1)\n", "setattr with a computed name"),
    ("infrastructure/x.py", "import os\ngetattr(os, 'system')\n", "forbidden attribute"),
    ("infrastructure/x.py", "x.__builtins__\n", "forbidden attribute"),
    ("infrastructure/x.py", "().__class__.__bases__[0].__subclasses__()\n", "forbidden attribute"),
    ("infrastructure/x.py", "def f(): pass\nf.__globals__\n", "forbidden attribute"),
    ("infrastructure/x.py", "import os as o\no.environ\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "from os import environ\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "from os import getenv as g\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "import os as o\no.putenv('A', 'b')\n", "only the composition layer reads the environment"),
    (
        "infrastructure/x.py",
        "import os\nos.execv('/bin/sh', ['sh'])\n",
        "may only be used by infrastructure/process.py",
    ),
    ("infrastructure/x.py", "import os\nos.spawnl(0, 'a')\n", "may only be used by infrastructure/process.py"),
    ("infrastructure/x.py", "import os\nos.fork()\n", "may only be used by infrastructure/process.py"),
    ("infrastructure/x.py", "from os import posix_spawn\n", "may only be used by infrastructure/process.py"),
    ("infrastructure/x.py", "import os as o\no.killpg(1, 9)\n", "may only be used by infrastructure/process.py"),
    ("infrastructure/process.py", "from os import system\n", "runs a shell"),
    ("infrastructure/process.py", "import os as o\no.popen('x')\n", "runs a shell"),
    ("infrastructure/x.py", "import posix\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "import pkgutil\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import timeit\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import cProfile\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import pdb\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import gc\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import inspect\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "import ctypes\n", "can execute or load arbitrary code"),
    ("domain/x.py", "import types\n", "domain must be pure"),
    ("infrastructure/x.py", "import os\nfn = os.__dict__['sys' + 'tem']\n", "forbidden attribute"),
    ("infrastructure/x.py", "import sys\nsys.modules['os'].system('x')\n", "forbidden attribute"),
]


@pytest.mark.parametrize(("rel", "source", "expected"), BAD)
def test_checker_flags_breaking_snippet(rel: str, source: str, expected: str) -> None:
    found = violations(rel, source)
    assert any(expected in line for line in found), found


GOOD: list[tuple[str, str]] = [
    (
        "domain/x.py",
        "import re\nimport json\nimport hashlib\nfrom dataclasses import dataclass\nfrom llm_eval_harness.domain import y\n",
    ),
    ("domain/grading/x.py", "from llm_eval_harness.domain.model import Case\n"),
    (
        "application/x.py",
        "import threading\nfrom concurrent.futures import ThreadPoolExecutor\nfrom llm_eval_harness.domain import y\n",
    ),
    ("infrastructure/process.py", "import subprocess\nsubprocess.run(['a'], shell=False)\n"),
    ("infrastructure/providers/http.py", "import http.client\nimport socket\n"),
    ("infrastructure/x.py", "import os\nimport time\nimport random\nfrom llm_eval_harness.application import ports\n"),
    ("composition/x.py", "import os\nx = os.environ.get('A')\nfrom llm_eval_harness.infrastructure import fs\n"),
    ("__main__.py", "from llm_eval_harness.composition.main import run\n"),
    (
        "infrastructure/process.py",
        "import os\nimport signal\nos.killpg(1, signal.SIGKILL)\nw = getattr(os, 'waitid', None)\n",
    ),
    ("infrastructure/process.py", "import os as o\no.waitid(0, 1, 2)\n"),
    ("composition/x.py", "import os as o\nfrom os import environ\no.environ.get('A')\n"),
    ("infrastructure/x.py", "import os\nos.path.join('a', 'b')\nos.replace('a', 'b')\nos.fdopen(1)\n"),
    ("domain/x.py", "x = getattr(obj, 'literal_name', None)\n"),
]


@pytest.mark.parametrize(("rel", "source"), GOOD)
def test_checker_accepts_clean_snippet(rel: str, source: str) -> None:
    assert violations(rel, source) == []


def test_relative_imports_are_resolved_against_the_layer() -> None:
    assert violations("domain/x.py", "from ..application import ports\n")
    assert violations("domain/grading/x.py", "from ...infrastructure import fs\n")
    assert violations("domain/grading/x.py", "from .. import model\n") == []
