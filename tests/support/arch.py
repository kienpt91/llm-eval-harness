"""The dependency rules of the architecture, as pure functions over source text (AST based).

Layers, from the inside out: domain <- application <- infrastructure <- composition. A layer imports only
from itself and from the layers inside it. Beyond that the rules are about what a layer may do:

* domain is pure: standard library modules that compute (no files, network, processes, clock, randomness,
  threads, environment) and nothing outside the package.
* application orchestrates over ports: the domain's modules plus threading primitives for the runner; no IO,
  no clock, no randomness (they come in through ports).
* infrastructure implements the ports with the standard library only (zero runtime dependencies).
  Process creation lives in one module and raw network access in one module.
* composition wires everything; it is the only place that reads the environment.
* nowhere: dynamic code execution, shell invocation, pickle-like loaders, wildcard imports.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

PACKAGE = "llm_eval_harness"
LAYERS = ("domain", "application", "infrastructure", "composition")

ALLOWED_INTERNAL: dict[str, tuple[str, ...]] = {
    "domain": ("domain",),
    "application": ("domain", "application"),
    "infrastructure": ("domain", "application", "infrastructure"),
    "composition": ("domain", "application", "infrastructure", "composition"),
    "entry": ("composition",),
}

# modules that only compute: usable in the pure layers
PURE_STDLIB = frozenset(
    {
        "__future__",
        "abc",
        "collections",
        "collections.abc",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "ipaddress",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "types",
        "typing",
        "unicodedata",
    }
)
# what the application layer may add: thread primitives for the concurrency-limited runner
APPLICATION_EXTRA = frozenset({"concurrent.futures", "threading"})

PROCESS_MODULES = frozenset({"subprocess", "multiprocessing", "pty", "selectors", "select"})
NETWORK_MODULES = frozenset(
    {
        "socket",
        "ssl",
        "http",
        "http.client",
        "http.server",
        "urllib.request",
        "ftplib",
        "smtplib",
        "asyncio",
        "xmlrpc",
    }
)
UNSAFE_MODULES = frozenset({"pickle", "marshal", "shelve", "ctypes", "importlib", "code", "codeop", "runpy", "imp"})

PROCESS_FILE = "infrastructure/process.py"
NETWORK_FILE = "infrastructure/providers/http.py"

FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
# builtins that touch the outside world: never in the pure layers
IO_BUILTINS = frozenset({"open", "print", "input", "breakpoint"})


@dataclass(frozen=True)
class Import:
    module: str  # absolute module name, or the resolved package-relative one
    internal: bool
    line: int


def layer_of(relpath: str) -> str:
    """The layer of a file given its path relative to src/llm_eval_harness."""
    first = PurePosixPath(relpath).parts[0]
    return first if first in LAYERS else "entry"


def _package_of(relpath: str) -> list[str]:
    parts = list(PurePosixPath(relpath).parts[:-1])
    return [PACKAGE, *parts]


def imports_of(tree: ast.AST, relpath: str) -> list[Import]:
    found: list[Import] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(Import(alias.name, alias.name.split(".")[0] == PACKAGE, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _package_of(relpath)
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
                found.append(Import(module, True, node.lineno))
            else:
                module = node.module or ""
                found.append(Import(module, module.split(".")[0] == PACKAGE, node.lineno))
                # `from pkg import sub` may import a submodule: check it as well
                if module.split(".")[0] == PACKAGE:
                    for alias in node.names:
                        found.append(Import(f"{module}.{alias.name}", True, node.lineno))
    return found


def _internal_layer(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == PACKAGE and parts[1] in LAYERS:
        return parts[1]
    if len(parts) >= 1 and parts[0] == PACKAGE:
        return "entry"
    return None


def _stdlib_ok(module: str, allowed: frozenset[str]) -> bool:
    return any(module == a or module.startswith(a + ".") for a in allowed)


def violations(relpath: str, source: str) -> list[str]:
    layer = layer_of(relpath)
    tree = ast.parse(source)
    found: list[str] = []
    for imp in imports_of(tree, relpath):
        top = imp.module.split(".")[0]
        where = f"{relpath}:{imp.line}"
        if imp.internal:
            target = _internal_layer(imp.module)
            if target is not None and target != "entry" and target not in ALLOWED_INTERNAL[layer]:
                found.append(f"{where}: {layer} must not import {target} ({imp.module})")
            continue
        if top not in sys.stdlib_module_names:
            found.append(f"{where}: {imp.module} is not in the standard library (zero runtime dependencies)")
            continue
        if layer == "domain" and not _stdlib_ok(imp.module, PURE_STDLIB):
            found.append(f"{where}: domain must be pure; {imp.module} is not on the allow-list")
        if layer == "application" and not _stdlib_ok(imp.module, PURE_STDLIB | APPLICATION_EXTRA):
            found.append(
                f"{where}: application reaches the outside world only through ports; {imp.module} is not on the allow-list"
            )
        if _stdlib_ok(imp.module, PROCESS_MODULES) and relpath != PROCESS_FILE:
            found.append(f"{where}: {imp.module} may only be imported by {PROCESS_FILE}")
        if _stdlib_ok(imp.module, NETWORK_MODULES) and relpath != NETWORK_FILE:
            found.append(f"{where}: {imp.module} may only be imported by {NETWORK_FILE}")
        if _stdlib_ok(imp.module, UNSAFE_MODULES):
            found.append(f"{where}: {imp.module} can execute or load arbitrary code")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            found.append(f"{relpath}:{node.lineno}: wildcard import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                found.append(
                    f"{relpath}:{node.lineno}: call to {node.func.id}() (model output and datasets are never executed)"
                )
            if node.func.id in IO_BUILTINS and layer in ("domain", "application"):
                found.append(f"{relpath}:{node.lineno}: {node.func.id}() in the {layer} layer")
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    found.append(f"{relpath}:{node.lineno}: shell= must be the literal False")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in ("system", "popen")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                found.append(f"{relpath}:{node.lineno}: os.{node.func.attr} runs a shell")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in ("environ", "getenv", "environb")
        ):
            if layer != "composition":
                found.append(f"{relpath}:{node.lineno}: only the composition layer reads the environment")
        if (
            isinstance(node, ast.Name)
            and node.id in ("__builtins__", "globals", "vars")
            and layer in ("domain", "application")
        ):
            found.append(f"{relpath}:{node.lineno}: reflection ({node.id}) in the {layer} layer")
    return found
