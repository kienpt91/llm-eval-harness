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
UNSAFE_MODULES = frozenset(
    {
        "pickle", "marshal", "shelve", "ctypes", "importlib", "code", "codeop", "runpy", "imp", "builtins", "posix", "nt",
        "pkgutil", "timeit", "cProfile", "profile", "pdb", "trace", "gc", "inspect",
    }
)  # fmt: skip

PROCESS_FILE = "infrastructure/process.py"
NETWORK_FILE = "infrastructure/providers/http.py"

FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})
# attributes that reach the interpreter's internals or other modules behind the import rules' back
FORBIDDEN_ATTRS = frozenset(
    {
        "__builtins__", "__globals__", "__code__", "__subclasses__", "__mro__", "__bases__", "__loader__", "__import__",
        "__dict__", "f_globals", "f_locals", "f_builtins", "gi_frame", "cr_frame", "modules",
    }
)  # fmt: skip
REFLECTION = frozenset({"getattr", "setattr", "delattr"})
ENV_NAMES = frozenset({"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"})
SHELL_NAMES = frozenset({"system", "popen", "startfile"})
PROCESS_ONLY_NAMES = frozenset(
    {
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp",
        "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "fork", "forkpty", "posix_spawn", "posix_spawnp", "kill",
        "killpg", "waitid", "waitpid", "wait", "wait3", "wait4", "setsid", "setpgid",
    }
)  # fmt: skip
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
    found.extend(_syntax_violations(relpath, layer, tree))
    return found


def _is_literal_str(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _syntax_violations(relpath: str, layer: str, tree: ast.AST) -> list[str]:
    found: list[str] = []
    os_names = {"os"}  # `import os`, `import os as o`
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(alias.asname or alias.name for alias in node.names if alias.name == "os")

    def process_only(name: str, where: str) -> None:
        if name in SHELL_NAMES:
            found.append(f"{where}: os.{name} runs a shell")
        elif name in PROCESS_ONLY_NAMES and relpath != PROCESS_FILE:
            found.append(f"{where}: os.{name} may only be used by {PROCESS_FILE}")
        elif name in ENV_NAMES and layer != "composition":
            found.append(f"{where}: only the composition layer reads the environment (os.{name})")

    for node in ast.walk(tree):
        where = f"{relpath}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                found.append(f"{where}: wildcard import")
            if node.level == 0 and node.module == "os":
                for alias in node.names:
                    process_only(alias.name, where)
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS and isinstance(node.ctx, ast.Load):
                is_call = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.Call) and parent.func is node:
                        is_call = True
                        break
                found.append(
                    f"{where}: call to {node.id}() (model output and datasets are never executed)"
                    if is_call
                    else f"{where}: reference to {node.id} (an alias would hide the call)"
                )
            if node.id in IO_BUILTINS and layer in ("domain", "application") and isinstance(node.ctx, ast.Load):
                found.append(f"{where}: {node.id}() in the {layer} layer")
            if node.id in ("__builtins__", "globals", "vars", "locals") and layer in ("domain", "application"):
                found.append(f"{where}: reflection ({node.id}) in the {layer} layer")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                found.append(f"{where}: forbidden attribute .{node.attr}: it reaches the interpreter's internals")
            if isinstance(node.value, ast.Name) and node.value.id in os_names:
                process_only(node.attr, where)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "shell" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    found.append(f"{where}: shell= must be the literal False")
            if isinstance(node.func, ast.Name) and node.func.id in REFLECTION:
                name_arg = node.args[1] if len(node.args) > 1 else None
                if name_arg is None or not _is_literal_str(name_arg):
                    found.append(f"{where}: {node.func.id} with a computed name")
                else:
                    assert isinstance(name_arg, ast.Constant)
                    literal = str(name_arg.value)
                    banned = FORBIDDEN_ATTRS | SHELL_NAMES | ENV_NAMES
                    if relpath != PROCESS_FILE:
                        banned |= PROCESS_ONLY_NAMES
                    if literal in banned:
                        found.append(f"{where}: forbidden attribute {literal!r} through {node.func.id}")
    return found
