"""The documentation and the repository's own hygiene, checked against the code and the files."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from llm_eval_harness import __version__
from llm_eval_harness.application.config import FakeProviderConfig, parse_config
from llm_eval_harness.domain import exitcodes
from llm_eval_harness.domain.dataset import parse_dataset
from llm_eval_harness.domain.grading.registry import GRADERS
from llm_eval_harness.domain.jsonio import parse_json
from llm_eval_harness.domain.pricing import parse_price_table
from llm_eval_harness.domain.safety_suite import starter_cases
from llm_eval_harness.infrastructure.providers.fake import FakeProvider

ROOT = Path(__file__).resolve().parents[2]


def tracked_files() -> list[Path]:
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return [ROOT / line for line in out.stdout.splitlines() if (ROOT / line).is_file()]
    skip = {".git", "out", "__pycache__", ".ruff_cache", ".mypy_cache"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


def text_files() -> list[Path]:
    suffixes = {".py", ".md", ".json", ".jsonl", ".toml", ".yml", ".yaml", ".sh", ".txt", ".in", ".mjs", ".cfg"}
    return [p for p in tracked_files() if p.suffix in suffixes or p.name in {"LICENSE", ".gitignore"}]


def blocks(markdown: Path, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", markdown.read_text(), flags=re.S)


# --- docs against code ----------------------------------------------------------------------------------------


def test_every_grader_is_documented_and_every_documented_grader_exists() -> None:
    headings = set(re.findall(r"^### `([a-z_0-9]+)`$", (ROOT / "docs/graders.md").read_text(), flags=re.M))
    assert headings == set(GRADERS)
    table = set(re.findall(r"^\| \[`([a-z_0-9]+)`\]\(#", (ROOT / "docs/graders.md").read_text(), flags=re.M))
    assert table == set(GRADERS)


def test_every_grader_has_a_failure_modes_paragraph_in_the_docs() -> None:
    text = (ROOT / "docs/graders.md").read_text()
    sections = re.split(r"^### `", text, flags=re.M)[1:]
    assert len(sections) == len(GRADERS)
    for section in sections:
        assert "ailure modes" in section, section.split("`")[0]


def test_the_exit_codes_in_the_docs_match_the_code() -> None:
    gate = (ROOT / "docs/gate.md").read_text()
    rows = dict(re.findall(r"^\| (\d) \| ([a-z ]+(?: error)?) \|", gate, flags=re.M))
    assert {int(k) for k in rows} == set(exitcodes.MEANING)
    assert rows["0"] == "pass" and rows["4"] == "safety failure" and rows["2"] == "usage error"
    assert rows["1"] == "quality regression" and rows["3"] == "provider or infrastructure error"
    readme = (ROOT / "README.md").read_text()
    for code, name in rows.items():
        assert f"| {code} | {name}" in readme, (code, name)


def test_the_safety_suite_doc_lists_exactly_the_starter_cases() -> None:
    documented = set(re.findall(r"^\| `(safety-[a-z-]+)` \|", (ROOT / "docs/safety-suite.md").read_text(), flags=re.M))
    assert documented == {c["id"] for c in starter_cases()}


def test_the_dataset_example_validates() -> None:
    [example] = [b for b in blocks(ROOT / "docs/dataset-format.md", "json") if b.lstrip().startswith('{"id"')]
    dataset = parse_dataset([("line 1", parse_json(example))], "docs")
    assert dataset.cases[0].id == "refund-window" and dataset.cases[0].expected.tool_calls is not None


def test_the_configuration_examples_are_valid() -> None:
    doc = ROOT / "docs/configuration.md"
    config_block = next(b for b in blocks(doc, "json") if b.lstrip().startswith('{\n  "dataset"'))
    config = parse_config(parse_json(config_block), "docs")
    assert config.gate.max_cost_increase_pct == 20 and config.redact_patterns == ("ORD-\\d{4,}",)
    fake_block = next(b for b in blocks(doc, "json") if '"type": "fake"' in b)
    fake = parse_json(fake_block)
    assert isinstance(fake, dict)
    FakeProvider(FakeProviderConfig({k: v for k, v in fake.items() if k != "type"}))
    price_block = next(b for b in blocks(doc, "json") if "input_per_mtok" in b)
    assert "my-model" in parse_price_table(parse_json(price_block), "docs").models


def test_the_wiring_doc_embeds_the_example_files_verbatim() -> None:
    doc = (ROOT / "docs/wiring.md").read_text()
    for name in ("pipeline.py", "pipeline.mjs"):
        assert (ROOT / "examples/wiring" / name).read_text().strip() in doc, name


def test_the_example_configs_and_dataset_are_valid() -> None:
    for name in ("llm-eval.json", "llm-eval.v2.json"):
        config = parse_config(parse_json((ROOT / "examples/support-agent" / name).read_text()), name)
        assert config.provider is not None and config.gate.deterministic
    lines = (ROOT / "examples/support-agent/datasets/support.jsonl").read_text().splitlines()
    dataset = parse_dataset([(f"line {i + 1}", parse_json(line)) for i, line in enumerate(lines)], "support.jsonl")
    assert sum(c.is_safety for c in dataset.cases) == 9 and len(dataset.cases) == 21


def test_the_committed_baseline_belongs_to_the_committed_dataset() -> None:
    baseline = json.loads((ROOT / "examples/support-agent/baseline.json").read_text())
    lines = (ROOT / "examples/support-agent/datasets/support.jsonl").read_text().splitlines()
    dataset = parse_dataset([(f"line {i + 1}", parse_json(line)) for i, line in enumerate(lines)], "support.jsonl")
    assert baseline["dataset"]["hash"] == dataset.hash
    assert {c.id: c.hash for c in dataset.cases} == {k: v["case_hash"] for k, v in baseline["cases"].items()}


def test_the_security_doc_names_tests_that_exist() -> None:
    text = (ROOT / "SECURITY.md").read_text()
    references = re.findall(r"`(test_[a-z_]+\.py)(?:::(test_[a-z_0-9]+))?`", text)
    assert references
    available = {p.name: p.read_text() for p in (ROOT / "tests").rglob("test_*.py")}
    for file, function in references:
        assert file in available, file
        if function:
            assert re.search(rf"^def {function}\(", available[file], flags=re.M), (file, function)


def test_the_changelog_and_manifest_agree() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.search(rf"^## {re.escape(__version__)}$", changelog, flags=re.M)
    assert f'version = "{__version__}"' in (ROOT / "pyproject.toml").read_text()


def test_the_readme_demo_is_real_captured_output() -> None:
    readme = (ROOT / "README.md").read_text()
    captured = (ROOT / "docs/demo-output.txt").read_text()
    shown = re.findall(r"<!-- demo:start -->\n```text\n(.*?)```\n<!-- demo:end -->", readme, flags=re.S)
    assert len(shown) >= 2
    for block in shown:
        assert block in captured, block[:80]
    assert "(exit code 4)" in captured and "gate      FAIL (exit 4)" in captured


# --- links ---------------------------------------------------------------------------------------------------


def slug(heading: str) -> str:
    text = re.sub(r"[`*]", "", heading.strip().lower())
    return re.sub(r"[^a-z0-9 _-]", "", text).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    return {slug(h) for h in re.findall(r"^#{1,6} (.+)$", path.read_text(), flags=re.M)}


@pytest.mark.parametrize(
    "doc", sorted(p for p in tracked_files() if p.suffix == ".md"), ids=lambda p: str(p.relative_to(ROOT))
)
def test_relative_links_resolve(doc: Path) -> None:
    for target in re.findall(r"\]\(([^)\s]+)\)", doc.read_text()):
        if re.match(r"[a-z]+:", target) or (target.startswith("#") and target[1:] in anchors(doc)):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = (doc.parent / path_part).resolve() if path_part else doc
        assert resolved.exists(), f"{doc.name}: {target} does not exist"
        if anchor and resolved.suffix == ".md":
            assert anchor in anchors(resolved), f"{doc.name}: no heading #{anchor} in {resolved.name}"


# --- repository hygiene --------------------------------------------------------------------------------------

# assembled from pieces so that this file does not contain them
VENDOR_WORDS = [
    "open" + "ai", "anth" + "ropic", "cla" + "ude", "chat" + "gpt", "gem" + "ini", "co" + "pilot", "co" + "dex",
    "mis" + "tral", "co" + "here", "bed" + "rock", "hugging" + "face", "gp" + "t-", "lla" + "ma", "dee" + "pseek",
]  # fmt: skip


def test_no_vendor_or_assistant_names_anywhere() -> None:
    pattern = re.compile("|".join(re.escape(w) for w in VENDOR_WORDS), re.IGNORECASE)
    abbreviation = re.compile(r"\bAI\b")
    found = []
    for path in text_files():
        if path.name == "check-commit-msg.sh":
            continue  # its list of words is the guard itself, assembled from pieces
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if pattern.search(line) or abbreviation.search(line):
                found.append(f"{path.relative_to(ROOT)}:{number}")
    assert found == []


def test_the_commit_history_is_clean_and_by_one_author() -> None:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("no git history in this environment (CI and the release script check it with git)")
    log = subprocess.run(
        ["git", "log", "--format=%an <%ae>%n%cn <%ce>%n%B"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if log.returncode != 0 or not log.stdout.strip():
        pytest.skip("no git history in this environment")
    pattern = re.compile("|".join(re.escape(w) for w in VENDOR_WORDS) + r"|co-authored-by|signed-off-by", re.IGNORECASE)
    assert not pattern.search(log.stdout) and not re.search(r"\bAI\b", log.stdout)
    authors = {line for line in log.stdout.splitlines() if line.endswith(">") and line.count("<") == 1}
    assert authors == {"kienpt <kienphan.dev@gmail.com>"}


def test_no_unfinished_markers_are_left() -> None:
    marker = re.compile(r"\b(" + "TO" + "DO|FIX" + "ME|XX" + "X)\b")
    assert [str(p.relative_to(ROOT)) for p in text_files() if marker.search(p.read_text(errors="replace"))] == []


def test_the_licence_is_mit_with_the_right_holder() -> None:
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License") and "Copyright (c) 2026 Kien Phan" in text


def test_workflows_pin_actions_by_sha_and_default_to_least_privilege() -> None:
    for workflow in [*(ROOT / ".github/workflows").glob("*.yml"), ROOT / "examples/support-agent/ci.yml"]:
        text = workflow.read_text()
        for use in re.findall(r"^\s+(?:- )?uses: (\S+)", text, flags=re.M):
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", use), f"{workflow.name}: {use} is not pinned by commit SHA"
        assert re.search(r"^permissions:", text, flags=re.M), f"{workflow.name} has no top-level permissions"
        assert "pull_request_target" not in text
        assert "persist-credentials: false" in text
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert "twine" not in release and "pypi" not in release.lower().replace("package registry (pypi included)", "")
    assert "scripts/check-release-tag.sh" in release and "attest-build-provenance" in release


def test_dependabot_covers_pip_and_actions_with_a_cooldown() -> None:
    text = (ROOT / ".github/dependabot.yml").read_text()
    assert "package-ecosystem: pip" in text and "package-ecosystem: github-actions" in text
    assert text.count("default-days: 7") == 2


def test_dev_requirements_are_exactly_pinned_with_hashes() -> None:
    entries = re.split(r"\n(?=\S)", (ROOT / "requirements-dev.txt").read_text().strip())
    pins = [e for e in entries if not e.startswith("#")]
    assert len(pins) >= 10
    for entry in pins:
        assert re.match(r"[A-Za-z0-9._-]+==[0-9][A-Za-z0-9.]*", entry), entry
        assert "--hash=sha256:" in entry and ">=" not in entry


def test_the_container_image_is_pinned_by_digest() -> None:
    text = (ROOT / "scripts/py.sh").read_text()
    digests = re.findall(r"default_image=python@sha256:[0-9a-f]{64}", text)
    assert len(digests) == 2
    assert "--cap-drop ALL" in text and "--network" in text and "--cpus 4" in text and "--memory 6g" in text
    assert "no-new-privileges" in text and "--read-only" in text and '-v "$here:/work:$mode"' in text


def test_secret_scan_rules_and_hooks_are_in_place() -> None:
    assert "secret-scan.sh --staged" in (ROOT / "lefthook.yml").read_text()
    assert "check-commit-msg.sh" in (ROOT / "lefthook.yml").read_text()
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        return
    scan = subprocess.run(
        ["sh", "scripts/secret-scan.sh", "--tree"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert scan.returncode == 0, scan.stderr
