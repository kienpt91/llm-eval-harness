#!/usr/bin/env python3
"""Run the worked example end to end and print a transcript (the text of docs/demo-output.txt).

    scripts/py.sh 'python scripts/demo.py' > docs/demo-output.txt
    PY_RW=1 scripts/py.sh 'python scripts/demo.py --update-baseline' > docs/demo-output.txt

The example is examples/support-agent: a small retrieval-augmented support assistant (pipeline.py) with two
versions. The demo records a baseline with v1, gates v1 again (passes), then gates v2, a "faster resolutions"
change that improves quality and silently lets a customer review trigger a refund: the gate exits 4.
Nothing here is mocked: the commands are the real command line and the pipeline is a real child process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "support-agent"


def run(workdir: Path, *args: str) -> int:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    print("$ llm-eval " + " ".join(args))
    done = subprocess.run(
        [sys.executable, "-m", "llm_eval_harness", *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(done.stdout)
    sys.stdout.write(done.stderr)
    print(f"(exit code {done.returncode})")
    print()
    return done.returncode


def main() -> int:
    update = "--update-baseline" in sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "support-agent"
        shutil.copytree(EXAMPLE, work, ignore=shutil.ignore_patterns("out", "out-*", "__pycache__"))
        print("# llm-eval worked example: examples/support-agent")
        print(f"# python {sys.version.split()[0]}; times are real and vary from run to run")
        print()
        print("# 1. the dataset: 12 quality cases and 9 safety cases")
        run(work, "validate")
        print("# 2. run pipeline v1 and record its baseline (this file is committed to git)")
        run(work, "run", "--write-baseline", "baseline.json", "--force")
        print("# 3. the gate on the unchanged pipeline: passes")
        run(work, "gate")
        print("# 4. the change: pipeline v2 ('faster resolutions'). Quality goes up. The gate looks at everything")
        run(work, "gate", "--config", "llm-eval.v2.json", "--markdown", "pr-comment.md")
        print("# 5. the Markdown that CI posts on the pull request")
        print((work / "pr-comment.md").read_text())
        if update:
            shutil.copyfile(work / "baseline.json", EXAMPLE / "baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
