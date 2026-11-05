"""Programs the command adapter is run against: well-behaved and hostile. Each is a small Python script."""

from __future__ import annotations

import sys
from pathlib import Path

ECHO = r"""
import json, sys
request = json.load(sys.stdin)
case = request["case"]
print(json.dumps({"text": "echo: " + str(case["input"]), "usage": {"model": "m", "input_tokens": 3, "output_tokens": 4}}))
"""

ENV_DUMP = r"""
import json, os, sys
json.load(sys.stdin)
print(json.dumps({"text": json.dumps(dict(os.environ), sort_keys=True)}))
"""

FD_DUMP = r"""
import json, os, sys
json.load(sys.stdin)
open_fds = sorted(int(n) for n in os.listdir("/dev/fd") if n.isdigit())
print(json.dumps({"text": json.dumps(open_fds)}))
"""

ARGV_DUMP = r"""
import json, sys
json.load(sys.stdin)
print(json.dumps({"text": json.dumps(sys.argv[1:])}))
"""

CWD_DUMP = r"""
import json, os, sys
json.load(sys.stdin)
print(json.dumps({"text": os.getcwd()}))
"""

HUGE_STDOUT = r"""
import sys
chunk = b"x" * 65536
while True:
    sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()
"""

HUGE_STDERR = r"""
import json, sys
sys.stderr.write("E" * 5_000_000)
sys.stderr.flush()
sys.exit(3)
"""

HANG = r"""
import time
time.sleep(600)
"""

NEVER_READS_STDIN = r"""
import time
time.sleep(600)
"""

INVALID_JSON = r"""
print("{not json")
"""

NOT_AN_OBJECT = r"""
print("[1, 2, 3]")
"""

EMPTY = r"""
"""

BINARY = r"""
import sys
sys.stdout.buffer.write(bytes(range(256)) * 10)
"""

EXIT_NONZERO = r"""
import sys
sys.stderr.write("boom: something went wrong\x1b[31m")
sys.exit(7)
"""

SELF_KILL = r"""
import os, signal
os.kill(os.getpid(), signal.SIGKILL)
"""

DEEP_JSON = r"""
print("[" * 100000 + "]" * 100000)
"""

DUPLICATE_KEYS = r"""
print('{"text": "a", "text": "b"}')
"""

NAN = r"""
print('{"text": "a", "usage": {"input_tokens": NaN}}')
"""

PROTO_KEYS = r"""
import json, sys
json.load(sys.stdin)
print('{"text": "ok", "structured": {"__proto__": {"admin": true}, "constructor": {"prototype": 1}}}')
"""

UNKNOWN_KEY = r"""
print('{"text": "a", "tool_call": []}')
"""

CONTROL_TEXT = r"""
import json
print(json.dumps({"text": "\x1b[31mred\x1b]0;pwned\x07 \u202e evil \x00 end"}))
"""

# starts a long-lived grandchild, records its pid, and exits at once
ORPHAN_MAKER = r"""
import json, subprocess, sys
pid_file = sys.argv[1]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
open(pid_file, "w").write(str(child.pid))
json.load(sys.stdin)
print(json.dumps({"text": "done"}))
"""

# the same, but the parent hangs: the timeout must take the grandchild with it
HANG_WITH_CHILD = r"""
import subprocess, sys, time
pid_file = sys.argv[1]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
open(pid_file, "w").write(str(child.pid))
time.sleep(600)
"""

# starts a helper in its OWN session (it leaves the process group) that inherits stdout and stderr, then answers and exits
ESCAPED_HOLDER = r"""
import json, subprocess, sys
pid_file = sys.argv[1]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
open(pid_file, "w").write(str(child.pid))
print(json.dumps({"text": "answered while an escaped helper holds the pipes"}))
"""

# the same, but the program itself hangs
ESCAPED_HOLDER_HANGS = r"""
import subprocess, sys, time
pid_file = sys.argv[1]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
open(pid_file, "w").write(str(child.pid))
time.sleep(600)
"""

CLOSES_STDIN_AND_ANSWERS = r"""
import json, os, sys
os.close(0)
print(json.dumps({"text": "answered without reading"}))
"""

SLOW_THEN_OK = r"""
import json, sys, time
json.load(sys.stdin)
time.sleep(0.3)
print(json.dumps({"text": "slow but fine"}))
"""

JUDGE_PASS = r"""
import json, sys
request = json.load(sys.stdin)
print(json.dumps({"verdict": "pass", "confidence": 0.8, "reason": "rubric: " + request["rubric"]}))
"""


def write_program(directory: Path, name: str, source: str) -> list[str]:
    """Write ``source`` as a script and return the argv that runs it with the current interpreter."""
    path = directory / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return [sys.executable, str(path)]


def is_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers signal 0: check its state
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True
