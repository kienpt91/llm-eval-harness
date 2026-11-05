# 9. Talking to a child process without threads or blocking calls

Status: accepted (replaces the reader-thread design of the first version)

## Context

The first version read a child's stdout and stderr in threads (buffered readers) and wrote its stdin from another. A
review showed a way to hang the harness forever: a helper started with `setsid` leaves the process group, keeps the
inherited pipe open, and the reader thread stays blocked in a read that holds the buffer lock; the final `close()` of
the same reader then waits for that lock with no timeout. The timeout the user set did not apply to it.

## Decision

`infrastructure/process.py` is single-threaded and never blocks on a pipe:

- the pipes are non-blocking; `selectors` waits for readiness with a timeout that never exceeds the time left to the
  deadline; `os.read` and `os.write` move at most 64 KiB at a time;
- stdout is capped while it is read (the program is killed at the cap); stderr beyond its cap is drained and discarded
  so the program cannot block on a full pipe; stdin is written as the program accepts it and closed at the end or on
  `BrokenPipeError`;
- when the program exits, what is already in the pipes is drained until the pipe is empty, not until end of file, so a
  helper that holds the pipe open does not delay the harness;
- the group is killed before the child is reaped (`os.waitid` with `WNOWAIT` to notice the exit without reaping it),
  so the pid cannot be recycled in between; without `waitid` (macOS) the old order applies, with a documented window;
- every descriptor is closed exactly once, through its stream object (a first attempt that also called `os.close`
  closed a descriptor number that another thread had been given in between, found by the concurrency test);
- reaping waits at most five seconds.

## Consequences

- The harness cannot hang on a pipe, whatever the program or its helpers do.
- A helper that leaves the session is **not killed**: the harness cannot reach it and says so (SECURITY.md, wiring.md).
  Containers are the control for that.
- One thread per run in the runner's pool, none inside the adapter: the "unable to start a thread" failure of the
  first version cannot happen mid-run.
