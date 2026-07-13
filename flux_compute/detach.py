"""Detach-and-poll: run a remote job so it survives the launching SSH session
dropping (laptop sleep), and follow it with a reconnect-tolerant poll loop.

The problem this solves is a real incident. A foreground ``ssh host 'job'`` binds
the remote job's lifetime to that one TCP session. When the operator's laptop
sleeps (a closed-lid commute), the session dies and sshd HUPs the remote job a
few minutes later, killing an in-flight run and aborting the shard. The fix has
two halves, both here as pure (network-free, unit-tested) building blocks that
``provision.py`` wires to real SSH:

1. **Launch the job DETACHED** (``launcher_script``). ``setsid`` puts the job in a
   new session with every file descriptor redirected to files, so sshd can close
   the channel the instant the launcher returns and a later disconnect cannot HUP
   the job. A ``timeout`` wrapper is the remote runaway backstop (enforced on the
   VM, independent of the laptop), and the integer return code is written to
   ``~/job.rc`` only when the job finishes -- its appearance is the done signal.

2. **Follow it with a POLL loop** (``poll_until_done``). Each poll is a fresh,
   short SSH that reads ``~/job.rc`` (finished?) and incrementally tails
   ``~/job.out`` (live log). A failed poll -- the laptop just woke, a network flap
   -- is retried with exponential backoff and is NEVER fatal; the sole abort is
   the local wall-clock deadline. On the rc appearing, the caller pulls the full
   ``~/job.out`` and the artifacts and tears down: exactly the old success path.

``AttachRecord`` persists the handful of facts needed to re-attach to a
still-running detached job after the local orchestrator fully restarts (a hard
kill, not just a sleep) -- the recovery the incident actually needed.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass

# ASCII record separator (0x1e) delimits a poll's output chunk from its status
# trailer. Job logs are text and never contain it, so the split is unambiguous.
RS = "\x1e"

# Home-relative remote filenames the launcher writes and the poller reads.
REMOTE_OUT = "job.out"        # combined stdout+stderr, tailed by the poller
REMOTE_RC = "job.rc"          # integer return code, written only on completion
REMOTE_PID = "job.pid"        # PID of the detached job supervisor
REMOTE_LAUNCHER = "job_launch.sh"


def launcher_script(remote_script, cap_seconds, *, kill_after_s=30):
    """Return the bash for the detached launcher (uploaded and run once).

    ``remote_script`` is the home-relative basename of the caller's uploaded job
    script; ``cap_seconds`` is the remote runaway cap ``timeout`` enforces on the
    VM. The job runs under a login shell (``bash -lc``) to match the old
    foreground path, so a baked venv activated from the profile still applies. The
    launcher itself is short and non-blocking: it spawns the ``setsid`` job with
    all descriptors off the SSH channel and returns, so the launching SSH closes
    at once and the job keeps running through any later disconnect.
    """
    remote_script = str(remote_script)
    cap = int(cap_seconds)
    kill_after = int(kill_after_s)
    return f"""#!/usr/bin/env bash
# flux-compute detached job launcher (generated; do not edit).
set -u
cd "$HOME"
chmod +x "$HOME/{remote_script}" 2>/dev/null || true
rm -f {REMOTE_RC} {REMOTE_PID}
: > {REMOTE_OUT}
# New session + every fd off the SSH channel: sshd closes the channel the moment
# this launcher returns, and a dropped connection (laptop sleep) can no longer
# HUP the job. The subshell records its own PID, runs the job under the remote
# runaway cap, and writes the return code only on completion -- its presence is
# the done signal the poller waits for.
setsid --fork bash -c '
  echo "$$" > "$HOME/{REMOTE_PID}"
  timeout --signal=TERM --kill-after={kill_after} {cap} bash -lc "$HOME/{remote_script}"
  echo "$?" > "$HOME/{REMOTE_RC}"
' >> "$HOME/{REMOTE_OUT}" 2>&1 < /dev/null &
disown || true
sleep 0.2
echo "flux-compute: detached job launched (pid $(cat "$HOME/{REMOTE_PID}" 2>/dev/null || echo '?'), cap {cap}s)"
"""


def poll_command(next_byte):
    """Return the bash for one poll: emit new ``job.out`` bytes since ``next_byte``
    (1-based), then a record separator, then ``SIZE=<bytes>;RC=<rc-or-empty>``.

    The size is captured *before* the tail and the tail is bounded to it
    (``head -c``), so a ``job.out`` still being appended cannot make the poller
    over-advance and silently drop bytes: the next poll picks up from the recorded
    size.
    """
    n = int(next_byte)
    return (
        f'sz=$(wc -c < "$HOME/{REMOTE_OUT}" 2>/dev/null || echo 0); '
        f'if [ -f "$HOME/{REMOTE_RC}" ]; then rc=$(cat "$HOME/{REMOTE_RC}" 2>/dev/null); else rc=""; fi; '
        f'tail -c "+{n}" "$HOME/{REMOTE_OUT}" 2>/dev/null | head -c "$(( sz - {n} + 1 ))"; '
        f"printf '{RS}SIZE=%s;RC=%s' \"$sz\" \"$rc\""
    )


@dataclass(frozen=True)
class PollResult:
    """A parsed poll: the new output chunk, the current ``job.out`` size, and the
    return code once the job has finished (else ``None``)."""

    chunk: str
    size: int
    rc: int | None


_TRAILER_RE = re.compile(r"SIZE=(\d+);RC=(-?\d*)\s*$")


def parse_poll_output(stdout):
    """Parse a poll command's stdout into a ``PollResult``, or ``None`` when the
    status trailer is absent or malformed (a truncated/garbled read -- the caller
    treats it as a soft failure and retries)."""
    if stdout is None:
        return None
    idx = stdout.rfind(RS)
    if idx < 0:
        return None
    chunk = stdout[:idx]
    m = _TRAILER_RE.match(stdout[idx + 1:])
    if m is None:
        return None
    size = int(m.group(1))
    rc = int(m.group(2)) if m.group(2) != "" else None
    return PollResult(chunk=chunk, size=size, rc=rc)


class PollAttempt:
    """One poll's outcome from the injected SSH runner: either a connected read
    (``ok=True`` with the remote command's ``stdout``) or a connection failure
    (``ok=False`` with a short ``error``). A connection failure is what the poll
    loop retries; a connected read -- even if the remote poll command itself
    exited nonzero -- still carries the status trailer and is parsed."""

    __slots__ = ("ok", "stdout", "error")

    def __init__(self, ok, stdout=None, error=None):
        self.ok = ok
        self.stdout = stdout
        self.error = error

    @classmethod
    def connected(cls, stdout):
        return cls(True, stdout=stdout)

    @classmethod
    def failed(cls, error):
        return cls(False, error=error)


@dataclass(frozen=True)
class PollOutcome:
    """The result of following a detached job to its end (or to a local abort).

    ``reason`` is ``"done"`` when the remote ``job.rc`` appeared (``rc`` is the
    job's return code, including 124/137 for a remote-cap kill) or ``"deadline"``
    when the local wall-clock deadline aborted the follow before completion
    (``rc`` is ``None``)."""

    rc: int | None
    reason: str
    output_size: int


def poll_until_done(run_poll, *, poll_interval, deadline_s,
                    backoff_base=5.0, backoff_max=60.0,
                    clock=time.monotonic, sleep=time.sleep,
                    on_chunk=None, on_status=None):
    """Poll a detached remote job to completion, tolerant of reconnection.

    ``run_poll(next_byte) -> PollAttempt`` performs one poll (a fresh short SSH in
    production). A failed attempt (connection error: the laptop just woke, a
    network flap) is retried with exponential backoff -- ``backoff_base`` doubling
    per consecutive failure, capped at ``backoff_max`` -- and is NEVER fatal. The
    single abort is the local wall-clock ``deadline_s`` measured from the first
    poll. Returns a ``PollOutcome``.

    ``clock``/``sleep`` are injected for deterministic tests. ``on_chunk(str)``
    receives each new ``job.out`` fragment (for a live local log / stream);
    ``on_status(str)`` receives a short human progress line per poll.
    """
    next_byte = 1
    size_seen = 0
    start = clock()
    consecutive_failures = 0

    def _remaining():
        return deadline_s - (clock() - start)

    while True:
        if _remaining() <= 0:
            return PollOutcome(rc=None, reason="deadline", output_size=size_seen)

        attempt = run_poll(next_byte)

        if not attempt.ok:
            consecutive_failures += 1
            if on_status is not None:
                on_status(f"poll failed ({attempt.error}); retry #{consecutive_failures}")
            backoff = min(backoff_max, backoff_base * (2 ** (consecutive_failures - 1)))
            remaining = _remaining()
            if remaining <= 0:
                return PollOutcome(rc=None, reason="deadline", output_size=size_seen)
            sleep(min(backoff, remaining))
            continue

        parsed = parse_poll_output(attempt.stdout)
        if parsed is None:
            # A garbled or truncated read on an otherwise-live connection: treat
            # as a soft failure (short retry), do not advance the byte cursor.
            consecutive_failures += 1
            if on_status is not None:
                on_status("poll returned unparseable status; retrying")
            remaining = _remaining()
            if remaining <= 0:
                return PollOutcome(rc=None, reason="deadline", output_size=size_seen)
            sleep(min(poll_interval, remaining))
            continue

        consecutive_failures = 0
        if parsed.chunk and on_chunk is not None:
            on_chunk(parsed.chunk)
        next_byte = parsed.size + 1
        size_seen = parsed.size

        if parsed.rc is not None:
            return PollOutcome(rc=parsed.rc, reason="done", output_size=size_seen)

        if on_status is not None:
            on_status(f"running ({parsed.size} bytes captured)")
        remaining = _remaining()
        if remaining <= 0:
            return PollOutcome(rc=None, reason="deadline", output_size=size_seen)
        sleep(min(poll_interval, remaining))


@dataclass(frozen=True)
class AttachRecord:
    """Everything needed to re-attach to a detached job on a still-running VM
    after the local orchestrator restarts (a hard process kill, where the
    teardown context manager never ran). Persisted per label under ``<into>``; its
    presence marks a job as not-yet-completed-and-torn-down."""

    label: str
    cloud: str | None
    region: str | None
    name: str            # instance / keypair / security-group name
    server_id: str
    ip: str
    keyfile: str         # path to the persisted (ephemeral) private key
    remote_script: str   # home-relative job-script basename
    fetch: str           # home-relative artifact dir pulled on completion
    into: str            # local base dir for fetched artifacts
    cap_seconds: int     # remote runaway cap
    launch_epoch: float  # wall-clock (time.time) at launch, for the resume deadline

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))
