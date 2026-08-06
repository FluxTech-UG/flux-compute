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
import sys
import time
from dataclasses import asdict, dataclass

# ASCII record separator (0x1e) delimits a poll's output chunk from its status
# trailer. Job logs are text and never contain it, so the split is unambiguous.
RS = "\x1e"

# Consecutive ssh-transport failures before the poll loop escalates to its
# ``on_stuck`` handler. Four polls is ~1 minute of sweep polling (or two
# backoff-doubled retries): long enough not to fire on a single network flap,
# short enough that a real blackout is surfaced in minutes rather than hours.
STUCK_AFTER_POLLS = 4

# A sleep that overshoots its intended duration by more than this many seconds
# means the machine was SUSPENDED, not merely slow: ``time.monotonic`` freezes
# across a suspend on both macOS and Linux while wall-clock time keeps running,
# so the gap between the two is a direct suspend detector.
#
# A wake is the single likeliest moment for the caller's public IP to have
# changed (the lid closed at the office and opened at home), and it is also the
# moment the poll loop is least entitled to conclude anything: it has been blind.
# So a detected wake forces the ingress check on the NEXT failure rather than
# after another ``STUCK_AFTER_POLLS`` of them.
WAKE_JUMP_MARGIN_S = 90.0


# The verdicts ``classify_blackout`` returns, and what the poll loop does with
# each. They are the whole decision surface of the self-heal, kept as plain
# strings so the pure function is trivially testable without importing the
# OpenStack side.
BLACKOUT_HEALED = "healed"          # ingress was reopened; keep polling, reset the clock
BLACKOUT_OFFLINE = "offline"        # WE are disconnected; never abort, keep polling
BLACKOUT_RETRY = "retry"            # inconclusive; keep polling
BLACKOUT_UNREACHABLE = "unreachable"  # verified online + ingress open + past the bound


def classify_blackout(status, seconds_unreachable, *, abort_after_s):
    """Decide what a sustained SSH blackout means, given the ingress check's verdict.

    This is the whole self-heal decision, isolated from both the OpenStack API and
    the SSH transport so it can be tested directly. ``status`` is the
    ``IngressCheck.status`` the caller's repair attempt produced
    (``"healed"`` / ``"open"`` / ``"unknown-ip"`` / ``"no-group"`` / ``"error"``).

    The discriminator that matters is **who is disconnected**. The poll loop's
    reconnect tolerance exists because a closed laptop lid must not kill a healthy
    remote job, so a blackout can never be fatal while we cannot even read our own
    public address (``unknown-ip``): that is us being offline, and the job is
    almost certainly fine. Once we can see our own address AND the instance's
    group demonstrably admits it (``open``), the remaining explanation is the
    instance, and a fail-fast rule applies -- silence past ``abort_after_s`` is
    reported as the unreachability it is instead of being retried until the wall
    cap expires hours later.

    ``no-group`` is also not fatal here: the group is gone, so the instance is
    almost certainly gone too, but the teardown path is the authority on that and
    it runs on the caller's schedule, not this loop's.

    ``abort_after_s`` of ``None`` disables the bound entirely (retry until the
    local deadline, the historical behavior).
    """
    if status == "healed":
        return BLACKOUT_HEALED
    if status == "unknown-ip":
        # We cannot see our own public IP: the network we are on is the broken
        # thing. Never conclude anything about the instance from here.
        return BLACKOUT_OFFLINE
    if status == "open" and abort_after_s is not None and seconds_unreachable >= abort_after_s:
        return BLACKOUT_UNREACHABLE
    return BLACKOUT_RETRY

# Home-relative remote filenames the launcher writes and the poller reads.
REMOTE_OUT = "job.out"        # combined stdout+stderr, tailed by the poller
REMOTE_RC = "job.rc"          # integer return code, written only on completion
REMOTE_PID = "job.pid"        # PID of the detached job supervisor
REMOTE_LAUNCHER = "job_launch.sh"

# Host-RAM mitigation applied to EVERY detached job, at the provision layer.
#
# On Linux glibc, a thread pool (XLA's, in every JAX consumer) spawns up to
# 8 x ncpu malloc arenas, and large transient buffers fragment across them and
# are never returned to the OS: process RSS ratchets far past the live set until
# the kernel OOM-killer fires (rc=137) on a VM whose actual working set fits.
# Capping the arena count and lowering the trim threshold collapses that
# fragmentation headroom. It is allocator tuning only -- no correctness impact,
# strictly less RSS -- so it belongs on every job rather than being re-derived in
# each consumer's job script.
#
# ``:-`` means a value already in the environment wins, so a job script's own
# ``export MALLOC_ARENA_MAX=...`` (or a caller's env prefix) still overrides and
# consumer-side settings keep working, redundantly.
#
# tcmalloc is preloaded only when it is ALREADY installed on the image: it avoids
# per-thread arena fragmentation outright, but installing it would put an apt
# round-trip in front of every job, so the launcher uses it opportunistically and
# a consumer that needs it keeps installing it in its own script.
_MALLOC_TUNING = """\
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
if [ -z "${LD_PRELOAD:-}" ]; then
  _tcm=$(ldconfig -p 2>/dev/null | grep -om1 '/[^ ]*libtcmalloc_minimal\\.so[^ ]*' || true)
  [ -n "$_tcm" ] && export LD_PRELOAD="$_tcm"
fi
"""


def launcher_script(remote_script, cap_seconds, *, kill_after_s=30):
    """Return the bash for the detached launcher (uploaded and run once).

    ``remote_script`` is the home-relative basename of the caller's uploaded job
    script; ``cap_seconds`` is the remote runaway cap ``timeout`` enforces on the
    VM. The job runs under a login shell (``bash -lc``) to match the old
    foreground path, so a baked venv activated from the profile still applies. The
    launcher itself is short and non-blocking: it spawns the ``setsid`` job with
    all descriptors off the SSH channel and returns, so the launching SSH closes
    at once and the job keeps running through any later disconnect.

    It also applies the universal glibc-allocator tuning (``_MALLOC_TUNING``)
    before spawning the job, so every consumer inherits the OOM mitigation
    without repeating it in its own script.
    """
    remote_script = str(remote_script)
    cap = int(cap_seconds)
    kill_after = int(kill_after_s)
    return f"""#!/usr/bin/env bash
# flux-compute detached job launcher (generated; do not edit).
set -u
cd "$HOME"
chmod +x "$HOME/{remote_script}" 2>/dev/null || true
{_MALLOC_TUNING}
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

    ``reason`` is one of:

    ``"done"``
        The remote ``job.rc`` appeared; ``rc`` is the job's return code
        (including 124/137 for a remote-cap kill).
    ``"deadline"``
        The local wall-clock deadline aborted the follow before completion.
    ``"unreachable"``
        SSH stayed dead past the blackout bound while the caller was verifiably
        online and the instance's security group verifiably admitted it -- so the
        instance, not the network, is the cause. Reported rather than retried to
        the deadline, because a dead VM should not consume its whole wall cap in
        silence.

    ``rc`` is ``None`` for everything but ``"done"``."""

    rc: int | None
    reason: str
    output_size: int


def poll_until_done(run_poll, *, poll_interval, deadline_s,
                    backoff_base=5.0, backoff_max=60.0,
                    clock=time.monotonic, sleep=time.sleep, wall_clock=time.time,
                    on_chunk=None, on_status=None, on_warn=None,
                    on_stuck=None, stuck_after=STUCK_AFTER_POLLS,
                    unreachable_abort_s=None):
    """Poll a detached remote job to completion, tolerant of reconnection.

    ``run_poll(next_byte) -> PollAttempt`` performs one poll (a fresh short SSH in
    production). A failed attempt (connection error: the laptop just woke, a
    network flap) is retried with exponential backoff -- ``backoff_base`` doubling
    per consecutive failure, capped at ``backoff_max``. Returns a ``PollOutcome``.

    ``clock``/``sleep``/``wall_clock`` are injected for deterministic tests.
    ``on_chunk(str)`` receives each new ``job.out`` fragment (for a live local log
    / stream); ``on_status(str)`` receives a short human progress line per poll.
    ``on_warn(str)`` receives the things an operator must not miss even when no
    status sink is wired -- it defaults to ``on_status`` and, failing that, to
    stderr, because the one report that was silently dropped here (a self-heal
    that itself raised) is exactly the report that explains a frozen fleet.

    **Escalation and self-heal.** ``on_stuck(n_failures, seconds_unreachable)``
    fires once every ``stuck_after`` CONSECUTIVE ssh-transport failures so a
    caller can repair the one fault that breaks every job at once -- the caller's
    public IP having moved out of the security group's allowed CIDR. Only genuine
    connection failures count toward it (a connected read whose status trailer was
    garbled is a live SSH and resets the counter), so it is a true "the host is
    unreachable" signal, never a parse hiccup. The handler may return an
    ``IngressCheck``-shaped object (anything with a ``.status``); its verdict is
    put through ``classify_blackout`` to decide what the blackout means.

    **Waking counts as a reconnect.** A sleep that overshoots its intended
    duration by more than ``WAKE_JUMP_MARGIN_S`` of wall-clock time means the
    machine was suspended. That is the likeliest moment for the public IP to have
    moved and the moment the loop knows least, so the next failure escalates
    immediately instead of waiting for another ``stuck_after`` failures.

    **The bound.** ``unreachable_abort_s`` (``None`` = no bound) is how long a
    blackout may persist *while the caller is verifiably online and the group
    verifiably admits it* before the follow gives up with ``reason="unreachable"``.
    It is deliberately not a blanket timeout: a blackout during which we cannot
    read our own public IP is US being offline (the closed-lid case the whole
    detach design exists for) and never counts against it.
    """
    next_byte = 1
    size_seen = 0
    start = clock()
    consecutive_failures = 0
    ssh_failures = 0            # consecutive TRANSPORT failures (the stuck signal)
    unreachable_since = None
    escalate_now = False        # set by a detected wake: check ingress at once

    if on_warn is None:
        on_warn = on_status if on_status is not None else (
            lambda msg: print(msg, file=sys.stderr))

    def _remaining():
        return deadline_s - (clock() - start)

    def _sleep_watching_for_wake(seconds):
        """Sleep, then report whether the machine was suspended while we did.

        ``clock`` (monotonic) freezes across a system suspend while ``wall_clock``
        keeps running, so an overshoot between the two is the suspend.
        """
        nonlocal escalate_now
        wall_before = wall_clock()
        sleep(seconds)
        overshoot = (wall_clock() - wall_before) - seconds
        if overshoot > WAKE_JUMP_MARGIN_S:
            escalate_now = True
            on_warn(f"woke after ~{int(overshoot)}s suspended; "
                    "re-checking SSH ingress before trusting a failed poll")

    while True:
        if _remaining() <= 0:
            return PollOutcome(rc=None, reason="deadline", output_size=size_seen)

        attempt = run_poll(next_byte)

        if not attempt.ok:
            consecutive_failures += 1
            ssh_failures += 1
            if unreachable_since is None:
                unreachable_since = clock()
            if on_status is not None:
                on_status(f"poll failed ({attempt.error}); retry #{consecutive_failures}")

            due = (on_stuck is not None and stuck_after > 0
                   and (escalate_now or ssh_failures % stuck_after == 0))
            if due:
                escalate_now = False
                blackout_s = clock() - unreachable_since
                # A self-heal that itself fails must never break the follow loop,
                # whose contract is that only the deadline and the verified-dead
                # bound end it -- but it must never be SILENT either. Reporting it
                # only when a status sink happened to be wired is what let a
                # fleet-wide lockout look exactly like a healthy long job.
                try:
                    check = on_stuck(ssh_failures, blackout_s)
                except Exception as exc:      # noqa: BLE001 - advisory hook
                    on_warn(f"stuck handler failed: {type(exc).__name__}: {str(exc)[:80]}")
                    check = None
                verdict = classify_blackout(
                    getattr(check, "status", None), blackout_s,
                    abort_after_s=unreachable_abort_s)
                if verdict == BLACKOUT_UNREACHABLE:
                    on_warn(f"giving up: SSH unreachable for {int(blackout_s)}s while this "
                            "machine was online and the instance's security group admitted "
                            "it -- the instance, not the network, is the cause")
                    return PollOutcome(rc=None, reason="unreachable", output_size=size_seen)
                if verdict == BLACKOUT_HEALED:
                    # The lockout was ours and it is repaired: the blackout clock
                    # restarts so the freshly reopened path gets a full grace
                    # period before anything is concluded from its silence.
                    unreachable_since = clock()

            # The doubling is clamped before it is applied, not after. The result
            # is capped at `backoff_max` either way, but `2 ** n` on an unbounded
            # n becomes an integer too large to convert to float and raises
            # OverflowError out of the follow loop -- at roughly 1024 consecutive
            # failures, which a long wall cap reaches during exactly the kind of
            # multi-hour blackout this loop exists to survive.
            backoff = min(backoff_max,
                          backoff_base * (2 ** min(consecutive_failures - 1, 32)))
            remaining = _remaining()
            if remaining <= 0:
                return PollOutcome(rc=None, reason="deadline", output_size=size_seen)
            _sleep_watching_for_wake(min(backoff, remaining))
            continue

        # The connection is live: whatever the payload, SSH is reachable again.
        ssh_failures = 0
        unreachable_since = None

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
            _sleep_watching_for_wake(min(poll_interval, remaining))
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
        _sleep_watching_for_wake(min(poll_interval, remaining))


@dataclass(frozen=True)
class AttachRecord:
    """Everything needed to re-attach to a detached job on a still-running VM
    after the local orchestrator restarts (a hard process kill, where the
    teardown context manager never ran). Persisted per label under ``<into>``; its
    presence marks a job as not-yet-completed-and-torn-down.

    The record is written in TWO stages, and the first one is written **before
    the instance boots**. A launcher killed between ``create_server`` and the
    post-launch record write would otherwise leave a booted, billing VM that no
    later ``--resume`` could even name -- an orphan recoverable only by hand. So a
    *pending* record (``label``/``cloud``/``region``/``name``, no ``server_id``,
    ``ip`` or ``keyfile``) lands first: ``name`` is generated locally and is the
    instance / keypair / security-group name, which is enough for resume to find
    the server and tear it down. The post-launch write fills in the boot-time
    facts and makes the job ``attachable`` -- collectable, not merely killable.
    """

    label: str
    cloud: str | None
    region: str | None
    name: str            # instance / keypair / security-group name
    remote_script: str   # home-relative job-script basename
    fetch: str           # home-relative artifact dir pulled on completion
    into: str            # local base dir for fetched artifacts
    cap_seconds: int     # remote runaway cap
    launch_epoch: float  # wall-clock (time.time) at launch, for the resume deadline
    # Boot-time facts: empty in a pending (pre-boot) record, filled in after the
    # instance is up and the job is launched.
    server_id: str = ""
    ip: str = ""
    keyfile: str = ""    # path to the persisted (ephemeral) private key

    @property
    def attachable(self) -> bool:
        """True when the job can be re-followed and collected (the VM's address
        and private key are known). A pending record is not attachable: its VM can
        still be found and torn down by name, but its results are unreachable."""
        return bool(self.ip and self.keyfile)

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))
