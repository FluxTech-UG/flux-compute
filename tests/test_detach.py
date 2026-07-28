"""Pure-logic tests for the detach-and-poll machinery. No network, no credentials.

Covers the two halves of the sleep-survival fix: the bash the launcher and poller
emit (launcher_script / poll_command), the poll-output parser, the
reconnection-tolerant poll loop driven by a fake SSH runner (no-rc-yet /
connection error / rc present), the SSH-command runner's error mapping against a
fake `_ssh`, and the AttachRecord round-trip used by sweep --resume.
"""
import subprocess

import pytest

from flux_compute import detach, provision
from flux_compute.detach import (
    AttachRecord,
    PollAttempt,
    launcher_script,
    parse_poll_output,
    poll_command,
    poll_until_done,
)


# --- launcher_script: detached, capped, records rc only on completion ---------

def test_launcher_detaches_into_a_new_session():
    ls = launcher_script("job.sh", 1800)
    assert "setsid --fork" in ls          # new session -> survives SSH drop
    assert "< /dev/null" in ls            # stdin off the channel
    assert "job.out" in ls and "2>&1" in ls   # combined stdout+stderr to a file


def test_launcher_caps_the_job_remotely_and_writes_rc_on_exit():
    ls = launcher_script("job.sh", 1800, kill_after_s=30)
    # timeout is the laptop-independent runaway backstop; rc written only after.
    assert "timeout --signal=TERM --kill-after=30 1800 bash -lc" in ls
    assert 'echo "$?" > "$HOME/job.rc"' in ls
    assert 'echo "$$" > "$HOME/job.pid"' in ls


def test_launcher_runs_the_named_script_under_a_login_shell():
    # -lc matches the old foreground path so a profile-activated venv still applies.
    ls = launcher_script("my_job.sh", 600)
    assert 'bash -lc "$HOME/my_job.sh"' in ls
    assert "600 bash -lc" in ls           # cap substituted as integer seconds


def test_launcher_makes_the_job_script_executable():
    # The job is run via its path, so the execute bit must be set (as the old
    # foreground `chmod +x ~/script && bash -lc '~/script'` did).
    ls = launcher_script("my_job.sh", 600)
    assert 'chmod +x "$HOME/my_job.sh"' in ls


def test_launcher_cap_is_coerced_to_int_seconds():
    assert "1234 bash -lc" in launcher_script("j.sh", 1234.9)


# --- poll_command: incremental, over-advance-proof -----------------------------

def test_poll_command_tails_from_the_byte_offset_and_bounds_to_size():
    pc = poll_command(42)
    assert 'wc -c < "$HOME/job.out"' in pc         # size captured first
    assert 'tail -c "+42"' in pc                    # from the 1-based offset
    assert 'head -c "$(( sz - 42 + 1 ))"' in pc     # bounded to the captured size
    assert "SIZE=%s;RC=%s" in pc                     # status trailer


def test_poll_command_reads_rc_only_when_present():
    pc = poll_command(1)
    assert '[ -f "$HOME/job.rc" ]' in pc


# --- parse_poll_output ---------------------------------------------------------

def test_parse_running_has_no_rc():
    r = parse_poll_output("hello\x1eSIZE=5;RC=")
    assert r.chunk == "hello" and r.size == 5 and r.rc is None


def test_parse_completed_returns_rc_zero():
    r = parse_poll_output("done\x1eSIZE=4;RC=0")
    assert r.chunk == "done" and r.size == 4 and r.rc == 0


def test_parse_remote_cap_kill_returns_124():
    r = parse_poll_output("partial output\x1eSIZE=14;RC=124")
    assert r.rc == 124 and r.size == 14


def test_parse_empty_chunk_is_fine():
    r = parse_poll_output("\x1eSIZE=100;RC=")
    assert r.chunk == "" and r.size == 100 and r.rc is None


def test_parse_missing_separator_is_none():
    # A truncated/garbled read: the caller treats None as a soft failure.
    assert parse_poll_output("no separator at all") is None


def test_parse_malformed_trailer_is_none():
    assert parse_poll_output("chunk\x1eGARBAGE") is None


def test_parse_none_input_is_none():
    assert parse_poll_output(None) is None


def test_parse_uses_the_last_separator_so_chunk_may_contain_one():
    # rfind picks the trailer we printed even if the log itself held a 0x1e.
    r = parse_poll_output("weird\x1elog\x1eSIZE=9;RC=0")
    assert r.chunk == "weird\x1elog" and r.size == 9 and r.rc == 0


# --- poll_until_done: the reconnection-tolerant loop, fake SSH runner ----------

class _Clock:
    """A fake monotonic clock advanced only by the loop's own sleeps, so tests
    are deterministic and instant."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _run(seq, **kw):
    """Drive poll_until_done over a scripted list of PollAttempts."""
    clk = _Clock()
    it = iter(seq)
    kw.setdefault("poll_interval", 10)
    kw.setdefault("deadline_s", 100000)
    return clk, poll_until_done(lambda n: next(it), clock=clk.now, sleep=clk.sleep, **kw)


def test_loop_normal_completion_streams_chunks_then_returns_rc():
    chunks = []
    _, out = _run(
        [PollAttempt.connected("a\x1eSIZE=1;RC="),
         PollAttempt.connected("bb\x1eSIZE=3;RC="),
         PollAttempt.connected("end\x1eSIZE=6;RC=0")],
        on_chunk=chunks.append,
    )
    assert out.reason == "done" and out.rc == 0 and out.output_size == 6
    assert chunks == ["a", "bb", "end"]        # each new fragment delivered once


def test_loop_retries_transient_connection_failures_with_backoff():
    # no-rc-yet, then three connection errors, then rc present: never fatal.
    stat = []
    clk, out = _run(
        [PollAttempt.connected("x\x1eSIZE=1;RC="),
         PollAttempt.failed("ssh 255"),
         PollAttempt.failed("ssh 255"),
         PollAttempt.failed("ssh 255"),
         PollAttempt.connected("done\x1eSIZE=5;RC=0")],
        backoff_base=5.0, backoff_max=60.0, on_status=stat.append,
    )
    assert out.reason == "done" and out.rc == 0
    # one poll_interval (10) + backoff 5 + 10 + 20 = 45; backoff doubled each fail.
    assert clk.t == pytest.approx(45.0)
    assert sum("retry #" in s for s in stat) == 3


def test_loop_backoff_is_capped():
    # Six straight failures then success: backoff caps at backoff_max, not 5*2^5.
    clk, out = _run(
        [PollAttempt.failed("down")] * 6 + [PollAttempt.connected("k\x1eSIZE=1;RC=0")],
        backoff_base=5.0, backoff_max=20.0,
    )
    assert out.rc == 0
    # 5 + 10 + 20 + 20 + 20 + 20 = 95 (third failure onward pinned at the cap).
    assert clk.t == pytest.approx(95.0)


def test_loop_reports_remote_cap_kill_as_done_with_rc_124():
    _, out = _run(
        [PollAttempt.connected("x\x1eSIZE=1;RC="),
         PollAttempt.connected("killed\x1eSIZE=7;RC=124")],
    )
    assert out.reason == "done" and out.rc == 124   # not hung: a real return code


def test_loop_local_deadline_aborts_a_never_finishing_job():
    # Always running, clock runs past the deadline -> deadline abort, rc None.
    clk = _Clock()
    out = poll_until_done(
        lambda n: PollAttempt.connected("r\x1eSIZE=1;RC="),
        poll_interval=10, deadline_s=55, clock=clk.now, sleep=clk.sleep,
    )
    assert out.reason == "deadline" and out.rc is None
    assert clk.t <= 55 + 10          # never sleeps far past the deadline


def test_loop_connection_loss_is_never_fatal_only_the_deadline_aborts():
    # The laptop stays asleep: every poll fails, yet the loop only ends at the
    # deadline (never on the failures themselves).
    clk = _Clock()
    out = poll_until_done(
        lambda n: PollAttempt.failed("host down"),
        poll_interval=10, deadline_s=200, backoff_base=5.0, backoff_max=60.0,
        clock=clk.now, sleep=clk.sleep,
    )
    assert out.reason == "deadline" and out.rc is None


def test_loop_unparseable_read_is_a_soft_retry_not_an_advance():
    # A connected-but-garbled read must not advance the byte cursor or crash.
    seen = []

    def run_poll(n):
        seen.append(n)
        return (PollAttempt.connected("junk-no-separator") if len(seen) == 1
                else PollAttempt.connected("ok\x1eSIZE=2;RC=0"))

    clk = _Clock()
    out = poll_until_done(run_poll, poll_interval=10, deadline_s=1000,
                          clock=clk.now, sleep=clk.sleep)
    assert out.rc == 0
    assert seen == [1, 1]            # cursor not advanced by the garbled read


# --- the SSH-command poll runner maps transport errors to retryable failures ---

class _FakeSSH:
    """A fake `_ssh(ip, keyfile, command, timeout, capture)` scripted with a list
    of responses: a CompletedProcess-like namespace, or a TimeoutExpired to
    raise. Records every command it was asked to run."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.commands = []

    def __call__(self, ip, keyfile, command, timeout=None, capture=True):
        self.commands.append(command)
        r = self._responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _completed(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_ssh_runner_maps_exit_255_to_a_retryable_failure():
    fake = _FakeSSH([_completed(255, stderr="Connection refused")])
    run_poll = provision._make_poll_runner("1.2.3.4", "/k", _ssh=fake)
    attempt = run_poll(1)
    assert attempt.ok is False and "transport" in attempt.error


def test_ssh_runner_maps_a_timeout_to_a_retryable_failure():
    fake = _FakeSSH([subprocess.TimeoutExpired(cmd="ssh", timeout=30)])
    run_poll = provision._make_poll_runner("1.2.3.4", "/k", _ssh=fake)
    attempt = run_poll(1)
    assert attempt.ok is False and "timed out" in attempt.error


def test_ssh_runner_passes_a_connected_read_through_to_the_parser():
    fake = _FakeSSH([_completed(0, stdout="out\x1eSIZE=3;RC=0")])
    run_poll = provision._make_poll_runner("1.2.3.4", "/k", _ssh=fake)
    attempt = run_poll(1)
    assert attempt.ok is True
    parsed = parse_poll_output(attempt.stdout)
    assert parsed.rc == 0 and parsed.chunk == "out"


def test_ssh_backed_runner_completes_through_a_transport_flap():
    # End to end: the real _make_poll_runner + the real loop reconnect through a
    # 255 flap and finish, exactly as the sleep/wake case needs.
    fake = _FakeSSH([
        _completed(0, stdout="a\x1eSIZE=1;RC="),
        _completed(255, stderr="reset by peer"),     # laptop asleep
        _completed(0, stdout="done\x1eSIZE=5;RC=0"),  # woke, reconnected
    ])
    run_poll = provision._make_poll_runner("1.2.3.4", "/k", _ssh=fake)
    clk = _Clock()
    out = poll_until_done(run_poll, poll_interval=10, deadline_s=10000,
                          clock=clk.now, sleep=clk.sleep)
    assert out.reason == "done" and out.rc == 0


# --- AttachRecord: the resume handoff -----------------------------------------

def test_attach_record_round_trips_through_json():
    rec = AttachRecord(
        label="alpha", cloud="flux-ovh", region="GRA11",
        name="flux-compute-sweep-abcd1234", server_id="srv-1", ip="1.2.3.4",
        keyfile="/tmp/k/id_key", remote_script="job.sh", fetch="out",
        into="cloud-sweep", cap_seconds=1800, launch_epoch=1_700_000_000.0)
    assert AttachRecord.from_json(rec.to_json()) == rec


def test_attach_record_tolerates_none_cloud_and_region():
    rec = AttachRecord(
        label="a", cloud=None, region=None, name="n", server_id="s", ip="i",
        keyfile="/k", remote_script="j.sh", fetch="out", into="cloud-sweep",
        cap_seconds=600, launch_epoch=1.0)
    assert AttachRecord.from_json(rec.to_json()) == rec


# --- universal allocator tuning (host-RAM mitigation at the provision layer) ---

def test_launcher_applies_the_glibc_arena_cap():
    """The OOM mitigation belongs to every job, not to each consumer's script."""
    ls = launcher_script("job.sh", 600)
    assert "MALLOC_ARENA_MAX" in ls and "MALLOC_TRIM_THRESHOLD_" in ls


def test_launcher_lets_a_preset_value_win():
    """':-' expansion: a job script's own export (or a caller's env prefix) still
    overrides, so consumer-side settings keep working, redundantly."""
    ls = launcher_script("job.sh", 600)
    assert 'MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"' in ls
    assert 'MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"' in ls


def test_launcher_preloads_tcmalloc_only_when_already_installed():
    """Opportunistic: no apt round-trip in front of every job, and never
    clobbering an LD_PRELOAD the caller set."""
    ls = launcher_script("job.sh", 600)
    assert "libtcmalloc_minimal" in ls and "ldconfig" in ls
    assert 'if [ -z "${LD_PRELOAD:-}" ]' in ls
    assert "apt-get" not in ls


def test_launcher_tuning_precedes_the_job_spawn():
    ls = launcher_script("job.sh", 600)
    assert ls.index("MALLOC_ARENA_MAX") < ls.index("setsid --fork")


# --- on_stuck: a sustained SSH blackout is surfaced, not retried in silence ----

def test_stuck_handler_fires_on_a_sustained_blackout():
    stuck = []
    _run([PollAttempt.failed("ssh timed out")] * 4 + [PollAttempt.connected("\x1eSIZE=0;RC=0")],
         on_stuck=lambda n, secs: stuck.append((n, secs)), stuck_after=4)
    assert len(stuck) == 1 and stuck[0][0] == 4
    assert stuck[0][1] > 0                    # reports how long it has been down


def test_stuck_handler_re_fires_while_the_blackout_continues():
    stuck = []
    _run([PollAttempt.failed("ssh timed out")] * 8 + [PollAttempt.connected("\x1eSIZE=0;RC=0")],
         on_stuck=lambda n, secs: stuck.append(n), stuck_after=4)
    assert stuck == [4, 8]


def test_stuck_handler_does_not_fire_on_a_brief_flap():
    stuck = []
    _run([PollAttempt.failed("flap"),
          PollAttempt.connected("\x1eSIZE=0;RC="),
          PollAttempt.failed("flap"),
          PollAttempt.connected("\x1eSIZE=0;RC=0")],
         on_stuck=lambda n, secs: stuck.append(n), stuck_after=4)
    assert stuck == []


def test_a_connected_read_resets_the_blackout_counter():
    """Only transport failures count: a garbled trailer is a LIVE ssh, so it must
    not be mistaken for an unreachable host."""
    stuck = []
    _run([PollAttempt.failed("down")] * 3
         + [PollAttempt.connected("garbled, no trailer")]     # connected but unparseable
         + [PollAttempt.failed("down")] * 3
         + [PollAttempt.connected("\x1eSIZE=0;RC=0")],
         on_stuck=lambda n, secs: stuck.append(n), stuck_after=4)
    assert stuck == []


def test_a_raising_stuck_handler_never_breaks_the_follow_loop():
    """Only the local deadline may abort the follow; a failing self-heal may not."""
    def boom(n, secs):
        raise RuntimeError("network API down")

    _, out = _run([PollAttempt.failed("down")] * 4 + [PollAttempt.connected("\x1eSIZE=0;RC=0")],
                  on_stuck=boom, stuck_after=4)
    assert out.reason == "done" and out.rc == 0


# --- pending (pre-boot) attach records -----------------------------------------

def test_pending_record_round_trips_without_boot_time_facts():
    rec = AttachRecord(
        label="alpha", cloud="flux-ovh", region="GRA11",
        name="flux-compute-sweep-abcd1234", remote_script="job.sh", fetch="out",
        into="cloud-sweep", cap_seconds=1800, launch_epoch=1.0)
    assert not rec.attachable                     # no ip / key: killable, not collectable
    assert rec.name and rec.server_id == "" and rec.ip == ""
    assert AttachRecord.from_json(rec.to_json()) == rec


def test_full_record_is_attachable():
    rec = AttachRecord(
        label="alpha", cloud=None, region=None, name="n", remote_script="j.sh",
        fetch="out", into="cloud-sweep", cap_seconds=600, launch_epoch=1.0,
        server_id="srv-1", ip="1.2.3.4", keyfile="/k")
    assert rec.attachable
