"""Tests for the command-line entry point. No network, no credentials.

The backgrounding flags are covered end to end through a real subprocess rather
than by forking inside pytest: `--detach` forks and calls `setsid`, and a test
process that did that to itself would take the test runner with it. The
subprocess drives `sweep --resume` against an empty results dir, which returns
before it ever reaches `connect()`, so the whole exercise stays offline.
"""
import os
import subprocess
import sys
import time

import pytest

from flux_compute.cli import main


def _run_cli(*args, cwd=None):
    """Run the CLI as a real process and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "flux_compute.cli", *args],
        capture_output=True, text=True, timeout=60, cwd=cwd,
        env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(__file__))})


# --- the flags exist and are documented ---------------------------------------

def test_sweep_help_documents_detach_and_log(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["sweep", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--detach" in out and "--log FILE" in out
    # The help must say what the flags replace, so nobody reaches for the shell.
    assert "nohup" in out


def test_detach_without_log_is_refused_with_the_remedy(capsys):
    """Fail fast, and in the foreground: a detached run whose output went nowhere
    would be indistinguishable from one that never started."""
    with pytest.raises(SystemExit) as exc:
        main(["sweep", "--detach"])
    assert exc.value.code == 2
    assert "--detach needs --log" in capsys.readouterr().err


# --- backgrounding, end to end ------------------------------------------------

@pytest.mark.skipif(not hasattr(os, "fork"), reason="--detach needs POSIX fork")
def test_detach_returns_at_once_and_the_work_lands_in_the_log(tmp_path):
    """The whole point of the flag: the caller gets its shell back immediately,
    names a pid, and the command's own output appears in the log written by a
    process that no longer has a terminal."""
    into = tmp_path / "empty"
    into.mkdir()
    log = tmp_path / "fleet.log"

    res = _run_cli("sweep", "--detach", "--log", str(log),
                   "--resume", "--into", str(into))

    assert res.returncode == 0
    assert "detached as pid" in res.stdout
    assert str(log) in res.stdout
    assert "tail -f" in res.stdout                 # tells the operator how to watch
    # The parent printed nothing of the command's own work ...
    assert "no in-flight jobs" not in res.stdout

    # ... which the daemon wrote to the log instead.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if log.exists() and "no in-flight jobs" in log.read_text():
            break
        time.sleep(0.1)
    assert "no in-flight jobs" in log.read_text()


def test_log_without_detach_redirects_the_output_to_the_file(tmp_path):
    into = tmp_path / "empty"
    into.mkdir()
    log = tmp_path / "run.log"

    res = _run_cli("sweep", "--log", str(log), "--resume", "--into", str(into))

    assert res.returncode == 0
    assert "no in-flight jobs" in log.read_text()   # the work went to the file
    assert "no in-flight jobs" not in res.stdout
    assert f"output -> {log}" in res.stderr         # said so on the terminal


def test_log_appends_so_a_resume_can_share_its_run_s_log(tmp_path):
    """A --resume pointed at the log of the run it continues must read as one
    story, not truncate the evidence for why the first attempt stopped."""
    into = tmp_path / "empty"
    into.mkdir()
    log = tmp_path / "run.log"
    log.write_text("FIRST ATTEMPT\n")

    _run_cli("sweep", "--log", str(log), "--resume", "--into", str(into))

    text = log.read_text()
    assert text.startswith("FIRST ATTEMPT")
    assert "no in-flight jobs" in text
