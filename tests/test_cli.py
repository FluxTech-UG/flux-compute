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


# --- a refused --detach sweep is refused in the foreground ---------------------
#
# The incident: a budget refusal raised inside the detached child, so the terminal
# showed "detached as pid ..." and exit 0 while the refusal sat in the log, and a
# fleet that never ran was recorded as running. The pre-flight now runs before
# the fork. The fork itself is stubbed, so these tests stay in-process and offline.

def _wire_offline_sweep(monkeypatch, tmp_path, price_eur_hr=0.80):
    """A one-region sweep resolved without network: a priced V100S shard, a
    four-job jobs file and a job script. Returns the argv that runs it."""
    from types import SimpleNamespace

    from flux_compute import sweep
    from flux_compute.sweep import Shard

    spec = SimpleNamespace(flavor="t2-le-45", est_cost_eur_hr=price_eur_hr,
                           gpu_model="Tesla V100S 32GB", image="img", network="Ext-Net")
    monkeypatch.setattr(sweep, "_prepare_shard",
                        lambda cloud, region, flavor, image, max_parallel:
                        Shard(region=region, spec=spec, vcpus=15, cap=4))
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("a = 1\nb = 2\nc = 3\nd = 4\n")
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    return ["sweep", "--region", "GRA11", "--flavor", "t2-le-45",
            "--jobs", str(jobs), "--script", str(script), "--fetch", "out",
            "--into", str(tmp_path / "into"), "--max-minutes", "60",
            "--detach", "--log", str(tmp_path / "fleet.log")]


def _forbid_detach(monkeypatch):
    from flux_compute import cli

    def _no_fork(log_path):
        raise AssertionError("a refused sweep must not detach")
    monkeypatch.setattr(cli, "_detach_into_background", _no_fork)


def test_detach_over_budget_is_refused_on_the_terminal_before_any_fork(
        monkeypatch, tmp_path, capsys):
    # 4 jobs x EUR 0.80/hr x 60 min = EUR 3.20 worst case, against EUR 1.
    argv = _wire_offline_sweep(monkeypatch, tmp_path) + ["--budget", "1"]
    _forbid_detach(monkeypatch)

    rc = main(argv)

    captured = capsys.readouterr()
    assert rc == 1
    assert "worst-case ~EUR 3.20 exceeds budget EUR 1.00" in captured.err
    assert "detached as pid" not in captured.out
    assert not (tmp_path / "fleet.log").exists()     # nothing was handed to a log


def test_detach_with_a_bad_spec_is_refused_before_any_fork(monkeypatch, tmp_path, capsys):
    argv = _wire_offline_sweep(monkeypatch, tmp_path)
    i = argv.index("--script")
    del argv[i:i + 2]                                  # no job script
    _forbid_detach(monkeypatch)

    rc = main(argv)

    assert rc == 1
    assert "sweep needs --script" in capsys.readouterr().err


def test_detach_within_budget_detaches_and_says_what_passed(monkeypatch, tmp_path, capsys):
    from flux_compute import cli

    argv = _wire_offline_sweep(monkeypatch, tmp_path) + ["--budget", "10"]
    forked = []
    monkeypatch.setattr(cli, "_detach_into_background",
                        lambda log_path: forked.append(log_path) or 4242)

    rc = main(argv)

    out = capsys.readouterr().out
    assert rc == 0
    assert forked == [str(tmp_path / "fleet.log")]
    assert "pre-flight passed: 4 job(s), worst-case ~EUR 3.20" in out
    assert "detached as pid 4242" in out


def test_detach_resume_leaves_the_region_fit_to_the_child(monkeypatch, tmp_path, capsys):
    """The in-flight fleet still holds the quota a resume frees as it collects,
    so a live quota read on the terminal refused exactly what the child would
    launch. A --resume pre-flight checks the spec and the static-price budget,
    and forks even when every region reads full."""
    from types import SimpleNamespace

    from flux_compute import cli, sweep
    from flux_compute.sweep import Shard

    argv = _wire_offline_sweep(monkeypatch, tmp_path) + ["--resume", "--budget", "10"]
    full = SimpleNamespace(flavor="t2-le-45", est_cost_eur_hr=0.80,
                           gpu_model="Tesla V100S 32GB", image="img", network="Ext-Net")
    monkeypatch.setattr(sweep, "_prepare_shard",
                        lambda cloud, region, flavor, image, max_parallel:
                        Shard(region=region, spec=full, vcpus=15, cap=0))
    forked = []
    monkeypatch.setattr(cli, "_detach_into_background",
                        lambda log_path: forked.append(log_path) or 4242)

    rc = main(argv)

    out = capsys.readouterr().out
    assert rc == 0
    assert forked == [str(tmp_path / "fleet.log")]
    assert "pre-flight passed: 4 job(s), worst-case ~EUR 3.20" in out


# --- --plan always prints in the foreground --------------------------------------
#
# The incident: a campaign's recorded launch line ends in `--detach --log FILE`,
# and a `--plan` dry run built from it went into the background log instead of
# the terminal where it was meant to be read.

def test_plan_ignores_detach_and_log_and_prints_to_the_terminal(
        monkeypatch, tmp_path, capsys):
    argv = _wire_offline_sweep(monkeypatch, tmp_path) + ["--plan"]
    _forbid_detach(monkeypatch)

    rc = main(argv)

    captured = capsys.readouterr()
    assert rc == 0
    assert "--plan prints to the terminal; ignoring --detach and --log" in captured.err
    assert "plan only: no instances launched." in captured.out
    assert not (tmp_path / "fleet.log").exists()


def test_plan_with_log_alone_names_only_log(monkeypatch, tmp_path, capsys):
    argv = _wire_offline_sweep(monkeypatch, tmp_path)
    argv.remove("--detach")
    assert main(argv + ["--plan"]) == 0
    captured = capsys.readouterr()
    assert "ignoring --log" in captured.err and "--detach" not in captured.err
    assert "plan only" in captured.out
    assert not (tmp_path / "fleet.log").exists()


# --- --wait-next and --rerun-failed-start are parsed and dispatched ---------------

def test_wait_next_dispatches_with_the_campaign_folder(monkeypatch, tmp_path):
    from flux_compute import sweep

    seen = []
    monkeypatch.setattr(sweep, "wait_next_job", lambda into: seen.append(into) or 0)
    assert main(["sweep", "--wait-next", "--into", str(tmp_path)]) == 0
    assert seen == [str(tmp_path)]


def test_wait_next_with_no_running_job_exits_nonzero(tmp_path, capsys):
    assert main(["sweep", "--wait-next", "--into", str(tmp_path)]) == 1
    assert "no running jobs" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--resume"], ["--plan"], ["--jobs", "j.txt"],
                                   ["--detach", "--log", "x.log"]])
def test_wait_next_refuses_flags_that_would_launch_or_background(extra, capsys):
    with pytest.raises(SystemExit):
        main(["sweep", "--wait-next", "--into", "cloud-sweep", *extra])
    assert "--wait-next only waits on a running campaign" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--resume"], ["--jobs", "j.txt"]])
def test_rerun_failed_start_needs_resume_and_jobs(extra, capsys):
    with pytest.raises(SystemExit):
        main(["sweep", "--rerun-failed-start", *extra])
    assert "--rerun-failed-start needs --resume and --jobs" in capsys.readouterr().err


def test_resume_passes_rerun_failed_start_through(monkeypatch, tmp_path):
    from flux_compute import sweep

    seen = {}
    monkeypatch.setattr(sweep, "run_sweep", lambda **kw: seen.update(kw) or 0)
    assert main(["sweep", "--resume", "--rerun-failed-start", "--jobs", "j.txt",
                 "--into", str(tmp_path)]) == 0
    assert seen["rerun_failed_start"] is True and seen["resume"] is True


def test_wait_next_ignores_log_and_prints_to_the_terminal(monkeypatch, tmp_path, capsys):
    from flux_compute import sweep

    monkeypatch.setattr(sweep, "wait_next_job",
                        lambda into: print("chen_nx1024 region=DE1 rc=0 ok") or 0)
    log = tmp_path / "fleet.log"
    assert main(["sweep", "--wait-next", "--into", str(tmp_path), "--log", str(log)]) == 0
    captured = capsys.readouterr()
    assert "chen_nx1024 region=DE1 rc=0 ok" in captured.out
    assert "--wait-next prints to the terminal; ignoring --log" in captured.err
    assert not log.exists()
