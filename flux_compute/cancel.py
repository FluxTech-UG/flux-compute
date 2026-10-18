"""`flux-compute cancel`: stop one in-flight sweep job, or every one under a folder.

`reap` works on the cloud side by TTL and cannot name a job; this works from a
job's local attach record (`<into>/<label>/.flux_attach/record.json`), which
names its instance, region and cloud. For each job, after connecting to its
cloud and region (a credential failure stops the cancel with the job untouched),
in this order:

  1. stop its follower: the sweep or `--resume` process named by the
     `controller.pid` sidecar, if it is alive and is a flux-compute process
     (same user; SIGTERM, then SIGKILL). It goes first so it cannot rewrite the
     job's outcome or folder after the cancel.
  2. fetch what the job wrote so far (its job log and the fetch dir), best
     effort: the instance is about to be deleted.
  3. tear the instance down with its keypair and security group (verified; a
     strand raises and the record stays for a retry).
  4. write `outcome.json` with status `cancelled` and the time, then remove
     `.flux_attach/` (the order every job end keeps).

A cancelled job reads as collected, so `sweep --resume --jobs` never launches it
again. Killing a follower stops its other jobs' follow too: their records stay,
the command names them, and `sweep --resume` follows them again.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from .auth import connect, credential_error
from .provision import pull_job_log, rsync_down_best_effort, teardown_by_name
from .sweep import (
    JOB_LOG, _close_job, _load_attach_records, _outcome, read_controller_pid,
)

CANCELLED = "cancelled"
TERM_WAIT_S = 10.0
KILL_WAIT_S = 5.0


class CancelRefused(RuntimeError):
    """A cancel that names no in-flight job; the CLI prints it as one line, exit 2."""


def _is_alive(pid, _kill=os.kill):
    """True when `pid` exists. A process this user may not signal raises: it is
    another user's, or the shell is sandboxed, and either way the check did not run."""
    try:
        _kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError(
            f"cancel: cannot signal pid {pid} (PermissionError): it belongs to another "
            "user, or this shell is sandboxed; send the command hatched") from exc
    return True


def _command_of(pid, _run=subprocess.run):
    """The command line of `pid`, or None when it has exited. A `ps` that cannot
    run raises."""
    res = _run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    out = res.stdout.strip()
    if res.returncode != 0 and not out:
        if res.stderr.strip():
            raise RuntimeError(f"cancel: `ps -p {pid}` failed: {res.stderr.strip()[:200]}")
        return None
    return out


def _is_flux_compute(command):
    return "flux-compute" in command or "flux_compute" in command


def stop_controller(pid, *, emit=print, _kill=os.kill, _run=subprocess.run,
                    _sleep=time.sleep, term_wait_s=TERM_WAIT_S, kill_wait_s=KILL_WAIT_S):
    """Stop the flux-compute process `pid`. Returns True when it was stopped
    here, False when it was already gone or the pid now names another program
    (left alone). A process that outlives SIGKILL raises."""
    if pid == os.getpid():
        raise RuntimeError(f"cancel: controller pid {pid} is this process")
    if not _is_alive(pid, _kill):
        emit(f"  controller pid {pid} already gone")
        return False
    command = _command_of(pid, _run)
    if command is None:
        emit(f"  controller pid {pid} already gone")
        return False
    if not _is_flux_compute(command):
        emit(f"  pid {pid} is now {command[:100]!r}, not a flux-compute process "
             "(the pid was reused); left alone")
        return False
    for sig, wait in ((signal.SIGTERM, term_wait_s), (signal.SIGKILL, kill_wait_s)):
        try:
            _kill(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not _is_alive(pid, _kill):
                break
            _sleep(0.2)
        if not _is_alive(pid, _kill):
            break
    if _is_alive(pid, _kill):
        raise RuntimeError(f"cancel: controller pid {pid} is still alive after SIGKILL")
    emit(f"  stopped controller pid {pid} ({command[:100]})")
    return True


def _select(into, labels, take_all):
    """The in-flight records the cancel acts on, and the controller pid of
    every in-flight job of the folder, by label."""
    if not os.path.isdir(into):
        raise CancelRefused(f"no campaign folder {into}/")
    records = {rec.label: rec for rec in _load_attach_records(into)}
    if take_all:
        if not records:
            raise CancelRefused(f"no in-flight job under {into}/ (no .flux_attach/ record)")
        chosen = sorted(records)
    else:
        missing = [label for label in labels if label not in records]
        if missing:
            raise CancelRefused(
                f"{', '.join(missing)} not in flight under {into}/ "
                "(no .flux_attach/ record)")
        chosen = list(dict.fromkeys(labels))
    pids = {label: read_controller_pid(os.path.join(into, label)) for label in records}
    return [records[label] for label in chosen], pids


def _print_plan(into, targets, pids, cloud):
    for rec in targets:
        pid = pids[rec.label]
        print(f"cancel --plan: {rec.label} ({rec.region or '(default region)'}, "
              f"cloud {rec.cloud or cloud!r}, instance {rec.name}"
              f"{', server ' + rec.server_id if rec.server_id else ''})")
        if pid is None:
            print("  no controller pid recorded: no process to stop")
        else:
            print(f"  would stop controller pid {pid} if it is alive and a flux-compute "
                  "process (checked when the cancel runs)")
        if rec.attachable:
            print(f"  would fetch ~/{rec.fetch} and the job log into {into}/{rec.label}/ "
                  "(best effort)")
        print("  would delete the instance, its keypair and its security group (verified)")
        print(f"  would write {into}/{rec.label}/outcome.json (status {CANCELLED}) "
              "and remove .flux_attach/")
    return 0


def run_cancel(into, labels=(), take_all=False, cloud=None, plan=False, *,
               _connect=connect, _teardown=teardown_by_name, _stop=stop_controller,
               _pull_log=pull_job_log, _fetch=rsync_down_best_effort, emit=print):
    """Cancel the named in-flight jobs under `into` (or all of them). Returns 0
    when every one is cancelled; raises `CancelRefused` when a label is not in
    flight and RuntimeError when a step fails (the job's record then stays)."""
    if take_all == bool(labels):
        raise CancelRefused("name the jobs with --label, or pass --all (not both)")
    targets, pids = _select(into, labels, take_all)
    if plan:
        return _print_plan(into, targets, pids, cloud)

    stopped, checked = set(), set()
    for rec in targets:
        dest = os.path.join(into, rec.label)
        c = rec.cloud or cloud
        emit(f"cancel: {rec.label} ({rec.region or '(default region)'}, instance {rec.name})")
        # Connect first: a credential failure then leaves the job exactly as it
        # was, its follower still running.
        try:
            conn = _connect(cloud=c, region=rec.region)
        except RuntimeError as exc:
            raise credential_error("cancel", c, rec.region, exc) from exc
        pid = pids[rec.label]
        if pid is not None and pid not in checked:
            checked.add(pid)
            if _stop(pid, emit=emit):
                stopped.add(pid)
        if rec.attachable:
            got_log = _pull_log(rec.ip, rec.keyfile, os.path.join(dest, JOB_LOG))
            got_out = _fetch(rec.ip, rec.keyfile, rec.fetch, dest)
            emit(f"  fetched: job log {'yes' if got_log else 'no'}, "
                 f"~/{rec.fetch} {'yes' if got_out else 'no'}")
        _teardown(conn, rec.name, rec.server_id or None)
        out = _outcome(rec.label, rec.region, -1, CANCELLED)
        _close_job(dest, out)
        when = datetime.fromtimestamp(out.ended_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        emit(f"  {rec.label}: {CANCELLED} at {when}")

    cancelled = {rec.label for rec in targets}
    for pid in sorted(stopped):
        orphaned = sorted(label for label, p in pids.items()
                          if p == pid and label not in cancelled)
        if orphaned:
            emit(f"note: stopped pid {pid} was also following {', '.join(orphaned)}; "
                 f"their records are kept, and `flux-compute sweep --resume --into {into}` "
                 "follows them again")
        emit(f"note: if pid {pid} was launching a jobs file, its jobs not yet started "
             "stay pending; `sweep --resume --jobs FILE` launches them")
    return 0


def main_cancel(args) -> int:
    """The CLI branch: a refusal is one line on stderr and exit 2."""
    try:
        return run_cancel(args.into, labels=args.label or (), take_all=args.take_all,
                          cloud=args.cloud, plan=args.plan)
    except CancelRefused as exc:
        print(f"flux-compute cancel: {exc}", file=sys.stderr)
        return 2
