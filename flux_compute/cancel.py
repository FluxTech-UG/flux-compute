"""`flux-compute cancel`: stop one in-flight sweep job, or every one under a folder.

`reap` works on the cloud side by TTL and cannot name a job; this works from a
job's local attach record (`<into>/<label>/.flux_attach/record.json`), which
names its instance, region and cloud. A job with no `controller.pid` (launched
before the sweep recorded its follower) is refused unless `--no-follower` says
its follower is already stopped: an unstopped follower would overwrite the
cancel. For each job, after connecting to its cloud and region (a credential
failure stops the cancel with the job untouched), in this order:

  1. stop its follower: the sweep or `--resume` process named by the
     `controller.pid` sidecar (its pid and start time), when a process with
     that pid AND that start time is alive (SIGTERM, then SIGKILL). A pid that
     now names a process started at another time was reused and is left alone.
     It goes first so it cannot rewrite the job's outcome or folder afterwards.
  2. fetch what the job wrote so far (its job log and the fetch dir), best
     effort: the instance is about to be deleted.
  3. tear the instance down with its keypair and security group, each verified;
     a failure of any of the three raises and the record stays for a retry.
  4. write `outcome.json` with status `cancelled` and the time, then remove
     `.flux_attach/` (the order every job end keeps).

A cancelled job reads as collected, so `sweep --resume --jobs` never launches it
again. Killing a follower stops its other jobs' follow too: their records stay,
the command names them, and `sweep --resume` follows them again. Those notes
print whether the cancel finished or stopped partway.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime, timezone

from .auth import connect, credential_error
from .provision import pull_job_log, rsync_down_best_effort, teardown_by_name
from .sweep import (
    JOB_LOG, _close_job, _load_attach_records, _outcome, process_start, read_controller,
)

CANCELLED = "cancelled"
TERM_WAIT_S = 10.0
KILL_WAIT_S = 5.0


class CancelRefused(RuntimeError):
    """A cancel the command will not run; the CLI prints it as one line, exit 2."""


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


def stop_controller(pid, start, *, emit=print, _kill=os.kill, _start_of=process_start,
                    _sleep=time.sleep, term_wait_s=TERM_WAIT_S, kill_wait_s=KILL_WAIT_S):
    """Stop the follower `pid`, started at `start` (`ps -o lstart=` text).
    Returns True when it was stopped here, False when it was already gone or
    the pid now names a process started at another time (reused; left alone).
    A process that outlives SIGKILL raises."""
    if pid == os.getpid():
        raise RuntimeError(f"cancel: controller pid {pid} is this process")
    if not _is_alive(pid, _kill):
        emit(f"  controller pid {pid} already gone")
        return False
    now = _start_of(pid)
    if now is None:
        emit(f"  controller pid {pid} already gone")
        return False
    if now != start:
        emit(f"  pid {pid} now names a process started {now}, not the follower started "
             f"{start} (the pid was reused); left alone")
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
    emit(f"  stopped controller pid {pid} (started {start})")
    return True


def _select(into, labels, take_all):
    """The in-flight records the cancel acts on, and the controller (pid, start)
    of every in-flight job of the folder, by label."""
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
    controllers = {label: read_controller(os.path.join(into, label)) for label in records}
    return [records[label] for label in chosen], controllers


def _no_follower_refusal(labels):
    return (f"{', '.join(labels)}: no controller.pid recorded (launched before the sweep "
            "recorded its follower), so the sweep or --resume process following it cannot "
            "be stopped here and would overwrite the cancel; stop that process first "
            "(kill <its pid>), then cancel again with --no-follower")


def _print_plan(into, targets, controllers, cloud, refusal):
    for rec in targets:
        ctl = controllers[rec.label]
        print(f"cancel --plan: {rec.label} ({rec.region or '(default region)'}, "
              f"cloud {rec.cloud or cloud!r}, instance {rec.name}"
              f"{', server ' + rec.server_id if rec.server_id else ''})")
        if ctl is None:
            print("  no controller pid recorded: no process to stop (--no-follower given)"
                  if refusal is None else "  no controller pid recorded: refused, see below")
        else:
            print(f"  would stop controller pid {ctl[0]} if a process with that pid started "
                  f"{ctl[1]} is alive (checked when the cancel runs)")
        if rec.attachable:
            print(f"  would fetch ~/{rec.fetch} and the job log into {into}/{rec.label}/ "
                  "(best effort)")
        print("  would delete the instance, its keypair and its security group (each verified)")
        print(f"  would write {into}/{rec.label}/outcome.json (status {CANCELLED}) "
              "and remove .flux_attach/")
    if refusal is not None:
        print(f"cancel --plan: would refuse: {refusal}", file=sys.stderr)
        return 2
    return 0


def run_cancel(into, labels=(), take_all=False, cloud=None, plan=False, no_follower=False,
               *, _connect=connect, _teardown=teardown_by_name, _stop=stop_controller,
               _pull_log=pull_job_log, _fetch=rsync_down_best_effort, emit=print):
    """Cancel the named in-flight jobs under `into` (or all of them). Returns 0
    when every one is cancelled; raises `CancelRefused` when a label is not in
    flight or has no recorded follower (without `no_follower`), and RuntimeError
    when a step fails (the job's record then stays)."""
    if take_all == bool(labels):
        raise CancelRefused("name the jobs with --label, or pass --all (not both)")
    targets, controllers = _select(into, labels, take_all)
    unowned = [rec.label for rec in targets if controllers[rec.label] is None]
    refusal = _no_follower_refusal(unowned) if unowned and not no_follower else None
    if plan:
        return _print_plan(into, targets, controllers, cloud, refusal)
    if refusal is not None:
        raise CancelRefused(refusal)

    stopped, checked, cancelled = set(), set(), set()
    try:
        for rec in targets:
            dest = os.path.join(into, rec.label)
            c = rec.cloud or cloud
            emit(f"cancel: {rec.label} ({rec.region or '(default region)'}, "
                 f"instance {rec.name})")
            # Connect first: a credential failure then leaves the job exactly as
            # it was, its follower still running.
            try:
                conn = _connect(cloud=c, region=rec.region)
            except RuntimeError as exc:
                raise credential_error("cancel", c, rec.region, exc) from exc
            ctl = controllers[rec.label]
            if ctl is not None and ctl not in checked:
                checked.add(ctl)
                if _stop(ctl[0], ctl[1], emit=emit):
                    stopped.add(ctl)
            if rec.attachable:
                got_log = _pull_log(rec.ip, rec.keyfile, os.path.join(dest, JOB_LOG))
                got_out = _fetch(rec.ip, rec.keyfile, rec.fetch, dest)
                emit(f"  fetched: job log {'yes' if got_log else 'no'}, "
                     f"~/{rec.fetch} {'yes' if got_out else 'no'}")
            _teardown(conn, rec.name, rec.server_id or None)
            out = _outcome(rec.label, rec.region, -1, CANCELLED)
            _close_job(dest, out)
            cancelled.add(rec.label)
            when = datetime.fromtimestamp(out.ended_epoch, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            emit(f"  {rec.label}: {CANCELLED} at {when}")
    finally:
        # On a partway failure too: a stopped follower's other jobs have lost
        # their follow whether or not this cancel finished.
        for ctl in sorted(stopped):
            orphaned = sorted(label for label, c in controllers.items()
                              if c == ctl and label not in cancelled)
            if orphaned:
                emit(f"note: stopped pid {ctl[0]} was also following {', '.join(orphaned)}; "
                     "their records are kept, and "
                     f"`flux-compute sweep --resume --into {into}` follows them again")
            emit(f"note: if pid {ctl[0]} was launching a jobs file, its jobs not yet "
                 "started stay pending; `sweep --resume --jobs FILE` launches them")
    return 0


def main_cancel(args) -> int:
    """The CLI branch: a refusal is one line on stderr and exit 2."""
    try:
        return run_cancel(args.into, labels=args.label or (), take_all=args.take_all,
                          cloud=args.cloud, plan=args.plan, no_follower=args.no_follower)
    except CancelRefused as exc:
        print(f"flux-compute cancel: {exc}", file=sys.stderr)
        return 2
