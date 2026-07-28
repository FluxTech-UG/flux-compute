"""Provision a GPU instance, run work on it, and always tear it down.

Phase 1 core. `_gpu_instance` is the shared machinery: boot the resolved flavor
(GPU or CPU) on its resolved image with an ephemeral keypair and an SSH security
group (ingress locked to the caller's public IP), wait for SSH, and delete every
created resource in a finally block on success and on failure. The server delete
is retried and verified (`_delete_server_verified`); a delete that cannot be
verified prints a stranded-instance banner with the exact cleanup commands and
raises TeardownStrandError so the CLI exits nonzero.

  smoke_test : boot, verify the device (nvidia-smi on a GPU, a boot + remote-exec
               check on a CPU flavor), tear down.
  run_job    : boot, rsync repos up, run an uploaded job script, fetch artifacts
               back, tear down.

run_job is consumer-agnostic: the consumer supplies the upload dirs, the job
script (its own setup + run commands), and the artifact paths. The package owns
provisioning; the consumer owns the script, so a consumer can iterate on its
bootstrap without editing this package.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import detach
from .auth import connect, resolve_region_name
from .launch import resolve_spec

# Provenance + TTL metadata stamped on every created server. `flux-compute
# reap` auto-takes only instances that carry the flux_created_by stamp AND are
# past their stamped expiry; a flux_keep stamp (from --keep) is never
# auto-taken; name-prefix matches without the stamp are report-only.
FLUX_CREATED_BY_KEY = "flux_created_by"
FLUX_CREATED_BY = "flux-compute"
FLUX_EXPIRES_KEY = "flux_expires_at"
FLUX_KEEP_KEY = "flux_keep"
FLUX_NAME_PREFIX = "flux-compute-"
# Minimum TTL margin (minutes) beyond a run's wall cap, for boot + upload +
# fetch + teardown overhead; ttl_minutes_for widens it to 25% of long caps.
TTL_MARGIN_MIN = 30

# Detach-and-poll parameters (see detach.py). Each poll is a fresh short SSH, so
# the interval trades local log latency against SSH churn across a wide fan-out.
# The local deadline is the remote runaway cap plus a grace covering the
# timeout -> SIGKILL -> rc-write tail and one poll interval, so a job killed by
# its own remote cap resolves via job.rc (rc 124/137) rather than tripping the
# local abort. The remote cap and `flux-compute reap`'s TTL stamp remain the two
# laptop-independent backstops.
POLL_INTERVAL_SWEEP_S = 15
POLL_INTERVAL_RUN_S = 5
CONNECT_TIMEOUT_S = 30
LOCAL_GRACE_S = 120
BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 60.0
# A resume after a full restart gets at least this long to reach a job whose
# original deadline may already have passed while the orchestrator was down.
RESUME_MIN_DEADLINE_S = 300

# Remote exit codes of the `timeout --signal=TERM --kill-after=N CAP ...` wrapper
# the detached launcher wraps every job in (detach.launcher_script).
#
# 124 is unambiguous: `timeout` TERM'd the job at its cap.
# 137 is 128+SIGKILL(9) and is AMBIGUOUS -- `timeout`'s own --kill-after
# escalation raises it, but so does the kernel OOM-killer, and the two mean
# opposite things. Reading every 137 as "job timed out (remote cap)" is what sent
# an OOM investigation chasing phantom slow jobs; a 137 far short of its cap was
# killed by something else, and saying so is the whole point of _classify_exit.
RC_CAP_TIMEOUT = 124
RC_SIGKILL = 137
# A genuine cap kill lands at (or just past) the cap. Below this fraction of the
# cap, "the cap did it" is not a tenable explanation.
CAP_KILL_MIN_FRACTION = 0.9
# Kernel-log markers the OOM-killer leaves behind.
_OOM_MARKERS = ("out of memory", "oom-kill", "oom_kill", "oom_reaper", "killed process")
# Read the kernel ring buffer however this host allows it: dmesg is often
# restricted to root (kernel.dmesg_restrict), and journald may or may not be the
# one carrying it. Best-effort by construction -- an empty read is "not
# confirmed", never "confirmed absent".
_KERNEL_LOG_CMD = (
    "{ sudo -n dmesg -T 2>/dev/null || sudo -n dmesg 2>/dev/null "
    "|| dmesg -T 2>/dev/null || dmesg 2>/dev/null "
    "|| sudo -n journalctl -k -n 400 --no-pager 2>/dev/null "
    "|| journalctl -k -n 400 --no-pager 2>/dev/null "
    "|| true; } | tail -n 400"
)


def ttl_minutes_for(cap_minutes):
    """TTL for a run with the given wall cap: cap + max(30 min, 25% of the cap).

    Deliberately generous, never tight: a reap that fires early once destroys
    trust in the whole mechanism, while firing late costs cents. The margin
    covers boot, rsync, install and fetch overhead beyond the remote-exec cap.
    """
    return int(cap_minutes) + max(TTL_MARGIN_MIN, -(-int(cap_minutes) // 4))


def ttl_metadata(ttl_minutes, keep=False, now=None):
    """The metadata stamped on every created server: provenance, expiry, and
    (for --keep runs) the keep flag that exempts it from auto-reap."""
    now = now or datetime.now(timezone.utc)
    expires = now + timedelta(minutes=ttl_minutes)
    md = {
        FLUX_CREATED_BY_KEY: FLUX_CREATED_BY,
        FLUX_EXPIRES_KEY: expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if keep:
        md[FLUX_KEEP_KEY] = "true"
    return md

SSH_USER = "ubuntu"
# What `_public_ip_cidr` returns when it cannot read the caller's address. At
# launch it means "could not lock the group down", and it is NEVER a value to
# repair an existing group with: adding it would open SSH to the whole internet
# on the strength of a failed lookup.
UNKNOWN_CIDR = "0.0.0.0/0"
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]
_GPU_SMOKE = (
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader "
    "&& python3 --version"
)
# A CPU flavor has no GPU, so nvidia-smi could never pass; verify boot + remote
# exec instead (host identity, core count, a working python3).
_CPU_SMOKE = (
    "python3 -c 'import platform; print(platform.platform())' "
    "&& echo \"cores: $(nproc)\" && python3 --version"
)
# Directories never worth uploading, plus one hazard directory: `cloud-sweep/`
# is where a live sweep persists its per-job `.flux_attach` records (sweep.py),
# and those records are created and deleted *while the fleet runs*. Uploading a
# repo that contains a live fleet's `cloud-sweep/` lets those records vanish
# mid-transfer, so rsync exits 24 and (with check=True) aborts the whole launch
# — this stranded two fleets. Excluding it removes the self-race at the source;
# `_rsync_up` additionally tolerates exit 24 as belt-and-suspenders.
#
# `.flux_attach` is the general form of that hazard and the one that actually
# holds: `--into` is configurable, so excluding only the default `cloud-sweep`
# left every non-default results tree exposed. The attach dirs are excluded by
# name wherever they live, and callers additionally pass their own `--into` dir
# via `extra_excludes` so fetched results never re-enter an upload either.
_RSYNC_EXCLUDES = (
    ".git", ".jax_cache", ".pche_cache", "outputs", "__pycache__",
    ".venv", ".pytest_cache", "*.egg-info", "cc-logs", "cloud-sweep",
    ".flux_attach",
)


class TeardownStrandError(RuntimeError):
    """Teardown could not verifiably delete a created server; it may still be
    running and billing. Subclasses RuntimeError so every CLI path that hits a
    strand exits nonzero via the standard handler in cli.py."""


def _region(conn, region):
    return resolve_region_name(conn, region)


def _cloud_name(conn):
    return getattr(getattr(conn, "config", None), "name", None) or "<cloud>"


def _delete_server_verified(conn, server, retries=3, wait=300, retry_delay=10):
    """Delete `server` and verify it is actually gone, retrying on failure.

    `wait_for_delete` is the verification: it returns only once the server has
    disappeared from the compute API. Any failure after all retries raises
    TeardownStrandError; a live billing instance is never reduced to a quiet
    log line.
    """
    last = None
    for attempt in range(1, retries + 1):
        try:
            conn.compute.delete_server(server.id, force=True)
            conn.compute.wait_for_delete(server, wait=wait)
            return
        except Exception as exc:
            last = exc
            if attempt < retries:
                print(f"  server delete attempt {attempt}/{retries} failed "
                      f"({type(exc).__name__}: {str(exc)[:80]}); retrying ...")
                time.sleep(retry_delay)
    raise TeardownStrandError(
        f"stranded: server {server.id} could not be verifiably deleted after "
        f"{retries} attempts (last: {type(last).__name__}: {str(last)[:120]})"
    ) from last


def _delete_sg_with_retry(conn, sg_id, attempts=6, delay=10):
    """Delete a security group, retrying 409s: the server's port can linger for
    a few seconds after the server delete. Returns True when deleted."""
    for attempt in range(attempts):
        try:
            conn.network.delete_security_group(sg_id, ignore_missing=True)
            print("  deleted security-group")
            return True
        except Exception as exc:
            if attempt == attempts - 1:
                print(f"  security-group: {type(exc).__name__}: {str(exc)[:120]} "
                      "(manual cleanup may be needed)")
            else:
                time.sleep(delay)
    return False


def _stranded_banner(cloud, name, server_id, reason):
    """Print an unmissable multi-line stranded-instance banner to stderr with
    the exact cleanup commands. Printed in addition to (never instead of) the
    TeardownStrandError that makes the CLI exit nonzero."""
    bar = "!" * 76
    print("\n".join([
        "",
        bar,
        "!!  STRANDED INSTANCE : TEARDOWN FAILED",
        f"!!  server  : {server_id}",
        f"!!  name    : {name}",
        f"!!  reason  : {reason}",
        "!!  This instance may still be RUNNING and BILLING. Clean up NOW:",
        f"!!    openstack --os-cloud {cloud} server delete {server_id}",
        f"!!    openstack --os-cloud {cloud} keypair delete {name}",
        f"!!    openstack --os-cloud {cloud} security group delete {name}",
        "!!  Then verify nothing is left:",
        f"!!    openstack --os-cloud {cloud} server list",
        bar,
        "",
    ]), file=sys.stderr)


def _name(kind):
    return f"flux-compute-{kind}-{uuid.uuid4().hex[:8]}"


def _print_plan(spec):
    cost = f"EUR {spec.est_cost_eur_hr:.2f}/hr" if spec.est_cost_eur_hr is not None else "price n/a"
    print(f"plan: {spec.flavor} [{spec.gpu_model or 'CPU'}] / {spec.image} / {spec.network} / {cost}")


def _smoke_command(gpu_model):
    """Choose the smoke check by device: a GPU card gets nvidia-smi, a CPU flavor
    a boot + remote-exec check. Returns (label, command)."""
    if gpu_model is not None:
        return "GPU", _GPU_SMOKE
    return "CPU", _CPU_SMOKE


def _public_ip_cidr():
    try:
        ip = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()
        socket.inet_aton(ip)
        return f"{ip}/32"
    except Exception:
        return UNKNOWN_CIDR


def current_ingress_cidr():
    """The /32 an instance's security group must admit for this caller to SSH in,
    or None when the address could not be read.

    The ONE public-IP read in the package: `_public_ip_cidr` is what the launch
    path uses to build a new group's original ingress rule, so every later repair
    must ask the same question the same way, or it would compare against a
    different notion of "here". A caller repairing many groups at once resolves
    this once and passes it down rather than re-reading per group.
    """
    cidr = _public_ip_cidr()
    return None if cidr == UNKNOWN_CIDR else cidr


def _wait_ssh(host, port=22, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except OSError:
            time.sleep(5)
    return False


def _server_ipv4(server):
    for _net, addrs in (server.addresses or {}).items():
        for a in addrs:
            if a.get("version") == 4:
                return a["addr"]
    return None


def _ssh_cmd(keyfile):
    return "ssh " + " ".join(_SSH_OPTS) + f" -i {keyfile}"


def _ssh(ip, keyfile, command, timeout=600, capture=True):
    args = ["ssh", *_SSH_OPTS, "-i", keyfile, f"{SSH_USER}@{ip}", command]
    if capture:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(args, timeout=timeout)


def _scp_up(local, ip, keyfile, remote):
    subprocess.run(["scp", *_SSH_OPTS, "-i", keyfile, local, f"{SSH_USER}@{ip}:{remote}"], check=True)


def _scp_down(ip, keyfile, remote, local, timeout=120):
    """Copy a single home-relative remote file down to a local path. Used for the
    authoritative final pull of ~/job.out into the local job.log."""
    return subprocess.run(
        ["scp", *_SSH_OPTS, "-i", keyfile, f"{SSH_USER}@{ip}:{remote}", local],
        capture_output=True, text=True, timeout=timeout)


def parse_upload_spec(spec):
    """Parse one ``--upload`` value into ``(local_src, remote_dest)``.

    Two forms, and the mapping form is why this exists:

      ``DIR``           -> uploaded to ``~/<basename of DIR>`` (unchanged behavior)
      ``SRC:DEST``      -> uploaded to ``~/DEST``, whatever SRC is named

    Without the mapping form, the only way to land a directory under a different
    remote name was to build a local symlink named like the destination and
    upload that -- a workaround at the call site for a missing feature here.

    Split on the LAST colon: DEST is a remote home-relative name and never
    contains one, while a local path may. A DEST that is absolute, escapes home
    (``..``), or is empty is refused, as is a source that does not exist -- so a
    mistyped spec fails at parse time with the remedy, rather than rsyncing
    nothing or writing outside the home dir.
    """
    text = str(spec)
    src, dest = (text.rsplit(":", 1) if ":" in text else (text, None))
    src = src.rstrip("/") or "/"
    if not os.path.isdir(src):
        hint = (f" (parsed as SRC={src!r} DEST={dest!r}; pass a bare path if the "
                "colon is part of the directory name)" if dest is not None else "")
        raise RuntimeError(f"--upload source {src!r} is not an existing directory{hint}")
    if dest is None:
        return src, os.path.basename(os.path.abspath(src))
    dest = dest.strip()
    # Checked BEFORE any normalization, so an absolute path cannot be normalized
    # into looking relative: the destination is joined onto the remote home dir.
    if not dest or dest.startswith("/") or ".." in dest.split("/"):
        raise RuntimeError(
            f"--upload destination {dest!r} must be a non-empty path relative to the "
            "remote home dir (no leading '/', no '..')")
    return src, dest.rstrip("/")


def _rsync_up(local, ip, keyfile, dest, extra_excludes=()):
    excludes = []
    for e in (*_RSYNC_EXCLUDES, *extra_excludes):
        excludes += ["--exclude", e]
    res = subprocess.run(
        ["rsync", "-az", "-e", _ssh_cmd(keyfile), *excludes,
         local.rstrip("/") + "/", f"{SSH_USER}@{ip}:{dest}/"])
    if res.returncode == 24:
        # rsync exit 24 = "some files vanished before they could be transferred"
        # — a source file (e.g. a live fleet's churning `.flux_attach` record)
        # disappeared mid-copy. That is benign for a launch upload: the transfer
        # otherwise completed. Warn and continue rather than aborting the launch
        # (which used to strand the freshly-booted VM).
        print(f"WARNING: rsync of {local} reported vanished source files (exit 24); "
              "continuing (files that disappeared mid-copy were skipped).",
              file=sys.stderr)
        return
    if res.returncode != 0:
        raise RuntimeError(
            f"rsync upload of {local} -> {dest} failed (exit {res.returncode}); "
            "check the local path and SSH connectivity to the instance.")


def _rsync_down(ip, keyfile, remote, local):
    subprocess.run(
        ["rsync", "-az", "-e", _ssh_cmd(keyfile),
         f"{SSH_USER}@{ip}:{remote.rstrip('/')}/", local.rstrip("/") + "/"],
        check=True)


def rsync_down_best_effort(ip, keyfile, remote, local, _rsync=None):
    """Fetch artifacts without letting a failure abort the caller. Returns True
    when the fetch succeeded.

    This is the fetch used on the paths where the job did NOT finish cleanly --
    a local-deadline abort, a job killed by its cap or the OOM-killer. Those runs
    still have partial results on disk (checkpoints, a resumable ledger, the
    rows computed before the kill), and the VM is about to be torn down, so the
    partial fetch is the last chance to keep them: skipping it turned a partial
    result into a total loss and destroyed a fleet's worth of work once. The
    remote dir may legitimately not exist yet, so failure here is reported and
    swallowed, never raised.
    """
    fetch = _rsync or _rsync_down
    try:
        fetch(ip, keyfile, remote, local)
        return True
    except Exception as exc:
        print(f"  partial fetch of ~/{remote} failed "
              f"({type(exc).__name__}: {str(exc)[:100]}); nothing recovered.",
              file=sys.stderr)
        return False


# --- why did the remote job die? (rc=137 is ambiguous) ------------------------

@dataclass(frozen=True)
class OomProbe:
    """A read of the VM's kernel log looking for OOM-killer evidence.

    ``confirmed`` False means "no evidence found", never "definitely not an OOM":
    the log may be root-restricted, rotated, or on a host whose journal was not
    reachable. Absence is reported as unknown, not as innocence.
    """

    confirmed: bool
    summary: str = ""
    read_ok: bool = False


def oom_evidence(kernel_log):
    """Pure: scan a kernel-log excerpt for OOM-killer markers -> ``OomProbe``."""
    text = kernel_log or ""
    hits = [ln.strip() for ln in text.splitlines()
            if any(m in ln.lower() for m in _OOM_MARKERS)]
    return OomProbe(confirmed=bool(hits), summary=(hits[-1][:200] if hits else ""),
                    read_ok=bool(text.strip()))


def probe_oom_kill(ip, keyfile, _ssh=_ssh):
    """Read the VM's kernel log over SSH and look for the OOM-killer. Best-effort:
    any failure returns an unconfirmed probe rather than raising, because this runs
    on the teardown path where the job's own outcome must survive."""
    try:
        res = _ssh(ip, keyfile, _KERNEL_LOG_CMD, timeout=60, capture=True)
    except Exception:
        return OomProbe(confirmed=False)
    return oom_evidence(getattr(res, "stdout", "") or "")


def looks_like_cap_kill(elapsed_s, cap_seconds, *, fraction=CAP_KILL_MIN_FRACTION):
    """Did this run live long enough for its own wall cap to be the killer?
    Returns True / False, or None when there is no basis to judge."""
    if elapsed_s is None or not cap_seconds or cap_seconds <= 0:
        return None
    return elapsed_s >= fraction * cap_seconds


def _elapsed_phrase(elapsed_s, cap_seconds):
    if elapsed_s is None:
        return ""
    cap = f" of {int(cap_seconds)}s cap" if cap_seconds else ""
    return f" at {int(elapsed_s)}s{cap}"


def classify_exit(rc, *, elapsed_s=None, cap_seconds=None, oom=None):
    """Explain a remote job's return code honestly -> a short status string.

    The rule that matters: a 137 is only called a cap timeout when the run
    actually reached its cap. An OOM-killed job that died at 8 minutes of a
    60-minute cap is reported as an OOM kill (or as an unexplained SIGKILL when
    the kernel log could not confirm it), never as a timeout -- misreporting it
    sends the next session tuning wall caps for a memory problem.
    """
    if rc == 0:
        return "ok"
    if rc == RC_CAP_TIMEOUT:
        return "job timed out (remote cap)"
    if rc != RC_SIGKILL:
        return "job nonzero"

    where = _elapsed_phrase(elapsed_s, cap_seconds)
    if oom is not None and oom.confirmed:
        detail = f": {oom.summary}" if oom.summary else ""
        return f"OOM-killed (rc=137, kernel oom-killer confirmed{where}){detail}"
    at_cap = looks_like_cap_kill(elapsed_s, cap_seconds)
    if at_cap:
        return f"job timed out (remote cap; SIGKILL after TERM{where})"
    if at_cap is False:
        why = ("no oom-killer evidence in the kernel log" if oom is not None and oom.read_ok
               else "kernel log unavailable")
        return (f"killed (rc=137, SIGKILL{where}, far short of the cap) - "
                f"cause unknown: {why}")
    return "killed (rc=137, SIGKILL) - cause unknown (no elapsed/cap basis to judge)"


def explain_remote_kill(ip, keyfile, rc, *, elapsed_s, cap_seconds, _ssh=_ssh):
    """Classify a finished job, probing the VM's kernel log first when the return
    code is an ambiguous sub-cap SIGKILL. Must be called BEFORE teardown -- the
    evidence dies with the instance."""
    oom = None
    if rc == RC_SIGKILL and looks_like_cap_kill(elapsed_s, cap_seconds) is not True:
        oom = probe_oom_kill(ip, keyfile, _ssh=_ssh)
    return classify_exit(rc, elapsed_s=elapsed_s, cap_seconds=cap_seconds, oom=oom)


@contextmanager
def _gpu_instance(conn, spec, name, ttl_minutes, keep=False):
    image = conn.compute.find_image(spec.image)
    flavor_obj = conn.compute.find_flavor(spec.flavor)
    network = conn.network.find_network(spec.network)

    tmp = tempfile.mkdtemp(prefix="flux-compute-")
    keyfile = os.path.join(tmp, "id_ed25519")
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", keyfile, "-N", "", "-q"], check=True)
    with open(keyfile + ".pub") as fh:
        pubkey = fh.read().strip()

    keypair = sg = server = None
    try:
        keypair = conn.compute.create_keypair(name=name, public_key=pubkey)
        cidr = _public_ip_cidr()
        sg = conn.network.create_security_group(name=name, description="flux-compute ssh")
        conn.network.create_security_group_rule(
            security_group_id=sg.id, direction="ingress", protocol="tcp",
            port_range_min=22, port_range_max=22, remote_ip_prefix=cidr, ethertype="IPv4")
        print(f"created keypair + SG '{name}'; SSH ingress from {cidr}")

        print("booting instance ...")
        server = conn.compute.create_server(
            name=name, image_id=image.id, flavor_id=flavor_obj.id,
            networks=[{"uuid": network.id}], key_name=name,
            security_groups=[{"name": name}],
            metadata=ttl_metadata(ttl_minutes, keep=keep))
        server = conn.compute.wait_for_server(server, status="ACTIVE", wait=900)
        ip = _server_ipv4(server)
        print(f"ACTIVE: {server.id} @ {ip}")

        if not _wait_ssh(ip):
            raise RuntimeError(f"SSH to {ip} never opened within timeout")
        print("SSH up.")
        yield server, ip, keyfile
    finally:
        if keep and server is not None:
            # No `return` here: a return inside finally swallows any with-body
            # exception, letting a failed --keep run exit 0 with the instance
            # left running. Print the banner and fall through so an in-flight
            # exception propagates. The tmp keydir is deliberately kept: the
            # ssh line below needs it.
            print("----- --keep set: instance LEFT RUNNING (tear down manually) -----")
            print(f"  ssh {' '.join(_SSH_OPTS)} -i {keyfile} {SSH_USER}@{_server_ipv4(server)}")
            print(f"  server={server.id}  keypair={name}  sg={name}")
            print("  it is stamped flux_keep=true: `flux-compute reap` lists it with its")
            print("  accrued cost but never auto-deletes it; tear it down when done.")
        else:
            print("----- teardown -----")
            strand = None
            if server is not None:
                try:
                    _delete_server_verified(conn, server)
                    print("  deleted server (verified gone)")
                except TeardownStrandError as exc:
                    strand = exc
                    _stranded_banner(_cloud_name(conn), name, server.id, str(exc))
            if keypair is not None:
                try:
                    conn.compute.delete_keypair(name, ignore_missing=True)
                    print("  deleted keypair")
                except Exception as exc:
                    print(f"  keypair: {type(exc).__name__}: {str(exc)[:120]}")
            if sg is not None:
                _delete_sg_with_retry(conn, sg.id)
            shutil.rmtree(tmp, ignore_errors=True)
            if strand is not None:
                # Propagate after the rest of the cleanup ran: the banner above
                # has the commands; this raise makes every CLI path exit
                # nonzero. If the with-body also raised, that exception rides
                # along as __context__.
                raise strand


# --- detach-and-poll: survive the launching SSH dropping (laptop sleep) --------

def _make_poll_runner(ip, keyfile, connect_timeout=CONNECT_TIMEOUT_S, _ssh=_ssh):
    """Build the SSH-backed poll runner the detach loop calls each iteration.

    A fresh, short SSH per poll. An ssh transport failure (exit 255: connection
    refused/reset/timed out -- the laptop asleep or a network flap) or a subprocess
    timeout is surfaced as a retryable `PollAttempt.failed`, never raised; any
    connected read (even if the remote poll command itself exited nonzero) still
    carries the status trailer and is handed to the parser.
    """
    def run_poll(next_byte):
        cmd = detach.poll_command(next_byte)
        try:
            res = _ssh(ip, keyfile, cmd, timeout=connect_timeout, capture=True)
        except subprocess.TimeoutExpired:
            return detach.PollAttempt.failed(f"ssh timed out after {connect_timeout}s")
        except Exception as exc:
            return detach.PollAttempt.failed(f"{type(exc).__name__}: {str(exc)[:80]}")
        if res.returncode == 255:
            return detach.PollAttempt.failed(
                f"ssh transport error: {(res.stderr or '').strip()[:80]}")
        return detach.PollAttempt.connected(res.stdout)
    return run_poll


def _launch_detached(ip, keyfile, remote_script, cap_seconds, env_prefix="",
                     launch_timeout=90, _ssh=_ssh):
    """Upload the generated launcher and run it to start the job detached.

    Short and non-blocking: the launcher spawns the setsid'd job (all descriptors
    off the SSH channel) and returns at once, so this SSH closes cleanly and the
    job keeps running through any later disconnect. env_prefix carries the caller's
    `$FLUX_LABEL=... $FLUX_JOB=...` assignments into the job's environment. Returns
    the launcher's stdout; raises on a nonzero launch.
    """
    script_text = detach.launcher_script(remote_script, cap_seconds)
    tmp = tempfile.mkdtemp(prefix="flux-launch-")
    try:
        path = os.path.join(tmp, detach.REMOTE_LAUNCHER)
        with open(path, "w") as fh:
            fh.write(script_text)
        _scp_up(path, ip, keyfile, detach.REMOTE_LAUNCHER)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    prefix = (env_prefix + " ") if env_prefix else ""
    res = _ssh(ip, keyfile,
               f"chmod +x ~/{detach.REMOTE_LAUNCHER} && {prefix}bash ~/{detach.REMOTE_LAUNCHER}",
               timeout=launch_timeout, capture=True)
    if res.returncode != 0:
        raise RuntimeError(
            "detached launch failed "
            f"(rc={res.returncode}): {(res.stderr or res.stdout or '').strip()[:200]}")
    return res.stdout


def _sg_allows_ssh_from(conn, sg, cidr):
    """Does this security group already permit SSH ingress from `cidr`?"""
    for rule in conn.network.security_group_rules(security_group_id=sg.id):
        if getattr(rule, "direction", None) != "ingress":
            continue
        if getattr(rule, "remote_ip_prefix", None) != cidr:
            continue
        lo = getattr(rule, "port_range_min", None)
        hi = getattr(rule, "port_range_max", None)
        if lo is None or hi is None or (lo <= 22 <= hi):
            return True
    return False


@dataclass(frozen=True)
class IngressCheck:
    """What `heal_ssh_ingress` found, and a line fit to print about it.

    The status is carried separately because the three non-repairing outcomes are
    different news: ``open`` means the group was checked and is fine, while
    ``unknown-ip`` and ``no-group`` mean the check could not be completed at all.
    A caller that cannot tell them apart reports a healthy group when nothing was
    verified -- the exact ambiguity that leaves an operator staring at a silent
    log wondering whether the fleet is reachable.
    """

    status: str          # "healed" | "open" | "unknown-ip" | "no-group"
    message: str

    def __bool__(self):
        """Truthy only when a rule was actually added, so ``if check:`` reads as
        "did we change anything"."""
        return self.status == "healed"


def heal_ssh_ingress(conn, sg_name, cidr=None):
    """Open SSH ingress for the caller's current address if the group lacks it.

    Every instance's security group is created allowing SSH from ONE address --
    the launcher's public IP at boot (`_public_ip_cidr`). If that address changes
    while a fleet is running (a different network, a VPN toggling, an ISP
    re-lease), every SSH to every job starts timing out at once: the fleet is
    alive, the work is fine, and nothing can be collected. Re-opening ingress for
    the new address is the fix, and it is the one repair worth attempting
    automatically, because the failure is global and self-inflicted rather than a
    property of any job.

    `cidr` defaults to a fresh read (`current_ingress_cidr`); a caller sweeping
    many groups resolves it once and passes it in. Additive only: the stale rule
    is left in place, so a flapping address does not lock anyone out, and a
    repeat call is a no-op rather than a pile of duplicate rules. Returns an
    `IngressCheck`; an unexpected API error is NOT caught here -- the caller
    decides whether its own contract can absorb one.
    """
    cidr = cidr or current_ingress_cidr()
    if cidr is None or cidr == UNKNOWN_CIDR:
        # The public-IP read failed. Never widen a group on a guess -- report the
        # gap and let the SSH attempt itself say what the real state is.
        return IngressCheck(
            "unknown-ip",
            "could not read this machine's public IP, so SSH ingress could not be "
            f"checked on security group {sg_name}; continuing anyway")
    sg = conn.network.find_security_group(sg_name)
    if sg is None:
        return IngressCheck(
            "no-group", f"security group {sg_name} no longer exists")
    if _sg_allows_ssh_from(conn, sg, cidr):
        return IngressCheck(
            "open", f"SSH ingress from {cidr} is already open on {sg_name}")
    conn.network.create_security_group_rule(
        security_group_id=sg.id, direction="ingress", protocol="tcp",
        port_range_min=22, port_range_max=22, remote_ip_prefix=cidr,
        ethertype="IPv4")
    return IngressCheck(
        "healed",
        f"public IP moved: opened SSH ingress from {cidr} on security group {sg_name}")


def make_stuck_handler(conn, sg_name, *, label=None, emit=None, now=None):
    """Build the poll loop's `on_stuck` callback: say the host is unreachable, and
    try the one repair that fixes it fleet-wide (`heal_ssh_ingress`).

    Visibility is half the point. A sustained SSH blackout used to look exactly
    like a healthy long job -- no output either way -- so a fleet sat frozen for
    hours before anyone suspected it. This prints an escalating, timestamped
    "SSH unreachable since ..." line instead.
    """
    emit = emit or (lambda msg: print(msg, file=sys.stderr))
    clock = now or time.time

    def _on_stuck(n_failures, seconds_unreachable):
        since = time.strftime("%H:%M:%S", time.localtime(clock() - seconds_unreachable))
        tag = f"[{label}] " if label else ""
        emit(f"WARNING: {tag}SSH unreachable since {since} "
             f"({n_failures} consecutive failed polls, {int(seconds_unreachable)}s); "
             "the job itself may be fine and still running -- checking whether our "
             "public IP moved out of the instance's allowed range ...")
        try:
            check = heal_ssh_ingress(conn, sg_name)
        except Exception as exc:
            emit(f"         {tag}ingress re-check failed "
                 f"({type(exc).__name__}: {str(exc)[:100]}); still retrying.")
            return
        if check:
            emit(f"         {tag}{check.message}; polling should recover.")
        elif check.status == "open":
            emit(f"         {tag}public IP unchanged and ingress still open; "
                 "the instance or the network is the cause. Retrying until the "
                 "local deadline.")
        else:
            emit(f"         {tag}{check.message}. Retrying until the local deadline.")

    return _on_stuck


def follow_detached_job(ip, keyfile, cap_seconds, *, deadline_s, poll_interval,
                        on_chunk=None, on_status=None, on_stuck=None, _ssh=_ssh):
    """Poll an already-launched detached job to completion (or a local abort).

    Returns the `PollOutcome`. Reconnection-tolerant: a failed poll is retried with
    backoff and never fatal; only `deadline_s` (measured from the first poll)
    aborts. `on_chunk` receives new job.out fragments for a live log/stream;
    `on_stuck` (see `make_stuck_handler`) escalates a sustained SSH blackout.
    """
    run_poll = _make_poll_runner(ip, keyfile, _ssh=_ssh)
    return detach.poll_until_done(
        run_poll, poll_interval=poll_interval, deadline_s=deadline_s,
        backoff_base=BACKOFF_BASE_S, backoff_max=BACKOFF_MAX_S,
        on_chunk=on_chunk, on_status=on_status, on_stuck=on_stuck)


def pull_job_log(ip, keyfile, log_path, _ssh=_ssh):
    """Pull the full remote ~/job.out down to log_path (the authoritative final
    job log). Best-effort: returns True on success, False if the file could not be
    fetched (e.g. the job never wrote it). Never raises -- a missing log must not
    mask the job's own return code or block teardown."""
    try:
        res = _scp_down(ip, keyfile, detach.REMOTE_OUT, log_path)
        return res.returncode == 0
    except Exception:
        return False


def _server_by_name_or_id(conn, name, server_id=None):
    """Find a server by id (preferred) or name, or None if it is already gone."""
    for lookup in (
        lambda: conn.compute.get_server(server_id) if server_id else None,
        lambda: conn.compute.find_server(name, ignore_missing=True),
    ):
        try:
            server = lookup()
        except Exception:
            server = None
        if server is not None:
            return server
    return None


def teardown_by_name(conn, name, server_id=None):
    """Delete a flux-compute instance and its same-named keypair + security group
    by name -- the standalone teardown used on resume, where the `_gpu_instance`
    context manager that normally owns teardown is gone (a fresh process). Verified
    and retried exactly like the in-context path; prints the stranded-instance
    banner and raises `TeardownStrandError` if the server cannot be confirmed gone.
    """
    print("----- teardown (resume) -----")
    server = _server_by_name_or_id(conn, name, server_id)
    if server is not None:
        try:
            _delete_server_verified(conn, server)
            print("  deleted server (verified gone)")
        except TeardownStrandError as exc:
            _stranded_banner(_cloud_name(conn), name, getattr(server, "id", server_id), str(exc))
            raise
    else:
        print(f"  server {name} already gone")
    try:
        conn.compute.delete_keypair(name, ignore_missing=True)
        print("  deleted keypair")
    except Exception as exc:
        print(f"  keypair: {type(exc).__name__}: {str(exc)[:120]}")
    sg = conn.network.find_security_group(name)
    if sg is not None:
        _delete_sg_with_retry(conn, sg.id)


def smoke_test(cloud=None, region=None, flavor=None) -> int:
    from .reap import warn_strays  # function-level: reap imports this module

    conn = connect(cloud=cloud, region=region)
    warn_strays(conn)
    spec = resolve_spec(conn, _region(conn, region), flavor=flavor)
    _print_plan(spec)
    label, check = _smoke_command(spec.gpu_model)
    with _gpu_instance(conn, spec, _name("smoke"),
                       ttl_minutes=ttl_minutes_for(2)) as (_server, ip, keyfile):
        print(f"running {label} check ...")
        out = _ssh(ip, keyfile, check, timeout=120)
        print("----- remote stdout -----")
        print(out.stdout.strip())
        if out.returncode != 0:
            print("----- remote stderr (tail) -----")
            print(out.stderr.strip()[-1500:])
            raise RuntimeError(f"remote command exited {out.returncode}")
        ok = bool(out.stdout.strip())
        print("SMOKE TEST:", "PASS" if ok else "INCONCLUSIVE")
        return 0 if ok else 1


def run_job(cloud=None, region=None, flavor=None, uploads=(), script=None,
            fetch=(), keep=False, exec_timeout=2400, image=None) -> int:
    from .reap import warn_strays  # function-level: reap imports this module

    conn = connect(cloud=cloud, region=region)
    warn_strays(conn)
    spec = resolve_spec(conn, _region(conn, region), flavor=flavor, image=image)
    _print_plan(spec)
    ttl = ttl_minutes_for(-(-exec_timeout // 60))
    upload_pairs = [parse_upload_spec(u) for u in uploads]
    name = _name("run")
    with _gpu_instance(conn, spec, name, ttl_minutes=ttl, keep=keep) as (_server, ip, keyfile):
        for local, dest in upload_pairs:
            _rsync_up(local, ip, keyfile, dest)
            print(f"uploaded {local} -> ~/{dest}/")

        rc = 0
        deadline_hit = False
        if script:
            remote = os.path.basename(script)
            _scp_up(script, ip, keyfile, remote)
            cap = exec_timeout
            print(f"running ~/{remote} detached (remote cap {cap}s); "
                  "streaming output as it lands, surviving local disconnection ...")
            started = time.time()
            _launch_detached(ip, keyfile, remote, cap)

            def _emit(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()

            outcome = follow_detached_job(
                ip, keyfile, cap,
                deadline_s=cap + LOCAL_GRACE_S, poll_interval=POLL_INTERVAL_RUN_S,
                on_chunk=_emit,
                on_stuck=make_stuck_handler(conn, name))
            elapsed = time.time() - started
            if outcome.reason == "deadline":
                # Do NOT raise yet: the artifacts below are the only surviving
                # trace of the work, and the instance is about to be torn down.
                deadline_hit = True
                rc = -1
                print(f"\nlocal deadline reached without the job finishing (remote cap {cap}s "
                      "+ grace); fetching whatever exists before teardown ...", file=sys.stderr)
            else:
                rc = outcome.rc
                print(f"\njob exited {rc}: "
                      + explain_remote_kill(ip, keyfile, rc, elapsed_s=elapsed,
                                            cap_seconds=cap))

        for spec_f in fetch:
            if ":" not in spec_f:
                raise RuntimeError(f"--fetch expects REMOTE:LOCAL (home-relative), got {spec_f!r}")
            remote, local = spec_f.split(":", 1)
            os.makedirs(local, exist_ok=True)
            if rc == 0:
                _rsync_down(ip, keyfile, remote, local)
                print(f"fetched ~/{remote} -> {local}")
            elif rsync_down_best_effort(ip, keyfile, remote, local):
                print(f"fetched ~/{remote} -> {local} (PARTIAL: the job did not finish cleanly)")

        if deadline_hit:
            raise RuntimeError(
                f"local deadline reached without the job finishing (remote cap {exec_timeout}s "
                "+ grace); ~/job.rc never appeared -- the VM may be wedged (torn down now). "
                "Any artifacts that existed were fetched first.")
        return rc
