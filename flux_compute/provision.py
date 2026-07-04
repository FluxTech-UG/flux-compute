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
from datetime import datetime, timedelta, timezone

from .auth import connect
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
_RSYNC_EXCLUDES = (
    ".git", ".jax_cache", ".pche_cache", "outputs", "__pycache__",
    ".venv", ".pytest_cache", "*.egg-info", "cc-logs",
)


class TeardownStrandError(RuntimeError):
    """Teardown could not verifiably delete a created server; it may still be
    running and billing. Subclasses RuntimeError so every CLI path that hits a
    strand exits nonzero via the standard handler in cli.py."""


def _region(conn, region):
    return (region
            or getattr(getattr(conn, "config", None), "region_name", None)
            or os.environ.get("OS_REGION_NAME") or "(unknown)")


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
        return "0.0.0.0/0"


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


def _rsync_up(local, ip, keyfile, dest):
    excludes = []
    for e in _RSYNC_EXCLUDES:
        excludes += ["--exclude", e]
    subprocess.run(
        ["rsync", "-az", "-e", _ssh_cmd(keyfile), *excludes,
         local.rstrip("/") + "/", f"{SSH_USER}@{ip}:{dest}/"],
        check=True)


def _rsync_down(ip, keyfile, remote, local):
    subprocess.run(
        ["rsync", "-az", "-e", _ssh_cmd(keyfile),
         f"{SSH_USER}@{ip}:{remote.rstrip('/')}/", local.rstrip("/") + "/"],
        check=True)


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
    with _gpu_instance(conn, spec, _name("run"), ttl_minutes=ttl, keep=keep) as (_server, ip, keyfile):
        for local in uploads:
            base = os.path.basename(os.path.abspath(local.rstrip("/")))
            _rsync_up(local, ip, keyfile, base)
            print(f"uploaded {local} -> ~/{base}/")

        rc = 0
        if script:
            remote = os.path.basename(script)
            _scp_up(script, ip, keyfile, remote)
            print(f"running ~/{remote} (streaming; up to {exec_timeout}s) ...")
            res = _ssh(ip, keyfile, f"chmod +x ~/{remote} && bash -lc '~/{remote}'",
                       timeout=exec_timeout, capture=False)
            rc = res.returncode
            print(f"job exited {rc}")

        for spec_f in fetch:
            if ":" not in spec_f:
                raise RuntimeError(f"--fetch expects REMOTE:LOCAL (home-relative), got {spec_f!r}")
            remote, local = spec_f.split(":", 1)
            os.makedirs(local, exist_ok=True)
            _rsync_down(ip, keyfile, remote, local)
            print(f"fetched ~/{remote} -> {local}")

        return rc
