"""Provision a GPU instance, run work on it, and always tear it down.

Phase 1 core. `_gpu_instance` is the shared machinery: boot the resolved GPU
flavor on the NVIDIA-driver image with an ephemeral keypair and an SSH security
group (ingress locked to the caller's public IP), wait for SSH, and delete every
created resource in a finally block on success and on failure.

  smoke_test : boot, confirm the GPU (nvidia-smi), tear down.
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
import tempfile
import time
import urllib.request
import uuid
from contextlib import contextmanager

from .auth import connect
from .launch import resolve_spec

SSH_USER = "ubuntu"
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]
_DEFAULT_SMOKE = (
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader "
    "&& python3 --version"
)
_RSYNC_EXCLUDES = (
    ".git", ".jax_cache", ".pche_cache", "outputs", "__pycache__",
    ".venv", ".pytest_cache", "*.egg-info", "cc-logs",
)


def _region(conn, region):
    return (region
            or getattr(getattr(conn, "config", None), "region_name", None)
            or os.environ.get("OS_REGION_NAME") or "(unknown)")


def _name(kind):
    return f"flux-compute-{kind}-{uuid.uuid4().hex[:8]}"


def _print_plan(spec):
    cost = f"EUR {spec.est_cost_eur_hr:.2f}/hr" if spec.est_cost_eur_hr is not None else "price n/a"
    print(f"plan: {spec.flavor} [{spec.gpu_model}] / {spec.image} / {spec.network} / {cost}")


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
def _gpu_instance(conn, spec, name, keep=False):
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
            security_groups=[{"name": name}])
        server = conn.compute.wait_for_server(server, status="ACTIVE", wait=900)
        ip = _server_ipv4(server)
        print(f"ACTIVE: {server.id} @ {ip}")

        if not _wait_ssh(ip):
            raise RuntimeError(f"SSH to {ip} never opened within timeout")
        print("SSH up.")
        yield server, ip, keyfile
    finally:
        if keep and server is not None:
            print("----- --keep set: instance LEFT RUNNING (tear down manually) -----")
            print(f"  ssh {' '.join(_SSH_OPTS)} -i {keyfile} {SSH_USER}@{_server_ipv4(server)}")
            print(f"  server={server.id}  keypair={name}  sg={name}")
            return
        print("----- teardown -----")
        if server is not None:
            try:
                conn.compute.delete_server(server.id, force=True)
                conn.compute.wait_for_delete(server, wait=300)
                print("  deleted server")
            except Exception as exc:
                print(f"  server: {type(exc).__name__}: {str(exc)[:120]}")
        if keypair is not None:
            try:
                conn.compute.delete_keypair(name, ignore_missing=True)
                print("  deleted keypair")
            except Exception as exc:
                print(f"  keypair: {type(exc).__name__}: {str(exc)[:120]}")
        if sg is not None:
            # The server's port can linger for a few seconds after delete, which
            # 409s the security-group delete; retry until the port is released.
            for attempt in range(6):
                try:
                    conn.network.delete_security_group(sg.id, ignore_missing=True)
                    print("  deleted security-group")
                    break
                except Exception as exc:
                    if attempt == 5:
                        print(f"  security-group: {type(exc).__name__}: {str(exc)[:120]} (manual cleanup may be needed)")
                    else:
                        time.sleep(10)
        shutil.rmtree(tmp, ignore_errors=True)


def smoke_test(cloud=None, region=None, flavor=None) -> int:
    conn = connect(cloud=cloud, region=region)
    spec = resolve_spec(conn, _region(conn, region), flavor=flavor)
    _print_plan(spec)
    with _gpu_instance(conn, spec, _name("smoke")) as (_server, ip, keyfile):
        print("running GPU check ...")
        out = _ssh(ip, keyfile, _DEFAULT_SMOKE, timeout=120)
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
    conn = connect(cloud=cloud, region=region)
    spec = resolve_spec(conn, _region(conn, region), flavor=flavor, image=image)
    _print_plan(spec)
    with _gpu_instance(conn, spec, _name("run"), keep=keep) as (_server, ip, keyfile):
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
