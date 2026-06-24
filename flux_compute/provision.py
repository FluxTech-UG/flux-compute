"""Provision a GPU instance, run a command on it, and always tear it down.

Phase 1 core. `smoke_test` boots the resolved GPU flavor on the NVIDIA-driver
image, runs a command over SSH (default: confirm the GPU via nvidia-smi), and
deletes every resource it created (instance, keypair, security group) in a
finally block, on success and on failure alike. Nothing is left running.
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

from .auth import connect
from .launch import resolve_spec

SSH_USER = "ubuntu"
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]
_DEFAULT_REMOTE = (
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader "
    "&& python3 --version"
)


def _public_ip_cidr() -> str:
    try:
        ip = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()
        socket.inet_aton(ip)
        return f"{ip}/32"
    except Exception:
        return "0.0.0.0/0"


def _wait_ssh(host: str, port: int = 22, timeout: int = 180) -> bool:
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


def smoke_test(cloud=None, region=None, flavor=None, remote_command: str | None = None) -> int:
    conn = connect(cloud=cloud, region=region)
    reg = (region
           or getattr(getattr(conn, "config", None), "region_name", None)
           or os.environ.get("OS_REGION_NAME") or "(unknown)")
    spec = resolve_spec(conn, reg, flavor=flavor)
    name = f"flux-compute-smoke-{uuid.uuid4().hex[:8]}"
    cost = f"EUR {spec.est_cost_eur_hr:.2f}/hr" if spec.est_cost_eur_hr is not None else "price n/a"
    print(f"plan: {spec.flavor} [{spec.gpu_model}] / {spec.image} / {spec.network} / {cost}")

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
        sg = conn.network.create_security_group(name=name, description="flux-compute smoke ssh")
        conn.network.create_security_group_rule(
            security_group_id=sg.id, direction="ingress", protocol="tcp",
            port_range_min=22, port_range_max=22, remote_ip_prefix=cidr, ethertype="IPv4")
        print(f"created keypair + SG '{name}'; SSH ingress from {cidr}")

        print("booting instance ...")
        server = conn.compute.create_server(
            name=name, image_id=image.id, flavor_id=flavor_obj.id,
            networks=[{"uuid": network.id}], key_name=name,
            security_groups=[{"name": name}])
        server = conn.compute.wait_for_server(server, status="ACTIVE", wait=600)
        ip = _server_ipv4(server)
        print(f"ACTIVE: {server.id} @ {ip}")

        if not _wait_ssh(ip):
            raise RuntimeError(f"SSH to {ip} never opened within timeout")
        print("SSH up; running GPU check ...")

        out = subprocess.run(
            ["ssh", *_SSH_OPTS, "-i", keyfile, f"{SSH_USER}@{ip}", remote_command or _DEFAULT_REMOTE],
            capture_output=True, text=True, timeout=600)
        print("----- remote stdout -----")
        print(out.stdout.strip())
        if out.returncode != 0:
            print("----- remote stderr (tail) -----")
            print(out.stderr.strip()[-1500:])
            raise RuntimeError(f"remote command exited {out.returncode}")
        ok = bool(out.stdout.strip())
        print("SMOKE TEST:", "PASS" if ok else "INCONCLUSIVE")
        return 0 if ok else 1
    finally:
        print("----- teardown -----")
        steps = (
            ("server", lambda: server and conn.compute.delete_server(server.id, force=True)),
            ("server-wait", lambda: server and conn.compute.wait_for_delete(server, wait=180)),
            ("keypair", lambda: keypair and conn.compute.delete_keypair(name, ignore_missing=True)),
            ("security-group", lambda: sg and conn.network.delete_security_group(sg.id, ignore_missing=True)),
        )
        for label, fn in steps:
            try:
                fn()
                print(f"  deleted {label}")
            except Exception as exc:
                print(f"  {label}: {type(exc).__name__}: {str(exc)[:120]}")
        shutil.rmtree(tmp, ignore_errors=True)
