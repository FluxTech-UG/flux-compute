"""Bake a reusable GPU image: provision, run a setup script, snapshot, tear down.

A baked image preinstalls the heavy common stack (CUDA jaxlib + sim deps) so that
`run`/`sweep --image <name>` start at boot time instead of paying the multi-minute
per-job install. The image is consumer-agnostic: bake whatever the setup script
installs (into ~/venv by convention); consumers then `pip install -e --no-deps`
their own repos on top, per job.

NOTE (OVH, measured 2026-06-25): booting from a custom snapshot on OVH is
dominated by image staging (~12 min to reach ACTIVE), which exceeds the ~5 min
install it was meant to replace (the stock NVIDIA image boots in ~2 min). A baked
image is therefore a net loss on OVH today; prefer the stock image + per-job
install. This command and `--image` are kept (correct and cloud-general) for
if/when snapshot staging improves, or for non-OVH use.
"""
from __future__ import annotations

import os

from .auth import connect
from .launch import resolve_spec
from .provision import _gpu_instance, _name, _print_plan, _region, _rsync_up, _scp_up, _ssh


def bake(cloud=None, region=None, name=None, script=None, flavor=None,
         uploads=(), setup_timeout=2400, replace=False) -> int:
    if not name:
        raise RuntimeError("bake needs --name (the image to create)")
    if not script:
        raise RuntimeError("bake needs --script (the setup script to bake in)")

    conn = connect(cloud=cloud, region=region)
    # Always bake from the stock NVIDIA base image, never a prior baked image.
    spec = resolve_spec(conn, _region(conn, region), flavor=flavor)
    _print_plan(spec)

    existing = [i for i in conn.image.images() if i.name == name]
    if existing and not replace:
        raise RuntimeError(
            f"image {name!r} already exists ({len(existing)} found). "
            "Use --replace to rebuild, or pick another --name.")

    with _gpu_instance(conn, spec, _name("bake")) as (server, ip, keyfile):
        for local in uploads:
            base = os.path.basename(os.path.abspath(local.rstrip("/")))
            _rsync_up(local, ip, keyfile, base)
            print(f"uploaded {local} -> ~/{base}/")
        remote = os.path.basename(script)
        _scp_up(script, ip, keyfile, remote)
        print(f"running setup ~/{remote} (streaming; up to {setup_timeout}s) ...")
        res = _ssh(ip, keyfile, f"chmod +x ~/{remote} && bash -lc '~/{remote}'",
                   timeout=setup_timeout, capture=False)
        if res.returncode != 0:
            raise RuntimeError(f"setup script exited {res.returncode}; image not created")
        print(f"snapshotting to image '{name}' (this can take a few minutes) ...")
        img = conn.compute.create_server_image(server, name=name, wait=True, timeout=1200)
        print(f"image created: {name} ({img.id})")

    if existing and replace:
        for old in existing:
            try:
                conn.image.delete_image(old.id, ignore_missing=True)
                print(f"deleted old image {old.id}")
            except Exception as exc:
                print(f"could not delete old image {old.id}: {exc}")
    print(f"done. Use it with: flux-compute run --image {name} ...")
    return 0
