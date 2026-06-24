"""Push artifacts to OVH Object Storage (Swift) for durable cloud copies.

Uploads a local directory to a container using the project's OpenStack credentials
from the laptop, so no secret ever lands on an instance. The container is created
if missing. (Object Storage is credit-eligible under the Startup Program.)

OVH splits region labels: compute is under numbered regions (GRA11) but
object-store is under 3-letter ones (GRA). openstacksdk's cloud config only knows
the pinned compute region, so rather than reconnect, we read the object-store
endpoint URL from the service catalog and talk to Swift over the existing
authenticated session.
"""
from __future__ import annotations

import os


def _object_store_endpoint(conn, prefer: str | None = None):
    """Return (region, storage_url) for an object-store endpoint, preferring the
    one geographically matching the compute region (GRA11 -> GRA)."""
    access = conn.session.auth.get_access(conn.session)
    eps = [
        (ep.get("region"), ep.get("url"))
        for entry in access.service_catalog.catalog if entry.get("type") == "object-store"
        for ep in entry.get("endpoints", []) if ep.get("interface") in ("public", None) and ep.get("url")
    ]
    if not eps:
        raise RuntimeError("no object-store (Swift) endpoint in this project's catalog.")
    if prefer:
        alpha = "".join(ch for ch in prefer if ch.isalpha())  # GRA11 -> GRA
        for region, url in eps:
            if region == alpha or (region and prefer.startswith(region)):
                return region, url
    return eps[0]


def push_dir(conn, container: str, local_dir: str, prefix: str = ""):
    if not os.path.isdir(local_dir):
        raise RuntimeError(f"not a directory: {local_dir}")
    region, base = _object_store_endpoint(conn, prefer=getattr(conn.config, "region_name", None))
    sess = conn.session
    sess.put(f"{base}/{container}")  # create container (idempotent: 201/202)
    n = 0
    for root, _dirs, files in os.walk(local_dir):
        for fname in files:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, local_dir).replace(os.sep, "/")
            obj = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            with open(path, "rb") as fh:
                sess.put(f"{base}/{container}/{obj}", data=fh.read())
            n += 1
    return n, region


def run_push(cloud=None, region=None, local_dir=None, container=None, prefix="") -> int:
    if not local_dir or not container:
        raise RuntimeError("push needs a local DIR and a CONTAINER")
    from .auth import connect
    conn = connect(cloud=cloud, region=region)
    n, osr = push_dir(conn, container, local_dir, prefix=prefix)
    print(f"uploaded {n} file(s) from {local_dir} -> container {container} (object-store region {osr})"
          + (f" under {prefix}/" if prefix else ""))
    return 0
