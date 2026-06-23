"""`flux-compute doctor`: verify OVH OpenStack API access.

The smallest end-to-end test that the API works: authenticate, list the compute
flavors and images the project can see, and report which flavors are both
Startup-Program credit-eligible and fp64-healthy for the FluxTech sims.
"""
from __future__ import annotations

import os

from .auth import connect
from .flavors import classify, recommended_for_sim


def run_doctor(cloud: str | None = None, region: str | None = None) -> int:
    conn = connect(cloud=cloud, region=region)

    print("Authenticated to OVH OpenStack.")
    print(f"  project: {_project_of(conn) or '(unknown)'}")
    print(f"  region : {_region_of(conn, region) or '(unknown)'}")
    print()

    try:
        flavors = list(conn.compute.flavors())
    except Exception as exc:
        raise RuntimeError(
            f"Authenticated, but listing compute flavors failed: {exc}\n"
            "Check the project has Public Cloud Compute enabled and the region is valid."
        ) from exc

    verdicts = sorted((classify(f.name) for f in flavors), key=lambda v: v.name)
    healthy_gpu = [v for v in verdicts if v.kind == "gpu" and v.usable_for_sim]
    crippled_gpu = [v for v in verdicts if v.kind == "gpu" and v.credit_eligible and not v.fp64_healthy]
    blocked_gpu = [v for v in verdicts if v.kind == "gpu" and not v.credit_eligible]
    cpu = [v for v in verdicts if v.kind == "cpu"]
    gpu_total = len(healthy_gpu) + len(crippled_gpu) + len(blocked_gpu)

    print(f"Visible flavors: {len(flavors)} ({len(cpu)} CPU, {gpu_total} GPU)")
    print()
    _print_group("Credit-eligible + fp64-healthy GPUs (use these)", healthy_gpu)
    _print_group("Credit-eligible but fp64-crippled (avoid for sims)", crippled_gpu)
    _print_group("GPUs present but NOT covered by credits", blocked_gpu)

    try:
        images = list(conn.image.images())
        ubuntu = [i for i in images if "ubuntu" in (i.name or "").lower()]
        print(f"Images visible: {len(images)} ({len(ubuntu)} Ubuntu).")
    except Exception as exc:
        print(f"Image list unavailable: {exc}")
    print()

    if healthy_gpu:
        pick = recommended_for_sim([v.name for v in healthy_gpu])
        print(f"API OK. Recommended sim flavor here: {pick}")
        return 0

    print("API reachable, but this region exposes no credit-eligible, fp64-healthy GPU.")
    print("Switch to a GPU region (GRA9, GRA11, BHS5) via --region or clouds.yaml.")
    return 1


def _print_group(title: str, verdicts) -> None:
    print(f"{title}: {len(verdicts)}")
    for v in verdicts:
        price = f"EUR {v.price_eur_hr:.2f}/hr" if v.price_eur_hr is not None else "price n/a"
        model = f"[{v.gpu_model}]" if v.gpu_model else ""
        print(f"  {v.name:<14} {model:<22} {price}")
    print()


def _region_of(conn, region):
    if region:
        return region
    cfg = getattr(conn, "config", None)
    return getattr(cfg, "region_name", None) or os.environ.get("OS_REGION_NAME")


def _project_of(conn):
    pid = getattr(conn, "current_project_id", None)
    if pid:
        return pid
    return (os.environ.get("OS_PROJECT_ID")
            or os.environ.get("OS_TENANT_ID")
            or os.environ.get("OS_PROJECT_NAME"))
