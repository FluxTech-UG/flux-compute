"""Resolve a launch spec for a GPU run, and (for now) plan it without launching.

`resolve_spec` turns a region into the concrete choices a launch needs: which
flavor (credit-eligible and fp64-healthy), which image (an NVIDIA-driver Ubuntu,
so the GPU is usable), which network. `plan` prints that spec as a dry run. The
actual provision / bootstrap / fetch / teardown step is billable and is wired
separately.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .flavors import classify, recommended_for_sim

PUBLIC_NETWORK = "Ext-Net"


def select_gpu_image(image_names) -> str:
    """Pick an NVIDIA-driver Ubuntu image, preferring the newest LTS.

    A GPU run needs the host NVIDIA driver present; OVH ships driver-included
    images named like "Ubuntu 24.04 - NVIDIA - v580". Launching a stock image on
    a GPU flavor would hand the sim a GPU it cannot use, so this raises rather
    than fall back to a driverless image.
    """
    nvidia = [n for n in image_names if "nvidia" in n.lower() and "ubuntu" in n.lower()]
    if not nvidia:
        raise RuntimeError(
            "No NVIDIA-driver Ubuntu image available in this region. A GPU run needs "
            "the host driver; a stock image would give the sim an unusable GPU."
        )
    for lts in ("24.04", "22.04"):
        match = [n for n in nvidia if lts in n]
        if match:
            return sorted(match, reverse=True)[0]
    return sorted(nvidia, reverse=True)[0]


@dataclass(frozen=True)
class LaunchSpec:
    region: str
    flavor: str
    gpu_model: str | None
    image: str
    network: str
    keypair: str
    est_cost_eur_hr: float | None


def resolve_spec(conn, region: str, flavor: str | None = None, keypair: str | None = None) -> LaunchSpec:
    """Resolve the concrete launch choices for `region`, or raise (fail-fast)."""
    names = [f.name for f in conn.compute.flavors()]
    chosen = flavor or recommended_for_sim(names)

    verdict = classify(chosen)
    if not verdict.usable_for_sim:
        raise RuntimeError(f"Flavor {chosen} is not usable for sims: {verdict.reason}")
    if chosen not in names:
        healthy = sorted(n for n in names if classify(n).kind == "gpu" and classify(n).usable_for_sim)
        raise RuntimeError(
            f"Flavor {chosen} is not available in region {region}. "
            f"Healthy GPU flavors here: {healthy or 'none'}."
        )

    nets = [n.name for n in conn.network.networks()]
    if PUBLIC_NETWORK not in nets:
        raise RuntimeError(f"Public network {PUBLIC_NETWORK!r} not found in {region}; available: {nets}.")

    image = select_gpu_image([i.name for i in conn.image.images()])

    return LaunchSpec(
        region=region,
        flavor=chosen,
        gpu_model=verdict.gpu_model,
        image=image,
        network=PUBLIC_NETWORK,
        keypair=keypair or "flux-compute-<generated-per-run>",
        est_cost_eur_hr=verdict.price_eur_hr,
    )


def plan(cloud: str | None = None, region: str | None = None, flavor: str | None = None) -> int:
    from .auth import connect

    conn = connect(cloud=cloud, region=region)
    reg = (region
           or getattr(getattr(conn, "config", None), "region_name", None)
           or os.environ.get("OS_REGION_NAME")
           or "(unknown)")
    spec = resolve_spec(conn, reg, flavor=flavor)

    cost = f"EUR {spec.est_cost_eur_hr:.2f}/hr" if spec.est_cost_eur_hr is not None else "price n/a"
    print("flux-compute run plan (dry run, no instance launched):")
    print(f"  region   : {spec.region}")
    print(f"  flavor   : {spec.flavor}  [{spec.gpu_model}]")
    print(f"  image    : {spec.image}")
    print(f"  network  : {spec.network} (public IP)")
    print(f"  keypair  : {spec.keypair}")
    print(f"  est cost : {cost}")
    print()
    print("Provision / bootstrap / fetch / teardown is billable and not wired yet.")
    return 0
