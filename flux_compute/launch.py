"""Resolve a launch spec for a run, and print it as a dry run.

`resolve_spec` turns a region into the concrete choices a launch needs: which
flavor (credit-eligible and fp64-healthy), which image (flavor-aware: an
NVIDIA-driver Ubuntu for a GPU flavor, a plain Ubuntu LTS for a CPU flavor,
unless `--image` overrides), which network. It is the single resolver every
launching path shares -- `provision.run_job`, `provision.smoke_test`,
`sweep._prepare_shard` and `image.bake` all call it -- so a dry run resolves
exactly what a real launch will use. `plan` prints that spec without launching.
"""
from __future__ import annotations

from dataclasses import dataclass

from .flavors import classify, recommended_for_sim

PUBLIC_NETWORK = "Ext-Net"


def _newest_lts(candidates, prefer="newest") -> str:
    """Return the newest-LTS Ubuntu image name (24.04, then 22.04, else the pick
    over all). Within an LTS, `prefer='newest'` takes the highest-sorting name
    (e.g. the newest NVIDIA driver "v580" over "v535"); `prefer='base'` takes the
    plainest base image (the shortest name, e.g. "Ubuntu 24.04" over "Ubuntu
    24.04 - UEFI"). Assumes `candidates` is already non-empty."""
    def pick(group):
        if prefer == "base":
            return min(group, key=lambda n: (len(n), n))
        return sorted(group, reverse=True)[0]

    for lts in ("24.04", "22.04"):
        match = [n for n in candidates if lts in n]
        if match:
            return pick(match)
    return pick(candidates)


def select_gpu_image(image_names) -> str:
    """Pick an NVIDIA-driver Ubuntu image, preferring the newest LTS and driver.

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
    return _newest_lts(nvidia, prefer="newest")


def select_cpu_image(image_names) -> str:
    """Pick a plain (non-NVIDIA) Ubuntu LTS base image, preferring the newest LTS.

    A CPU flavor has no GPU, so a driver-included image only wastes boot time and
    disk. This selects a stock Ubuntu LTS base image, excluding NVIDIA, baremetal
    and variant (e.g. UEFI) builds in favour of the plain image; it raises rather
    than fall back to a driver image.
    """
    plain = [n for n in image_names
             if "ubuntu" in n.lower()
             and "nvidia" not in n.lower()
             and "baremetal" not in n.lower()]
    if not plain:
        raise RuntimeError(
            "No plain Ubuntu image available in this region for a CPU flavor. "
            "Pass --image to name one explicitly."
        )
    return _newest_lts(plain, prefer="base")


def select_image(kind: str, image_names) -> str:
    """Dispatch image selection by flavor kind: CPU flavors get a plain Ubuntu
    LTS, GPU flavors an NVIDIA-driver Ubuntu. Any other kind is a policy bug
    (resolve_spec has already rejected non-usable flavors)."""
    if kind == "cpu":
        return select_cpu_image(image_names)
    if kind == "gpu":
        return select_gpu_image(image_names)
    raise RuntimeError(f"cannot select an image for flavor kind {kind!r}")


@dataclass(frozen=True)
class LaunchSpec:
    region: str
    flavor: str
    gpu_model: str | None
    image: str
    network: str
    keypair: str
    est_cost_eur_hr: float | None


def resolve_spec(conn, region: str, flavor: str | None = None, keypair: str | None = None,
                 image: str | None = None) -> LaunchSpec:
    """Resolve the concrete launch choices for `region`, or raise (fail-fast).

    `image` overrides image selection (e.g. a baked image); otherwise the newest
    NVIDIA-driver Ubuntu image is chosen.
    """
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

    if image:
        if conn.compute.find_image(image) is None:
            raise RuntimeError(f"Image {image!r} not found in region {region}.")
        img_name = image
    else:
        img_name = select_image(verdict.kind, [i.name for i in conn.image.images()])

    return LaunchSpec(
        region=region,
        flavor=chosen,
        gpu_model=verdict.gpu_model,
        image=img_name,
        network=PUBLIC_NETWORK,
        keypair=keypair or "flux-compute-<generated-per-run>",
        est_cost_eur_hr=verdict.price_eur_hr,
    )


def plan(cloud: str | None = None, region: str | None = None, flavor: str | None = None) -> int:
    from .auth import connect, resolve_region_name
    from .reap import warn_strays  # function-level: reap imports provision, which imports this module

    conn = connect(cloud=cloud, region=region)
    warn_strays(conn)
    reg = resolve_region_name(conn, region)
    spec = resolve_spec(conn, reg, flavor=flavor)

    cost = f"EUR {spec.est_cost_eur_hr:.2f}/hr" if spec.est_cost_eur_hr is not None else "price n/a"
    print("flux-compute run plan (dry run, no instance launched):")
    print(f"  region   : {spec.region}")
    print(f"  flavor   : {spec.flavor}  [{spec.gpu_model or 'CPU'}]")
    print(f"  image    : {spec.image}")
    print(f"  network  : {spec.network} (public IP)")
    print(f"  keypair  : {spec.keypair}")
    print(f"  est cost : {cost}")
    print()
    print("Dry run: nothing launched, nothing billed. To actually run work here:")
    print("  flux-compute run   --upload DIR[:DEST] --script FILE --fetch REMOTE:LOCAL")
    print("  flux-compute sweep --jobs FILE --script FILE --fetch REMOTE --budget EUR")
    return 0
