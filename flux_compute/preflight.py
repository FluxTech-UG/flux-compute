"""`flux-compute preflight`: read-only check that a region can actually launch.

doctor answers "does the API work and what GPUs are covered". preflight answers
the next question: "can this project provision a GPU instance right now" (quota,
public network, an NVIDIA-driver image), without launching anything.
"""
from __future__ import annotations

from .auth import connect
from .flavors import classify, recommended_for_sim
from .launch import PUBLIC_NETWORK

# A single V100S (t2-le-45) is 15 vCPU; treat that as the core-quota floor.
_MIN_CORES = 15


def gather(conn) -> dict:
    lim = conn.get_compute_limits()
    g = lambda k: getattr(lim, k, None)
    flavor_names = [f.name for f in conn.compute.flavors()]
    image_names = [i.name for i in conn.image.images()]
    return {
        "cores": (g("total_cores_used"), g("max_total_cores")),
        "instances": (g("total_instances_used"), g("max_total_instances")),
        "networks": [n.name for n in conn.network.networks()],
        "keypairs": [k.name for k in conn.compute.keypairs()],
        "security_groups": [s.name for s in conn.network.security_groups()],
        "healthy_gpu_flavors": sorted(
            n for n in flavor_names if classify(n).kind == "gpu" and classify(n).usable_for_sim
        ),
        "nvidia_images": [n for n in image_names if "nvidia" in n.lower()],
    }


def run_preflight(cloud: str | None = None, region: str | None = None) -> int:
    conn = connect(cloud=cloud, region=region)
    d = gather(conn)

    print("flux-compute preflight (read-only launch-readiness check):")
    print(f"  cores used/max     : {d['cores'][0]}/{d['cores'][1]}")
    print(f"  instances used/max : {d['instances'][0]}/{d['instances'][1]}")
    print(f"  public network     : {'Ext-Net OK' if PUBLIC_NETWORK in d['networks'] else 'MISSING ' + str(d['networks'])}")
    print(f"  keypairs           : {d['keypairs'] or 'none (a run will create one)'}")
    print(f"  security groups    : {d['security_groups']}")
    print(f"  healthy GPU flavors: {d['healthy_gpu_flavors'] or 'NONE (switch region)'}")
    print(f"  NVIDIA images      : {d['nvidia_images'] or 'NONE (driver-install required)'}")
    print()

    problems = []
    if not d["healthy_gpu_flavors"]:
        problems.append("no credit-eligible, fp64-healthy GPU in this region")
    if PUBLIC_NETWORK not in d["networks"]:
        problems.append(f"{PUBLIC_NETWORK} network missing")
    if not d["nvidia_images"]:
        problems.append("no NVIDIA-driver image")
    max_cores = d["cores"][1] or 0
    if max_cores < _MIN_CORES:
        problems.append(f"core quota {max_cores} below a V100S ({_MIN_CORES} vCPU); request an increase")

    if problems:
        print("NOT launch-ready:")
        for p in problems:
            print(f"  - {p}")
        return 1

    rec = recommended_for_sim(d["healthy_gpu_flavors"])
    print(f"Launch-ready. Recommended flavor: {rec}. Run `flux-compute run --plan` for the full spec.")
    return 0
