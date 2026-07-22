"""OVH Public Cloud flavor policy for the FluxTech Startup Program.

Two independent gates decide whether a flavor may run a FluxTech simulation:

  1. credit_eligible: the OVHcloud Startup Program covers the flavor's cost.
     Per the March 2026 product-eligibility list, GPU credits cover only the
     V100, V100S and RTX5000 cards. H100, A100, L40S, L4 and A10 are blocked.

  2. fp64_healthy: the GPU runs double precision at a usable rate. The FluxTech
     sims (1DSim3, LumpedSim2) force jax_enable_x64 and are roughly 95% EOS
     transcendental work, so a card whose fp64 throughput is ~1/32 of fp32 is
     unusable for them. Volta (V100, V100S) runs fp64 at ~1/2 of fp32. Turing
     (RTX5000) runs it at ~1/32, so RTX5000 is credit-eligible but not fp64
     healthy, and is refused for sims by default.

CPU flavors are always fp64 healthy and are the right choice for small runs, and
for a wide fan-out of one-job-per-VM CPU work, where GPU kernel-launch latency
would dominate. The classifier treats them as usable and prices them (b3-*/c3-*)
from the OVH catalog. CPU instance spend IS covered by Startup Program credits:
the program's product-eligibility guide (March 2026, archived at
docs/product-eligibility-startup-program-2026-03.html) marks every b-series and
c-series instance "Covered" at both program levels, with GPU instances as the
only restricted family — so for CPU flavors credit_eligible is a sourced billing
fact, same as for the GPU cards (README, "CPU credit coverage").
"""
from __future__ import annotations

from dataclasses import dataclass

# V100S (t2-le) is available across EU regions (GRA11, DE1, UK1, WAW1) and BHS5;
# plain V100 (t1-le, slightly cheaper, 16GB) exists only in BHS5 (Canada). The
# EU-wide V100S is the right default; recommended_for_sim() still picks the
# cheapest fp64-healthy GPU actually present in the target region.
DEFAULT_SIM_FLAVOR = "t2-le-45"

# Public list prices (EUR/hr, ex VAT) from the OVHcloud public cloud order
# catalog for the DE subsidiary (the account's billing subsidiary), read
# 2026-07-04 from the `<flavor>.consumption` hourly pricing at
#   https://api.ovh.com/1.0/order/catalog/public/cloud?ovhSubsidiary=DE
# The whole table is single-sourced from that catalog, so GPU and CPU rows are
# directly comparable. Used for cost display, to rank the cheapest fp64-healthy
# GPU, and to bound worst-case sweep spend. Re-verify against the account catalog
# when prices move; an unpriced flavor is refused under a sweep --budget rather
# than silently skipping the guard.
_KNOWN_PRICE_EUR_HR = {
    # GPU (credit-eligible cards)
    "t1-le-45": 0.70, "t1-le-90": 1.40, "t1-le-180": 2.80,
    "t2-le-45": 0.80, "t2-le-90": 1.60, "t2-le-180": 3.20,
    "rtx5000-28": 0.36, "rtx5000-56": 0.72, "rtx5000-84": 1.08,
    # CPU — General Purpose (b3-*, 1 vCPU : 4 GB RAM)
    "b3-8": 0.0512, "b3-16": 0.1023, "b3-32": 0.2046, "b3-64": 0.4092,
    "b3-128": 0.8190, "b3-256": 1.6370, "b3-512": 3.2740, "b3-640": 4.0920,
    # CPU — Compute Optimized (c3-*, 1 vCPU : 2 GB RAM)
    "c3-4": 0.0457, "c3-8": 0.0913, "c3-16": 0.1825, "c3-32": 0.3650,
    "c3-64": 0.7301, "c3-128": 1.4610, "c3-256": 2.9210, "c3-320": 3.6510,
}

# GPU flavor-name prefix -> (card model, credit_eligible, fp64_healthy, reason).
# Order matters: longer/more-specific prefixes ("l40s") precede shorter ones
# ("l4") so the first match wins.
_GPU_RULES = (
    ("t1-le",   ("Tesla V100 16GB",  True,  True,
                 "V100 (Volta): credit-eligible and fp64-healthy.")),
    ("t2-le",   ("Tesla V100S 32GB", True,  True,
                 "V100S (Volta): credit-eligible and fp64-healthy.")),
    ("rtx5000", ("Quadro RTX5000",   True,  False,
                 "RTX5000 (Turing): credit-eligible but fp64 ~1/32 of fp32; "
                 "refused for x64 sims by default.")),
    ("h100",    ("H100 80GB",        False, True,
                 "H100: not covered by Startup Program credits.")),
    ("a100",    ("A100 80GB",        False, True,
                 "A100: not covered by Startup Program credits.")),
    ("l40s",    ("L40S 48GB",        False, False,
                 "L40S: not covered by Startup Program credits.")),
    ("l4",      ("L4 24GB",          False, False,
                 "L4: not covered by Startup Program credits.")),
    ("a10",     ("A10 24GB",         False, False,
                 "A10: not covered by Startup Program credits.")),
)

# CPU flavor families on Public Cloud: fp64-healthy, usable, and covered by
# Startup Program credits (the archived eligibility guide marks every b-/c-series
# instance "Covered"; only GPU instances are restricted — module docstring).
_CPU_PREFIXES = ("b3-", "b2-", "c3-", "c2-", "r3-", "r2-", "d2-", "i1-", "bm-")

# Host RAM per vCPU by CPU family, from the OVH catalog (docs/product-eligibility-
# startup-program-2026-03.html): b3 (General Purpose) is 4 GB/vCPU, c3 (Compute
# Optimized) is 2 GB/vCPU. For those two families the flavor's numeric suffix is
# also its total host RAM in GB (b3-8 = 8 GB / 2 vCPU, c3-4 = 4 GB / 2 vCPU), so
# the offline path derives both vCPU and RAM from the name and these ratios with
# no live lookup. Only the families whose ratio is sourced appear here; a CPU
# family that is not listed has no static RAM and is a fail-fast in the pure path
# (the live path reads .ram off the flavor object instead).
_CPU_RAM_PER_VCPU_GB = {"b3-": 4.0, "c3-": 2.0}

# GPU flavor -> (vCPUs, host RAM GB), from the OVH catalog (same archived source,
# cross-checked against README "Multi-region sweeps"). Host RAM, not GPU VRAM:
# the planner's RAM model bounds how many job processes/members a VM's *host*
# memory can hold. The suffix is the RAM in GB but vCPUs do not follow a single
# ratio across the GPU families (t2-le-45 is 15 vCPU, t1-le-45 is 8), so the GPU
# rows are tabulated rather than derived. Unknown GPU flavor -> fail fast.
_GPU_SPECS_VCPU_RAM = {
    "t1-le-45": (8, 45),   "t1-le-90": (16, 90),  "t1-le-180": (32, 180),
    "t2-le-45": (15, 45),  "t2-le-90": (30, 90),  "t2-le-180": (60, 180),
    "rtx5000-28": (4, 28), "rtx5000-56": (8, 56), "rtx5000-84": (16, 84),
}


@dataclass(frozen=True)
class FlavorVerdict:
    """The policy verdict for a single flavor name."""

    name: str
    kind: str                 # "gpu", "cpu", or "unknown"
    gpu_model: str | None
    credit_eligible: bool
    fp64_healthy: bool
    price_eur_hr: float | None
    reason: str

    @property
    def usable_for_sim(self) -> bool:
        """True only when both gates pass: covered by credits and fp64-healthy."""
        return self.kind in ("gpu", "cpu") and self.credit_eligible and self.fp64_healthy


def classify(name: str) -> FlavorVerdict:
    """Classify a flavor name against the credit + fp64 policy.

    Works from the flavor-name family prefix, so it covers any flavor OVH
    returns, not only the priced ones in the static table. An unrecognized
    family yields an "unknown" verdict that is not usable, never a silent pass.
    """
    n = name.strip().lower()
    price = _KNOWN_PRICE_EUR_HR.get(n)

    for prefix, (model, eligible, fp64, reason) in _GPU_RULES:
        if n.startswith(prefix):
            return FlavorVerdict(name, "gpu", model, eligible, fp64, price, reason)

    for prefix in _CPU_PREFIXES:
        if n.startswith(prefix):
            return FlavorVerdict(
                name, "cpu", None, True, True, price,
                "CPU flavor: fp64-healthy, covered by Startup Program credits; best "
                "for small runs and wide one-job-per-VM fan-out.",
            )

    return FlavorVerdict(
        name, "unknown", None, False, False, price,
        "Unrecognized flavor family; verify against the eligibility list before use.",
    )


def recommended_for_sim(available_names) -> str:
    """Return the cheapest credit-eligible, fp64-healthy GPU among those available.

    Raises if none qualify (for example, a region that exposes no covered GPU).
    The caller should switch to a GPU-enabled region (GRA9, GRA11, BHS5) rather
    than silently fall back to a crippled or uncovered card.
    """
    gpus = [v for v in (classify(n) for n in available_names)
            if v.kind == "gpu" and v.usable_for_sim]
    if not gpus:
        raise RuntimeError(
            "No credit-eligible, fp64-healthy GPU flavor is available here. "
            "Covered fp64-healthy GPUs are V100 (t1-le-*) and V100S (t2-le-*); "
            "RTX5000 is covered but fp64-crippled. Try a GPU region: GRA9, GRA11, BHS5."
        )
    gpus.sort(key=lambda v: (v.price_eur_hr if v.price_eur_hr is not None else float("inf"), v.name))
    return gpus[0].name


@dataclass(frozen=True)
class FlavorSpec:
    """A flavor's resource shape: the fields the fleet planner sizes against.

    `vcpus` and `ram_gb` are the two capacity axes a VM offers; `price_eur_hr`
    bounds spend (None when the flavor is unpriced, which the budget guard
    refuses). `kind`/`gpu_model` come straight from the policy `classify`, so a
    spec of a non-usable flavor still describes it — the planner filters on
    `usable_for_sim` itself.
    """

    name: str
    kind: str                 # "gpu", "cpu", or "unknown"
    gpu_model: str | None
    vcpus: int
    ram_gb: float
    price_eur_hr: float | None
    usable_for_sim: bool


def _parse_cpu_suffix(name: str):
    """(family_prefix, numeric suffix) for a b3-*/c3-* flavor, or raise.

    The suffix is the flavor's host RAM in GB (b3-8 = 8 GB); the family prefix
    keys the RAM-per-vCPU ratio. Raises for a CPU family whose ratio is not
    sourced, or a suffix that is not a bare integer (e.g. a `-flex` variant),
    rather than guessing — the offline path is fail-fast, the live path reads
    `.ram`/`.vcpus` off the flavor object.
    """
    n = name.strip().lower()
    for prefix, ratio in _CPU_RAM_PER_VCPU_GB.items():
        if n.startswith(prefix):
            suffix = n[len(prefix):]
            if not suffix.isdigit():
                raise RuntimeError(
                    f"cannot size CPU flavor {name!r} offline: its RAM/vCPU are read "
                    f"from the '{prefix}N' suffix (N = GB of RAM), but {suffix!r} is not "
                    f"a bare integer. Pass a catalog flavor, or resolve it live."
                )
            return prefix, ratio, int(suffix)
    raise RuntimeError(
        f"no sourced RAM-per-vCPU ratio for CPU flavor {name!r} "
        f"(known families: {', '.join(sorted(_CPU_RAM_PER_VCPU_GB))}). "
        f"Add its catalog ratio to _CPU_RAM_PER_VCPU_GB, or resolve it live."
    )


def static_flavor_spec(name: str) -> FlavorSpec:
    """Resolve a flavor's (vCPU, RAM, price) shape offline from the catalog tables.

    The pure/offline counterpart to reading a live OpenStack flavor object. GPU
    flavors come from the tabulated `_GPU_SPECS_VCPU_RAM`; CPU flavors are derived
    from the family ratio and the name's GB suffix. An unknown flavor family, or a
    GPU name not in the catalog table, raises (fail fast — never a guessed shape).
    """
    verdict = classify(name)
    if verdict.kind == "gpu":
        key = name.strip().lower()
        if key not in _GPU_SPECS_VCPU_RAM:
            raise RuntimeError(
                f"no catalog RAM/vCPU for GPU flavor {name!r} "
                f"(known: {', '.join(sorted(_GPU_SPECS_VCPU_RAM))}). "
                f"Add it to _GPU_SPECS_VCPU_RAM, or resolve it live."
            )
        vcpus, ram_gb = _GPU_SPECS_VCPU_RAM[key]
        return FlavorSpec(name, "gpu", verdict.gpu_model, vcpus, float(ram_gb),
                          verdict.price_eur_hr, verdict.usable_for_sim)
    if verdict.kind == "cpu":
        prefix, ratio, ram_gb = _parse_cpu_suffix(name)
        vcpus = int(round(ram_gb / ratio))
        return FlavorSpec(name, "cpu", None, vcpus, float(ram_gb),
                          verdict.price_eur_hr, verdict.usable_for_sim)
    raise RuntimeError(
        f"cannot size unknown flavor {name!r} offline: {verdict.reason}"
    )


def flavor_ram_gb(flavor_obj) -> float:
    """Host RAM in GB read from a live OpenStack flavor object.

    Mirrors `sweep._flavor_vcpus`: the compute API reports `.ram` in **MiB**, so
    this divides by 1024. Raises if the object exposes no readable RAM rather than
    defaulting to a fabricated size.
    """
    ram_mib = getattr(flavor_obj, "ram", None)
    if ram_mib is None:
        raise RuntimeError(
            f"could not read RAM for flavor {getattr(flavor_obj, 'name', flavor_obj)!r} "
            f"from the compute API"
        )
    return float(ram_mib) / 1024.0


def live_flavor_spec(flavor_obj) -> FlavorSpec:
    """Resolve a FlavorSpec from a live OpenStack flavor object.

    Reads `.vcpus` and `.ram` off the object (the authoritative live shape) and
    takes kind/model/price from the policy `classify`. The live counterpart to
    `static_flavor_spec`; used by the fleet planner's live wrapper so a launch
    sizes against the region's actual flavor, not the catalog snapshot.
    """
    name = getattr(flavor_obj, "name", None)
    if not name:
        raise RuntimeError("flavor object has no name")
    vcpus = getattr(flavor_obj, "vcpus", None)
    if vcpus is None:
        raise RuntimeError(f"could not read the vCPU count for flavor {name!r} from the compute API")
    verdict = classify(name)
    return FlavorSpec(name, verdict.kind, verdict.gpu_model, int(vcpus),
                      flavor_ram_gb(flavor_obj), verdict.price_eur_hr, verdict.usable_for_sim)
