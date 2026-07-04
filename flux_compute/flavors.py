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
from the OVH catalog. Whether CPU instance spend is *covered by the Startup
Program credits*, however, is unconfirmed: the March 2026 eligibility list cited
above names only GPU cards and is silent on CPU. So for a CPU flavor the
credit_eligible flag means "not a blocked flavor / usable", not a sourced billing
guarantee. Do not present CPU as credit-covered until it is confirmed with OVH
(see README, "Open question for John — CPU credit coverage").
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

# CPU flavor families on Public Cloud: fp64-healthy and treated as usable. Their
# Startup Program credit coverage is unconfirmed (see the module docstring and
# the README open question); classify() sets credit_eligible=True to keep them
# usable, not as a sourced billing claim.
_CPU_PREFIXES = ("b3-", "b2-", "c3-", "c2-", "r3-", "r2-", "d2-", "i1-", "bm-")


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
                "CPU flavor: fp64-healthy and usable; best for small runs and wide "
                "one-job-per-VM fan-out. Startup Program credit coverage unconfirmed.",
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
