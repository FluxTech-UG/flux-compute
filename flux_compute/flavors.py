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

CPU flavors are always fp64 healthy and credit eligible; they are the right
choice for small runs where GPU kernel-launch latency would dominate.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SIM_FLAVOR = "t1-le-45"

# Public list prices (EUR/hr, ex VAT) for the credit-eligible flavors, from the
# OVHcloud Startup Program product-eligibility list (March 2026). Used for cost
# display and to rank the cheapest fp64-healthy GPU.
_KNOWN_PRICE_EUR_HR = {
    "t1-le-45": 0.70, "t1-le-90": 1.40, "t1-le-180": 2.80,
    "t2-le-45": 0.80, "t2-le-90": 1.60, "t2-le-180": 3.20,
    "rtx5000-28": 0.36, "rtx5000-56": 0.72, "rtx5000-84": 1.08,
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

# CPU flavor families on Public Cloud: all credit-eligible and fp64-healthy.
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
                "CPU flavor: credit-eligible and fp64-healthy "
                "(best for small runs where GPU launch latency dominates).",
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
