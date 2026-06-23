"""flux-compute: run FluxTech simulations on OVH Public Cloud GPU instances.

A package in the FluxTech family (consumers import it; it imports nothing back).
Phase 0 surface: the flavor-eligibility policy plus the `doctor` API health
check. `connect()` lives in `flux_compute.auth` and is imported there so that
this policy module stays importable without openstacksdk installed.
"""
from .flavors import (
    DEFAULT_SIM_FLAVOR,
    FlavorVerdict,
    classify,
    recommended_for_sim,
)

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_SIM_FLAVOR",
    "FlavorVerdict",
    "classify",
    "recommended_for_sim",
    "__version__",
]
