"""Authenticated OpenStack connection to the OVH Public Cloud project.

Credentials come from the standard OpenStack sources: a clouds.yaml entry, or
the OS_* environment variables from a sourced OVH openrc.sh / exported
application credentials. Nothing is defaulted; with no credentials present this
raises with the exact remedy rather than guessing.
"""
from __future__ import annotations

import os

_REMEDY = (
    "No OVH OpenStack credentials found.\n"
    "Provide them one of two ways:\n"
    "  1. A clouds.yaml (see examples/clouds.yaml.example), then pass --cloud <name>.\n"
    "  2. Source an OVH openrc.sh, or export application-credential OS_* vars.\n"
    "Mint either in the OVH manager: Public Cloud project > Users & Roles\n"
    "(application credentials preferred: scoped and revocable)."
)


def connect(cloud: str | None = None, region: str | None = None):
    """Open an authenticated connection to the OVH project, or raise the remedy.

    `cloud` selects a clouds.yaml entry; otherwise the OS_* environment
    variables are used. `region` overrides the region from those sources.
    """
    if cloud is None and not os.environ.get("OS_AUTH_URL"):
        raise RuntimeError(_REMEDY)

    import openstack

    kwargs = {}
    if cloud is not None:
        kwargs["cloud"] = cloud
    if region is not None:
        kwargs["region_name"] = region

    try:
        return openstack.connect(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialise the OVH OpenStack connection: {exc}\n\n{_REMEDY}"
        ) from exc
