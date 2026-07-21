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


_REGION_PIN_REMEDY = (
    "Region {region!r} was refused by the local clouds.yaml, not by OVH:\n"
    "  {exc}\n\n"
    "A clouds.yaml entry with a single `region_name:` pins that cloud to ONE region,\n"
    "so --region / --regions for any other is rejected before a request is sent.\n"
    "Compute quota at OVH is per region, so that pin also caps how wide a sweep can\n"
    "run. Replace the single pin with the list of regions you use:\n\n"
    "    regions:\n"
    "      - GRA11\n"
    "      - DE1\n"
    "      - UK1\n"
    "      - WAW1\n"
    "      - BHS5\n\n"
    "(or drop `region_name:` entirely and always pass --region). See\n"
    "examples/clouds.yaml.example."
)


def configured_regions(cloud: str | None = None):
    """Every region this cloud entry is configured for, or [None] if it names one.

    A clouds.yaml `regions:` list expands to one config per region, which is what
    makes a project-wide stray hunt possible: OVH quota, servers and therefore
    STRAYS are per region, so a reap that looked only at the default region would
    report "no strays" while an instance billed in another. Returns [None] (the
    default region) when the cloud names a single region or when the config
    cannot be read -- callers then behave exactly as they did single-region.
    """
    if cloud is None:
        return [None]
    try:
        import openstack.config
        cfg = openstack.config.OpenStackConfig()
        names = [r.region_name for r in cfg.get_all()
                 if r.name == cloud and r.region_name]
    except Exception:
        return [None]
    return names or [None]


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
        if "is not a valid region name" in str(exc):
            raise RuntimeError(_REGION_PIN_REMEDY.format(region=region, exc=exc)) from exc
        raise RuntimeError(
            f"Could not initialise the OVH OpenStack connection: {exc}\n\n{_REMEDY}"
        ) from exc
