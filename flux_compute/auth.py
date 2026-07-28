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


def parse_region_list(text):
    """Parse a `--regions A,B,C` value into an ordered, de-duplicated region list.

    The one definition of that flag's syntax, shared by `sweep`, `regions` and
    `reap` -- three commands that previously each re-implemented it, and only one
    of which de-duplicated, so `--regions DE1,DE1` scanned DE1 twice under `reap`
    and once under `sweep`. Fails fast on an empty or blank-only value rather than
    silently falling back to the single-region path: `--regions ''` is a mistake,
    not a default.
    """
    names = [s.strip() for s in str(text).split(",")]
    names = [n for n in names if n]
    if not names:
        raise RuntimeError("--regions was given but named no region")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def resolve_region_name(conn, region: str | None = None) -> str:
    """The region a connection is actually working in, for display and records.

    Explicit `--region` wins, then the connection's own resolved config, then the
    environment. One definition, because four modules carried byte-identical
    copies of this chain and a fifth a subtly different one.
    """
    return (region
            or getattr(getattr(conn, "config", None), "region_name", None)
            or os.environ.get("OS_REGION_NAME")
            or "(unknown)")


def configured_regions(cloud: str | None = None):
    """Every region this cloud entry is configured for, or [None] if it names one.

    A clouds.yaml `regions:` list expands to one config per region, which is what
    makes a project-wide stray hunt possible: OVH quota, servers and therefore
    STRAYS are per region, so a reap that looked only at the default region would
    report "no strays" while an instance billed in another region.

    An unreadable config RAISES. Returning the single default region there would
    silently produce exactly the partial stray hunt this function exists to
    prevent -- and it would print "no strays" while an instance billed elsewhere,
    which is the expensive direction to be wrong in. [None] is returned only when
    the config is read successfully and genuinely names no region list.
    """
    if cloud is None:
        return [None]
    try:
        import openstack.config
        cfg = openstack.config.OpenStackConfig()
        names = [r.region_name for r in cfg.get_all()
                 if r.name == cloud and r.region_name]
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the region list for cloud {cloud!r} from the OpenStack "
            f"config: {type(exc).__name__}: {exc}\n"
            "Quota, servers and strays are all PER REGION, so continuing against a "
            "single default region would hide instances billing elsewhere. Fix the "
            "clouds.yaml (see examples/clouds.yaml.example), or name the regions "
            "explicitly with --regions/--region."
        ) from exc
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
