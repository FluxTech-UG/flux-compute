"""`flux-compute regions`: a live, read-only per-region occupancy view.

For each region it reports the compute quota (vCPU / instances / RAM used vs the
region's own total), the running flux-compute instances occupying it (name,
flavor, age and TTL bucket, via the same positive-identification `reap` uses)
plus a count of foreign servers, and how many of a given flavor still fit the
remaining headroom. Everything here is **read-only** — `get_compute_limits`,
`compute.servers`, `compute.find_flavor` — so it is safe to run against live
fleets. It is the one surface both the sweep pre-flight (graceful-degrade: drop
regions that cannot fit the work) and the frontend region-status button read;
`--json` is the machine-readable shape for those programmatic consumers.

The split mirrors the rest of the package: `build_region_status` is pure (flavor
spec + quota + server list -> a `RegionStatus`, unit-testable with no
credentials), and `gather_region_status` gathers the live per-region reads.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from .auth import configured_regions, connect
from .flavors import live_flavor_spec
from .reap import find_candidates

# The default flavor the `fits` column is measured against: a representative
# general-purpose VM (b3-32 = 8 vCPU, 32 GB) present across every region. The
# fits count is "how many of THIS flavor the remaining headroom holds"; pass
# --flavor to measure against the flavor a specific fleet will actually launch.
DEFAULT_FITS_FLAVOR = "b3-32"


@dataclass(frozen=True)
class RegionQuota:
    """One region's compute quota, both axes read straight off the limits API.

    Every field is used/max on its axis. An OpenStack `-1` max is the unlimited
    sentinel (preserved as a negative value here); `None` means the API did not
    report that axis. RAM is in GB (the API reports MiB; the reader converts)."""

    cores_used: int | None
    cores_max: int | None
    instances_used: int | None
    instances_max: int | None
    ram_used_gb: float | None
    ram_max_gb: float | None


@dataclass(frozen=True)
class RegionStatus:
    """A region's live occupancy: quota, who occupies it, and how many `flavor` fit.

    `ok` is False with `error` set when the region could not be read (unreachable
    endpoint, no credentials for it); such a row is surfaced, never silently
    dropped. `flavor_available` is False when the fits flavor does not exist in
    the region (then `fits` is None). `instances` are the flux-compute-identified
    servers (`reap.Candidate`, carrying name/flavor/age/bucket/cost); everything
    else the region runs is counted in `foreign_count`."""

    region: str
    ok: bool
    error: str | None
    flavor: str | None
    flavor_available: bool
    quota: RegionQuota | None
    fits: int | None
    instances: tuple
    foreign_count: int


def _axis_fits(used, mx, per_unit) -> int | None:
    """How many units of size `per_unit` fit a quota axis, or None if unbounded.

    A `None` max (axis not reported) or a negative one (OpenStack's `-1`
    unlimited sentinel) is unbounded -> None, so it drops out of the min() that
    combines the axes rather than falsely pinning the fit to zero."""
    if mx is None or mx < 0:
        return None
    free = max(0.0, mx - (used or 0))
    return int(free // per_unit)


def fits_count(spec, quota: RegionQuota) -> int | None:
    """How many VMs of `spec` fit `quota`'s remaining headroom, over all axes.

    The binding axis wins (min across cores/instances/RAM). Returns 0 when the
    headroom holds none — never raises, unlike the launch-path `_region_cap`, so
    the occupancy view can report a full region as `fits 0` rather than an error.
    None only when every axis is unbounded (no binding axis to count against)."""
    bounds = [
        _axis_fits(quota.cores_used, quota.cores_max, spec.vcpus),
        _axis_fits(quota.instances_used, quota.instances_max, 1),
        _axis_fits(quota.ram_used_gb, quota.ram_max_gb, spec.ram_gb),
    ]
    bounded = [b for b in bounds if b is not None]
    return min(bounded) if bounded else None


def build_region_status(region, *, flavor, spec, quota, servers, now) -> RegionStatus:
    """Assemble a `RegionStatus` from already-read pieces. Pure: no network.

    `spec` is the fits flavor's shape (None when the flavor is absent in the
    region); `servers` is the region's live server list; `now` dates the ages.
    """
    cands = find_candidates(servers, now)
    foreign = max(0, len(servers) - len(cands))
    fits = fits_count(spec, quota) if (spec is not None and quota is not None) else None
    return RegionStatus(
        region=region, ok=True, error=None, flavor=flavor,
        flavor_available=spec is not None, quota=quota, fits=fits,
        instances=tuple(cands), foreign_count=foreign)


def _region_name(conn, fallback):
    return (fallback
            or getattr(getattr(conn, "config", None), "region_name", None)
            or os.environ.get("OS_REGION_NAME") or "(unknown)")


def _read_quota(lim) -> RegionQuota:
    """Read a `RegionQuota` off a compute-limits object (the same fields
    `fleet.plan_fleet_live` reads). RAM is reported in MiB and converted to GB;
    a `-1` unlimited max is preserved as `-1.0` so the fits math treats it as
    unbounded rather than a fractional negative size."""
    gq = lambda k: getattr(lim, k, None)

    def ram_gb(mib):
        if mib is None:
            return None
        return -1.0 if mib < 0 else float(mib) / 1024.0

    return RegionQuota(
        cores_used=gq("total_cores_used"),
        cores_max=gq("max_total_cores"),
        instances_used=gq("total_instances_used"),
        instances_max=gq("max_total_instances"),
        ram_used_gb=ram_gb(gq("total_ram_used")),
        ram_max_gb=ram_gb(gq("max_total_ram_size")),
    )


def _one_region_status(cloud, region, flavor, now) -> RegionStatus:
    """Live-read one region into a `RegionStatus`, or an error row (never raises,
    except a clouds.yaml region pin, which is one global config fault fixing every
    region and so surfaces whole)."""
    try:
        conn = connect(cloud=cloud, region=region)
        reg = _region_name(conn, region)
        quota = _read_quota(conn.get_compute_limits())
        servers = list(conn.compute.servers(details=True))
        spec = None
        if flavor:
            flavor_obj = conn.compute.find_flavor(flavor)
            spec = live_flavor_spec(flavor_obj) if flavor_obj is not None else None
        return build_region_status(reg, flavor=flavor, spec=spec, quota=quota,
                                   servers=servers, now=now)
    except Exception as exc:                      # noqa: BLE001 — surfaced as a row
        if "refused by the local clouds.yaml" in str(exc):
            raise
        return RegionStatus(
            region=region or "(default)", ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:160]}", flavor=flavor,
            flavor_available=False, quota=None, fits=None, instances=(),
            foreign_count=0)


def gather_region_status(cloud, regions, flavor=DEFAULT_FITS_FLAVOR, *, now=None):
    """Read every region into a `RegionStatus`. Read-only; one row per region."""
    now = now or datetime.now(timezone.utc)
    return [_one_region_status(cloud, r, flavor, now) for r in regions]


def occupancy_summary(status: RegionStatus) -> str | None:
    """A short 'who occupies this region' line from a `RegionStatus`, grouping the
    flux-compute instances by TTL bucket and counting foreign servers. None when
    the region is empty (or unreadable)."""
    if not status.ok:
        return None
    parts = []
    for bucket, n in sorted(Counter(c.bucket for c in status.instances).items()):
        parts.append(f"{n}x flux-compute [{bucket}]")
    if status.foreign_count:
        parts.append(f"{status.foreign_count} foreign server(s)")
    return ", ".join(parts) if parts else None


def occupancy_line(cloud, region) -> str | None:
    """Best-effort one-line occupancy for `region`, for a drop warning. Read-only;
    never raises (a region that cannot be read yields None) — the caller already
    knows the region is being dropped and only wants the 'who' when it is cheap."""
    try:
        st = _one_region_status(cloud, region, None, datetime.now(timezone.utc))
    except Exception:
        return None
    return occupancy_summary(st)


# --- rendering ---------------------------------------------------------------

def _quota_cells(q: RegionQuota):
    """(vCPU, instances, RAM) 'used/max' cells for the text table, with '∞' for an
    unlimited axis and '?' for one the API did not report."""
    def cell(used, mx, unit=""):
        u = "?" if used is None else f"{used:g}"
        m = "?" if mx is None else ("∞" if mx < 0 else f"{mx:g}")
        return f"{u}/{m}{unit}"
    return (cell(q.cores_used, q.cores_max),
            cell(q.instances_used, q.instances_max),
            cell(q.ram_used_gb, q.ram_max_gb, " GB"))


def format_regions(statuses, flavor=DEFAULT_FITS_FLAVOR) -> str:
    """Render the region statuses as a human-readable block. Pure: returns a
    string, never prints — the CLI prints it, a consumer can reuse it."""
    lines = [
        "flux-compute regions — live per-region occupancy (read-only):",
        f"  fits column: how many {flavor} fit each region's remaining headroom.",
        "",
    ]
    for st in statuses:
        if not st.ok:
            lines.append(f"{st.region:<10}  ERROR: {st.error}")
            lines.append("")
            continue
        vcpu, inst, ram = _quota_cells(st.quota)
        fits = ("n/a (flavor absent)" if not st.flavor_available
                else "∞" if st.fits is None else f"{st.fits}")
        lines.append(
            f"{st.region:<10}  vCPU {vcpu:<9} inst {inst:<8} RAM {ram:<12} "
            f"fits {fits}x {st.flavor}")
        if st.instances:
            lines.append(f"    running flux-compute instances ({len(st.instances)}):")
            for c in st.instances:
                age = f"{c.age_hr:.1f}h" if c.age_hr is not None else "age ?"
                cost = f"~EUR {c.cost_eur:.2f}" if c.cost_eur is not None else "cost ?"
                lines.append(
                    f"      - {c.name}  {c.flavor or '?'}  {age}  "
                    f"[{c.bucket}]  {cost}")
        if st.foreign_count:
            lines.append(f"    foreign servers (not flux-compute): {st.foreign_count}")
        if not st.instances and not st.foreign_count:
            lines.append("    (idle: no running servers)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _quota_json(q: RegionQuota):
    if q is None:
        return None
    return {
        "vcpus": {"used": q.cores_used, "max": q.cores_max},
        "instances": {"used": q.instances_used, "max": q.instances_max},
        "ram_gb": {"used": q.ram_used_gb, "max": q.ram_max_gb},
    }


def regions_json(statuses, flavor=DEFAULT_FITS_FLAVOR) -> dict:
    """The machine-readable shape for the frontend button / agents.

    `{flavor, regions: [{region, ok, error, quota{vcpus,instances,ram_gb}, fits,
    flavor_available, foreign_count, instances:[{name,flavor,age_hr,bucket,
    cost_eur}]}]}`. `fits` is the headroom count for `flavor` (the field a
    launcher keys its region choice off), null when the flavor is absent."""
    out = []
    for st in statuses:
        out.append({
            "region": st.region,
            "ok": st.ok,
            "error": st.error,
            "flavor_available": st.flavor_available,
            "fits": st.fits,
            "foreign_count": st.foreign_count,
            "quota": _quota_json(st.quota),
            "instances": [
                {"name": c.name, "flavor": c.flavor, "age_hr": c.age_hr,
                 "bucket": c.bucket, "cost_eur": c.cost_eur}
                for c in st.instances
            ],
        })
    return {"flavor": flavor, "regions": out}


def _resolve_targets(cloud, region, regions):
    """The regions to show: an explicit --regions list, a single --region, else
    every region the cloud entry is configured for (like `reap`'s default scan)."""
    if regions:
        targets = [s.strip() for s in str(regions).split(",") if s.strip()]
        if not targets:
            raise RuntimeError("--regions was given but named no region")
        return targets
    if region:
        return [region]
    return configured_regions(cloud)


def run_regions(cloud=None, region=None, regions=None,
                flavor=DEFAULT_FITS_FLAVOR, as_json=False) -> int:
    """CLI entry: gather live occupancy for the target regions and print it."""
    targets = _resolve_targets(cloud, region, regions)
    statuses = gather_region_status(cloud, targets, flavor)
    if as_json:
        print(json.dumps(regions_json(statuses, flavor), indent=2))
    else:
        print(format_regions(statuses, flavor), end="")
    return 0
