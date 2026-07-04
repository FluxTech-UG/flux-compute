"""`flux-compute reap`: find flux-compute instances and delete the expired ones.

Every server this package creates is stamped with provenance + TTL metadata
(provision.ttl_metadata). Reap sorts what it finds into four buckets and
annotates each listing with the decision basis:

  expired-stamped : flux_created_by stamp present AND past the stamped expiry.
                    The only bucket reap auto-deletes (confirm, or --yes).
  within-ttl      : stamped, not yet expired (a run in flight). Listed only;
                    taken only by --all with interactive confirmation.
  keep            : stamped flux_keep=true (a --keep run). Never auto-taken
                    regardless of age; listed prominently with accrued cost;
                    taken only by --all with interactive confirmation.
  unstamped-legacy: name-prefix match without the metadata stamp (a stray from
                    before TTL stamping). Report-only; taken only by --all with
                    interactive confirmation.

A server that is neither stamped nor name-prefixed is never listed as reapable
and never touched. --yes skips confirmation for the expired-stamped bucket
only; everything --all adds always requires the interactive confirmation.
Exit is nonzero when expired-stamped or unstamped-legacy strays remain.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .auth import connect
from .flavors import classify
from .provision import (
    FLUX_CREATED_BY,
    FLUX_CREATED_BY_KEY,
    FLUX_EXPIRES_KEY,
    FLUX_KEEP_KEY,
    FLUX_NAME_PREFIX,
    _cloud_name,
    _delete_server_verified,
    _delete_sg_with_retry,
    _stranded_banner,
)

AUTO_BUCKET = "expired-stamped"


def parse_utc(s):
    """Parse an ISO-8601 UTC timestamp (Z or offset) to an aware datetime, or
    return None for a missing/malformed value."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Candidate:
    """One flux-compute instance and the decision basis for its bucket."""

    server_id: str
    name: str
    flavor: str | None
    bucket: str            # "expired-stamped" | "within-ttl" | "keep" | "unstamped-legacy"
    why: str               # human-readable decision basis for the listing
    age_hr: float | None
    price_eur_hr: float | None
    cost_eur: float | None

    @property
    def auto_reapable(self) -> bool:
        return self.bucket == AUTO_BUCKET

    @property
    def is_stray(self) -> bool:
        """Strays drive the nonzero exit and the per-command warnings: a
        past-expiry stamped instance or a legacy one with no trustworthy TTL.
        A keep-flagged instance is deliberate, not a stray, however old."""
        return self.bucket in (AUTO_BUCKET, "unstamped-legacy")


def assess(server_id, name, metadata, created_at, flavor_name, now) -> Candidate | None:
    """Classify one server, or return None when it is not positively
    identifiable as flux-compute-created (such a server is never listed).

    Auto-reap requires the metadata stamp AND a parseable, passed expiry; a
    stamped instance whose expiry stamp is missing or malformed is demoted to
    the report-only legacy bucket rather than risking a false kill.
    """
    md = metadata or {}
    stamped = md.get(FLUX_CREATED_BY_KEY) == FLUX_CREATED_BY
    named = (name or "").startswith(FLUX_NAME_PREFIX)
    if not stamped and not named:
        return None

    created = parse_utc(created_at)
    age_hr = (now - created).total_seconds() / 3600.0 if created else None
    price = classify(flavor_name).price_eur_hr if flavor_name else None
    cost = age_hr * price if (age_hr is not None and price is not None) else None

    if stamped and md.get(FLUX_KEEP_KEY) == "true":
        bucket, why = "keep", "keep-flagged (--keep): never auto-reaped; take with --all + confirm"
    elif stamped:
        expires = parse_utc(md.get(FLUX_EXPIRES_KEY))
        if expires is None:
            bucket, why = "unstamped-legacy", (
                "stamped but no valid expiry stamp: report-only, never a false kill; "
                "take with --all + confirm")
        elif now >= expires:
            bucket, why = AUTO_BUCKET, f"stamped TTL passed {expires.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        else:
            bucket, why = "within-ttl", f"stamped, expires {expires.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    else:
        bucket, why = "unstamped-legacy", (
            "name-prefix match, no metadata stamp (pre-TTL stray): report-only; "
            "take with --all + confirm")

    return Candidate(server_id, name or "", flavor_name, bucket, why, age_hr, price, cost)


def _server_flavor_name(server):
    fl = getattr(server, "flavor", None)
    if fl is None:
        return None
    if isinstance(fl, dict):
        return fl.get("original_name") or fl.get("id")
    return getattr(fl, "original_name", None) or getattr(fl, "id", None)


def find_candidates(servers, now):
    """Assess every server; only positively identified flux-compute instances
    come back. Everything else is left alone by construction."""
    out = []
    for s in servers:
        c = assess(s.id, getattr(s, "name", "") or "",
                   getattr(s, "metadata", None) or {},
                   getattr(s, "created_at", None), _server_flavor_name(s), now)
        if c is not None:
            out.append(c)
    return out


def describe(c: Candidate) -> str:
    age = f"{c.age_hr:.1f}h old" if c.age_hr is not None else "age unknown"
    price = f"EUR {c.price_eur_hr:.2f}/hr" if c.price_eur_hr is not None else "price n/a"
    cost = f"~EUR {c.cost_eur:.2f} accrued" if c.cost_eur is not None else "cost n/a"
    return (f"{c.name} ({c.server_id})  {c.flavor or '?'}  {age}  {price}  {cost}\n"
            f"      [{c.bucket}] {c.why}")


def _confirm(prompt) -> bool:
    try:
        ans = input(prompt)
    except EOFError:
        return False
    return ans.strip().lower() in ("y", "yes")


def _reap_one(conn, c: Candidate) -> bool:
    """Delete one candidate's server (verified) plus its same-named keypair and
    security group. Returns True when the server is verifiably gone."""
    try:
        server = conn.compute.get_server(c.server_id)
        _delete_server_verified(conn, server)
        print(f"  deleted server {c.name} ({c.server_id}) (verified gone)")
    except Exception as exc:
        _stranded_banner(_cloud_name(conn), c.name, c.server_id, str(exc))
        return False
    try:
        conn.compute.delete_keypair(c.name, ignore_missing=True)
        print(f"  deleted keypair {c.name}")
    except Exception as exc:
        print(f"  keypair {c.name}: {type(exc).__name__}: {str(exc)[:120]}")
    sg = conn.network.find_security_group(c.name)
    if sg is not None:
        _delete_sg_with_retry(conn, sg.id)
    return True


def run_reap(cloud=None, region=None, yes=False, take_all=False) -> int:
    conn = connect(cloud=cloud, region=region)
    now = datetime.now(timezone.utc)
    servers = list(conn.compute.servers(details=True))
    cands = find_candidates(servers, now)
    foreign = len(servers) - len(cands)

    print(f"flux-compute reap: {len(cands)} flux-compute instance(s), "
          f"{foreign} foreign server(s) (never touched).")
    for c in cands:
        print(f"  - {describe(c)}")
    if not cands:
        return 0

    auto = [c for c in cands if c.auto_reapable]
    extra = [c for c in cands if not c.auto_reapable] if take_all else []
    reaped = set()

    if auto:
        if yes or _confirm(f"Delete {len(auto)} expired-stamped instance(s)? [y/N] "):
            for c in auto:
                if _reap_one(conn, c):
                    reaped.add(c.server_id)
        else:
            print("reap: expired-stamped instances left in place (not confirmed).")

    if extra:
        # Everything beyond the expired-stamped bucket always requires the
        # interactive confirmation; --yes deliberately does not extend to it.
        print(f"--all: {len(extra)} non-expired candidate(s) "
              "(keep-flagged / within-ttl / unstamped-legacy).")
        if _confirm(f"Also delete these {len(extra)} instance(s)? [y/N] "):
            for c in extra:
                if _reap_one(conn, c):
                    reaped.add(c.server_id)
        else:
            print("reap: --all candidates left in place (not confirmed).")

    strays_left = [c for c in cands if c.is_stray and c.server_id not in reaped]
    if strays_left:
        print(f"reap: {len(strays_left)} stray(s) remain and are still billing:", file=sys.stderr)
        for c in strays_left:
            print(f"  - {describe(c)}", file=sys.stderr)
        return 1
    if reaped:
        print(f"reap: done, {len(reaped)} instance(s) removed; no strays remain.")
    else:
        print("reap: no strays; nothing to do.")
    return 0
