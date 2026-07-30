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
only; everything --all adds needs a confirmation of its own -- interactive by
default, or `--all --force` for the non-interactive case (stopping a runaway
fleet from a script or a session with no tty).
Exit is nonzero when expired-stamped or unstamped-legacy strays remain.

`--sweep-local DIR` (repeatable) additionally reconciles the persisted
`.flux_attach*` records under DIR against the live listings: sweep persists a
per-job record plus a copy of the ephemeral SSH private key so a killed
orchestrator can re-attach, and a clean teardown deletes them -- so an
orchestrator that dies and is never resumed leaves record + key on disk
indefinitely. The sweep removes every attach dir whose instance is verifiably
absent from its (scanned) region; a record in an unscanned region, a live
instance's record, and an unreadable record are all left alone and said so.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from .auth import configured_regions, connect, parse_region_list
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

# Local attach-state layout, shared with sweep.py (the writer): a per-job
# directory <into>/<label>/.flux_attach/ holding record.json and the persisted
# ephemeral key id_key. Renamed variants (".flux_attach_stopped") are swept by
# the same prefix match.
ATTACH_DIR_PREFIX = ".flux_attach"
ATTACH_RECORD_NAME = "record.json"
ATTACH_KEY_NAME = "id_key"


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


def warn_strays(conn, full=False, now=None):
    """Advisory stray check run at the start of every CLI command that
    connects: surface flux-compute instances that are past TTL, legacy, or
    keep-flagged, with their accrued cost, and point at `flux-compute reap`.

    Never deletes anything (an unrelated command must not turn destructive).
    A failed server list is reported and skipped rather than breaking the
    command the user actually asked for. Returns the surfaced candidates.
    """
    try:
        now = now or datetime.now(timezone.utc)
        cands = find_candidates(list(conn.compute.servers(details=True)), now)
    except Exception as exc:
        print(f"note: stray-instance check skipped "
              f"({type(exc).__name__}: {str(exc)[:80]})", file=sys.stderr)
        return []
    noisy = [c for c in cands if c.is_stray or c.bucket == "keep"]
    if not noisy:
        return []
    if full:
        print("flux-compute instances needing attention:", file=sys.stderr)
        for c in noisy:
            print(f"  - {describe(c)}", file=sys.stderr)
        print("  run: flux-compute reap", file=sys.stderr)
        return noisy
    for c in noisy:
        cost = f"~EUR {c.cost_eur:.2f} burned" if c.cost_eur is not None else "cost n/a"
        age = f"{c.age_hr:.1f}h old" if c.age_hr is not None else "age unknown"
        kind = "stranded" if c.is_stray else "kept (--keep)"
        print(f"WARNING: {kind} instance {c.name} ({c.flavor or '?'}, {age}, {cost}) "
              f"- run: flux-compute reap", file=sys.stderr)
    return noisy


@dataclass(frozen=True)
class LocalAttach:
    """One persisted `.flux_attach*` directory found under a --sweep-local root."""

    path: str                     # the attach dir itself
    label: str | None
    region: str | None
    server_id: str | None
    name: str | None
    has_key: bool                 # an id_key private-key copy is present
    parse_error: str | None = None  # set when record.json could not be read


def find_local_attach_dirs(roots):
    """Every `.flux_attach*` dir holding a record.json under the given roots.

    A root that is not a directory raises (fail fast: a typo'd path must not
    read as "nothing to sweep"). A record that cannot be parsed comes back with
    `parse_error` set -- surfaced and left alone by the sweep, never guessed at.
    """
    out = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise RuntimeError(f"--sweep-local: not a directory: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            if not os.path.basename(dirpath).startswith(ATTACH_DIR_PREFIX):
                continue
            dirnames[:] = []      # an attach dir has no nested attach dirs
            if ATTACH_RECORD_NAME not in filenames:
                continue
            has_key = ATTACH_KEY_NAME in filenames
            try:
                with open(os.path.join(dirpath, ATTACH_RECORD_NAME)) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError) as exc:
                out.append(LocalAttach(dirpath, None, None, None, None, has_key,
                                       parse_error=f"{type(exc).__name__}: {str(exc)[:120]}"))
                continue
            out.append(LocalAttach(
                dirpath, rec.get("label") or None, rec.get("region") or None,
                rec.get("server_id") or None, rec.get("name") or None, has_key))
    return sorted(out, key=lambda e: e.path)


def sweep_local_attach(entries, live_by_region, emit=print, union_ok=False) -> int:
    """Reconcile persisted attach dirs against the live per-region listings.

    An attach dir whose instance is absent from its region's listing is
    finished-or-lost bookkeeping holding a now-useless private key: the whole
    dir is removed, matching the clean-teardown semantics of
    `sweep._clear_attach_record` (job_state then reads collected/pending from
    what was actually fetched). Three cases are never removed, each said out
    loud: a record whose region carries no evidence this run (not scanned, or
    recorded as null by an older schema when the scan was partial), a record
    whose instance is LIVE (an in-flight job -- collect it with
    `sweep --resume`, do not strip its key), and an unreadable record.

    `union_ok=True` asserts the scan was exhaustive -- every configured region
    listed successfully -- which is the one condition under which a
    null-region record may be judged against the union of all listings (a
    server can only exist in some region, and every region was looked at).
    Returns the number of dirs removed.
    """
    removed = 0
    for e in entries:
        if e.parse_error:
            emit(f"sweep-local: unreadable record left alone: {e.path} ({e.parse_error})")
            continue
        if e.region is None:
            if not union_ok:
                emit(f"sweep-local: left alone (no region recorded, and this run did "
                     f"not scan every configured region): {e.path}")
                continue
            live = set().union(*live_by_region.values()) if live_by_region else set()
            where = "every scanned region"
        elif e.region not in live_by_region:
            emit(f"sweep-local: left alone (region {e.region!r} not scanned this run): {e.path}")
            continue
        else:
            live = live_by_region[e.region]
            where = e.region
        if (e.server_id and e.server_id in live) or (e.name and e.name in live):
            emit(f"sweep-local: instance {e.name or e.server_id} is LIVE; left alone "
                 f"(in-flight job -- collect with `sweep --resume`): {e.path}")
            continue
        shutil.rmtree(e.path)
        removed += 1
        what = "record + id_key" if e.has_key else "record"
        emit(f"sweep-local: removed {e.path} ({what}; instance "
             f"{e.name or e.server_id or '?'} gone from {where})")
    return removed


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


def run_reap(cloud=None, region=None, regions=None, yes=False, take_all=False,
             force=False, sweep_local=()) -> int:
    """Hunt strays across regions, because servers and quota are BOTH per region.

    With no --region/--regions, every region the cloud entry is configured for is
    scanned: a multi-region sweep leaves instances in several regions, and a reap
    that looked only at the default one would print "no strays" while an instance
    billed elsewhere. A region that cannot be scanned is reported and the exit
    code is nonzero, but the remaining regions are still swept -- one unreachable
    region must never mask strays in the others.

    `force` is the explicit non-interactive confirmation for the --all buckets
    (see `_reap_region`).

    `sweep_local` roots are reconciled AFTER the region scans, against what those
    scans actually saw (`sweep_local_attach`): only a scanned region's absence
    counts as evidence that a persisted attach record's instance is gone.
    """
    if regions:
        targets = parse_region_list(regions)
    elif region:
        targets = [region]
    else:
        targets = configured_regions(cloud)

    if force and not take_all:
        raise RuntimeError(
            "--force only means anything with --all: on its own the expired-stamped "
            "bucket is already non-interactive via --yes. Use `--all --force` to take "
            "within-TTL / keep-flagged / legacy instances without a prompt.")

    live_by_region = {}
    if len(targets) == 1:
        rc = _reap_region(cloud, targets[0], yes, take_all, force,
                          live_out=live_by_region)
    else:
        print(f"reap: scanning {len(targets)} configured region(s): {', '.join(targets)}\n")
        rc = 0
        for r in targets:
            print(f"--- region {r}")
            try:
                rc |= _reap_region(cloud, r, yes, take_all, force,
                                   live_out=live_by_region)
            except Exception as exc:
                print(f"reap: region {r} could not be scanned: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
                rc = 1
            print()

    if sweep_local:
        # A null-region record (older schema) may be judged against the union of
        # listings only when this run scanned EVERY configured region cleanly.
        exhaustive = (not regions and not region
                      and set(live_by_region) == set(targets))
        entries = find_local_attach_dirs(sweep_local)
        if not entries:
            print("sweep-local: no attach records found; nothing to sweep.")
        else:
            n = sweep_local_attach(entries, live_by_region, union_ok=exhaustive)
            print(f"sweep-local: {n} of {len(entries)} attach record(s) removed.")
    return rc


def _reap_region(cloud, region, yes, take_all, force=False, live_out=None) -> int:
    conn = connect(cloud=cloud, region=region)
    now = datetime.now(timezone.utc)
    servers = list(conn.compute.servers(details=True))
    cands = find_candidates(servers, now)
    foreign = len(servers) - len(cands)
    reaped = set()

    def _record_live():
        # What this scan leaves alive, by id AND name (attach records may carry
        # either), for the --sweep-local reconciliation. Written even when the
        # region has zero servers: an empty listing is positive evidence.
        if live_out is None:
            return
        live = set()
        for s in servers:
            if s.id in reaped:
                continue
            live.add(s.id)
            n = getattr(s, "name", "") or ""
            if n:
                live.add(n)
        live_out[region] = live

    print(f"flux-compute reap: {len(cands)} flux-compute instance(s), "
          f"{foreign} foreign server(s) (never touched).")
    for c in cands:
        print(f"  - {describe(c)}")
    if not cands:
        _record_live()
        return 0

    auto = [c for c in cands if c.auto_reapable]
    extra = [c for c in cands if not c.auto_reapable] if take_all else []

    if auto:
        if yes or force or _confirm(f"Delete {len(auto)} expired-stamped instance(s)? [y/N] "):
            for c in auto:
                if _reap_one(conn, c):
                    reaped.add(c.server_id)
        else:
            print("reap: expired-stamped instances left in place (not confirmed).")

    if extra:
        # Everything beyond the expired-stamped bucket needs an explicit
        # confirmation, because it can kill a live fleet: --yes deliberately does
        # not extend to it. `--force` is that confirmation given up front, for the
        # case the interactive prompt cannot serve -- stopping a runaway fleet
        # from a script or a session with no tty. It is a separate flag rather
        # than a widened --yes so that "skip the routine prompt" and "kill running
        # work" can never be the same keystroke, and it beats the alternative that
        # was actually reached for (`yes | flux-compute reap --all`), which
        # answers every prompt in the command blind, including ones added later.
        print(f"--all: {len(extra)} non-expired candidate(s) "
              "(keep-flagged / within-ttl / unstamped-legacy).")
        if force:
            print(f"--force: taking all {len(extra)} without confirmation.")
        if force or _confirm(f"Also delete these {len(extra)} instance(s)? [y/N] "):
            for c in extra:
                if _reap_one(conn, c):
                    reaped.add(c.server_id)
        else:
            print("reap: --all candidates left in place (not confirmed).")

    _record_live()
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
