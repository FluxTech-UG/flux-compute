"""Pure-logic tests for reap selection: expiry math, positive identification,
never-touch-foreign, keep exemption. No network, no credentials."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from flux_compute.provision import ttl_metadata, ttl_minutes_for
from flux_compute.reap import assess, find_candidates, parse_utc, warn_strays

NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(expires_delta_min, keep=False):
    md = {"flux_created_by": "flux-compute",
          "flux_expires_at": _iso(NOW + timedelta(minutes=expires_delta_min))}
    if keep:
        md["flux_keep"] = "true"
    return md


# --- TTL stamp creation --------------------------------------------------------

def test_ttl_minutes_margin_is_at_least_30():
    assert ttl_minutes_for(2) == 32        # short cap: fixed 30 min floor
    assert ttl_minutes_for(30) == 60


def test_ttl_minutes_margin_widens_to_quarter_of_long_caps():
    assert ttl_minutes_for(240) == 300     # 240 + max(30, 60) = 300


def test_ttl_metadata_round_trips_and_stamps_provenance():
    md = ttl_metadata(45, now=NOW)
    assert md["flux_created_by"] == "flux-compute"
    assert parse_utc(md["flux_expires_at"]) == NOW + timedelta(minutes=45)
    assert "flux_keep" not in md


def test_ttl_metadata_keep_flag():
    assert ttl_metadata(45, keep=True, now=NOW)["flux_keep"] == "true"


# --- positive identification ---------------------------------------------------

def test_foreign_server_is_never_a_candidate():
    # No stamp, no flux-compute name prefix: invisible to reap by construction.
    assert assess("id1", "web-1", {}, _iso(NOW), "b3-8", NOW) is None
    assert assess("id2", "gitlab-runner", {"owner": "ops"}, _iso(NOW), "c3-8", NOW) is None


def test_stamped_server_is_identified_even_without_the_name_prefix():
    c = assess("id3", "custom-name", _stamp(-10), _iso(NOW - timedelta(hours=1)), "b3-8", NOW)
    assert c is not None
    assert c.bucket == "expired-stamped"


def test_name_prefix_without_stamp_is_report_only_legacy():
    c = assess("id4", "flux-compute-run-2c60228b", {}, _iso(NOW - timedelta(hours=46)),
               "t2-le-45", NOW)
    assert c.bucket == "unstamped-legacy"
    assert not c.auto_reapable
    assert c.is_stray                       # drives nonzero exit + warnings
    assert "report-only" in c.why


# --- expiry math ---------------------------------------------------------------

def test_stamped_past_expiry_is_auto_reapable():
    c = assess("id5", "flux-compute-sweep-aa", _stamp(-1), _iso(NOW - timedelta(hours=2)),
               "b3-8", NOW)
    assert c.bucket == "expired-stamped"
    assert c.auto_reapable and c.is_stray


def test_stamped_within_ttl_is_not_taken():
    c = assess("id6", "flux-compute-sweep-bb", _stamp(+30), _iso(NOW), "b3-8", NOW)
    assert c.bucket == "within-ttl"
    assert not c.auto_reapable and not c.is_stray


def test_stamped_with_malformed_expiry_is_never_a_false_kill():
    md = {"flux_created_by": "flux-compute", "flux_expires_at": "not-a-date"}
    c = assess("id7", "flux-compute-run-cc", md, _iso(NOW), "b3-8", NOW)
    assert c.bucket == "unstamped-legacy"   # report-only, not auto-deleted
    assert not c.auto_reapable


# --- keep exemption -------------------------------------------------------------

def test_keep_flagged_is_never_auto_reaped_no_matter_how_old():
    c = assess("id8", "flux-compute-run-dd", _stamp(-60 * 24 * 7, keep=True),
               _iso(NOW - timedelta(days=7)), "t2-le-45", NOW)
    assert c.bucket == "keep"
    assert not c.auto_reapable
    assert not c.is_stray                   # deliberate, but listed prominently
    assert "never auto-reaped" in c.why


# --- cost estimate + fleet scan --------------------------------------------------

def test_cost_estimate_age_times_catalog_price():
    c = assess("id9", "flux-compute-run-ee", _stamp(-60), _iso(NOW - timedelta(hours=2)),
               "t2-le-45", NOW)
    assert c.age_hr == pytest.approx(2.0)
    assert c.price_eur_hr == pytest.approx(0.80)
    assert c.cost_eur == pytest.approx(1.60)


def test_find_candidates_partitions_a_mixed_fleet():
    servers = [
        SimpleNamespace(id="s1", name="web-1", metadata={}, created_at=_iso(NOW),
                        flavor={"original_name": "b3-8"}),                     # foreign
        SimpleNamespace(id="s2", name="flux-compute-run-old", metadata={},
                        created_at=_iso(NOW - timedelta(hours=46)),
                        flavor={"original_name": "t2-le-45"}),                 # legacy
        SimpleNamespace(id="s3", name="flux-compute-sweep-x", metadata=_stamp(-5),
                        created_at=_iso(NOW - timedelta(hours=1)),
                        flavor={"original_name": "b3-8"}),                     # expired
        SimpleNamespace(id="s4", name="flux-compute-run-keepme", metadata=_stamp(-5, keep=True),
                        created_at=_iso(NOW - timedelta(hours=9)),
                        flavor={"original_name": "t2-le-45"}),                 # keep
    ]
    cands = find_candidates(servers, NOW)
    by_id = {c.server_id: c for c in cands}
    assert set(by_id) == {"s2", "s3", "s4"}          # the foreign server never appears
    assert by_id["s2"].bucket == "unstamped-legacy"
    assert by_id["s3"].bucket == "expired-stamped"
    assert by_id["s4"].bucket == "keep"
    assert [c.server_id for c in cands if c.auto_reapable] == ["s3"]


# --- per-command stray warnings ---------------------------------------------------

def _conn_with(servers):
    return SimpleNamespace(compute=SimpleNamespace(servers=lambda details=True: servers))


def test_warn_strays_surfaces_expired_legacy_and_keep_but_not_inflight(capsys):
    servers = [
        SimpleNamespace(id="s1", name="web-1", metadata={}, created_at=_iso(NOW),
                        flavor={"original_name": "b3-8"}),                     # foreign: silent
        SimpleNamespace(id="s2", name="flux-compute-sweep-live", metadata=_stamp(+25),
                        created_at=_iso(NOW), flavor={"original_name": "b3-8"}),  # in flight: silent
        SimpleNamespace(id="s3", name="flux-compute-run-old", metadata={},
                        created_at=_iso(NOW - timedelta(hours=46)),
                        flavor={"original_name": "t2-le-45"}),                 # legacy stray
        SimpleNamespace(id="s4", name="flux-compute-run-keepme", metadata=_stamp(-5, keep=True),
                        created_at=_iso(NOW - timedelta(hours=2)),
                        flavor={"original_name": "t2-le-45"}),                 # kept: surfaced
    ]
    surfaced = warn_strays(_conn_with(servers), now=NOW)
    err = capsys.readouterr().err
    assert {c.server_id for c in surfaced} == {"s3", "s4"}
    assert "flux-compute-run-old" in err
    assert "flux-compute reap" in err                # points at the remedy
    assert "web-1" not in err                        # foreign never mentioned
    assert "sweep-live" not in err                   # in-flight run not flagged


def test_warn_strays_never_breaks_the_calling_command(capsys):
    def boom(details=True):
        raise RuntimeError("compute API down")
    conn = SimpleNamespace(compute=SimpleNamespace(servers=boom))
    assert warn_strays(conn, now=NOW) == []          # advisory: no raise
    assert "stray-instance check skipped" in capsys.readouterr().err
