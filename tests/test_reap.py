"""Pure-logic tests for reap selection: expiry math, positive identification,
never-touch-foreign, keep exemption. No network, no credentials."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from flux_compute import reap as reap_mod
from flux_compute.provision import ttl_metadata, ttl_minutes_for
from flux_compute.reap import assess, find_candidates, parse_utc, run_reap, warn_strays

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


# --- run_reap orchestration: --yes / --all scoping ---------------------------
#
# These lock the safety contract of the flags themselves: --yes must never
# extend beyond the expired-stamped bucket, --all always needs the interactive
# confirmation, and a foreign server is never deleted under any flag combination.

class _FakeReapConn:
    """Fake connection driving run_reap end to end; records every delete."""

    def __init__(self, servers):
        self._by_id = {s.id: s for s in servers}
        self.deleted = []
        self.compute = SimpleNamespace(
            servers=lambda details=True: list(self._by_id.values()),
            get_server=lambda sid: self._by_id[sid],
            delete_server=lambda sid, force=False: self.deleted.append(sid),
            wait_for_delete=lambda server, wait=0: None,
            delete_keypair=lambda name, ignore_missing=True: None,
        )
        self.network = SimpleNamespace(find_security_group=lambda name: None)
        self.config = SimpleNamespace(name="fake-cloud")


def _live_fleet(*, expired=True, within=True, keep=True, legacy=True, foreign=True):
    """A fleet stamped relative to the real clock (run_reap reads now itself)."""
    now = datetime.now(timezone.utc)

    def stamp(mins, keep_flag=False):
        md = {"flux_created_by": "flux-compute",
              "flux_expires_at": _iso(now + timedelta(minutes=mins))}
        if keep_flag:
            md["flux_keep"] = "true"
        return md

    created = _iso(now - timedelta(hours=1))
    mk = lambda sid, name, md: SimpleNamespace(
        id=sid, name=name, metadata=md, created_at=created,
        flavor={"original_name": "b3-8"})
    fleet = []
    if expired:
        fleet.append(mk("e1", "flux-compute-sweep-e1", stamp(-10)))
    if within:
        fleet.append(mk("w1", "flux-compute-sweep-w1", stamp(+60)))
    if keep:
        fleet.append(mk("k1", "flux-compute-run-k1", stamp(-10, keep_flag=True)))
    if legacy:
        fleet.append(mk("l1", "flux-compute-run-l1", {}))
    if foreign:
        fleet.append(mk("f1", "web-1", {}))
    return fleet


def _wire(monkeypatch, conn, confirm=None):
    monkeypatch.setattr(reap_mod, "connect", lambda cloud=None, region=None: conn)
    if confirm is not None:
        monkeypatch.setattr(reap_mod, "_confirm", confirm)


def _no_prompt(prompt):
    raise AssertionError(f"unexpected interactive prompt: {prompt!r}")


def test_run_reap_yes_deletes_only_expired_stamped_without_prompt(monkeypatch):
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn, confirm=_no_prompt)     # any prompt fails the test
    rc = run_reap(yes=True)
    assert conn.deleted == ["e1"]                    # not within-ttl/keep/legacy, never foreign
    assert rc == 1                                   # the legacy stray remains


def test_run_reap_yes_all_declined_still_deletes_only_expired(monkeypatch):
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn, confirm=lambda prompt: False)   # "n" to the --all prompt
    rc = run_reap(yes=True, take_all=True)
    assert conn.deleted == ["e1"]                    # --yes must never extend to the extra buckets
    assert rc == 1


def test_run_reap_all_confirmed_takes_extra_buckets_but_never_foreign(monkeypatch):
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn, confirm=lambda prompt: True)
    rc = run_reap(take_all=True)
    assert set(conn.deleted) == {"e1", "w1", "k1", "l1"}
    assert "f1" not in conn.deleted                  # foreign untouched under every flag
    assert rc == 0                                   # nothing stray remains


def test_run_reap_non_interactive_without_yes_deletes_nothing(monkeypatch):
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn)                         # real _confirm

    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)       # no tty: confirm reads EOF
    rc = run_reap(yes=False)
    assert conn.deleted == []                        # nothing deleted without consent
    assert rc == 1                                   # strays found and left


def test_run_reap_exit_zero_when_only_keep_flagged_remains(monkeypatch):
    conn = _FakeReapConn(_live_fleet(expired=False, within=False, legacy=False))
    _wire(monkeypatch, conn, confirm=_no_prompt)
    rc = run_reap(yes=True)
    assert conn.deleted == []
    assert rc == 0                                   # kept instance is deliberate, not a stray


# --- multi-region stray hunt --------------------------------------------------

def test_run_reap_scans_every_configured_region_by_default(monkeypatch):
    """With no --region/--regions, reap must sweep ALL configured regions: a
    multi-region sweep strands instances per region, and scanning only the
    default one would report 'no strays' while another region billed."""
    from flux_compute import reap as reap_mod

    seen = []
    monkeypatch.setattr(reap_mod, "configured_regions",
                        lambda cloud: ["GRA11", "DE1", "UK1"])
    monkeypatch.setattr(reap_mod, "_reap_region",
                        lambda cloud, region, yes, take_all, force: seen.append(region) or 0)

    rc = reap_mod.run_reap(cloud="flux-ovh")
    assert rc == 0
    assert seen == ["GRA11", "DE1", "UK1"]


def test_run_reap_explicit_region_scans_only_that_one(monkeypatch):
    from flux_compute import reap as reap_mod
    seen = []
    monkeypatch.setattr(reap_mod, "_reap_region",
                        lambda cloud, region, yes, take_all, force: seen.append(region) or 0)
    reap_mod.run_reap(cloud="flux-ovh", region="DE1")
    assert seen == ["DE1"]


def test_run_reap_unscannable_region_does_not_mask_the_others(monkeypatch):
    """One dead region must not abort the sweep -- the remaining regions are
    still scanned, and the exit code still reports the failure."""
    from flux_compute import reap as reap_mod
    seen = []

    def _fake(cloud, region, yes, take_all, force):
        seen.append(region)
        if region == "DE1":
            raise RuntimeError("no compute endpoint")
        return 0

    monkeypatch.setattr(reap_mod, "configured_regions",
                        lambda cloud: ["GRA11", "DE1", "UK1"])
    monkeypatch.setattr(reap_mod, "_reap_region", _fake)

    rc = reap_mod.run_reap(cloud="flux-ovh")
    assert seen == ["GRA11", "DE1", "UK1"]    # UK1 still scanned after DE1 failed
    assert rc == 1                            # and the failure is surfaced


def test_run_reap_empty_regions_string_raises():
    import pytest
    from flux_compute.reap import run_reap
    with pytest.raises(RuntimeError, match="named no region"):
        run_reap(cloud="flux-ovh", regions=" , ")


# --- --force: the non-interactive confirmation for the --all buckets ----------
#
# The alternative that was actually reached for to stop a runaway fleet was
# `yes | flux-compute reap --all`, which answers every prompt in the command
# blind. These lock the explicit flag's scope instead.

def test_force_with_all_takes_within_ttl_instances_without_a_prompt(monkeypatch):
    conn = _FakeReapConn(_live_fleet(foreign=True))
    _wire(monkeypatch, conn, confirm=_no_prompt)     # any prompt fails the test
    reap_mod.run_reap(cloud="c", region="DE1", take_all=True, force=True)
    assert set(conn.deleted) == {"e1", "w1", "k1", "l1"}   # every flux-compute bucket
    assert "f1" not in conn.deleted                        # foreign still never touched


def test_force_implies_yes_for_the_expired_bucket(monkeypatch):
    conn = _FakeReapConn(_live_fleet(within=False, keep=False, legacy=False))
    _wire(monkeypatch, conn, confirm=_no_prompt)
    reap_mod.run_reap(cloud="c", region="DE1", take_all=True, force=True)
    assert conn.deleted == ["e1"]


def test_force_without_all_is_refused(monkeypatch):
    """--force alone would be a confusing synonym for --yes; make it say so."""
    import pytest
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn, confirm=_no_prompt)
    with pytest.raises(RuntimeError, match="only means anything with --all"):
        reap_mod.run_reap(cloud="c", region="DE1", force=True)


def test_all_without_force_still_prompts(monkeypatch):
    """The interactive default is unchanged: declining leaves everything extra."""
    conn = _FakeReapConn(_live_fleet())
    _wire(monkeypatch, conn, confirm=lambda prompt: prompt.startswith("Delete"))
    reap_mod.run_reap(cloud="c", region="DE1", take_all=True)
    assert conn.deleted == ["e1"]      # expired only; the --all prompt was declined


# --- shared --regions parsing (was three implementations) --------------------

def test_reap_dedupes_a_repeated_region(monkeypatch):
    """reap and regions each re-implemented the comma parser WITHOUT sweep's
    de-duplication, so `--regions DE1,DE1` scanned DE1 twice."""
    seen = []
    monkeypatch.setattr(reap_mod, "_reap_region",
                        lambda cloud, region, yes, take_all, force: seen.append(region) or 0)
    reap_mod.run_reap(cloud="flux-ovh", regions="DE1,DE1,GRA11,DE1")
    assert seen == ["DE1", "GRA11"]


def test_configured_regions_refuses_to_degrade_to_one_region(monkeypatch):
    """An unreadable config used to silently become a single-region scan -- the
    exact partial stray hunt the function exists to prevent."""
    import pytest
    from flux_compute import auth

    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("clouds.yaml is malformed")

    import openstack.config
    monkeypatch.setattr(openstack.config, "OpenStackConfig", _Boom)
    with pytest.raises(RuntimeError, match="PER REGION"):
        auth.configured_regions("flux-ovh")
