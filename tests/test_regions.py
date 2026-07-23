"""Tests for the live per-region occupancy view. The pure core (fits math,
status assembly, rendering, JSON) needs no network; the live gather is driven by
a fake `connect`. No credentials."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from flux_compute import regions as regions_mod
from flux_compute.flavors import static_flavor_spec
from flux_compute.regions import (
    DEFAULT_FITS_FLAVOR,
    RegionQuota,
    RegionStatus,
    _axis_fits,
    _read_quota,
    build_region_status,
    fits_count,
    format_regions,
    gather_region_status,
    occupancy_line,
    occupancy_summary,
    regions_json,
    run_regions,
)

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(expires_delta_min, keep=False):
    md = {"flux_created_by": "flux-compute",
          "flux_expires_at": _iso(NOW + timedelta(minutes=expires_delta_min))}
    if keep:
        md["flux_keep"] = "true"
    return md


# --- fits math ----------------------------------------------------------------

def test_axis_fits_bounded_and_unbounded():
    assert _axis_fits(30, 34, 8) == 0          # 4 free / 8 vCPU -> 0
    assert _axis_fits(0, 34, 8) == 4           # 34 free / 8 -> 4
    assert _axis_fits(2, 10, 1) == 8           # 8 instance slots free
    assert _axis_fits(0, None, 8) is None      # axis unreported -> unbounded
    assert _axis_fits(0, -1, 8) is None        # OpenStack -1 unlimited -> unbounded


def test_fits_count_takes_the_binding_axis():
    spec = static_flavor_spec("b3-32")         # 8 vCPU, 32 GB
    # cores free 4 -> 0 fit; instances 8; RAM 60 GB -> 1. The min (cores) binds.
    q = RegionQuota(cores_used=30, cores_max=34, instances_used=2, instances_max=10,
                    ram_used_gb=360.0, ram_max_gb=420.0)
    assert fits_count(spec, q) == 0
    # Empty region: cores 34/8=4, inst 10, RAM 420/32=13 -> 4 fit.
    empty = RegionQuota(0, 34, 0, 10, 0.0, 420.0)
    assert fits_count(spec, empty) == 4


def test_fits_count_all_unbounded_is_none():
    spec = static_flavor_spec("b3-32")
    q = RegionQuota(0, -1, 0, -1, 0.0, -1.0)
    assert fits_count(spec, q) is None


# --- status assembly (pure) ---------------------------------------------------

def _server(sid, name, md, age_hr, flavor="t2-le-45"):
    return SimpleNamespace(id=sid, name=name, metadata=md,
                           created_at=_iso(NOW - timedelta(hours=age_hr)),
                           flavor={"original_name": flavor})


def test_build_region_status_partitions_and_counts_fits():
    spec = static_flavor_spec("b3-32")
    q = RegionQuota(30, 34, 2, 10, 360.0, 420.0)
    servers = [
        _server("s1", "web-1", {}, 1.0, "b3-8"),                       # foreign
        _server("s2", "flux-compute-sweep-aa", _stamp(+60), 3.0),      # within-ttl
        _server("s3", "flux-compute-sweep-bb", _stamp(+60), 3.0),      # within-ttl
    ]
    st = build_region_status("GRA11", flavor="b3-32", spec=spec, quota=q,
                             servers=servers, now=NOW)
    assert st.ok and st.flavor_available
    assert st.fits == 0                       # occupied: no b3-32 fits
    assert st.foreign_count == 1
    assert {c.name for c in st.instances} == {"flux-compute-sweep-aa",
                                              "flux-compute-sweep-bb"}
    assert all(c.bucket == "within-ttl" for c in st.instances)


def test_build_region_status_absent_flavor_has_no_fits():
    q = RegionQuota(0, 34, 0, 10, 0.0, 420.0)
    st = build_region_status("SBG5", flavor="t2-le-45", spec=None, quota=q,
                             servers=[], now=NOW)
    assert st.ok and not st.flavor_available and st.fits is None


# --- occupancy summary --------------------------------------------------------

def test_occupancy_summary_groups_buckets_and_foreign():
    spec = static_flavor_spec("b3-32")
    q = RegionQuota(30, 34, 3, 10, 360.0, 420.0)
    servers = [
        _server("s1", "web-1", {}, 1.0, "b3-8"),                       # foreign
        _server("s2", "flux-compute-sweep-aa", _stamp(+60), 3.0),      # within-ttl
        _server("s3", "flux-compute-sweep-bb", _stamp(+60), 3.0),      # within-ttl
        _server("s4", "flux-compute-run-cc", _stamp(-5, keep=True), 5.0),  # keep
    ]
    st = build_region_status("GRA11", flavor="b3-32", spec=spec, quota=q,
                             servers=servers, now=NOW)
    summ = occupancy_summary(st)
    assert "2x flux-compute [within-ttl]" in summ
    assert "1x flux-compute [keep]" in summ
    assert "1 foreign server(s)" in summ


def test_occupancy_summary_none_when_idle_or_unreadable():
    q = RegionQuota(0, 34, 0, 10, 0.0, 420.0)
    idle = build_region_status("DE1", flavor="b3-32",
                               spec=static_flavor_spec("b3-32"), quota=q,
                               servers=[], now=NOW)
    assert occupancy_summary(idle) is None
    err = RegionStatus("DE1", ok=False, error="boom", flavor="b3-32",
                       flavor_available=False, quota=None, fits=None,
                       instances=(), foreign_count=0)
    assert occupancy_summary(err) is None


# --- quota reader -------------------------------------------------------------

def test_read_quota_converts_ram_mib_and_preserves_unlimited():
    lim = SimpleNamespace(
        total_cores_used=30, max_total_cores=34,
        total_instances_used=2, max_total_instances=10,
        total_ram_used=360 * 1024, max_total_ram_size=420 * 1024)
    q = _read_quota(lim)
    assert q.cores_used == 30 and q.cores_max == 34
    assert q.ram_used_gb == pytest.approx(360.0)
    assert q.ram_max_gb == pytest.approx(420.0)
    # -1 unlimited RAM stays a negative sentinel (unbounded), not a tiny fraction.
    lim2 = SimpleNamespace(total_ram_used=0, max_total_ram_size=-1,
                           total_cores_used=0, max_total_cores=-1,
                           total_instances_used=0, max_total_instances=-1)
    assert _read_quota(lim2).ram_max_gb == -1.0


# --- rendering ----------------------------------------------------------------

def _sample_statuses():
    spec = static_flavor_spec("b3-32")
    occupied = build_region_status(
        "GRA11", flavor="b3-32", spec=spec,
        quota=RegionQuota(30, 34, 2, 10, 360.0, 420.0),
        servers=[_server("s2", "flux-compute-sweep-aa", _stamp(+60), 3.0)], now=NOW)
    free = build_region_status(
        "DE1", flavor="b3-32", spec=spec,
        quota=RegionQuota(0, 34, 0, 10, 0.0, 420.0), servers=[], now=NOW)
    err = RegionStatus("UK1", ok=False, error="EndpointNotFound: no compute",
                       flavor="b3-32", flavor_available=False, quota=None,
                       fits=None, instances=(), foreign_count=0)
    return [occupied, free, err]


def test_format_regions_shows_fits_instances_and_errors():
    text = format_regions(_sample_statuses(), "b3-32")
    assert "fits 0x b3-32" in text                      # occupied region
    assert "fits 4x b3-32" in text                      # free region
    assert "flux-compute-sweep-aa" in text              # instance surfaced
    assert "[within-ttl]" in text
    assert "UK1" in text and "ERROR" in text            # error row surfaced
    assert "(idle: no running servers)" in text         # DE1 empty


def test_regions_json_shape_is_stable_for_the_frontend():
    payload = regions_json(_sample_statuses(), "b3-32")
    assert payload["flavor"] == "b3-32"
    by_region = {r["region"]: r for r in payload["regions"]}
    gra = by_region["GRA11"]
    assert gra["ok"] is True and gra["fits"] == 0
    assert gra["quota"]["vcpus"] == {"used": 30, "max": 34}
    assert gra["instances"][0]["name"] == "flux-compute-sweep-aa"
    assert gra["instances"][0]["bucket"] == "within-ttl"
    assert by_region["UK1"]["ok"] is False and by_region["UK1"]["error"]
    # Round-trips through json (no non-serializable objects leaked in).
    assert json.loads(json.dumps(payload))["flavor"] == "b3-32"


# --- live gather via a fake connect -------------------------------------------

class _FakeConn:
    def __init__(self, region, lim, servers, flavors):
        self.config = SimpleNamespace(region_name=region)
        self._lim = lim
        self._servers = servers
        self._flavors = flavors        # name -> flavor object (or absent)

    def get_compute_limits(self):
        return self._lim

    @property
    def compute(self):
        return SimpleNamespace(
            servers=lambda details=True: list(self._servers),
            find_flavor=lambda name: self._flavors.get(name))


def _flavor_obj(name, vcpus, ram_gb):
    return SimpleNamespace(name=name, vcpus=vcpus, ram=int(ram_gb * 1024))


def test_gather_region_status_reads_each_region(monkeypatch):
    lim = SimpleNamespace(
        total_cores_used=30, max_total_cores=34,
        total_instances_used=2, max_total_instances=10,
        total_ram_used=360 * 1024, max_total_ram_size=420 * 1024)
    servers = [_server("s2", "flux-compute-sweep-aa", _stamp(+60), 3.0)]
    flavors = {"b3-32": _flavor_obj("b3-32", 8, 32.0)}

    def fake_connect(cloud=None, region=None):
        return _FakeConn(region, lim, servers, flavors)

    monkeypatch.setattr(regions_mod, "connect", fake_connect)
    statuses = gather_region_status("flux-ovh", ["GRA11", "DE1"], "b3-32", now=NOW)
    assert [s.region for s in statuses] == ["GRA11", "DE1"]
    assert all(s.ok for s in statuses)
    assert statuses[0].fits == 0                          # 4 cores free / 8 -> 0
    assert statuses[0].instances[0].name == "flux-compute-sweep-aa"


def test_gather_region_status_unreadable_region_is_an_error_row(monkeypatch):
    def boom(cloud=None, region=None):
        raise RuntimeError("no compute endpoint in this region")

    monkeypatch.setattr(regions_mod, "connect", boom)
    statuses = gather_region_status("flux-ovh", ["GRA11"], "b3-32", now=NOW)
    assert statuses[0].ok is False
    assert "no compute endpoint" in statuses[0].error   # surfaced, not raised


def test_gather_region_status_clouds_yaml_pin_surfaces_whole(monkeypatch):
    def pinned(cloud=None, region=None):
        raise RuntimeError("Region 'DE1' was refused by the local clouds.yaml")

    monkeypatch.setattr(regions_mod, "connect", pinned)
    with pytest.raises(RuntimeError, match="refused by the local clouds.yaml"):
        gather_region_status("flux-ovh", ["DE1"], "b3-32", now=NOW)


def test_occupancy_line_best_effort_never_raises(monkeypatch):
    def boom(cloud=None, region=None):
        raise RuntimeError("down")

    monkeypatch.setattr(regions_mod, "connect", boom)
    assert occupancy_line("flux-ovh", "GRA11") is None   # swallowed to None


# --- run_regions CLI dispatch -------------------------------------------------

def test_run_regions_json_output(monkeypatch, capsys):
    statuses = _sample_statuses()
    monkeypatch.setattr(regions_mod, "gather_region_status",
                        lambda cloud, targets, flavor, **kw: statuses)
    monkeypatch.setattr(regions_mod, "_resolve_targets",
                        lambda cloud, region, regions: ["GRA11", "DE1", "UK1"])
    rc = run_regions(cloud="flux-ovh", flavor="b3-32", as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["flavor"] == "b3-32" and len(payload["regions"]) == 3


def test_run_regions_default_flavor_is_b3_32():
    assert DEFAULT_FITS_FLAVOR == "b3-32"


def test_resolve_targets_empty_regions_string_raises():
    from flux_compute.regions import _resolve_targets
    with pytest.raises(RuntimeError, match="named no region"):
        _resolve_targets("flux-ovh", None, " , ")
