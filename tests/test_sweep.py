"""Pure-logic tests for the sweep helpers. No network, no credentials."""
import os
import threading
import time
from types import SimpleNamespace

import pytest

from flux_compute import provision, sweep
from flux_compute.detach import AttachRecord, PollOutcome
from flux_compute.provision import IngressCheck
from flux_compute.sweep import (
    RegionDrop,
    Shard,
    _clear_attach_record,
    _failure_status,
    _fan_out,
    _finalize,
    _load_attach_records,
    _prepare_shards,
    _status_for_outcome,
    _write_attach_record,
    allocate_concurrency,
    budget_guard,
    budget_guard_shards,
    clamp_concurrency,
    parse_jobs,
    parse_regions,
    run_sweep,
    shard_jobs,
    worst_case_eur,
)


def test_parse_label_equals_params():
    jobs = parse_jobs("alpha = N_x=128\nbeta = N_x=256\n")
    assert jobs == [("alpha", "N_x=128"), ("beta", "N_x=256")]


def test_parse_skips_blanks_and_comments():
    jobs = parse_jobs("# header\n\nonly = x\n  # indented comment\n")
    assert jobs == [("only", "x")]


def test_parse_line_without_equals_is_label_and_params():
    assert parse_jobs("spec_operating_point\n") == [("spec_operating_point", "spec_operating_point")]


def test_duplicate_label_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("a = 1\na = 2\n")


def test_label_with_slash_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("bad/label = 1\n")


def test_empty_jobs_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("# only comments\n\n")


# --- inline comments in the jobs file (the 16-VM fleet that did no work) ------

def test_parse_strips_an_inline_comment_from_the_params():
    """The incident: the comment reached the remote inside $FLUX_JOB, so the
    job's own selector matched nothing and every VM in the fleet ran empty."""
    jobs = parse_jobs("heavy = --select nx128   # rerun: OOM'd on b3-32\n")
    assert jobs == [("heavy", "--select nx128")]


def test_parse_strips_an_inline_comment_from_a_bare_label():
    assert parse_jobs("spec_point    # the anchor cell\n") == [
        ("spec_point", "spec_point")]


def test_parse_strips_an_inline_comment_before_the_equals():
    """Stripping happens before the label/params split, so a comment containing
    '=' cannot smuggle itself into the parse."""
    assert parse_jobs("alpha  # note: N_x=128 was the old value\n") == [
        ("alpha", "alpha")]


def test_parse_keeps_an_uncommented_line_byte_for_byte():
    """The uncommented forms every existing jobs file uses must be untouched."""
    assert parse_jobs("a = --flag x --other y\n") == [("a", "--flag x --other y")]
    assert parse_jobs("a=x\n") == [("a", "x")]


def test_parse_keeps_a_hash_that_is_not_a_comment():
    """A '#' glued to a value is part of the value (the shell reads it that way
    too); only a whitespace- or line-initial '#' opens a comment."""
    assert parse_jobs("a = --tag run#3\n") == [("a", "--tag run#3")]
    assert parse_jobs('a = --note "phase # two"\n') == [("a", '--note "phase # two"')]
    assert parse_jobs("a = --note 'x # y' --z 1\n") == [("a", "--note 'x # y' --z 1")]


def test_parse_strips_trailing_whitespace_left_by_a_comment():
    jobs = parse_jobs("a = x   #c\nb = y\t# c2\n")
    assert jobs == [("a", "x"), ("b", "y")]


def test_strip_inline_comment_edge_forms():
    from flux_compute.sweep import strip_inline_comment
    assert strip_inline_comment("") == ""
    assert strip_inline_comment("#whole line") == ""
    assert strip_inline_comment("   # indented") == ""
    assert strip_inline_comment("value#glued") == "value#glued"
    assert strip_inline_comment("value #detached") == "value"
    assert strip_inline_comment("a = b # c # d") == "a = b"
    assert strip_inline_comment('"#quoted"') == '"#quoted"'


# --- job state: what --resume may launch --------------------------------------

def test_job_state_distinguishes_pending_in_flight_and_collected(tmp_path):
    from flux_compute.sweep import ATTACH_DIR, ATTACH_RECORD, JOB_LOG, job_state
    into = str(tmp_path / "cloud-sweep")

    assert job_state(into, "never_started") == "pending"

    os.makedirs(os.path.join(into, "running", ATTACH_DIR))
    open(os.path.join(into, "running", ATTACH_DIR, ATTACH_RECORD), "w").close()
    assert job_state(into, "running") == "in_flight"

    os.makedirs(os.path.join(into, "finished"))
    open(os.path.join(into, "finished", JOB_LOG), "w").close()
    assert job_state(into, "finished") == "collected"


# --- upload excludes: the results tree must never re-enter an upload ----------

def test_upload_excludes_covers_an_into_dir_inside_the_upload(tmp_path):
    from flux_compute.sweep import _upload_excludes
    src = str(tmp_path / "repo")
    assert _upload_excludes(src, os.path.join(src, "cloud-sweep")) == ("/cloud-sweep",)
    assert _upload_excludes(src, os.path.join(src, "outputs", "fleet")) == ("/outputs/fleet",)


def test_upload_excludes_is_empty_when_into_is_outside_the_upload(tmp_path):
    from flux_compute.sweep import _upload_excludes
    src = str(tmp_path / "repo")
    assert _upload_excludes(src, str(tmp_path / "elsewhere")) == ()


def test_worst_case_cost():
    assert worst_case_eur(3, 0.80, 30) == pytest.approx(1.20)   # 3 * 0.80 * 0.5
    assert worst_case_eur(10, 0.80, 6) == pytest.approx(0.80)   # 10 * 0.80 * 0.1


def test_worst_case_price_unknown_is_none():
    assert worst_case_eur(5, None, 30) is None


def test_budget_guard_priced_under_budget_returns_worst_case():
    # 200 b3-8 jobs at 0.0512/hr, 30-min cap -> 200 * 0.0512 * 0.5 = 5.12.
    wc = budget_guard("b3-8", 0.0512, 200, 30, budget_eur=10.0)
    assert wc == pytest.approx(5.12)


def test_budget_guard_priced_over_budget_raises():
    with pytest.raises(RuntimeError, match="exceeds budget"):
        budget_guard("b3-8", 0.0512, 200, 30, budget_eur=4.0)


def test_budget_guard_unpriced_with_budget_refuses_and_names_flavor():
    with pytest.raises(RuntimeError) as exc:
        budget_guard("b3-8-flex", None, 200, 30, budget_eur=10.0)
    msg = str(exc.value)
    assert "b3-8-flex" in msg          # names the offending flavor
    assert "no known price" in msg     # and says why it refuses


def test_budget_guard_unpriced_without_budget_returns_none():
    # No budget set: an unknown price is not fatal, the guard just cannot bound it.
    assert budget_guard("b3-8-flex", None, 200, 30, budget_eur=None) is None


# --- quota-aware concurrency clamp -------------------------------------------

def test_clamp_concurrency_bounded_by_core_quota():
    # 8 cores free, 2 vCPU each -> 4 instances fit; well under the 10 ceiling.
    assert clamp_concurrency(10, 2, cores_used=0, cores_max=8,
                             instances_used=0, instances_max=100) == 4


def test_clamp_concurrency_bounded_by_instance_quota():
    # Cores allow 50, but only 3 instance slots free.
    assert clamp_concurrency(10, 2, cores_used=0, cores_max=100,
                             instances_used=0, instances_max=3) == 3


def test_clamp_concurrency_user_ceiling_wins_when_quota_is_ample():
    assert clamp_concurrency(4, 2, cores_used=0, cores_max=1000,
                             instances_used=0, instances_max=1000) == 4


def test_clamp_concurrency_no_core_headroom_raises():
    # 5 cores free but the flavor needs 15 -> not even one fits.
    with pytest.raises(RuntimeError, match="cannot fit even one"):
        clamp_concurrency(4, 15, cores_used=10, cores_max=15,
                          instances_used=0, instances_max=100)


def test_clamp_concurrency_no_instance_headroom_raises():
    with pytest.raises(RuntimeError, match="cannot fit even one"):
        clamp_concurrency(4, 2, cores_used=0, cores_max=100,
                          instances_used=5, instances_max=5)


def test_clamp_concurrency_unlimited_quota_sentinel_is_no_bound():
    # OpenStack reports unlimited as -1: must not falsely refuse or go negative.
    assert clamp_concurrency(6, 4, cores_used=30, cores_max=-1,
                             instances_used=9, instances_max=-1) == 6
    # Mixed: cores unlimited, instances still bind.
    assert clamp_concurrency(10, 4, cores_used=30, cores_max=-1,
                             instances_used=0, instances_max=3) == 3


# --- fan-out respects the clamp and completes every job ----------------------

def test_fan_out_bounds_concurrency_and_runs_every_job():
    K = 4
    jobs = list(range(3 * K))          # a 3K-job sweep
    lock = threading.Lock()
    state = {"live": 0, "peak": 0}

    def run_one(job):
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        time.sleep(0.02)
        with lock:
            state["live"] -= 1
        return (job, 0, "ok")

    results = _fan_out(jobs, run_one, K)
    assert len(results) == len(jobs)   # all jobs completed
    assert state["peak"] <= K          # never exceeded the clamped concurrency


# --- create-failure labelling ------------------------------------------------

def test_failure_status_flags_quota_and_capacity():
    assert _failure_status(Exception("Quota exceeded for instances")).startswith("quota/capacity")
    assert _failure_status(Exception("No valid host was found")).startswith("quota/capacity")


def test_failure_status_generic_error_stays_generic():
    assert _failure_status(RuntimeError("SSH never opened within timeout")).startswith("error")


def test_failure_status_flags_teardown_strand():
    from flux_compute.provision import TeardownStrandError
    exc = TeardownStrandError("stranded: server srv-1 could not be verifiably deleted")
    assert _failure_status(exc).startswith("STRANDED")


# --- follow-outcome -> (rc, status) ------------------------------------------

def test_status_ok_on_clean_exit():
    assert _status_for_outcome(PollOutcome(rc=0, reason="done", output_size=10)) == (0, "ok")


def test_status_nonzero_job():
    rc, status = _status_for_outcome(PollOutcome(rc=3, reason="done", output_size=10))
    assert rc == 3 and "nonzero" in status


def test_status_rc124_is_always_a_cap_timeout():
    """124 is `timeout`'s own TERM code: unambiguous, no elapsed basis needed."""
    rc, status = _status_for_outcome(PollOutcome(rc=124, reason="done", output_size=10))
    assert rc == 124 and "timed out" in status


def test_status_rc137_at_the_cap_is_a_timeout():
    """137 that ran (almost) its whole cap IS the cap's kill-after escalation."""
    rc, status = _status_for_outcome(
        PollOutcome(rc=137, reason="done", output_size=10),
        elapsed_s=1795, cap_seconds=1800)
    assert rc == 137 and "timed out" in status


def test_status_rc137_far_short_of_the_cap_is_never_called_a_timeout():
    """The misclassification that sent an OOM hunt after phantom slow jobs: a
    SIGKILL at 3 minutes of a 30-minute cap is not a timeout, whatever else it is."""
    rc, status = _status_for_outcome(
        PollOutcome(rc=137, reason="done", output_size=10),
        elapsed_s=180, cap_seconds=1800)
    assert rc == 137
    assert "timed out" not in status and "timeout" not in status
    assert "cause unknown" in status


def test_status_rc137_with_confirmed_oom_says_so():
    from flux_compute.provision import OomProbe
    rc, status = _status_for_outcome(
        PollOutcome(rc=137, reason="done", output_size=10),
        elapsed_s=180, cap_seconds=1800,
        oom=OomProbe(confirmed=True, summary="Out of memory: Killed process 941 (python)",
                     read_ok=True))
    assert rc == 137
    assert "OOM-killed" in status and "oom-killer confirmed" in status
    assert "timed out" not in status


def test_status_local_deadline_is_a_failure_record():
    rc, status = _status_for_outcome(PollOutcome(rc=None, reason="deadline", output_size=10))
    assert rc == -1 and "DEADLINE" in status


def test_status_unreachable_names_the_instance_as_the_cause():
    """Distinct from a deadline: the job did not merely run out of local time,
    the instance stopped answering while we were demonstrably able to reach it."""
    rc, status = _status_for_outcome(
        PollOutcome(rc=None, reason="unreachable", output_size=10))
    assert rc == -1 and "UNREACHABLE" in status
    assert "instance is the cause" in status


# --- attach-record persistence (the sweep --resume handoff) -------------------

def _make_keyfile(tmp_path):
    kf = tmp_path / "id_ed25519"
    kf.write_text("PRIVATE KEY MATERIAL")
    return str(kf)


def test_attach_record_write_load_clear_round_trip(tmp_path):
    into = str(tmp_path / "cloud-sweep")
    dest = os.path.join(into, "alpha")
    os.makedirs(dest, exist_ok=True)
    keyfile = _make_keyfile(tmp_path)

    rec = _write_attach_record(
        dest, label="alpha", cloud="flux-ovh", region="GRA11",
        name="flux-compute-sweep-abcd1234", server_id="srv-1", ip="1.2.3.4",
        keyfile=keyfile, remote_script="job.sh", fetch="out", into=into,
        cap_seconds=1800)

    # The key is copied durably (0600) into the attach dir, not left in temp.
    assert os.path.isfile(rec.keyfile)
    assert rec.keyfile.startswith(dest)
    assert (os.stat(rec.keyfile).st_mode & 0o777) == 0o600

    loaded = _load_attach_records(into)
    assert [r.label for r in loaded] == ["alpha"]
    assert loaded[0].server_id == "srv-1" and loaded[0].cap_seconds == 1800

    _clear_attach_record(dest)
    assert _load_attach_records(into) == []      # gone: job finalized


def test_load_attach_records_empty_when_no_into_dir(tmp_path):
    assert _load_attach_records(str(tmp_path / "nope")) == []


def _wire_finalize(monkeypatch):
    """Count the log pull, the strict fetch and the best-effort (partial) fetch."""
    calls = {"log": 0, "fetch": 0, "partial": 0}
    monkeypatch.setattr(sweep, "pull_job_log",
                        lambda ip, kf, path: calls.__setitem__("log", calls["log"] + 1))
    monkeypatch.setattr(sweep, "_rsync_down",
                        lambda ip, kf, remote, local: calls.__setitem__("fetch", calls["fetch"] + 1))
    monkeypatch.setattr(sweep, "rsync_down_best_effort",
                        lambda ip, kf, remote, local: calls.__setitem__("partial", calls["partial"] + 1))
    monkeypatch.setattr(sweep, "probe_oom_kill", lambda ip, kf: None)
    return calls


def test_finalize_clean_run_pulls_log_and_artifacts(tmp_path, monkeypatch):
    dest = str(tmp_path / "alpha")
    os.makedirs(dest, exist_ok=True)
    calls = _wire_finalize(monkeypatch)

    rc, status = _finalize("ip", "kf", dest, "out", PollOutcome(rc=0, reason="done", output_size=5))
    assert rc == 0 and status == "ok"
    assert calls == {"log": 1, "fetch": 1, "partial": 0}


def test_finalize_fetches_partial_artifacts_before_teardown_on_deadline(tmp_path, monkeypatch):
    """A deadline abort used to fetch NOTHING, so every checkpoint the job had
    already written died with the VM. It must fetch best-effort instead."""
    dest = str(tmp_path / "alpha")
    os.makedirs(dest, exist_ok=True)
    calls = _wire_finalize(monkeypatch)

    rc, status = _finalize("ip", "kf", dest, "out",
                           PollOutcome(rc=None, reason="deadline", output_size=5))
    assert rc == -1 and "DEADLINE" in status
    assert calls == {"log": 1, "fetch": 0, "partial": 1}   # partial, never the strict fetch


def test_finalize_fetches_partial_artifacts_on_a_nonzero_exit(tmp_path, monkeypatch):
    dest = str(tmp_path / "alpha")
    os.makedirs(dest, exist_ok=True)
    calls = _wire_finalize(monkeypatch)

    rc, status = _finalize("ip", "kf", dest, "out", PollOutcome(rc=2, reason="done", output_size=5))
    assert rc == 2 and "nonzero" in status
    assert calls == {"log": 1, "fetch": 0, "partial": 1}


def test_finalize_probes_the_kernel_log_for_a_sub_cap_sigkill(tmp_path, monkeypatch):
    """The OOM evidence dies with the instance, so the probe must happen here --
    before teardown -- and only for the ambiguous sub-cap 137."""
    from flux_compute.provision import OomProbe
    dest = str(tmp_path / "alpha")
    os.makedirs(dest, exist_ok=True)
    _wire_finalize(monkeypatch)
    probed = []
    monkeypatch.setattr(sweep, "probe_oom_kill", lambda ip, kf: probed.append(ip) or OomProbe(
        confirmed=True, summary="Out of memory: Killed process 941 (python)", read_ok=True))

    rc, status = _finalize("ip", "kf", dest, "out", PollOutcome(rc=137, reason="done", output_size=5),
                           elapsed_s=120, cap_seconds=3600)
    assert probed == ["ip"] and "OOM-killed" in status

    # A 137 that reached its cap needs no probe: the cap explains it.
    probed.clear()
    rc, status = _finalize("ip", "kf", dest, "out", PollOutcome(rc=137, reason="done", output_size=5),
                           elapsed_s=3599, cap_seconds=3600)
    assert probed == [] and "timed out" in status


# --- multi-region sharding ----------------------------------------------------

def test_parse_regions_orders_and_dedupes():
    assert parse_regions("GRA11, DE1 ,UK1") == ["GRA11", "DE1", "UK1"]
    assert parse_regions("DE1,DE1,GRA11") == ["DE1", "GRA11"]


def test_parse_regions_empty_raises():
    # An empty --regions is a mistake, not a silent fallback to one region.
    with pytest.raises(RuntimeError, match="named no region"):
        parse_regions(" , ")


def test_allocate_concurrency_spreads_across_regions_before_stacking():
    # Ceiling below the region count: distinct regions first, not one saturated.
    assert allocate_concurrency([2, 2, 2], 2) == [1, 1, 0]
    assert allocate_concurrency([2, 2, 2], 3) == [1, 1, 1]
    assert allocate_concurrency([2, 2, 2], 5) == [2, 2, 1]


def test_allocate_concurrency_never_exceeds_a_regions_cap():
    alloc = allocate_concurrency([1, 4], 10)
    assert alloc == [1, 4]                    # capped per region
    assert sum(alloc) == 5                    # and by the sum of caps


def test_allocate_concurrency_five_gpu_regions_at_two_each():
    # The live OVH shape: 34 vCPU/region / 15 vCPU per V100S = 2 per region.
    assert allocate_concurrency([2, 2, 2, 2, 2], 10) == [2, 2, 2, 2, 2]


def test_allocate_concurrency_rejects_nonpositive_ceiling():
    with pytest.raises(RuntimeError, match="at least 1"):
        allocate_concurrency([2, 2], 0)


def test_shard_jobs_deals_in_proportion_to_concurrency():
    jobs = [(f"j{i}", "p") for i in range(12)]
    shards = shard_jobs(jobs, [2, 1])
    assert len(shards[0]) == 8 and len(shards[1]) == 4      # 2:1 split
    assert sum(len(s) for s in shards) == len(jobs)         # nothing dropped


def test_shard_jobs_skips_zero_weight_regions():
    jobs = [(f"j{i}", "p") for i in range(4)]
    shards = shard_jobs(jobs, [2, 0, 2])
    assert shards[1] == []
    assert len(shards[0]) == 2 and len(shards[2]) == 2


def test_shard_jobs_fewer_jobs_than_regions_spreads_out():
    shards = shard_jobs([("a", "p"), ("b", "p")], [1, 1, 1])
    assert [len(s) for s in shards] == [1, 1, 0]


def test_shard_jobs_no_allocation_raises():
    with pytest.raises(RuntimeError, match="no region has any concurrency"):
        shard_jobs([("a", "p")], [0, 0])


def test_budget_guard_shards_sums_across_regions():
    # Same flavor in two regions: the cap is on the whole sweep, not per region.
    entries = [("b3-8", 0.0512, 100), ("b3-8", 0.0512, 100)]
    total = budget_guard_shards(entries, 30, budget_eur=10.0)
    assert total == pytest.approx(200 * 0.0512 * 0.5)


def test_budget_guard_shards_over_budget_raises_on_the_total():
    # Each shard alone is under the cap; together they exceed it.
    entries = [("b3-8", 0.0512, 100), ("b3-8", 0.0512, 100)]
    with pytest.raises(RuntimeError, match="exceeds budget"):
        budget_guard_shards(entries, 30, budget_eur=3.0)


def test_budget_guard_shards_unpriced_region_named_when_budget_set():
    entries = [("b3-8", 0.0512, 10), ("t2-le-45-flex", None, 10)]
    with pytest.raises(RuntimeError) as exc:
        budget_guard_shards(entries, 30, budget_eur=10.0)
    assert "t2-le-45-flex" in str(exc.value)
    assert "no known price" in str(exc.value)


# --- graceful-degrade region pre-flight ---------------------------------------
#
# The incident this closes: a multi-region sweep was refused ENTIRELY because a
# couple of its regions had no headroom (other fleets live there). Now unfit
# regions are dropped with a warning and the sweep runs on the rest; it refuses
# only when NONE fit, or under --strict-regions.

def _fake_shard(region, cap=2):
    spec = SimpleNamespace(flavor="t2-le-45", est_cost_eur_hr=0.80,
                           gpu_model="Tesla V100S 32GB", image="img",
                           network="Ext-Net")
    return Shard(region=region, spec=spec, vcpus=15, cap=cap)


def _wire_prepare(monkeypatch, fit_regions, drop_reason="RuntimeError: quota fits zero"):
    """Drive _prepare_shards without network: `fit_regions` succeed, the rest
    raise. occupancy_line is stubbed so no connection is attempted."""
    def fake_prepare_shard(cloud, region, flavor, image, max_parallel):
        if region in fit_regions:
            return _fake_shard(region)
        raise RuntimeError("quota fits zero")
    monkeypatch.setattr(sweep, "_prepare_shard", fake_prepare_shard)
    monkeypatch.setattr("flux_compute.regions.occupancy_line",
                        lambda cloud, region: "2x flux-compute [within-ttl]")


def _jobs_file(tmp_path):
    p = tmp_path / "jobs.txt"
    p.write_text("a = 1\nb = 2\nc = 3\nd = 4\n")
    return str(p)


def test_prepare_shards_partitions_fit_and_unfit(monkeypatch):
    _wire_prepare(monkeypatch, fit_regions={"DE1", "BHS5"})
    shards, drops = _prepare_shards(None, ["GRA11", "DE1", "UK1", "BHS5"], None, None, 12)
    assert {s.region for s in shards} == {"DE1", "BHS5"}
    assert {d.region for d in drops} == {"GRA11", "UK1"}
    assert all("quota fits zero" in d.reason for d in drops)


def test_prepare_shards_clouds_yaml_pin_still_raises(monkeypatch):
    def pinned(cloud, region, flavor, image, max_parallel):
        raise RuntimeError("Region 'UK1' was refused by the local clouds.yaml")
    monkeypatch.setattr(sweep, "_prepare_shard", pinned)
    with pytest.raises(RuntimeError, match="refused by the local clouds.yaml"):
        _prepare_shards(None, ["UK1"], None, None, 12)


def test_sweep_drops_unfit_regions_and_proceeds(monkeypatch, tmp_path, capsys):
    _wire_prepare(monkeypatch, fit_regions={"DE1"})
    rc = run_sweep(cloud=None, regions="GRA11,DE1,UK1", jobs_file=_jobs_file(tmp_path),
                   plan_only=True)
    out = capsys.readouterr().out
    assert rc == 0                                          # ran the plan on DE1
    assert "dropping region GRA11" in out
    assert "dropping region UK1" in out
    assert "occupied by: 2x flux-compute [within-ttl]" in out
    assert "proceeding on 1 of 3 requested region(s)" in out


def test_sweep_strict_regions_refuses_on_any_unfit(monkeypatch, tmp_path):
    _wire_prepare(monkeypatch, fit_regions={"DE1"})
    with pytest.raises(RuntimeError, match="these requested regions cannot run this sweep"):
        run_sweep(cloud=None, regions="GRA11,DE1", jobs_file=_jobs_file(tmp_path),
                  plan_only=True, strict_regions=True)


def test_sweep_refuses_when_no_region_fits(monkeypatch, tmp_path):
    _wire_prepare(monkeypatch, fit_regions=set())           # every region unfit
    with pytest.raises(RuntimeError, match="no requested region can run this sweep"):
        run_sweep(cloud=None, regions="GRA11,UK1", jobs_file=_jobs_file(tmp_path),
                  plan_only=True)


def test_region_drop_label_defaults_when_region_is_none():
    assert RegionDrop(region=None, reason="x").label == "(default region)"
    assert RegionDrop(region="DE1", reason="x").label == "DE1"


# --- the record is written BEFORE boot (no unfetchable orphans) ---------------

def test_pending_record_exists_before_the_instance_boots(tmp_path, monkeypatch):
    """A launcher killed during boot must leave a record naming the VM, or the
    instance becomes an orphan that no --resume can even find."""
    from contextlib import contextmanager
    from flux_compute.sweep import ATTACH_DIR, ATTACH_RECORD, _make_run_one
    from flux_compute.detach import AttachRecord

    into = str(tmp_path / "cloud-sweep")
    seen = {}

    @contextmanager
    def fake_instance(conn, spec, name, ttl_minutes, keep=False):
        path = os.path.join(into, "alpha", ATTACH_DIR, ATTACH_RECORD)
        seen["existed_at_boot"] = os.path.isfile(path)
        with open(path) as fh:
            seen["rec"] = AttachRecord.from_json(fh.read())
        raise RuntimeError("boot failed")       # never yields

    monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: object())
    monkeypatch.setattr(sweep, "_gpu_instance", fake_instance)

    shard = Shard(region="GRA11", spec=SimpleNamespace(flavor="b3-8"), vcpus=8, cap=1)
    run_one = _make_run_one("flux-ovh", shard, [], "job.sh", "out", into, 30)
    label, rc, status = run_one(("alpha", "--x 1"))

    assert seen["existed_at_boot"] is True
    assert seen["rec"].name.startswith("flux-compute-sweep-")   # the teardown handle
    assert seen["rec"].attachable is False                      # no key yet, by design
    assert rc == -1 and "boot failed" in status


def test_a_teardown_strand_keeps_the_record_for_resume(tmp_path, monkeypatch):
    """The record IS the handle --resume uses to finish a failed teardown, so a
    strand must never clear it -- while an ordinary failure (teardown ran) must."""
    from contextlib import contextmanager
    from flux_compute.provision import TeardownStrandError
    from flux_compute.sweep import _load_attach_records, _make_run_one

    into = str(tmp_path / "cloud-sweep")

    def _run_with(exc):
        @contextmanager
        def fake_instance(conn, spec, name, ttl_minutes, keep=False):
            raise exc
        monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: object())
        monkeypatch.setattr(sweep, "_gpu_instance", fake_instance)
        shard = Shard(region="GRA11", spec=SimpleNamespace(flavor="b3-8"), vcpus=8, cap=1)
        return _make_run_one("flux-ovh", shard, [], "job.sh", "out", into, 30)

    _run_with(TeardownStrandError("stranded: srv-9 could not be deleted"))(("kept", "p"))
    assert [r.label for r in _load_attach_records(into)] == ["kept"]

    _run_with(RuntimeError("ssh never opened"))(("dropped", "p"))
    assert [r.label for r in _load_attach_records(into)] == ["kept"]   # 'dropped' cleared


# --- --resume heals SSH ingress before it reconnects --------------------------
#
# The roaming incident: a 16-VM fleet was launched from one network and resumed
# from another, so every per-VM security group still admitted only the old /32.
# Every re-attach hung, the fleet sat FINISHED and idle-billing for hours, and
# the log said nothing because silence is also what a healthy long job looks
# like. The repair below runs before the first SSH of every re-attach.

def _rec(**kw):
    base = dict(label="alpha", cloud="flux-ovh", region="UK1",
                name="flux-compute-sweep-abcd1234", remote_script="job.sh",
                fetch="out", into="cloud-sweep", cap_seconds=1800,
                launch_epoch=time.time(), server_id="srv-1", ip="1.2.3.4",
                keyfile="/tmp/id_key")
    base.update(kw)
    return AttachRecord(**base)


def test_resume_heal_announces_a_reopened_group(monkeypatch):
    monkeypatch.setattr(provision, "heal_ssh_ingress",
                        lambda conn, sg, cidr: IngressCheck("healed", f"opened {cidr} on {sg}"))
    lines = []
    sweep._heal_ingress_before_reattach(None, _rec(), "9.9.9.9/32", emit=lines.append)
    assert len(lines) == 1
    assert "opened 9.9.9.9/32" in lines[0] and "[alpha]" in lines[0]


def test_resume_heal_is_quiet_when_ingress_is_already_open(monkeypatch):
    """The common case (same network) must not add noise to a 100-job resume."""
    monkeypatch.setattr(provision, "heal_ssh_ingress",
                        lambda conn, sg, cidr: IngressCheck("open", "already open"))
    lines = []
    sweep._heal_ingress_before_reattach(None, _rec(), "1.2.3.4/32", emit=lines.append)
    assert lines == []


def test_resume_heal_reports_an_unreadable_public_ip_and_continues(monkeypatch):
    """Cannot repair without knowing where we are -- say so, then let the SSH
    attempt be the authority on whether it actually matters."""
    monkeypatch.setattr(provision, "heal_ssh_ingress",
                        lambda conn, sg, cidr: IngressCheck("unknown-ip", "could not read public IP"))
    lines = []
    sweep._heal_ingress_before_reattach(None, _rec(), None, emit=lines.append)
    assert len(lines) == 1 and "could not read public IP" in lines[0]


def test_resume_heal_surfaces_an_api_error_without_failing_the_reattach(monkeypatch):
    """A network-API hiccup must not abandon a live, billing VM: the check is
    precautionary, so it is surfaced loudly and stepped past."""
    def _boom(conn, sg, cidr):
        raise RuntimeError("neutron 503")

    monkeypatch.setattr(provision, "heal_ssh_ingress", _boom)
    lines = []
    sweep._heal_ingress_before_reattach(None, _rec(), "9.9.9.9/32", emit=lines.append)
    assert len(lines) == 1
    assert "WARNING" in lines[0] and "neutron 503" in lines[0]
    assert "flux-compute-sweep-abcd1234" in lines[0]     # names the group


def test_reattach_heals_every_group_before_any_ssh(tmp_path, monkeypatch, capsys):
    """End to end: the ingress repair for each VM lands BEFORE that VM's first
    poll, and the address is read once for the whole fleet rather than per job."""
    events, reads = [], []

    monkeypatch.setattr(sweep, "_load_attach_records",
                        lambda into: [_rec(label="a", name="sg-a"),
                                      _rec(label="b", name="sg-b")])
    monkeypatch.setattr(sweep, "connect", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(sweep, "warn_strays", lambda conn: None)
    monkeypatch.setattr(sweep, "_server_by_name_or_id",
                        lambda conn, name, sid: SimpleNamespace(id=sid))
    monkeypatch.setattr(sweep, "current_ingress_cidr",
                        lambda: reads.append(1) or "9.9.9.9/32")

    def _heal(conn, sg, cidr):
        assert cidr == "9.9.9.9/32"
        events.append(("heal", sg))
        return IngressCheck("healed", f"opened {cidr} on {sg}")

    def _follow(ip, keyfile, cap, **kw):
        events.append(("ssh", kw.get("on_stuck") is not None))
        return PollOutcome(rc=0, reason="done", output_size=0)

    monkeypatch.setattr(provision, "heal_ssh_ingress", _heal)
    monkeypatch.setattr(sweep, "follow_detached_job", _follow)
    monkeypatch.setattr(sweep, "make_stuck_handler", lambda *a, **kw: (lambda n, s: None))
    monkeypatch.setattr(sweep, "_finalize", lambda *a, **kw: (0, "ok"))
    monkeypatch.setattr(sweep, "teardown_by_name", lambda conn, name, sid: None)
    monkeypatch.setattr(sweep, "_clear_attach_record", lambda dest: None)

    assert sweep._reattach_records(None, None, str(tmp_path), 1) == 0

    # Both VMs healed, each before its own SSH, and one public-IP read in total.
    assert [e for e in events if e[0] == "heal"] == [("heal", "sg-a"), ("heal", "sg-b")]
    assert events[0][0] == "heal" and events[1][0] == "ssh"
    assert events[2][0] == "heal" and events[3][0] == "ssh"
    assert len(reads) == 1
    assert "opened 9.9.9.9/32 on sg-a" in capsys.readouterr().out


def test_reattach_warns_once_when_the_public_ip_cannot_be_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "_load_attach_records", lambda into: [_rec()])
    monkeypatch.setattr(sweep, "connect", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(sweep, "warn_strays", lambda conn: None)
    monkeypatch.setattr(sweep, "_server_by_name_or_id",
                        lambda conn, name, sid: SimpleNamespace(id=sid))
    monkeypatch.setattr(sweep, "current_ingress_cidr", lambda: None)
    monkeypatch.setattr(provision, "heal_ssh_ingress",
                        lambda conn, sg, cidr: IngressCheck("unknown-ip", "could not read public IP"))
    monkeypatch.setattr(sweep, "follow_detached_job",
                        lambda *a, **kw: PollOutcome(rc=0, reason="done", output_size=0))
    monkeypatch.setattr(sweep, "make_stuck_handler", lambda *a, **kw: (lambda n, s: None))
    monkeypatch.setattr(sweep, "_finalize", lambda *a, **kw: (0, "ok"))
    monkeypatch.setattr(sweep, "teardown_by_name", lambda conn, name, sid: None)
    monkeypatch.setattr(sweep, "_clear_attach_record", lambda dest: None)

    assert sweep._reattach_records(None, None, str(tmp_path), 1) == 0
    out = capsys.readouterr().out
    assert "could not read this machine's public IP" in out   # the fleet-wide note
    assert "first thing to suspect" in out


# --- --resume continues the jobs file -----------------------------------------

def _stub_resume(monkeypatch, launched):
    monkeypatch.setattr(sweep, "_reattach_records", lambda *a, **kw: 0)
    monkeypatch.setattr(sweep, "_launch_jobs",
                        lambda **kw: launched.append(kw["jobs"]) or 0)


def test_resume_without_a_jobs_file_only_reattaches(tmp_path, monkeypatch):
    """Backward compatibility: the existing `sweep --resume --into X` invocation
    must behave exactly as it did."""
    launched = []
    _stub_resume(monkeypatch, launched)
    assert sweep.resume_sweep(into=str(tmp_path)) == 0
    assert launched == []


def test_resume_launches_only_the_jobs_that_never_started(tmp_path, monkeypatch):
    from flux_compute.sweep import ATTACH_DIR, ATTACH_RECORD, JOB_LOG

    into = tmp_path / "cloud-sweep"
    (into / "done_one" ).mkdir(parents=True)
    (into / "done_one" / JOB_LOG).write_text("collected")
    (into / "in_flight" / ATTACH_DIR).mkdir(parents=True)
    (into / "in_flight" / ATTACH_DIR / ATTACH_RECORD).write_text("{}")

    jobs_file = tmp_path / "jobs.txt"
    jobs_file.write_text("done_one = a\nin_flight = b\nnever_ran = c\nalso_new = d\n")

    launched = []
    _stub_resume(monkeypatch, launched)
    rc = sweep.resume_sweep(into=str(into), jobs_file=str(jobs_file),
                            script="job.sh", fetch="out")
    assert rc == 0
    assert launched == [[("never_ran", "c"), ("also_new", "d")]]


def test_resume_with_a_fully_accounted_jobs_file_launches_nothing(tmp_path, monkeypatch):
    from flux_compute.sweep import JOB_LOG
    into = tmp_path / "cloud-sweep"
    (into / "a").mkdir(parents=True)
    (into / "a" / JOB_LOG).write_text("x")
    jobs_file = tmp_path / "jobs.txt"
    jobs_file.write_text("a = 1\n")

    launched = []
    _stub_resume(monkeypatch, launched)
    assert sweep.resume_sweep(into=str(into), jobs_file=str(jobs_file),
                              script="job.sh", fetch="out") == 0
    assert launched == []


def test_resume_needs_script_and_fetch_to_launch_the_remainder(tmp_path, monkeypatch):
    jobs_file = tmp_path / "jobs.txt"
    jobs_file.write_text("never_ran = c\n")
    _stub_resume(monkeypatch, [])
    with pytest.raises(RuntimeError, match="needs --script and --fetch"):
        sweep.resume_sweep(into=str(tmp_path / "cloud-sweep"), jobs_file=str(jobs_file))


def test_run_sweep_resume_passes_the_jobs_file_through(tmp_path, monkeypatch):
    """`sweep --resume --jobs ...` must reach resume_sweep with the file."""
    seen = {}
    monkeypatch.setattr(sweep, "resume_sweep", lambda **kw: seen.update(kw) or 0)
    run_sweep(resume=True, into="cloud-sweep", jobs_file="jobs.txt",
              script="job.sh", fetch="out", regions="DE1,UK1")
    assert seen["jobs_file"] == "jobs.txt" and seen["regions"] == "DE1,UK1"
    assert seen["script"] == "job.sh" and seen["fetch"] == "out"
