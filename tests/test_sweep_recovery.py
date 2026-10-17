"""Campaign recovery: outcome records, the upload retry in another region,
`--resume --rerun-failed-start`, and `--wait-next`.

The incident these close: one job's upload to UK1 died mid-rsync ("ssh: connect
to host ... port 22: Operation timed out", rc=-1). `--resume` skipped it because
it had left a log, so the session hand-built a one-job jobs file and a second
results folder, and hand-made log watchers replayed old failure lines. All
offline: the OpenStack and SSH layers are stubbed the way test_sweep.py does.
"""
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from flux_compute import sweep
from flux_compute.detach import PollOutcome
from flux_compute.sweep import (
    ATTACH_DIR,
    ATTACH_RECORD,
    JOB_LOG,
    OUTCOME,
    Shard,
    _load_attach_records,
    _make_run_one,
    _outcome,
    _record_outcome,
    _write_pending_record,
    job_state,
    pick_retry_shard,
    read_outcome,
    wait_next_job,
)


def _shard(region, price=0.80, flavor="t2-le-45"):
    spec = SimpleNamespace(flavor=flavor, est_cost_eur_hr=price,
                           gpu_model="Tesla V100S 32GB", image="img", network="Ext-Net")
    return Shard(region=region, spec=spec, vcpus=15, cap=2)


# --- the outcome record ---------------------------------------------------------

def test_outcome_round_trips_and_is_absent_until_written(tmp_path):
    dest = str(tmp_path / "alpha")
    assert read_outcome(dest) is None
    written = _record_outcome(dest, _outcome("alpha", "UK1", -1, "x", "upload"))
    assert read_outcome(dest) == written
    assert written.failed_before_start


def test_an_unreadable_outcome_raises_with_its_path(tmp_path):
    dest = tmp_path / "alpha"
    dest.mkdir()
    (dest / OUTCOME).write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable outcome record"):
        read_outcome(str(dest))


def test_an_unknown_prestart_phase_is_refused():
    with pytest.raises(ValueError, match="unknown pre-start phase"):
        _outcome("a", "UK1", -1, "x", "follow")


def test_job_state_tells_a_failed_start_from_a_job_that_ran(tmp_path):
    into = str(tmp_path)
    _record_outcome(os.path.join(into, "upload_died"),
                    _outcome("upload_died", "UK1", -1, "x", "upload"))
    _record_outcome(os.path.join(into, "ran_and_failed"),
                    _outcome("ran_and_failed", "UK1", 1, "job nonzero"))
    assert job_state(into, "upload_died") == "failed_start"
    assert job_state(into, "ran_and_failed") == "collected"


# --- the per-job runner: phases, outcomes and the upload retry -------------------

def _fake_prepare(no_quota=(), prices=None):
    """A `_prepare_shard` stand-in: regions in `no_quota` refuse the way the
    quota clamp does, the rest resolve to a priced V100S shard."""
    def prepare(cloud, region, flavor, image, max_parallel, warn=True):
        if region in no_quota:
            raise RuntimeError("compute quota cannot fit even one instance")
        return _shard(region, price=(prices or {}).get(region, 0.80))
    return prepare


class _Cloud:
    """Stubs the instance and SSH layer. `upload_fails_in` names the regions
    whose rsync raises the way the UK1 upload did; every instance created and
    torn down is recorded. `account` is the cloud entry's configured region
    list, `no_quota` the regions whose live quota fits nothing."""

    def __init__(self, monkeypatch, tmp_path, *, upload_fails_in=(), follow=None,
                 account=(), no_quota=()):
        self.instances, self.torn_down = [], []
        self.keyfile = str(tmp_path / "id_ed25519")
        with open(self.keyfile, "w") as fh:
            fh.write("KEY")
        cloud = self

        @contextmanager
        def fake_instance(conn, spec, name, ttl_minutes, keep=False):
            cloud.instances.append((conn.region, name))
            try:
                yield SimpleNamespace(id=f"srv-{name}"), "1.2.3.4", cloud.keyfile
            finally:
                cloud.torn_down.append(name)

        def fake_rsync_up(local, ip, keyfile, dest, extra_excludes=()):
            region = cloud.instances[-1][0]
            if region in upload_fails_in:
                raise RuntimeError(f"rsync upload of {local} -> {dest} failed (exit 255)")

        monkeypatch.setattr(sweep, "connect",
                            lambda cloud=None, region=None: SimpleNamespace(region=region))
        monkeypatch.setattr(sweep, "_gpu_instance", fake_instance)
        monkeypatch.setattr(sweep, "_rsync_up", fake_rsync_up)
        monkeypatch.setattr(sweep, "_scp_up", lambda *a, **kw: None)
        monkeypatch.setattr(sweep, "_launch_detached", lambda *a, **kw: "")
        monkeypatch.setattr(sweep, "make_stuck_handler", lambda *a, **kw: None)
        monkeypatch.setattr(sweep, "follow_detached_job", follow or (
            lambda *a, **kw: PollOutcome(rc=0, reason="done", output_size=0)))
        monkeypatch.setattr(sweep, "_finalize", lambda *a, **kw: (0, "ok"))
        monkeypatch.setattr(sweep, "_prepare_shard", _fake_prepare(no_quota))
        monkeypatch.setattr(sweep, "configured_regions", lambda cloud: list(account))


def _runner(tmp_path, shard, alternates=(), budget_eur=None):
    into = str(tmp_path / "into")
    src = tmp_path / "repo"
    src.mkdir(exist_ok=True)
    return into, _make_run_one("flux-ovh", shard, [(str(src), "repo")], "job.sh",
                               "out", into, 30, alternates=alternates,
                               budget_eur=budget_eur)


def test_an_upload_failure_is_retried_once_in_another_region(tmp_path, monkeypatch):
    cloud = _Cloud(monkeypatch, tmp_path, upload_fails_in={"UK1"})
    uk1, de1 = _shard("UK1"), _shard("DE1")
    into, run_one = _runner(tmp_path, uk1, alternates=(uk1, de1))

    label, rc, status = run_one(("chen_nx1024", "--nx 1024"))

    assert rc == 0
    assert "retried in DE1 after an upload failure in UK1" in status
    assert [r for r, _ in cloud.instances] == ["UK1", "DE1"]
    assert cloud.torn_down[0] == cloud.instances[0][1]   # UK1 gone before the retry
    out = read_outcome(os.path.join(into, "chen_nx1024"))
    assert out.region == "DE1" and out.rc == 0 and not out.failed_before_start
    assert _load_attach_records(into) == []


def test_with_no_other_region_the_upload_failure_ends_as_a_failed_start(
        tmp_path, monkeypatch):
    cloud = _Cloud(monkeypatch, tmp_path, upload_fails_in={"UK1"})
    uk1 = _shard("UK1")
    into, run_one = _runner(tmp_path, uk1, alternates=(uk1,))

    _, rc, status = run_one(("chen_nx1024", "--nx 1024"))

    assert rc == -1
    assert "failed before start (upload, UK1)" in status
    assert "not retried: no other region in this sweep or this cloud entry" in status
    assert len(cloud.instances) == 1 and cloud.torn_down == [cloud.instances[0][1]]
    assert job_state(into, "chen_nx1024") == "failed_start"
    assert read_outcome(os.path.join(into, "chen_nx1024")).prestart_phase == "upload"
    assert _load_attach_records(into) == []


def test_a_retry_whose_upload_fails_too_is_not_retried_again(tmp_path, monkeypatch):
    cloud = _Cloud(monkeypatch, tmp_path, upload_fails_in={"UK1", "DE1"})
    uk1, de1, waw1 = _shard("UK1"), _shard("DE1"), _shard("WAW1")
    into, run_one = _runner(tmp_path, uk1, alternates=(uk1, de1, waw1))

    _, rc, status = run_one(("chen_nx1024", "--nx 1024"))

    assert rc == -1
    assert [r for r, _ in cloud.instances] == ["UK1", "DE1"]     # once, not twice
    assert "failed before start (upload, DE1)" in status
    out = read_outcome(os.path.join(into, "chen_nx1024"))
    assert out.region == "DE1" and out.prestart_phase == "upload"


def test_a_boot_failure_is_a_failed_start_but_a_failure_after_launch_is_not(
        tmp_path, monkeypatch):
    def follow_raises(*a, **kw):
        raise RuntimeError("poll loop crashed")

    _Cloud(monkeypatch, tmp_path, follow=follow_raises)
    into, run_one = _runner(tmp_path, _shard("UK1"))
    run_one(("ran", "p"))
    assert job_state(into, "ran") == "collected"
    assert read_outcome(os.path.join(into, "ran")).prestart_phase is None

    monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: (_ for _ in ()).throw(
        RuntimeError("keystone unreachable")))
    _, rc, status = run_one(("never_booted", "p"))
    assert rc == -1 and "failed before start (boot, UK1)" in status
    assert job_state(into, "never_booted") == "failed_start"


def test_a_rerun_attempt_clears_the_failed_attempts_records(tmp_path, monkeypatch):
    _Cloud(monkeypatch, tmp_path)
    into, run_one = _runner(tmp_path, _shard("DE1"))
    dest = os.path.join(into, "chen_nx1024")
    os.makedirs(dest)
    with open(os.path.join(dest, JOB_LOG), "w") as fh:
        fh.write("flux-compute: failed before start (upload, UK1): ...\n")
    _record_outcome(dest, _outcome("chen_nx1024", "UK1", -1, "old", "upload"))

    assert run_one(("chen_nx1024", "--nx 1024"))[1] == 0
    out = read_outcome(dest)
    assert out.region == "DE1" and out.status == "ok"
    assert not os.path.exists(os.path.join(dest, JOB_LOG))   # stale status gone


# --- choosing the retry region ---------------------------------------------------

def test_retry_region_skips_a_region_without_headroom_and_says_why(monkeypatch):
    monkeypatch.setattr(sweep, "configured_regions", lambda cloud: [])
    uk1, de1, waw1 = _shard("UK1"), _shard("DE1"), _shard("WAW1")
    alt, reasons = pick_retry_shard(None, uk1, (uk1, de1, waw1), None,
                                    _prepare=_fake_prepare(no_quota={"DE1"}))
    assert alt.region == "WAW1"
    assert reasons == ["DE1: RuntimeError: compute quota cannot fit even one instance"]


def test_under_a_budget_a_dearer_retry_region_is_refused(monkeypatch):
    monkeypatch.setattr(sweep, "configured_regions", lambda cloud: ["BHS5"])
    uk1 = _shard("UK1", price=0.80)
    dear = _fake_prepare(prices={"BHS5": 1.50})
    alt, reasons = pick_retry_shard(None, uk1, (uk1,), 10.0, _prepare=dear)
    assert alt is None
    assert "outside the budget this job was priced at" in reasons[0]
    # With no budget the same region is taken.
    assert pick_retry_shard(None, uk1, (uk1,), None, _prepare=dear)[0].region == "BHS5"


def test_retry_tries_the_sweeps_regions_before_the_accounts(monkeypatch):
    monkeypatch.setattr(sweep, "configured_regions",
                        lambda cloud: ["GRA11", "UK1", "WAW1", "DE1"])
    uk1, waw1 = _shard("UK1"), _shard("WAW1")
    names, notes = sweep._retry_regions("flux-ovh", uk1, (uk1, waw1))
    assert names == ["WAW1", "GRA11", "DE1"] and notes == []


def test_an_unreadable_account_region_list_is_a_reason_not_a_crash(monkeypatch):
    def unreadable(cloud):
        raise RuntimeError("Could not read the region list\nmore")
    monkeypatch.setattr(sweep, "configured_regions", unreadable)
    uk1 = _shard("UK1")
    alt, reasons = pick_retry_shard("flux-ovh", uk1, (uk1,), None,
                                    _prepare=_fake_prepare())
    assert alt is None
    assert reasons[0] == "account regions unreadable: Could not read the region list"


def test_a_one_region_sweep_retries_in_another_account_region(tmp_path, monkeypatch):
    """The UK1 case: a one-shard sweep whose upload failed now has a retry,
    in a region of the cloud entry with live headroom for the same flavor."""
    cloud = _Cloud(monkeypatch, tmp_path, upload_fails_in={"UK1"},
                   account=["UK1", "GRA11", "DE1"], no_quota={"GRA11"})
    uk1 = _shard("UK1")
    into, run_one = _runner(tmp_path, uk1, alternates=(uk1,))

    _, rc, status = run_one(("chen_nx1024", "--nx 1024"))

    assert rc == 0
    assert [r for r, _ in cloud.instances] == ["UK1", "DE1"]
    assert "retried in DE1 after an upload failure in UK1" in status
    assert read_outcome(os.path.join(into, "chen_nx1024")).region == "DE1"


def test_the_upload_failure_is_recorded_before_the_retry_region_is_picked(
        tmp_path, monkeypatch):
    """A kill while the retry region is chosen must leave a job --resume reads
    as failed before start, not a bare pending record."""
    _Cloud(monkeypatch, tmp_path, upload_fails_in={"UK1"})
    seen = {}

    def pick(cloud, failed, alternates, budget_eur, image=None):
        into = str(tmp_path / "into")
        seen["outcome"] = read_outcome(os.path.join(into, "chen_nx1024"))
        seen["record"] = os.path.isfile(
            os.path.join(into, "chen_nx1024", ATTACH_DIR, ATTACH_RECORD))
        raise KeyboardInterrupt        # the orchestrator dies here

    monkeypatch.setattr(sweep, "pick_retry_shard", pick)
    into, run_one = _runner(tmp_path, _shard("UK1"))
    with pytest.raises(KeyboardInterrupt):
        run_one(("chen_nx1024", "--nx 1024"))
    assert seen["record"] is True
    assert seen["outcome"].prestart_phase == "upload"


# --- --resume --rerun-failed-start -----------------------------------------------

def _campaign(tmp_path):
    into = tmp_path / "into"
    for label, phase, rc in (("upload_died", "upload", -1), ("ran_nonzero", None, 1),
                             ("ok_one", None, 0)):
        dest = into / label
        dest.mkdir(parents=True)
        (dest / JOB_LOG).write_text("x")
        _record_outcome(str(dest), _outcome(label, "UK1", rc, f"status of {label}", phase))
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("upload_died = a\nran_nonzero = b\nok_one = c\nnever_ran = d\n")
    return str(into), str(jobs)


def _stub_launch(monkeypatch):
    launched = []
    monkeypatch.setattr(sweep, "_reattach_records", lambda *a, **kw: 0)
    monkeypatch.setattr(sweep, "_launch_jobs",
                        lambda **kw: launched.append((kw["jobs"], kw["into"])) or 0)
    return launched


def test_resume_skips_failed_starts_and_names_the_flag(tmp_path, monkeypatch, capsys):
    into, jobs = _campaign(tmp_path)
    launched = _stub_launch(monkeypatch)
    sweep.resume_sweep(into=into, jobs_file=jobs, script="job.sh", fetch="out")
    assert launched == [([("never_ran", "d")], into)]
    out = capsys.readouterr().out
    assert "add --rerun-failed-start" in out and "upload_died" in out


def test_rerun_failed_start_relaunches_only_pre_start_failures_into_the_same_folder(
        tmp_path, monkeypatch, capsys):
    into, jobs = _campaign(tmp_path)
    launched = _stub_launch(monkeypatch)
    sweep.resume_sweep(into=into, jobs_file=jobs, script="job.sh", fetch="out",
                       rerun_failed_start=True)
    # The job that ran and failed is NOT rerun; the results folder is the same.
    assert launched == [([("upload_died", "a"), ("never_ran", "d")], into)]
    assert "rerunning [upload_died], which failed before start: status of upload_died" \
        in capsys.readouterr().out


def test_resume_preflight_counts_the_reruns(tmp_path):
    into, jobs = _campaign(tmp_path)
    kw = dict(resume=True, into=into, jobs_file=jobs, script="job.sh", fetch="out",
              flavor="t2-le-45")
    assert sweep.preflight_sweep(**kw).n_jobs == 1
    assert sweep.preflight_sweep(rerun_failed_start=True, **kw).n_jobs == 2


def test_rerun_failed_start_without_resume_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="needs --resume"):
        sweep.run_sweep(jobs_file="jobs.txt", rerun_failed_start=True)


# --- --wait-next -------------------------------------------------------------------

def _launch(into, label, region="UK1"):
    dest = os.path.join(into, label)
    os.makedirs(dest, exist_ok=True)
    _write_pending_record(dest, label=label, cloud="flux-ovh", region=region,
                          name=f"flux-compute-sweep-{label}", remote_script="job.sh",
                          fetch="out", into=into, cap_seconds=1800)
    return dest


def _end(into, label, rc, region="UK1", status="ok"):
    sweep._close_job(os.path.join(into, label), _outcome(label, region, rc, status))


def test_wait_next_with_no_running_job_exits_nonzero(tmp_path):
    into = str(tmp_path)
    _launch(into, "done")
    _end(into, "done", 1, status="job nonzero")
    with pytest.raises(RuntimeError, match="no running jobs"):
        wait_next_job(into, _sleep=lambda s: None)


def test_wait_next_reports_the_next_end_and_never_replays_an_old_one(tmp_path):
    into = str(tmp_path)
    _launch(into, "old_failure")
    _end(into, "old_failure", -1, status="failed before start (upload, UK1): ...")
    _launch(into, "chen_nx512", region="DE1")
    _launch(into, "chen_nx1024")
    polls, lines = [], []

    def sleep(_):
        polls.append(1)
        if len(polls) == 3:                       # a job ends while we wait
            _end(into, "chen_nx512", 0, region="DE1")

    assert wait_next_job(into, _sleep=sleep, emit=lines.append) == 0
    assert lines == ["chen_nx512 region=DE1 rc=0 ok"]
    assert len(polls) == 3


def test_wait_next_sees_a_job_launched_after_it_started(tmp_path):
    into = str(tmp_path)
    _launch(into, "first")
    polls, lines = [], []

    def sleep(_):
        polls.append(1)
        if len(polls) == 1:
            _launch(into, "second")
        if len(polls) == 2:
            _end(into, "second", 137, status="OOM-killed")

    assert wait_next_job(into, _sleep=sleep, emit=lines.append) == 0
    assert lines == ["second region=UK1 rc=137 OOM-killed"]


def test_wait_next_raises_when_a_running_record_vanishes_without_an_outcome(tmp_path):
    into = str(tmp_path)
    dest = _launch(into, "alpha")

    def sleep(_):
        sweep._clear_attach_record(dest)          # e.g. reap --sweep-local

    with pytest.raises(RuntimeError, match="without writing an outcome"):
        wait_next_job(into, _sleep=sleep)


def test_wait_next_counts_a_stranded_job_as_ended(tmp_path):
    """A teardown strand keeps its record for --resume AND writes its outcome:
    it has ended, so it is not a running job to wait on."""
    into = str(tmp_path)
    dest = _launch(into, "stranded")
    _record_outcome(dest, _outcome("stranded", "UK1", -1, "STRANDED: ..."))
    assert os.path.isfile(os.path.join(dest, ATTACH_DIR, ATTACH_RECORD))
    with pytest.raises(RuntimeError, match="no running jobs"):
        wait_next_job(into, _sleep=lambda s: None)


def test_wait_next_on_a_missing_folder_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no campaign folder"):
        wait_next_job(str(tmp_path / "nope"))


def test_a_new_attempt_keeps_the_log_of_a_run_whose_script_ran(tmp_path):
    dest = tmp_path / "ran"
    dest.mkdir()
    (dest / JOB_LOG).write_text("real remote output\n")
    _record_outcome(str(dest), _outcome("ran", "UK1", 1, "job nonzero"))
    sweep._clear_previous_attempt(str(dest))
    assert (dest / JOB_LOG).read_text() == "real remote output\n"
    assert read_outcome(str(dest)).status == "job nonzero"


# --- a recorded pre-start outcome survives a re-attach ---------------------------

def _killed_during_retry_pick(tmp_path):
    """The state a kill between the upload failure and the retry leaves: the
    pending record of the torn-down instance, plus the upload outcome."""
    into = str(tmp_path / "into")
    dest = _launch(into, "chen_nx1024")
    _record_outcome(dest, _outcome(
        "chen_nx1024", "UK1", -1, "failed before start (upload, UK1): rsync", "upload"))
    return into, dest


@pytest.mark.parametrize("server_alive", [False, True])
def test_reattach_keeps_a_recorded_prestart_failure(tmp_path, monkeypatch, server_alive):
    into, dest = _killed_during_retry_pick(tmp_path)
    torn = []
    monkeypatch.setattr(sweep, "connect", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(sweep, "warn_strays", lambda conn: None)
    monkeypatch.setattr(sweep, "current_ingress_cidr", lambda: "9.9.9.9/32")
    monkeypatch.setattr(sweep, "_server_by_name_or_id", lambda conn, name, sid:
                        SimpleNamespace(id="srv-1") if server_alive else None)
    monkeypatch.setattr(sweep, "teardown_by_name",
                        lambda conn, name, sid: torn.append(name))

    assert sweep._reattach_records(None, None, into, 1) == 1

    assert job_state(into, "chen_nx1024") == "failed_start"
    out = read_outcome(dest)
    assert out.prestart_phase == "upload" and "launcher died" not in out.status
    assert out.status.startswith("failed before start (upload, UK1)")
    assert _load_attach_records(into) == []
    assert torn == (["flux-compute-sweep-chen_nx1024"] if server_alive else [])


def test_reattach_ignores_an_outcome_older_than_the_record(tmp_path, monkeypatch):
    """An outcome from an earlier attempt says nothing about this one."""
    into = str(tmp_path / "into")
    dest = os.path.join(into, "a")
    _record_outcome(dest, _outcome("a", "UK1", -1, "old", "upload"))
    rec = _write_pending_record(dest, label="a", cloud=None, region="UK1",
                                name="n", remote_script="job.sh", fetch="out",
                                into=into, cap_seconds=60)
    assert sweep._recorded_prestart(dest, rec) is None


# --- legacy folders: a one-line pre-start job.log and no outcome -------------------

_UK1_LINE = ("flux-compute: error: RuntimeError: rsync upload of /Users/x/1DSim3 -> "
             "1DSim3 failed (exit 255); check the local path and SSH conn\n")


def test_a_legacy_one_line_upload_failure_reads_as_failed_start(tmp_path):
    dest = tmp_path / "chen_nx1024"
    dest.mkdir()
    (dest / JOB_LOG).write_text(_UK1_LINE)
    assert job_state(str(tmp_path), "chen_nx1024") == "failed_start"
    prior = sweep.prior_failed_start(str(dest))
    assert prior.prestart_phase == "upload" and prior.status.startswith("error: ")
    sweep._clear_previous_attempt(str(dest))        # a rerun clears it
    assert not (dest / JOB_LOG).exists()


@pytest.mark.parametrize("text", [
    _UK1_LINE + "more output\n",                                   # not one line
    "flux-compute: lost: VM gone (reaped?); cannot re-attach\n",   # may have run
    "rsync upload of x failed\n",                                  # not our prefix
    "flux-compute: error: RuntimeError: poll loop crashed\n",      # after the start
])
def test_other_logs_never_read_as_failed_start(tmp_path, text):
    dest = tmp_path / "j"
    dest.mkdir()
    (dest / JOB_LOG).write_text(text)
    assert job_state(str(tmp_path), "j") == "collected"


def test_a_legacy_boot_failure_reads_as_failed_start(tmp_path):
    dest = tmp_path / "j"
    dest.mkdir()
    (dest / JOB_LOG).write_text(
        "flux-compute: error: RuntimeError: SSH to 1.2.3.4 never opened within timeout\n")
    assert sweep.prior_failed_start(str(dest)).prestart_phase == "boot"


# --- a launch failure is pre-start only when the job provably never spawned -------

def _real_launch(monkeypatch, *, ssh=None, scp=None):
    from flux_compute import provision
    monkeypatch.setattr(provision, "_scp_up", scp or (lambda *a, **kw: None))

    def launch(ip, keyfile, remote, cap, env_prefix=""):
        return provision._launch_detached(ip, keyfile, remote, cap,
                                          env_prefix=env_prefix, _ssh=ssh)
    monkeypatch.setattr(sweep, "_launch_detached", launch)


def test_a_launch_ssh_timeout_is_recorded_as_unknown_not_prestart(tmp_path, monkeypatch):
    import subprocess

    def ssh_times_out(ip, keyfile, command, timeout=600, capture=True):
        raise subprocess.TimeoutExpired(command, timeout)

    _Cloud(monkeypatch, tmp_path)
    _real_launch(monkeypatch, ssh=ssh_times_out)
    into, run_one = _runner(tmp_path, _shard("UK1"))

    _, rc, status = run_one(("chen_nx1024", "p"))

    assert rc == -1
    assert "whether the script started is unknown" in status
    out = read_outcome(os.path.join(into, "chen_nx1024"))
    assert out.prestart_phase is None
    assert job_state(into, "chen_nx1024") == "collected"     # never rerun


def test_a_launcher_refusal_before_the_fork_is_a_failed_start(tmp_path, monkeypatch):
    from flux_compute.detach import LAUNCH_PREFORK_RC

    def ssh_prefork(ip, keyfile, command, timeout=600, capture=True):
        return SimpleNamespace(returncode=LAUNCH_PREFORK_RC, stdout="",
                               stderr="job script missing; not launched")

    _Cloud(monkeypatch, tmp_path)
    _real_launch(monkeypatch, ssh=ssh_prefork)
    into, run_one = _runner(tmp_path, _shard("UK1"))
    _, rc, status = run_one(("j", "p"))
    assert "failed before start (launch, UK1)" in status
    assert job_state(into, "j") == "failed_start"


def test_a_launcher_copy_failure_is_a_failed_start(tmp_path, monkeypatch):
    def scp_fails(*a, **kw):
        raise RuntimeError("scp: connection reset")

    _Cloud(monkeypatch, tmp_path)
    _real_launch(monkeypatch, ssh=lambda *a, **kw: None, scp=scp_fails)
    into, run_one = _runner(tmp_path, _shard("UK1"))
    run_one(("j", "p"))
    assert read_outcome(os.path.join(into, "j")).prestart_phase == "launch"


def test_the_launcher_exits_the_prefork_code_only_before_its_fork():
    from flux_compute.detach import LAUNCH_PREFORK_RC, launcher_script
    text = launcher_script("job.sh", 600)
    fork = text.index("setsid --fork")
    assert text.count(f"exit {LAUNCH_PREFORK_RC}") == 2
    assert text.rindex(f"exit {LAUNCH_PREFORK_RC}") < fork


# --- the campaign scan tolerates a record that vanishes mid-read --------------------

def test_campaign_state_reads_a_record_gone_mid_scan_as_ended(tmp_path, monkeypatch):
    import builtins
    into = str(tmp_path)
    dest = _launch(into, "alpha")
    real_open = builtins.open

    def racing_open(path, *a, **kw):
        if str(path).endswith(ATTACH_RECORD):
            _end(into, "alpha", 0)                   # the job ends right now
            raise FileNotFoundError(path)
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", racing_open)
    state = sweep._campaign_state(into)
    monkeypatch.setattr(builtins, "open", real_open)
    rec, out = state["alpha"]
    assert rec is None and out.rc == 0
