"""`flux-compute cancel`: stop one job by label (or every one in a folder) from
its local attach record. All offline: the OpenStack client, the teardown, the
fetch and the process signals are stubbed.

The incidents this closes: a superseded sweep controller stayed alive and could
rewrite a job folder, and a campaign's jobs ran on under their wall cap with no
way to cancel one label; the global `reap --yes` was the only lever.
"""
import os
import signal
from types import SimpleNamespace

import pytest

from flux_compute import cancel, cli, provision
from flux_compute.cancel import CANCELLED, CancelRefused, run_cancel, stop_controller
from flux_compute.provision import TeardownStrandError
from flux_compute.sweep import (
    ATTACH_DIR, CONTROLLER_PID, _load_attach_records, _write_attach_record,
    _write_pending_record, job_state, read_outcome,
)


START = "Sat Oct 18 09:00:00 2026"


def _job(into, label, *, pid=None, cloud="flux-ovh", region="UK1", attachable=True):
    dest = os.path.join(into, label)
    if attachable:
        key = os.path.join(str(into), f"{label}.key")
        os.makedirs(into, exist_ok=True)
        with open(key, "w") as fh:
            fh.write("KEY")
        _write_attach_record(dest, label=label, cloud=cloud, region=region,
                             name=f"flux-compute-sweep-{label}", server_id=f"srv-{label}",
                             ip="1.2.3.4", keyfile=key, remote_script="job.sh",
                             fetch="out", into=into, cap_seconds=60)
    else:
        _write_pending_record(dest, label=label, cloud=cloud, region=region,
                              name=f"flux-compute-sweep-{label}", remote_script="job.sh",
                              fetch="out", into=into, cap_seconds=60)
    if pid is not None:
        with open(os.path.join(dest, ATTACH_DIR, CONTROLLER_PID), "w") as fh:
            fh.write(f"{pid}\n{START}\n")
    return dest


class _Fleet:
    """Records every call the cancel makes, in order."""

    def __init__(self, strand=False, connect_fails=False):
        self.calls = []
        self.strand, self.connect_fails = strand, connect_fails   # strand: a label set or True

    def connect(self, cloud=None, region=None):
        if self.connect_fails:
            raise RuntimeError("No OVH OpenStack credentials found.")
        self.calls.append(("connect", cloud, region))
        return SimpleNamespace(region=region)

    def teardown(self, conn, name, server_id):
        self.calls.append(("teardown", name, server_id))
        if self.strand is True or (self.strand and name.removeprefix(
                "flux-compute-sweep-") in self.strand):
            raise TeardownStrandError(f"stranded: server {server_id}")

    def stop(self, pid, start, emit=print):
        self.calls.append(("stop", pid))
        return True

    def pull_log(self, ip, keyfile, path):
        self.calls.append(("pull_log", path))
        return True

    def fetch(self, ip, keyfile, remote, local):
        self.calls.append(("fetch", remote, local))
        return True

    def kw(self):
        return dict(_connect=self.connect, _teardown=self.teardown, _stop=self.stop,
                    _pull_log=self.pull_log, _fetch=self.fetch, emit=lambda m: None)

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


def test_cancel_one_label_stops_its_follower_tears_down_and_records_it(tmp_path):
    into = str(tmp_path / "into")
    dest = _job(into, "chen_nx1024", pid=4242)
    _job(into, "chen_nx512", pid=5151)
    fleet = _Fleet()

    assert run_cancel(into, labels=["chen_nx1024"], **fleet.kw()) == 0

    kinds = [c[0] for c in fleet.calls]
    assert kinds == ["connect", "stop", "pull_log", "fetch", "teardown"]
    assert fleet.of("connect") == [("connect", "flux-ovh", "UK1")]
    assert fleet.of("stop") == [("stop", 4242)]
    assert fleet.of("teardown") == [("teardown", "flux-compute-sweep-chen_nx1024",
                                     "srv-chen_nx1024")]
    out = read_outcome(dest)
    assert out.status == CANCELLED and out.label == "chen_nx1024" and out.ended_epoch > 0
    assert not os.path.exists(os.path.join(dest, ATTACH_DIR))
    assert job_state(into, "chen_nx1024") == "collected"      # never relaunched
    assert [r.label for r in _load_attach_records(into)] == ["chen_nx512"]


def test_a_label_not_in_flight_is_refused_in_one_line_with_exit_2(tmp_path, capsys):
    into = str(tmp_path / "into")
    _job(into, "running")
    os.makedirs(os.path.join(into, "finished"))

    assert cli.main(["cancel", "--into", into, "--label", "finished"]) == 2

    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "finished not in flight" in err
    assert len(_load_attach_records(into)) == 1


def test_all_with_nothing_in_flight_is_refused_with_exit_2(tmp_path, capsys):
    into = tmp_path / "into"
    into.mkdir()
    assert cli.main(["cancel", "--into", str(into), "--all"]) == 2
    assert "no in-flight job" in capsys.readouterr().err


def test_label_and_all_are_exclusive_and_one_is_required(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["cancel", "--into", str(tmp_path), "--all", "--label", "a"])
    with pytest.raises(SystemExit):
        cli.main(["cancel", "--into", str(tmp_path)])


def test_all_cancels_every_job_and_stops_a_shared_follower_once(tmp_path):
    into = str(tmp_path / "into")
    for label in ("a", "b", "c"):
        _job(into, label, pid=777)
    fleet = _Fleet()

    assert run_cancel(into, take_all=True, **fleet.kw()) == 0

    assert fleet.of("stop") == [("stop", 777)]
    assert len(fleet.of("teardown")) == 3
    assert _load_attach_records(into) == []
    assert all(read_outcome(os.path.join(into, l)).status == CANCELLED for l in "abc")


def test_stopping_a_shared_follower_names_the_jobs_it_leaves(tmp_path):
    into = str(tmp_path / "into")
    _job(into, "a", pid=777)
    _job(into, "b", pid=777)
    lines = []
    fleet = _Fleet()
    kw = fleet.kw()
    kw["emit"] = lines.append

    run_cancel(into, labels=["a"], **kw)

    note = next(line for line in lines if "also following" in line)
    assert "b" in note and f"sweep --resume --into {into}" in note
    assert [r.label for r in _load_attach_records(into)] == ["b"]


def test_plan_prints_and_touches_nothing(tmp_path, capsys):
    into = str(tmp_path / "into")
    dest = _job(into, "a", pid=777)
    fleet = _Fleet()
    kw = fleet.kw()
    del kw["emit"]

    assert run_cancel(into, labels=["a"], plan=True, **kw) == 0

    out = capsys.readouterr().out
    assert "would stop controller pid 777" in out and "would delete the instance" in out
    assert fleet.calls == []
    assert read_outcome(dest) is None and len(_load_attach_records(into)) == 1


def test_a_pending_record_is_torn_down_without_a_fetch(tmp_path):
    into = str(tmp_path / "into")
    _job(into, "booting", attachable=False, pid=777)
    fleet = _Fleet()
    run_cancel(into, labels=["booting"], **fleet.kw())
    fleet.calls = [c for c in fleet.calls if c[0] != "stop"]
    assert [c[0] for c in fleet.calls] == ["connect", "teardown"]


def test_a_teardown_strand_raises_and_keeps_the_record(tmp_path):
    into = str(tmp_path / "into")
    dest = _job(into, "a", pid=777)
    with pytest.raises(TeardownStrandError):
        run_cancel(into, labels=["a"], **_Fleet(strand=True).kw())
    assert read_outcome(dest) is None and len(_load_attach_records(into)) == 1


def test_missing_credentials_raise_before_the_follower_is_touched(tmp_path):
    into = str(tmp_path / "into")
    _job(into, "a", pid=777)
    fleet = _Fleet(connect_fails=True)
    with pytest.raises(RuntimeError) as exc:
        run_cancel(into, labels=["a"], **fleet.kw())
    assert "hatched" in str(exc.value) and "clouds.yaml" in str(exc.value)
    assert fleet.calls == [] and len(_load_attach_records(into)) == 1


def test_a_records_own_cloud_wins_over_the_flag(tmp_path):
    into = str(tmp_path / "into")
    _job(into, "a", cloud="flux-ovh", pid=777)
    _job(into, "b", cloud=None, pid=777)
    fleet = _Fleet()
    run_cancel(into, take_all=True, cloud="other", **fleet.kw())
    assert fleet.of("connect") == [("connect", "flux-ovh", "UK1"), ("connect", "other", "UK1")]


# --- stopping the follower ---------------------------------------------------------

class _Procs:
    """A process table: `alive` pids answer signal 0 and `starts` gives each its
    start time; SIGTERM ends a pid unless it is in `ignores_term`, SIGKILL ends
    it unless it is in `unkillable`. `commands` is what the pid runs, which the
    stop never trusts."""

    def __init__(self, alive, commands, starts=None, ignores_term=(), unkillable=(),
                 denied=()):
        self.alive, self.commands = set(alive), commands
        self.starts = starts if starts is not None else {pid: START for pid in alive}
        self.ignores_term, self.unkillable, self.denied = (
            set(ignores_term), set(unkillable), set(denied))
        self.sent = []

    def kill(self, pid, sig):
        if pid in self.denied:
            raise PermissionError(1, "Operation not permitted")
        if pid not in self.alive:
            raise ProcessLookupError
        if sig == 0:
            return
        self.sent.append(sig)
        if sig == signal.SIGTERM and pid not in self.ignores_term:
            self.alive.discard(pid)
        if sig == signal.SIGKILL and pid not in self.unkillable:
            self.alive.discard(pid)

    def start_of(self, pid):
        return self.starts.get(pid) if pid in self.alive else None


def _stop(procs, pid, start=START, **kw):
    lines = []
    got = stop_controller(pid, start, emit=lines.append, _kill=procs.kill,
                          _start_of=procs.start_of, _sleep=lambda s: None, **kw)
    return got, lines


def test_a_live_follower_with_a_matching_start_is_stopped_with_sigterm():
    procs = _Procs({30568}, {30568: "/opt/py/bin/flux-compute sweep --resume"})
    got, lines = _stop(procs, 30568)
    assert got is True and procs.sent == [signal.SIGTERM] and "stopped" in lines[0]


def test_a_follower_that_ignores_sigterm_gets_sigkill():
    procs = _Procs({1}, {1: "flux-compute sweep"}, ignores_term={1})
    got, _ = _stop(procs, 1, term_wait_s=0.0)
    assert got is True and procs.sent == [signal.SIGTERM, signal.SIGKILL]


def test_a_follower_that_outlives_sigkill_raises():
    procs = _Procs({1}, {1: "flux-compute sweep"}, ignores_term={1}, unkillable={1})
    with pytest.raises(RuntimeError, match="still alive after SIGKILL"):
        _stop(procs, 1, term_wait_s=0.0, kill_wait_s=0.0)


def test_a_gone_follower_is_reported_not_signalled():
    procs = _Procs(set(), {})
    got, lines = _stop(procs, 99)
    assert got is False and procs.sent == [] and "already gone" in lines[0]


def test_a_reused_pid_running_another_flux_compute_is_left_alone():
    """The pid now names another campaign's sweep: a flux-compute command line,
    started at another time. Nothing is sent."""
    procs = _Procs({5}, {5: "flux-compute sweep --into other"},
                   starts={5: "Sat Oct 18 11:30:00 2026"})
    got, lines = _stop(procs, 5)
    assert got is False and procs.sent == [] and "reused" in lines[0]


def test_a_pid_this_shell_may_not_signal_raises_with_the_hatch_advice():
    procs = _Procs({5}, {5: "flux-compute sweep"}, denied={5})
    with pytest.raises(RuntimeError, match="hatched"):
        _stop(procs, 5)


def test_process_start_reads_ps_and_reads_a_missing_pid_as_none():
    from flux_compute.sweep import process_start
    ok = SimpleNamespace(returncode=0, stdout=START + "\n", stderr="")
    gone = SimpleNamespace(returncode=1, stdout="", stderr="")
    assert process_start(5, _run=lambda *a, **kw: ok) == START
    assert process_start(5, _run=lambda *a, **kw: gone) is None

    def denied(*a, **kw):
        raise PermissionError(1, "Operation not permitted")
    with pytest.raises(RuntimeError, match="hatched"):
        process_start(5, _run=denied)


# --- a job with no recorded follower ------------------------------------------------

def test_a_job_with_no_recorded_follower_is_refused_and_kept(tmp_path):
    into = str(tmp_path / "into")
    dest = _job(into, "old_job")
    fleet = _Fleet()
    with pytest.raises(CancelRefused) as exc:
        run_cancel(into, labels=["old_job"], **fleet.kw())
    msg = str(exc.value)
    assert "old_job" in msg and "no controller.pid" in msg and "--no-follower" in msg
    assert "\n" not in msg
    assert fleet.calls == [] and read_outcome(dest) is None
    assert len(_load_attach_records(into)) == 1


def test_plan_prints_the_no_follower_refusal_and_exits_2(tmp_path, capsys):
    into = str(tmp_path / "into")
    _job(into, "old_job")
    assert cli.main(["cancel", "--into", into, "--label", "old_job", "--plan"]) == 2
    err = capsys.readouterr().err
    assert "would refuse" in err and "--no-follower" in err
    assert len(_load_attach_records(into)) == 1


def test_no_follower_acknowledges_it_and_the_cancel_runs(tmp_path):
    into = str(tmp_path / "into")
    dest = _job(into, "old_job")
    fleet = _Fleet()
    assert run_cancel(into, labels=["old_job"], no_follower=True, **fleet.kw()) == 0
    assert fleet.of("stop") == [] and read_outcome(dest).status == CANCELLED


# --- a teardown that leaves the keypair or group behind ------------------------------

class _LeftoverGroup:
    """A cloud whose server is already gone and whose security group delete
    keeps failing."""

    def __init__(self):
        self.compute = SimpleNamespace(get_server=lambda sid: None,
                                       find_server=lambda name, ignore_missing=True: None,
                                       delete_keypair=lambda name, ignore_missing=True: None)
        self.network = SimpleNamespace(
            find_security_group=lambda name: SimpleNamespace(id="sg-9"),
            delete_security_group=self._refuse)
        self.config = SimpleNamespace(name="fake-cloud")

    @staticmethod
    def _refuse(sg_id, ignore_missing=True):
        raise RuntimeError("409 Conflict: security group in use")


def test_a_group_that_cannot_be_deleted_raises_and_the_record_is_kept(
        tmp_path, monkeypatch):
    monkeypatch.setattr(provision.time, "sleep", lambda s: None)
    into = str(tmp_path / "into")
    dest = _job(into, "a", pid=777)
    fleet = _Fleet()
    kw = fleet.kw()
    del kw["_teardown"]                         # the real teardown_by_name
    kw["_connect"] = lambda cloud=None, region=None: _LeftoverGroup()

    with pytest.raises(TeardownStrandError, match="security group .*sg-9"):
        run_cancel(into, labels=["a"], **kw)

    assert read_outcome(dest) is None and len(_load_attach_records(into)) == 1


def test_a_keypair_that_cannot_be_deleted_raises(monkeypatch):
    conn = _LeftoverGroup()
    conn.network.find_security_group = lambda name: None

    def refuse(name, ignore_missing=True):
        raise RuntimeError("503")
    conn.compute.delete_keypair = refuse
    with pytest.raises(TeardownStrandError, match="keypair flux-compute-sweep-a"):
        provision.teardown_by_name(conn, "flux-compute-sweep-a")


# --- a cancel that stops partway still names the jobs left without a follower -------

def test_a_partway_failure_still_names_the_jobs_the_stopped_follower_left(tmp_path):
    into = str(tmp_path / "into")
    _job(into, "a", pid=777)
    _job(into, "b", pid=777)
    lines = []
    fleet = _Fleet(strand={"a"})
    kw = fleet.kw()
    kw["emit"] = lines.append

    with pytest.raises(TeardownStrandError):
        run_cancel(into, labels=["a"], **kw)

    note = next(line for line in lines if "also following" in line)
    assert "also following a, b;" in note
    assert f"sweep --resume --into {into}" in note


def test_an_unreadable_pid_file_raises_with_its_path(tmp_path):
    into = str(tmp_path / "into")
    dest = _job(into, "a")
    with open(os.path.join(dest, ATTACH_DIR, CONTROLLER_PID), "w") as fh:
        fh.write("not a pid")
    with pytest.raises(RuntimeError, match="unreadable controller pid file"):
        run_cancel(into, labels=["a"], **_Fleet().kw())


def test_cancel_is_a_refusal_type_the_cli_maps_to_exit_2():
    assert issubclass(CancelRefused, RuntimeError)
    assert cancel.main_cancel(SimpleNamespace(
        into="/nonexistent/flux", label=["a"], take_all=False, cloud=None, plan=False,
        no_follower=False)) == 2
