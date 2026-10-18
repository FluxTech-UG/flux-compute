"""An instance never outlives its local handle, and the resume's stray check
never reads as run when it did not.

The incidents these close: a VM left billing about 15 hours after its sweep's
log ended in a teardown banner with no "deleted server" line (the job's attach
record had already been dropped, so no `--resume` could name the VM), and a
hatched `sweep --resume` that printed "stray check skipped (No OVH OpenStack
credentials found)" and carried on, because it connected with the resume line's
`--cloud` (none) where every job re-attach used the cloud its launch recorded.
All offline: the OpenStack client and the SSH layer are stubbed.
"""
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from flux_compute import cli, provision, reap, sweep
from flux_compute.detach import PollOutcome
from flux_compute.provision import TeardownStrandError
from flux_compute.sweep import (
    ATTACH_DIR, ATTACH_RECORD, CONTROLLER_PID, Shard, _load_attach_records,
    _make_run_one, _write_attach_record, job_state, read_controller, read_outcome,
)


def _shard(region="UK1"):
    spec = SimpleNamespace(flavor="c3-4", est_cost_eur_hr=0.05, gpu_model=None,
                           image="Ubuntu 24.04", network="Ext-Net")
    return Shard(region=region, spec=spec, vcpus=2, cap=1)


# --- a fake OpenStack client that records every delete -----------------------------

class _Compute:
    def __init__(self):
        self.deleted_servers, self.deleted_keypairs = [], []

    def find_image(self, name):
        return SimpleNamespace(id="img-1")

    def find_flavor(self, name):
        return SimpleNamespace(id="fl-1")

    def create_keypair(self, name, public_key):
        return SimpleNamespace(name=name)

    def create_server(self, **kwargs):
        return SimpleNamespace(id="srv-1",
                               addresses={"Ext-Net": [{"version": 4, "addr": "10.0.0.1"}]})

    def wait_for_server(self, server, status=None, wait=None):
        return server

    def delete_server(self, server_id, force=False):
        self.deleted_servers.append(server_id)

    def wait_for_delete(self, server, wait=0):
        pass

    def delete_keypair(self, name, ignore_missing=True):
        self.deleted_keypairs.append(name)


class _Network:
    def __init__(self):
        self.deleted_groups = []

    def find_network(self, name):
        return SimpleNamespace(id="net-1")

    def create_security_group(self, name=None, description=None):
        return SimpleNamespace(id=f"sg-{name}")

    def create_security_group_rule(self, **kwargs):
        pass

    def delete_security_group(self, sg_id, ignore_missing=True):
        self.deleted_groups.append(sg_id)


def _client():
    return SimpleNamespace(compute=_Compute(), network=_Network(),
                           config=SimpleNamespace(name="fake-cloud"))


@pytest.fixture
def local_boot(monkeypatch):
    """Cut the real-network steps of `_gpu_instance` out."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    monkeypatch.setattr(provision, "_wait_ssh", lambda ip, **kw: True)


def _rsync_255(*a, **kw):
    raise RuntimeError("rsync upload of /src -> repo failed (exit 255)")


# --- 1. an upload failure tears the instance and its group down, then raises ------

def test_a_sweep_upload_failure_tears_down_before_the_failure_is_recorded(
        tmp_path, monkeypatch, local_boot):
    conn = _client()
    monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: conn)
    monkeypatch.setattr(sweep, "_rsync_up", _rsync_255)
    monkeypatch.setattr(sweep, "configured_regions", lambda cloud: [])
    into = str(tmp_path / "into")
    run_one = _make_run_one("flux-ovh", _shard(), [("/src", "repo")], "job.sh", "out",
                            into, 30)

    label, rc, status = run_one(("chen_nx1024", "--nx 1024"))

    assert conn.compute.deleted_servers == ["srv-1"]
    assert len(conn.network.deleted_groups) == 1
    assert len(conn.compute.deleted_keypairs) == 1
    out = read_outcome(os.path.join(into, label))
    assert rc == -1 and out.prestart_phase == "upload" and out.label == "chen_nx1024"
    assert "rsync upload of /src -> repo failed (exit 255)" in out.status
    assert job_state(into, label) == "failed_start"
    assert _load_attach_records(into) == []


def test_a_run_upload_failure_tears_down_and_the_error_propagates(
        tmp_path, monkeypatch, local_boot):
    conn = _client()
    monkeypatch.setattr(provision, "connect", lambda cloud=None, region=None: conn)
    monkeypatch.setattr(reap, "warn_strays", lambda conn, **kw: [])
    monkeypatch.setattr(provision, "resolve_spec", lambda *a, **kw: _shard().spec)
    monkeypatch.setattr(provision, "_print_plan", lambda spec: None)
    monkeypatch.setattr(provision, "_rsync_up", _rsync_255)
    src = tmp_path / "repo"
    src.mkdir()

    with pytest.raises(RuntimeError, match="exit 255"):
        provision.run_job(cloud="flux-ovh", uploads=[str(src)], script=None)

    assert conn.compute.deleted_servers == ["srv-1"]
    assert len(conn.network.deleted_groups) == 1


# --- the attach record outlives the teardown -----------------------------------

class _Instance:
    """A `_gpu_instance` stand-in that sees the job folder at teardown time and
    can end its teardown with `fail`: a strand, or a process kill."""

    def __init__(self, monkeypatch, tmp_path, dest_of, fail=None):
        self.record_at_teardown = None
        keyfile = tmp_path / "id_ed25519"
        keyfile.write_text("KEY")
        me = self

        @contextmanager
        def fake(conn, spec, name, ttl_minutes, keep=False):
            yield SimpleNamespace(id="srv-1"), "1.2.3.4", str(keyfile)
            me.record_at_teardown = os.path.isfile(
                os.path.join(dest_of(), ATTACH_DIR, ATTACH_RECORD))
            if fail is not None:
                raise fail

        monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: SimpleNamespace())
        monkeypatch.setattr(sweep, "_gpu_instance", fake)
        monkeypatch.setattr(sweep, "_rsync_up", lambda *a, **kw: None)
        monkeypatch.setattr(sweep, "_scp_up", lambda *a, **kw: None)
        monkeypatch.setattr(sweep, "_launch_detached", lambda *a, **kw: "")
        monkeypatch.setattr(sweep, "make_stuck_handler", lambda *a, **kw: None)
        monkeypatch.setattr(sweep, "follow_detached_job",
                            lambda *a, **kw: PollOutcome(rc=0, reason="done", output_size=0))
        monkeypatch.setattr(sweep, "_finalize", lambda *a, **kw: (0, "ok"))


class _ProcessKilled(BaseException):
    """What a SIGKILL of the controller looks like from inside: nothing after it runs."""


def _run(tmp_path, monkeypatch, fail=None):
    into = str(tmp_path / "into")
    dest = os.path.join(into, "chen_nx512")
    inst = _Instance(monkeypatch, tmp_path, lambda: dest, fail=fail)
    run_one = _make_run_one("flux-ovh", _shard("DE1"), [], "job.sh", "out", into, 30)
    return into, dest, inst, run_one


def test_a_collected_job_keeps_its_record_until_the_teardown_is_verified(
        tmp_path, monkeypatch):
    into, dest, inst, run_one = _run(tmp_path, monkeypatch)
    assert run_one(("chen_nx512", "--nx 512")) == ("chen_nx512", 0, "ok")
    assert inst.record_at_teardown is True
    assert _load_attach_records(into) == []
    assert read_outcome(dest).rc == 0


def test_a_kill_during_the_teardown_leaves_a_record_and_the_outcome(tmp_path, monkeypatch):
    into, dest, inst, run_one = _run(tmp_path, monkeypatch, fail=_ProcessKilled())
    with pytest.raises(_ProcessKilled):
        run_one(("chen_nx512", "--nx 512"))
    [rec] = _load_attach_records(into)
    assert rec.name.startswith("flux-compute-sweep-")
    assert read_outcome(dest).status == "ok"


def test_a_teardown_strand_after_a_collected_job_is_never_ok_but_keeps_the_result(
        tmp_path, monkeypatch):
    into, dest, inst, run_one = _run(
        tmp_path, monkeypatch, fail=TeardownStrandError("stranded: server srv-1 ..."))
    label, rc, status = run_one(("chen_nx512", "--nx 512"))
    assert rc == -1 and status.startswith("ok (job rc=0); ") and "stranded" in status
    assert read_outcome(dest).rc == -1
    assert len(_load_attach_records(into)) == 1          # the handle survives


def test_the_launching_process_names_itself_as_the_follower(tmp_path, monkeypatch):
    seen = []
    into, dest, inst, run_one = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(sweep, "_finalize",
                        lambda *a, **kw: seen.append(read_controller(dest)) or (0, "ok"))
    run_one(("chen_nx512", "--nx 512"))
    assert seen == [(os.getpid(), "Sat Oct 18 09:00:00 2026")]
    assert not os.path.exists(os.path.join(dest, ATTACH_DIR, CONTROLLER_PID))


@pytest.mark.parametrize("server_alive", [True, False])
def test_resume_finishes_the_teardown_of_a_collected_job_and_keeps_its_outcome(
        tmp_path, monkeypatch, server_alive):
    into, dest, inst, run_one = _run(tmp_path, monkeypatch, fail=_ProcessKilled())
    with pytest.raises(_ProcessKilled):
        run_one(("chen_nx512", "--nx 512"))
    torn, followed = [], []
    monkeypatch.setattr(sweep, "warn_strays", lambda conn, **kw: [])
    monkeypatch.setattr(sweep, "current_ingress_cidr", lambda: "9.9.9.9/32")
    monkeypatch.setattr(sweep, "_server_by_name_or_id", lambda conn, name, sid:
                        SimpleNamespace(id="srv-1") if server_alive else None)
    monkeypatch.setattr(sweep, "teardown_by_name", lambda conn, name, sid: torn.append(name))
    monkeypatch.setattr(sweep, "follow_detached_job", lambda *a, **kw: followed.append(1))

    assert sweep._reattach_records(None, None, into, 1) == 0

    out = read_outcome(dest)
    assert out.rc == 0 and out.status == "ok"
    assert followed == []
    assert len(torn) == (1 if server_alive else 0)
    assert _load_attach_records(into) == []


# --- 3. the resume's stray check raises when it cannot run ---------------------------

def _in_flight(into, label="a", cloud="flux-ovh", region="UK1"):
    dest = os.path.join(into, label)
    key = os.path.join(into, "key")
    os.makedirs(into, exist_ok=True)
    with open(key, "w") as fh:
        fh.write("KEY")
    return _write_attach_record(dest, label=label, cloud=cloud, region=region,
                                name=f"flux-compute-sweep-{label}", server_id="srv-1",
                                ip="1.2.3.4", keyfile=key, remote_script="job.sh",
                                fetch="out", into=into, cap_seconds=60)


def _no_credentials(cloud=None, region=None):
    raise RuntimeError("No OVH OpenStack credentials found.\nProvide them ...")


def test_resume_raises_when_the_stray_check_cannot_read_credentials(tmp_path, monkeypatch):
    into = str(tmp_path / "into")
    _in_flight(into)
    monkeypatch.setattr(sweep, "connect", _no_credentials)
    followed = []
    monkeypatch.setattr(sweep, "follow_detached_job", lambda *a, **kw: followed.append(1))

    with pytest.raises(RuntimeError) as exc:
        sweep._reattach_records(None, None, into, 1)

    msg = str(exc.value)
    assert "stray check" in msg and "No OVH OpenStack credentials found." in msg
    assert "~/.config/openstack/clouds.yaml" in msg and "'flux-ovh'" in msg
    assert "hatched" in msg
    assert followed == [] and len(_load_attach_records(into)) == 1


def test_the_cli_resume_exits_nonzero_with_the_credential_message(
        tmp_path, monkeypatch, capsys):
    into = str(tmp_path / "into")
    _in_flight(into)
    monkeypatch.setattr(sweep, "connect", _no_credentials)
    assert cli.main(["sweep", "--resume", "--into", into]) == 1
    err = capsys.readouterr().err
    assert "hatched" in err and "skipped" not in err


def test_the_stray_check_connects_with_each_records_launch_cloud(tmp_path, monkeypatch):
    """The launch connected with `--cloud flux-ovh`; a resume line without
    `--cloud` must reach the same credentials, as the per-job re-attach does."""
    into = str(tmp_path / "into")
    _in_flight(into, "a", region="UK1")
    _in_flight(into, "b", region="DE1")
    _in_flight(into, "c", region="UK1")
    connects, checks = [], []
    monkeypatch.setattr(sweep, "connect",
                        lambda cloud=None, region=None: connects.append((cloud, region))
                        or SimpleNamespace(region=region))
    monkeypatch.setattr(sweep, "warn_strays",
                        lambda conn, strict=False: checks.append((conn.region, strict)))

    sweep._check_strays_for_records(_load_attach_records(into), None, None)

    assert connects == [("flux-ovh", "DE1"), ("flux-ovh", "UK1")]
    assert checks == [("DE1", True), ("UK1", True)]


def test_a_strict_stray_check_raises_when_the_server_list_fails(capsys):
    class _Broken:
        def servers(self, details=True):
            raise ConnectionError("neutron 503")
    conn = SimpleNamespace(compute=_Broken())
    with pytest.raises(RuntimeError, match="stray-instance check failed.*neutron 503"):
        reap.warn_strays(conn, strict=True)
    assert reap.warn_strays(conn) == []                     # advisory elsewhere
    assert "skipped" in capsys.readouterr().err


# --- the in-context teardown verifies the keypair and the group too ------------------

def test_a_sweep_whose_group_delete_keeps_failing_keeps_the_record_and_exits_nonzero(
        tmp_path, monkeypatch, local_boot):
    conn = _client()

    def refuse(sg_id, ignore_missing=True):
        raise RuntimeError("409 Conflict: security group in use")
    conn.network.delete_security_group = refuse
    monkeypatch.setattr(provision.time, "sleep", lambda s: None)
    monkeypatch.setattr(sweep, "connect", lambda cloud=None, region=None: conn)
    monkeypatch.setattr(sweep, "warn_strays", lambda conn, **kw: [])
    monkeypatch.setattr(sweep, "_prepare_shard",
                        lambda cloud, region, flavor, image, max_parallel, warn=True:
                        _shard(region))
    monkeypatch.setattr(sweep, "_scp_up", lambda *a, **kw: None)
    monkeypatch.setattr(sweep, "_launch_detached", lambda *a, **kw: "")
    monkeypatch.setattr(sweep, "make_stuck_handler", lambda *a, **kw: None)
    monkeypatch.setattr(sweep, "follow_detached_job",
                        lambda *a, **kw: PollOutcome(rc=0, reason="done", output_size=0))
    monkeypatch.setattr(sweep, "_finalize", lambda *a, **kw: (0, "ok"))
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("chen_nx512 = --nx 512\n")
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    into = str(tmp_path / "into")

    rc = sweep.run_sweep(region="UK1", jobs_file=str(jobs), script=str(script),
                         fetch="out", into=into, max_parallel=1)

    assert rc != 0
    assert conn.compute.deleted_servers == ["srv-1"]          # the server did go
    [rec] = _load_attach_records(into)                         # the handle survives
    assert rec.label == "chen_nx512"
    out = read_outcome(os.path.join(into, "chen_nx512"))
    assert out.rc == -1 and "security group" in out.status and "ok (job rc=0)" in out.status


def test_the_in_context_teardown_raises_on_a_keypair_left_behind(monkeypatch, local_boot):
    conn = _client()

    def refuse(name, ignore_missing=True):
        raise RuntimeError("503")
    conn.compute.delete_keypair = refuse
    with pytest.raises(TeardownStrandError, match="keypair flux-compute-run-t1"):
        with provision._gpu_instance(conn, _shard().spec, "flux-compute-run-t1",
                                     ttl_minutes=30):
            pass
    assert conn.compute.deleted_servers == ["srv-1"]
