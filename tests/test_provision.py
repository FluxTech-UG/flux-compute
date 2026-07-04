"""Pure-logic tests for provision helpers. No network, no credentials."""
from types import SimpleNamespace

import pytest

from flux_compute import provision
from flux_compute.provision import (
    TeardownStrandError,
    _delete_server_verified,
    _gpu_instance,
    _smoke_command,
    _stranded_banner,
)


def test_gpu_smoke_uses_nvidia_smi():
    label, cmd = _smoke_command("Tesla V100S 32GB")
    assert label == "GPU"
    assert "nvidia-smi" in cmd


def test_cpu_smoke_verifies_boot_and_exec_without_gpu():
    label, cmd = _smoke_command(None)
    assert label == "CPU"
    assert "nvidia-smi" not in cmd      # could never pass on a CPU flavor
    assert "python3" in cmd             # boot + remote-exec check
    assert "nproc" in cmd


# --- verified, retrying teardown ----------------------------------------------

class _FakeCompute:
    """Fake compute API: fail the first `fail_deletes` delete calls and the
    first `fail_waits` wait_for_delete calls, then succeed."""

    def __init__(self, fail_deletes=0, fail_waits=0):
        self.fail_deletes = fail_deletes
        self.fail_waits = fail_waits
        self.delete_calls = 0
        self.wait_calls = 0

    def delete_server(self, server_id, force=False):
        self.delete_calls += 1
        if self.delete_calls <= self.fail_deletes:
            raise RuntimeError("API 500 on delete")

    def wait_for_delete(self, server, wait=0):
        self.wait_calls += 1
        if self.wait_calls <= self.fail_waits:
            raise RuntimeError("server still present after wait")


def _conn(compute):
    return SimpleNamespace(compute=compute)


_SERVER = SimpleNamespace(id="srv-123")


def test_delete_retries_transient_failure_then_verifies():
    compute = _FakeCompute(fail_deletes=2)
    _delete_server_verified(_conn(compute), _SERVER, retries=3, retry_delay=0)
    assert compute.delete_calls == 3          # two failures, then success
    assert compute.wait_calls == 1            # verified gone exactly once


def test_unverified_delete_is_a_strand():
    # delete_server "succeeds" but the server never disappears: wait_for_delete
    # failing every time must be treated as a strand, not a quiet log line.
    compute = _FakeCompute(fail_waits=99)
    with pytest.raises(TeardownStrandError) as exc:
        _delete_server_verified(_conn(compute), _SERVER, retries=3, retry_delay=0)
    assert compute.delete_calls == 3
    assert "srv-123" in str(exc.value)


def test_persistent_delete_failure_is_a_strand():
    compute = _FakeCompute(fail_deletes=99)
    with pytest.raises(TeardownStrandError):
        _delete_server_verified(_conn(compute), _SERVER, retries=3, retry_delay=0)


def test_strand_error_is_a_runtimeerror_so_cli_exits_nonzero():
    # cli.py catches RuntimeError and returns 1; the strand must ride that path.
    assert issubclass(TeardownStrandError, RuntimeError)


def test_stranded_banner_is_unmissable_and_actionable(capsys):
    _stranded_banner("flux-ovh", "flux-compute-run-abcd1234", "srv-123", "boom")
    err = capsys.readouterr().err
    assert "STRANDED INSTANCE" in err
    assert err.count("\n") >= 8                                   # multi-line, not one quiet line
    assert "openstack --os-cloud flux-ovh server delete srv-123" in err
    assert "keypair delete flux-compute-run-abcd1234" in err
    assert "security group delete flux-compute-run-abcd1234" in err


# --- _gpu_instance keep branch: exceptions must not be swallowed ---------------

class _FakeInstanceCompute:
    """Fake compute API for driving _gpu_instance end to end."""

    def __init__(self):
        self.deleted = []

    def find_image(self, name):
        return SimpleNamespace(id="img-1")

    def find_flavor(self, name):
        return SimpleNamespace(id="fl-1")

    def create_keypair(self, name, public_key):
        return SimpleNamespace(name=name)

    def create_server(self, **kwargs):
        return SimpleNamespace(
            id="srv-1",
            addresses={"Ext-Net": [{"version": 4, "addr": "10.0.0.1"}]},
        )

    def wait_for_server(self, server, status=None, wait=None):
        return server

    def delete_server(self, server_id, force=False):
        self.deleted.append(server_id)

    def wait_for_delete(self, server, wait=0):
        pass

    def delete_keypair(self, name, ignore_missing=True):
        pass


class _FakeInstanceNetwork:
    def find_network(self, name):
        return SimpleNamespace(id="net-1")

    def create_security_group(self, name=None, description=None):
        return SimpleNamespace(id="sg-1")

    def create_security_group_rule(self, **kwargs):
        pass

    def delete_security_group(self, sg_id, ignore_missing=True):
        pass


def _instance_conn():
    return SimpleNamespace(compute=_FakeInstanceCompute(),
                           network=_FakeInstanceNetwork(),
                           config=SimpleNamespace(name="fake-cloud"))


_SPEC = SimpleNamespace(image="Ubuntu 24.04", flavor="b3-8", network="Ext-Net")


@pytest.fixture
def local_boot(monkeypatch):
    """Cut the two real-network steps out of _gpu_instance."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    monkeypatch.setattr(provision, "_wait_ssh", lambda ip, **kw: True)


def test_keep_does_not_swallow_a_body_exception(local_boot, capsys):
    # Regression: a `return` inside finally used to swallow the with-body
    # exception under --keep, so a failed upload/SSH exited 0 with the
    # instance left running.
    conn = _instance_conn()
    with pytest.raises(RuntimeError, match="upload failed"):
        with _gpu_instance(conn, _SPEC, "flux-compute-run-t1", ttl_minutes=30, keep=True):
            raise RuntimeError("upload failed")
    assert conn.compute.deleted == []                    # keep: no delete attempted
    assert "LEFT RUNNING" in capsys.readouterr().out     # keep banner still printed


def test_body_exception_still_tears_down_without_keep(local_boot, capsys):
    conn = _instance_conn()
    with pytest.raises(RuntimeError, match="job failed"):
        with _gpu_instance(conn, _SPEC, "flux-compute-run-t2", ttl_minutes=30):
            raise RuntimeError("job failed")
    assert conn.compute.deleted == ["srv-1"]             # teardown ran, then the error propagated
