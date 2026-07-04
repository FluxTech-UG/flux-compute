"""Pure-logic tests for provision helpers. No network, no credentials."""
from types import SimpleNamespace

import pytest

from flux_compute.provision import (
    TeardownStrandError,
    _delete_server_verified,
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
