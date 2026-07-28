"""Pure-logic tests for provision helpers. No network, no credentials."""
from types import SimpleNamespace

import pytest

from flux_compute import provision
from flux_compute.provision import (
    _RSYNC_EXCLUDES,
    TeardownStrandError,
    _delete_server_verified,
    _gpu_instance,
    _rsync_up,
    _smoke_command,
    _stranded_banner,
    classify_exit,
    heal_ssh_ingress,
    looks_like_cap_kill,
    make_stuck_handler,
    oom_evidence,
    parse_upload_spec,
    probe_oom_kill,
    rsync_down_best_effort,
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


# --- rsync upload: exclude the live-fleet records dir, tolerate exit 24 --------
#
# Regression: uploading a repo that contains a live fleet's `cloud-sweep/`
# .flux_attach records let those records vanish mid-transfer, rsync exited 24,
# and check=True aborted the launch (stranding the freshly-booted VM).

def test_cloud_sweep_is_excluded_from_uploads():
    # The records dir a live fleet churns is never uploaded in the first place.
    assert "cloud-sweep" in _RSYNC_EXCLUDES


def _fake_rsync(monkeypatch, returncode):
    calls = {}

    def run(args, **kwargs):
        calls["args"] = args
        assert kwargs.get("check") is not True   # must not abort on nonzero itself
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(provision.subprocess, "run", run)
    return calls


def test_rsync_up_tolerates_exit_24_as_a_warning(monkeypatch, capsys):
    _fake_rsync(monkeypatch, 24)
    _rsync_up("./repo", "1.2.3.4", "/tmp/key", "repo")     # must not raise
    assert "exit 24" in capsys.readouterr().err           # warned, continued


def test_rsync_up_still_raises_on_a_real_failure(monkeypatch):
    _fake_rsync(monkeypatch, 23)
    with pytest.raises(RuntimeError, match="rsync upload of ./repo"):
        _rsync_up("./repo", "1.2.3.4", "/tmp/key", "repo")


def test_rsync_up_zero_is_success(monkeypatch):
    _fake_rsync(monkeypatch, 0)
    _rsync_up("./repo", "1.2.3.4", "/tmp/key", "repo")     # returns cleanly


# --- rc=137 is ambiguous: OOM vs cap kill vs unknown --------------------------

_OOM_LOG = """\
[12345.6] python3 invoked oom-killer: gfp_mask=0x140dca(GFP_HIGHUSER_MOVABLE)
[12345.7] Out of memory: Killed process 941 (python3) total-vm:41203400kB
"""


def test_oom_evidence_finds_the_kernel_oom_killer():
    probe = oom_evidence(_OOM_LOG)
    assert probe.confirmed and probe.read_ok
    assert "Killed process 941" in probe.summary


def test_oom_evidence_on_a_clean_log_is_unconfirmed_but_read():
    probe = oom_evidence("[1.0] eth0: link up\n[2.0] systemd: started\n")
    assert not probe.confirmed and probe.read_ok


def test_oom_evidence_on_an_unreadable_log_is_unconfirmed_and_unread():
    """Absence of evidence is reported as unknown, never as innocence."""
    probe = oom_evidence("")
    assert not probe.confirmed and not probe.read_ok


def test_looks_like_cap_kill_needs_a_basis():
    assert looks_like_cap_kill(1795, 1800) is True
    assert looks_like_cap_kill(60, 1800) is False
    assert looks_like_cap_kill(None, 1800) is None      # no elapsed -> no verdict
    assert looks_like_cap_kill(60, None) is None        # no cap -> no verdict


def test_classify_exit_never_calls_a_sub_cap_sigkill_a_timeout():
    status = classify_exit(137, elapsed_s=200, cap_seconds=7200)
    assert "timed out" not in status and "timeout" not in status
    assert "cause unknown" in status and "137" in status


def test_classify_exit_reports_a_confirmed_oom_kill():
    status = classify_exit(137, elapsed_s=200, cap_seconds=7200,
                           oom=oom_evidence(_OOM_LOG))
    assert "OOM-killed" in status and "oom-killer confirmed" in status


def test_classify_exit_distinguishes_a_read_log_from_an_unreadable_one():
    read = classify_exit(137, elapsed_s=200, cap_seconds=7200, oom=oom_evidence("clean\n"))
    unread = classify_exit(137, elapsed_s=200, cap_seconds=7200, oom=oom_evidence(""))
    assert "no oom-killer evidence" in read
    assert "kernel log unavailable" in unread


def test_classify_exit_ordinary_codes_are_unchanged():
    assert classify_exit(0) == "ok"
    assert "timed out" in classify_exit(124)
    assert "nonzero" in classify_exit(3)


def test_probe_oom_kill_never_raises_when_ssh_fails():
    def boom(ip, keyfile, command, timeout=None, capture=True):
        raise OSError("connection reset")
    probe = probe_oom_kill("1.2.3.4", "/k", _ssh=boom)
    assert not probe.confirmed and not probe.read_ok


def test_probe_oom_kill_reads_the_kernel_log_over_ssh():
    seen = {}

    def fake(ip, keyfile, command, timeout=None, capture=True):
        seen["cmd"] = command
        return SimpleNamespace(returncode=0, stdout=_OOM_LOG, stderr="")

    assert probe_oom_kill("1.2.3.4", "/k", _ssh=fake).confirmed
    assert "dmesg" in seen["cmd"] and "journalctl" in seen["cmd"]


# --- best-effort artifact fetch (never lose partial results) ------------------

def test_rsync_down_best_effort_swallows_a_failure(capsys):
    def boom(ip, kf, remote, local):
        raise RuntimeError("no such remote dir")
    assert rsync_down_best_effort("ip", "/k", "out", "/tmp/x", _rsync=boom) is False
    assert "partial fetch" in capsys.readouterr().err


def test_rsync_down_best_effort_reports_success():
    assert rsync_down_best_effort("ip", "/k", "out", "/tmp/x",
                                  _rsync=lambda *a: None) is True


# --- --upload SRC:DEST -------------------------------------------------------

def test_parse_upload_bare_dir_keeps_basename_behavior(tmp_path):
    src = tmp_path / "1DSim3"
    src.mkdir()
    assert parse_upload_spec(str(src)) == (str(src), "1DSim3")


def test_parse_upload_bare_dir_tolerates_a_trailing_slash(tmp_path):
    src = tmp_path / "1DSim3"
    src.mkdir()
    assert parse_upload_spec(str(src) + "/") == (str(src), "1DSim3")


def test_parse_upload_maps_src_to_a_different_remote_name(tmp_path):
    """The feature that replaces the symlink-named-like-the-destination trick."""
    src = tmp_path / "1DSim3-worktree"
    src.mkdir()
    assert parse_upload_spec(f"{src}:1DSim3") == (str(src), "1DSim3")


def test_parse_upload_rejects_a_missing_source(tmp_path):
    with pytest.raises(RuntimeError, match="not an existing directory"):
        parse_upload_spec(str(tmp_path / "nope"))


def test_parse_upload_rejects_an_escaping_or_absolute_destination(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    for bad in ("/etc", "../evil", "a/../../b"):
        with pytest.raises(RuntimeError, match="relative to the remote home"):
            parse_upload_spec(f"{src}:{bad}")


def test_rsync_excludes_cover_attach_records_under_any_into_dir():
    """`--into` is configurable, so excluding only the default name left every
    other results tree exposed to the mid-transfer record race."""
    assert ".flux_attach" in _RSYNC_EXCLUDES
    assert "cloud-sweep" in _RSYNC_EXCLUDES      # the default, still excluded


def test_rsync_up_passes_extra_excludes_through(monkeypatch, tmp_path):
    seen = {}

    def run(args, **kw):
        seen["args"] = args
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", run)
    _rsync_up(str(tmp_path), "1.2.3.4", "/k", "repo", extra_excludes=("/outputs/fleet",))
    assert "/outputs/fleet" in seen["args"]


# --- SSH self-heal: our public IP moved out of the security group -------------

class _FakeRule:
    def __init__(self, cidr, lo=22, hi=22, direction="ingress"):
        self.remote_ip_prefix, self.port_range_min = cidr, lo
        self.port_range_max, self.direction = hi, direction


class _FakeNet:
    def __init__(self, sg, rules):
        self.sg, self.rules, self.created = sg, rules, []

    def find_security_group(self, name):
        return self.sg

    def security_group_rules(self, security_group_id=None):
        return list(self.rules)

    def create_security_group_rule(self, **kw):
        self.created.append(kw)


def _conn_with(rules, sg=SimpleNamespace(id="sg-1")):
    return SimpleNamespace(network=_FakeNet(sg, rules))


def test_heal_opens_ingress_when_the_public_ip_moved(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "9.9.9.9/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    check = heal_ssh_ingress(conn, "flux-compute-sweep-abcd")
    assert check and check.status == "healed" and "9.9.9.9/32" in check.message
    assert conn.network.created[0]["remote_ip_prefix"] == "9.9.9.9/32"
    assert conn.network.created[0]["port_range_min"] == 22


def test_heal_appends_and_never_replaces_the_launch_time_rule(monkeypatch):
    """The stale /32 stays: a flapping address (tethering, a VPN toggling back)
    must not lock out the very machine that launched the fleet."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "9.9.9.9/32")
    original = _FakeRule("1.2.3.4/32")
    conn = _conn_with([original])
    assert heal_ssh_ingress(conn, "sg")
    assert original in conn.network.rules            # untouched, still present
    assert len(conn.network.created) == 1            # added, not swapped


def test_heal_is_a_noop_when_ingress_is_already_open(monkeypatch):
    """Idempotent: a repeated escalation must not pile up duplicate rules."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    check = heal_ssh_ingress(conn, "sg")
    assert not check and check.status == "open"
    assert conn.network.created == []


def test_heal_declines_when_the_public_ip_read_failed(monkeypatch):
    """0.0.0.0/0 is _public_ip_cidr's failure value; never open the world on it.
    The status must say the check did not run, not that the group looked fine."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: provision.UNKNOWN_CIDR)
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    check = heal_ssh_ingress(conn, "sg")
    assert not check and check.status == "unknown-ip"
    assert "public IP" in check.message
    assert conn.network.created == []


def test_heal_declines_when_the_security_group_is_gone(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "9.9.9.9/32")
    conn = _conn_with([], sg=None)
    check = heal_ssh_ingress(conn, "sg")
    assert not check and check.status == "no-group"


def test_heal_uses_a_caller_supplied_cidr_without_re_reading(monkeypatch):
    """A fleet-wide repair resolves the address once and passes it down."""
    def _boom():
        raise AssertionError("must not re-read the public IP")

    monkeypatch.setattr(provision, "_public_ip_cidr", _boom)
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    assert heal_ssh_ingress(conn, "sg", "5.5.5.5/32")
    assert conn.network.created[0]["remote_ip_prefix"] == "5.5.5.5/32"


def test_current_ingress_cidr_is_none_when_the_read_failed(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: provision.UNKNOWN_CIDR)
    assert provision.current_ingress_cidr() is None
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "7.7.7.7/32")
    assert provision.current_ingress_cidr() == "7.7.7.7/32"


def test_stuck_handler_surfaces_the_blackout_and_tries_to_heal(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "9.9.9.9/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    lines = []
    make_stuck_handler(conn, "sg", label="alpha", emit=lines.append)(8, 300.0)
    text = "\n".join(lines)
    assert "SSH unreachable since" in text and "[alpha]" in text
    assert "9.9.9.9/32" in text                       # healed, and said so
    assert conn.network.created                       # the rule really was added


def test_stuck_handler_says_so_when_the_public_ip_could_not_be_read(monkeypatch):
    """The blackout report must not claim a healthy group when the check never ran."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: provision.UNKNOWN_CIDR)
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    lines = []
    make_stuck_handler(conn, "sg", emit=lines.append)(4, 60.0)
    text = "\n".join(lines)
    assert "could not read" in text
    assert "ingress still open" not in text        # the misleading old claim
    assert conn.network.created == []


def test_stuck_handler_reports_a_failed_heal_without_raising():
    """The follow loop's contract is that only the deadline aborts it, so the
    handler must swallow its own failure."""
    class _Boom:
        def find_security_group(self, name):
            raise RuntimeError("network API down")

    lines = []
    make_stuck_handler(SimpleNamespace(network=_Boom()), "sg", emit=lines.append)(4, 60.0)
    assert any("ingress re-check failed" in ln for ln in lines)
