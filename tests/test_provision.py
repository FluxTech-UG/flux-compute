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

    def get_server(self, server_id):
        return SimpleNamespace(
            id=server_id, status="ACTIVE", task_state=None, fault=None,
            addresses={"Ext-Net": [{"version": 4, "addr": "10.0.0.1"}]},
        )

    def get_server_console_output(self, server, length=None):
        return {"output": "[    0.000000] Linux version 6.8\ncloud-init: starting\n"}

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
    monkeypatch.setattr(provision, "_wait_ssh",
                        lambda ip, **kw: provision.SshWait(True, 12.0, "1.2.3.4/32", "1.2.3.4/32", False, []))


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


def test_stuck_handler_reports_a_failed_heal_without_raising(monkeypatch):
    """The follow loop tolerates a blackout it cannot explain, so the handler must
    swallow its own failure -- and still hand back a status saying the check never
    ran, so the loop cannot mistake an unchecked group for a verified-open one."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    class _Boom:
        def find_security_group(self, name):
            raise RuntimeError("network API down")

    lines = []
    check = make_stuck_handler(
        SimpleNamespace(network=_Boom()), "sg", emit=lines.append)(4, 60.0)
    assert check.status == "error"
    assert any("could not check SSH ingress" in ln and "network API down" in ln
               for ln in lines)


def test_stuck_handler_hands_the_verdict_back_to_the_poll_loop(monkeypatch):
    """The loop cannot decide what a blackout means without knowing whether the
    group was verified open, repaired, or never actually checked."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    check = make_stuck_handler(conn, "sg", emit=lambda _m: None)(4, 60.0)
    assert check.status == "open"


# --- ensure_ssh_ingress: the one check-and-repair both callers share ----------
#
# The re-attach path and the steady-state poll loop each used to wrap
# heal_ssh_ingress in their own try/except with their own wording, so the same
# event was reported two different ways depending on which one noticed it.

def test_ensure_reports_a_repair_and_returns_the_healed_status(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "9.9.9.9/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    lines = []
    check = provision.ensure_ssh_ingress(conn, "sg", label="alpha", emit=lines.append)
    assert check.status == "healed"
    assert len(lines) == 1 and "9.9.9.9/32" in lines[0] and "[alpha]" in lines[0]
    assert conn.network.created                     # the rule really was added


def test_ensure_is_silent_on_the_common_no_op(monkeypatch):
    """Same network, nothing to do: a 100-job resume must not print 100 lines."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    lines = []
    check = provision.ensure_ssh_ingress(
        _conn_with([_FakeRule("1.2.3.4/32")]), "sg", emit=lines.append)
    assert check.status == "open" and lines == []


def test_ensure_announces_the_no_op_when_the_caller_asks(monkeypatch):
    """Mid-blackout, "ingress is fine" is the informative half of the answer, so
    the poll loop opts into hearing it."""
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    lines = []
    provision.ensure_ssh_ingress(
        _conn_with([_FakeRule("1.2.3.4/32")]), "sg",
        emit=lines.append, announce_open=True)
    assert len(lines) == 1 and "ingress still open" in lines[0]


def test_ensure_turns_an_api_error_into_a_status_and_never_raises():
    """The check is precautionary; the SSH attempt that follows is the authority.
    Abandoning a live, billing VM over a neutron hiccup would be far worse."""
    class _Boom:
        def find_security_group(self, name):
            raise RuntimeError("neutron 503")

    lines = []
    check = provision.ensure_ssh_ingress(
        SimpleNamespace(network=_Boom()), "sg-xyz", cidr="9.9.9.9/32", emit=lines.append)
    assert check.status == "error"                  # NOT "open": nothing was verified
    assert len(lines) == 1
    assert "WARNING" in lines[0] and "neutron 503" in lines[0] and "sg-xyz" in lines[0]


def test_ensure_uses_the_caller_supplied_cidr_without_re_reading(monkeypatch):
    """A fleet-wide resume resolves the address once and passes it down; every
    group must be compared against that same notion of "here"."""
    monkeypatch.setattr(provision, "_public_ip_cidr",
                        lambda: pytest.fail("re-read the public IP per group"))
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    check = provision.ensure_ssh_ingress(conn, "sg", cidr="9.9.9.9/32", emit=lambda _m: None)
    assert check.status == "healed"
    assert conn.network.created[0]["remote_ip_prefix"] == "9.9.9.9/32"


# --- the boot: ACTIVE is polled and narrated, SSH waits long and reopens the door ----
#
# Both launches of 21 October died at OVH before the first image: one reached ACTIVE and
# port 22 never opened in the fixed 180 s, the other sat in BUILD for fifteen minutes and
# ended in a bare SDK timeout. The launcher's public address is in a mobile NAT pool and
# the group admits only the /32 read at launch; the repair existed but never ran in the
# first wait. Neither failure said what the server had been doing.

def _ticking():
    """A clock that advances only when the code under test sleeps."""
    t = [0.0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))


class _BootingCompute(_FakeInstanceCompute):
    """get_server walks a scripted list of (status, task_state, fault), holding the last."""

    def __init__(self, states, console="kernel: hello\n"):
        super().__init__()
        self.states, self.console, self.polls = list(states), console, 0

    def get_server(self, server_id):
        self.polls += 1
        status, task, fault = self.states[min(self.polls, len(self.states)) - 1]
        return SimpleNamespace(id=server_id, status=status, task_state=task, fault=fault,
                               addresses={"Ext-Net": [{"version": 4, "addr": "10.0.0.1"}]})

    def get_server_console_output(self, server, length=None):
        return {"output": self.console}


def test_ssh_wait_is_ten_minutes_rechecking_the_address_every_minute():
    assert provision.SSH_WAIT_S == 600
    assert provision.SSH_INGRESS_RECHECK_S == 60
    assert provision.ACTIVE_WAIT_S == 900


def test_wait_ssh_reopens_ingress_when_the_public_ip_moved(monkeypatch):
    """The door opens the moment the moved address is admitted: the probe succeeds only
    after the heal added its rule, and the result says the address moved."""
    clock, sleep = _ticking()
    reads = iter(["1.2.3.4/32", "9.9.9.9/32"])
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: next(reads, "9.9.9.9/32"))
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    door = {"open": False}
    add = conn.network.create_security_group_rule

    def _create(**kw):
        add(**kw)
        door["open"] = True
    conn.network.create_security_group_rule = _create
    lines = []
    res = provision._wait_ssh("10.0.0.1", conn=conn, sg_name="sg", launch_cidr="1.2.3.4/32",
                              probe=lambda host, port: door["open"], clock=clock, sleep=sleep,
                              emit=lines.append)
    assert res.opened and res.moved
    assert res.launch_cidr == "1.2.3.4/32" and res.last_cidr == "9.9.9.9/32"
    assert conn.network.created[0]["remote_ip_prefix"] == "9.9.9.9/32"
    assert res.waited_s >= 2 * provision.SSH_INGRESS_RECHECK_S     # the second minute's read saw the move
    assert any("9.9.9.9/32" in line for line in lines)


def test_wait_ssh_timeout_says_the_address_stayed(monkeypatch):
    clock, sleep = _ticking()
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    conn = _conn_with([_FakeRule("1.2.3.4/32")])
    res = provision._wait_ssh("10.0.0.1", timeout=130, conn=conn, sg_name="sg", launch_cidr="1.2.3.4/32",
                              probe=lambda host, port: False, clock=clock, sleep=sleep, emit=lambda m: None)
    assert not res.opened and not res.moved and res.waited_s >= 130
    assert conn.network.created == []                     # same address: nothing to repair


def test_wait_ssh_returns_at_once_when_the_port_is_open(monkeypatch):
    def _boom():
        raise AssertionError("no address read when the door is already open")
    monkeypatch.setattr(provision, "_public_ip_cidr", _boom)
    clock, sleep = _ticking()
    res = provision._wait_ssh("10.0.0.1", conn=_conn_with([]), sg_name="sg", launch_cidr="1.2.3.4/32",
                              probe=lambda host, port: True, clock=clock, sleep=sleep, emit=lambda m: None)
    assert res.opened and res.waited_s == 0


def test_wait_active_narrates_each_change_and_returns_the_active_server():
    comp = _BootingCompute([("BUILD", "scheduling", None), ("BUILD", "scheduling", None),
                            ("BUILD", "spawning", None), ("ACTIVE", None, None)])
    clock, sleep = _ticking()
    lines = []
    server = provision._wait_active(SimpleNamespace(compute=comp), SimpleNamespace(id="srv-1"),
                                    wait=900, poll=10, emit=lines.append, clock=clock, sleep=sleep)
    assert server.status == "ACTIVE" and server.id == "srv-1"
    assert len(lines) == 3                                # a line per change, not per poll
    assert "scheduling" in lines[0] and "spawning" in lines[1] and "ACTIVE" in lines[2]
    assert "min" in lines[1]                              # each line carries when it happened


def test_wait_active_timeout_names_the_last_state_and_the_console():
    comp = _BootingCompute([("BUILD", "spawning", None)], console="kernel: hello\nnetworkd: no lease\n")
    clock, sleep = _ticking()
    with pytest.raises(RuntimeError, match="never reached ACTIVE within 60 s") as exc:
        provision._wait_active(SimpleNamespace(compute=comp), SimpleNamespace(id="srv-1"),
                               wait=60, poll=10, emit=lambda m: None, clock=clock, sleep=sleep)
    msg = str(exc.value)
    assert "status BUILD" in msg and "task state spawning" in msg
    assert "networkd: no lease" in msg


def test_wait_active_error_state_raises_at_once_with_the_fault():
    comp = _BootingCompute([("ERROR", None, {"message": "No valid host was found", "code": 500})])
    with pytest.raises(RuntimeError, match="No valid host was found"):
        provision._wait_active(SimpleNamespace(compute=comp), SimpleNamespace(id="srv-1"),
                               wait=60, poll=10, emit=lambda m: None, clock=lambda: 0.0, sleep=lambda s: None)
    assert comp.polls == 1


def test_console_tail_never_raises():
    class _NoConsole:
        def get_server_console_output(self, server, length=None):
            raise RuntimeError("409 Conflict")
    tail = provision._console_tail(SimpleNamespace(compute=_NoConsole()), SimpleNamespace(id="srv-1"))
    assert "unavailable" in tail and "409" in tail


def test_gpu_instance_ssh_failure_names_the_cause_and_tears_down(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    monkeypatch.setattr(provision, "_wait_ssh",
                        lambda ip, **kw: provision.SshWait(False, 600.0, "1.2.3.4/32", "1.2.3.4/32", False, []))
    conn = _instance_conn()
    with pytest.raises(RuntimeError, match="never opened within 600 s") as exc:
        with _gpu_instance(conn, _SPEC, "flux-compute-run-t3", ttl_minutes=30):
            raise AssertionError("the body must not run")
    msg = str(exc.value)
    assert "stayed 1.2.3.4/32" in msg and "cloud-init: starting" in msg
    assert conn.compute.deleted == ["srv-1"]


def test_gpu_instance_passes_the_group_and_launch_address_to_the_ssh_wait(monkeypatch):
    monkeypatch.setattr(provision, "_public_ip_cidr", lambda: "1.2.3.4/32")
    seen = {}

    def _wait(ip, **kw):
        seen.update(kw, ip=ip)
        return provision.SshWait(True, 1.0, "1.2.3.4/32", "1.2.3.4/32", False, [])
    monkeypatch.setattr(provision, "_wait_ssh", _wait)
    conn = _instance_conn()
    with _gpu_instance(conn, _SPEC, "flux-compute-run-t4", ttl_minutes=30):
        pass
    assert seen["ip"] == "10.0.0.1" and seen["sg_name"] == "flux-compute-run-t4"
    assert seen["launch_cidr"] == "1.2.3.4/32" and seen["conn"] is conn
