"""Pure-logic tests for launch-spec helpers. No network, no credentials."""
from types import SimpleNamespace

import pytest

from flux_compute.launch import resolve_spec, select_cpu_image, select_gpu_image


def _fake_conn(flavor_names, image_names, networks=("Ext-Net",)):
    """A minimal stand-in for an openstack connection for resolve_spec tests."""
    flavors = [SimpleNamespace(name=n) for n in flavor_names]
    images = [SimpleNamespace(name=n) for n in image_names]
    nets = [SimpleNamespace(name=n) for n in networks]
    return SimpleNamespace(
        compute=SimpleNamespace(
            flavors=lambda: flavors,
            find_image=lambda name: next((i for i in images if i.name == name), None),
        ),
        network=SimpleNamespace(networks=lambda: nets),
        image=SimpleNamespace(images=lambda: images),
    )


_IMAGES = ["Ubuntu 22.04", "Ubuntu 24.04", "Ubuntu 24.04 - NVIDIA - v580",
           "Ubuntu 22.04 - NVIDIA - v535", "Baremetal - Ubuntu 24.04"]


def test_prefers_2404_nvidia_driver_image():
    names = [
        "Ubuntu 22.04", "Ubuntu 24.04",
        "Ubuntu 24.04 - NVIDIA - v580", "Ubuntu 22.04 - NVIDIA - v535",
        "NVIDIA GPU Cloud (NGC)",
    ]
    assert select_gpu_image(names) == "Ubuntu 24.04 - NVIDIA - v580"


def test_falls_back_to_2204_when_no_2404():
    names = ["Ubuntu 22.04 - NVIDIA - v535", "Ubuntu 20.04 - NVIDIA - v470"]
    assert select_gpu_image(names) == "Ubuntu 22.04 - NVIDIA - v535"


def test_ngc_alone_is_not_an_ubuntu_image():
    # NGC contains "nvidia" but not "ubuntu", so it is not a base OS image here.
    with pytest.raises(RuntimeError):
        select_gpu_image(["NVIDIA GPU Cloud (NGC)", "Debian 12"])


def test_raises_when_no_nvidia_image():
    with pytest.raises(RuntimeError):
        select_gpu_image(["Ubuntu 24.04", "Debian 12"])


# --- CPU image selection ------------------------------------------------------

def test_cpu_image_prefers_2404_plain_ubuntu():
    assert select_cpu_image(_IMAGES) == "Ubuntu 24.04"


def test_cpu_image_excludes_nvidia_and_baremetal():
    # Only a driver image and a baremetal image present -> neither is a plain VM
    # Ubuntu, so refuse rather than boot a wasteful GPU-driver image on a CPU VM.
    with pytest.raises(RuntimeError):
        select_cpu_image(["Ubuntu 24.04 - NVIDIA - v580", "Baremetal - Ubuntu 24.04"])


def test_cpu_image_falls_back_to_2204_when_no_2404():
    assert select_cpu_image(["Ubuntu 22.04", "Debian 12"]) == "Ubuntu 22.04"


def test_cpu_image_prefers_base_over_uefi_variant():
    # GRA11 ships both "Ubuntu 24.04" and "Ubuntu 24.04 - UEFI"; the plain base
    # image is the safe default, not the UEFI variant.
    assert select_cpu_image(["Ubuntu 24.04", "Ubuntu 24.04 - UEFI"]) == "Ubuntu 24.04"


# --- resolve_spec image dispatch by flavor kind -------------------------------

def test_resolve_spec_cpu_flavor_gets_plain_ubuntu():
    conn = _fake_conn(["c3-8", "t2-le-45"], _IMAGES)
    spec = resolve_spec(conn, "GRA11", flavor="c3-8")
    assert spec.image == "Ubuntu 24.04"       # plain, non-NVIDIA
    assert spec.gpu_model is None             # CPU flavor
    assert spec.est_cost_eur_hr == pytest.approx(0.0913)


def test_resolve_spec_gpu_flavor_gets_nvidia_image():
    conn = _fake_conn(["c3-8", "t2-le-45"], _IMAGES)
    spec = resolve_spec(conn, "GRA11", flavor="t2-le-45")
    assert spec.image == "Ubuntu 24.04 - NVIDIA - v580"
    assert spec.gpu_model.startswith("Tesla V100S")


def test_resolve_spec_image_override_wins_for_cpu():
    conn = _fake_conn(["c3-8"], _IMAGES)
    spec = resolve_spec(conn, "GRA11", flavor="c3-8", image="Ubuntu 22.04")
    assert spec.image == "Ubuntu 22.04"


# --- CLI output streaming (an empty log must mean "nothing happened") ---------

def test_stream_output_line_buffers_a_piped_stdout(monkeypatch):
    """Piped stdout block-buffers by default, so hours of fleet progress sat in a
    4 KiB buffer and `| tee run.log` showed nothing. Line buffering is set at the
    entry point so no caller has to remember PYTHONUNBUFFERED."""
    import io
    import sys
    from flux_compute.cli import _stream_output

    pipe = io.TextIOWrapper(io.BufferedWriter(io.BytesIO()), line_buffering=False)
    monkeypatch.setattr(sys, "stdout", pipe)
    monkeypatch.setattr(sys, "stderr", pipe)
    assert pipe.line_buffering is False
    _stream_output()
    assert pipe.line_buffering is True


def test_stream_output_tolerates_a_stream_that_cannot_reconfigure(monkeypatch):
    """pytest's capture and other stand-ins are not TextIOWrapper; the entry point
    must not die on them."""
    import io
    import sys
    from flux_compute.cli import _stream_output

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _stream_output()          # must not raise


# --- shared region-name resolution (was four near-copies) --------------------

def test_resolve_region_name_prefers_the_explicit_override():
    from types import SimpleNamespace
    from flux_compute.auth import resolve_region_name
    conn = SimpleNamespace(config=SimpleNamespace(region_name="DE1"))
    assert resolve_region_name(conn, "UK1") == "UK1"


def test_resolve_region_name_falls_back_to_the_connection_then_env(monkeypatch):
    from types import SimpleNamespace
    from flux_compute.auth import resolve_region_name
    conn = SimpleNamespace(config=SimpleNamespace(region_name="DE1"))
    assert resolve_region_name(conn, None) == "DE1"

    bare = SimpleNamespace(config=None)
    monkeypatch.setenv("OS_REGION_NAME", "WAW1")
    assert resolve_region_name(bare, None) == "WAW1"
    monkeypatch.delenv("OS_REGION_NAME")
    assert resolve_region_name(bare, None) == "(unknown)"


def test_every_module_shares_one_region_resolver():
    """Four modules carried byte-identical copies of this chain; they must now
    all route through the one definition."""
    from flux_compute import auth, doctor, launch, provision, regions
    assert provision._region.__module__ == "flux_compute.provision"
    conn = type("C", (), {"config": type("Cfg", (), {"region_name": "GRA11"})()})()
    assert provision._region(conn, None) == "GRA11"
    assert regions._region_name(conn, None) == "GRA11"
    assert doctor._region_of(conn, None) == "GRA11"


def test_plan_no_longer_claims_launching_is_unwired():
    """Living-documents: `run --plan` used to tell the user the product it is
    part of does not exist yet."""
    import inspect
    from flux_compute import launch
    src = inspect.getsource(launch)
    assert "not wired yet" not in src and "for now" not in src
