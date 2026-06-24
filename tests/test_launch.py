"""Pure-logic tests for launch-spec helpers. No network, no credentials."""
import pytest

from flux_compute.launch import select_gpu_image


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
