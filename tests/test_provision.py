"""Pure-logic tests for provision helpers. No network, no credentials."""
from flux_compute.provision import _smoke_command


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
