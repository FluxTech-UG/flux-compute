"""Pure-logic tests for the flavor policy. No network, no credentials."""
from types import SimpleNamespace

import pytest

from flux_compute.flavors import (
    DEFAULT_SIM_FLAVOR,
    classify,
    flavor_ram_gb,
    live_flavor_spec,
    recommended_for_sim,
    static_flavor_spec,
)


def test_v100_is_eligible_and_fp64_healthy():
    v = classify("t1-le-45")
    assert v.kind == "gpu"
    assert v.credit_eligible
    assert v.fp64_healthy
    assert v.usable_for_sim


def test_v100s_is_eligible_and_fp64_healthy():
    v = classify("t2-le-90")
    assert v.credit_eligible and v.fp64_healthy and v.usable_for_sim


def test_rtx5000_is_eligible_but_not_fp64_healthy():
    v = classify("rtx5000-28")
    assert v.kind == "gpu"
    assert v.credit_eligible            # covered by credits ...
    assert not v.fp64_healthy           # ... but fp64-crippled
    assert not v.usable_for_sim         # so refused for sims by default


@pytest.mark.parametrize("name", ["h100-380", "a100-180", "l40s-90", "l4-90", "a10-45"])
def test_blocked_gpus_are_not_credit_eligible(name):
    v = classify(name)
    assert v.kind == "gpu"
    assert not v.credit_eligible
    assert not v.usable_for_sim


def test_l40s_is_matched_before_l4():
    assert classify("l40s-90").gpu_model.startswith("L40S")
    assert classify("l4-90").gpu_model.startswith("L4 ")


def test_cpu_flavor_is_usable():
    v = classify("c3-8")
    assert v.kind == "cpu"
    assert v.usable_for_sim


def test_cpu_flavors_are_priced():
    # The b3-* / c3-* families we fan CPU batches through carry catalog prices,
    # so a --budget guard is not blind on them.
    assert classify("b3-8").price_eur_hr == pytest.approx(0.0512)
    assert classify("b3-16").price_eur_hr == pytest.approx(0.1023)
    assert classify("c3-8").price_eur_hr == pytest.approx(0.0913)
    assert classify("c3-32").price_eur_hr == pytest.approx(0.3650)


def test_unpriced_cpu_flavor_is_cpu_but_priceless():
    # A CPU family we did not price (e.g. a -flex variant) still classifies as a
    # usable CPU flavor, but its price is unknown, which the budget guard catches.
    v = classify("b3-8-flex")
    assert v.kind == "cpu"
    assert v.usable_for_sim
    assert v.price_eur_hr is None


def test_unknown_flavor_is_not_usable():
    v = classify("zz-9000")
    assert v.kind == "unknown"
    assert not v.usable_for_sim


def test_default_sim_flavor_is_a_healthy_v100():
    v = classify(DEFAULT_SIM_FLAVOR)
    assert v.usable_for_sim
    assert v.gpu_model.startswith("Tesla V100")


def test_recommended_picks_cheapest_healthy_gpu():
    # t1-le-45 (0.70) beats t2-le-45 (0.80); rtx5000 excluded (fp64), h100 excluded (credits).
    available = ["d2-2", "rtx5000-28", "t2-le-45", "t1-le-45", "h100-380"]
    assert recommended_for_sim(available) == "t1-le-45"


def test_recommended_raises_when_no_healthy_gpu():
    with pytest.raises(RuntimeError):
        recommended_for_sim(["rtx5000-28", "h100-380", "c3-8"])


# --- RAM model: static (offline) derivation -----------------------------------

@pytest.mark.parametrize("name,vcpus,ram_gb", [
    ("c3-4", 2, 4), ("c3-8", 4, 8), ("c3-16", 8, 16), ("c3-256", 128, 256),
    ("b3-8", 2, 8), ("b3-16", 4, 16), ("b3-32", 8, 32), ("b3-512", 128, 512),
])
def test_static_cpu_spec_from_ratio_and_suffix(name, vcpus, ram_gb):
    # b3 = 4 GB/vCPU, c3 = 2 GB/vCPU; the suffix is the total RAM in GB.
    spec = static_flavor_spec(name)
    assert spec.kind == "cpu"
    assert spec.vcpus == vcpus
    assert spec.ram_gb == ram_gb
    assert spec.usable_for_sim


@pytest.mark.parametrize("name,vcpus,ram_gb", [
    ("t2-le-45", 15, 45), ("t2-le-90", 30, 90), ("t2-le-180", 60, 180),
    ("t1-le-45", 8, 45), ("t1-le-180", 32, 180),
    ("rtx5000-28", 4, 28), ("rtx5000-84", 16, 84),
])
def test_static_gpu_spec_from_catalog_table(name, vcpus, ram_gb):
    spec = static_flavor_spec(name)
    assert spec.kind == "gpu"
    assert spec.vcpus == vcpus and spec.ram_gb == ram_gb


@pytest.mark.parametrize("name,vram_gb", [
    ("t2-le-45", 32.0), ("t2-le-90", 64.0), ("t2-le-180", 128.0),
    ("t1-le-45", 16.0), ("t1-le-180", 64.0),
])
def test_static_gpu_spec_carries_vram(name, vram_gb):
    # VRAM is the device-memory axis; V100S = 32 GB/card, V100 = 16 GB/card,
    # scaled by card count on the multi-GPU flavors.
    assert static_flavor_spec(name).vram_gb == vram_gb


def test_static_cpu_spec_has_no_vram():
    assert static_flavor_spec("c3-4").vram_gb is None
    assert static_flavor_spec("b3-16").vram_gb is None


def test_static_spec_carries_price_and_usability():
    v100s = static_flavor_spec("t2-le-45")
    assert v100s.price_eur_hr == pytest.approx(0.80) and v100s.usable_for_sim
    rtx = static_flavor_spec("rtx5000-28")
    assert not rtx.usable_for_sim              # fp64-crippled, still describable


def test_static_spec_unknown_gpu_name_fails_fast():
    # A GPU family we recognize but a size not in the catalog table -> no guess.
    with pytest.raises(RuntimeError, match="no catalog RAM/vCPU"):
        static_flavor_spec("t2-le-999")


def test_static_spec_unsourced_cpu_family_fails_fast():
    with pytest.raises(RuntimeError, match="no sourced RAM-per-vCPU"):
        static_flavor_spec("d2-4")


def test_static_spec_nonnumeric_cpu_suffix_fails_fast():
    with pytest.raises(RuntimeError, match="not\n?.*integer|bare integer"):
        static_flavor_spec("b3-8-flex")


def test_static_spec_unknown_family_fails_fast():
    with pytest.raises(RuntimeError, match="cannot size unknown flavor"):
        static_flavor_spec("zz-9000")


# --- RAM model: live read off an OpenStack flavor object ----------------------

def test_flavor_ram_gb_converts_mib_to_gib():
    # OpenStack reports .ram in MiB; 46080 MiB = 45 GB.
    assert flavor_ram_gb(SimpleNamespace(name="t2-le-45", ram=46080)) == pytest.approx(45.0)


def test_flavor_ram_gb_missing_ram_fails_fast():
    with pytest.raises(RuntimeError, match="could not read RAM"):
        flavor_ram_gb(SimpleNamespace(name="t2-le-45"))


def test_live_flavor_spec_reads_vcpus_and_ram():
    obj = SimpleNamespace(name="t2-le-45", vcpus=15, ram=46080)
    spec = live_flavor_spec(obj)
    assert spec.kind == "gpu" and spec.vcpus == 15
    assert spec.ram_gb == pytest.approx(45.0)
    assert spec.price_eur_hr == pytest.approx(0.80)  # price still from the policy
    assert spec.vram_gb == 32.0                      # tabulated device memory


def test_live_cpu_flavor_spec_has_no_vram():
    spec = live_flavor_spec(SimpleNamespace(name="c3-8", vcpus=4, ram=8192))
    assert spec.kind == "cpu" and spec.vram_gb is None


def test_live_flavor_spec_missing_vcpus_fails_fast():
    with pytest.raises(RuntimeError, match="could not read the vCPU count"):
        live_flavor_spec(SimpleNamespace(name="c3-8", ram=8192))
