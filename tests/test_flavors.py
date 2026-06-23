"""Pure-logic tests for the flavor policy. No network, no credentials."""
import pytest

from flux_compute.flavors import DEFAULT_SIM_FLAVOR, classify, recommended_for_sim


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
