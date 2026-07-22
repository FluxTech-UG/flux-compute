"""Pure-logic tests for the fleet planner. No network, no credentials."""
import math

import pytest

from flux_compute.flavors import static_flavor_spec
from flux_compute.fleet import (
    ALL_REGIONS,
    GPU_REGIONS,
    FleetPlan,
    JobRequirements,
    RegionUnit,
    choose_flavor,
    format_plan,
    jobs_per_vm,
    plan_fleet,
    plan_fleet_core,
    _resolve_device,
)


# --- JobRequirements validation ----------------------------------------------

def test_requirements_defaults():
    r = JobRequirements(job_count=10, ram_gb_per_job=2.0)
    assert r.device == "either" and r.minutes_per_job == 30.0
    assert not r.batchable and r.batch_width is None


def test_requirements_bad_device_raises():
    with pytest.raises(RuntimeError, match="device must be"):
        JobRequirements(job_count=1, ram_gb_per_job=1.0, device="tpu")


@pytest.mark.parametrize("kw", [
    {"job_count": 0, "ram_gb_per_job": 1.0},
    {"job_count": 1, "ram_gb_per_job": 0.0},
    {"job_count": 1, "ram_gb_per_job": 1.0, "minutes_per_job": 0.0},
])
def test_requirements_nonpositive_fields_raise(kw):
    with pytest.raises(RuntimeError):
        JobRequirements(**kw)


def test_batch_width_without_batchable_raises():
    with pytest.raises(RuntimeError, match="only means something for batchable"):
        JobRequirements(job_count=1, ram_gb_per_job=1.0, batch_width=64)


def test_batch_width_must_be_positive():
    with pytest.raises(RuntimeError, match="batch_width must be"):
        JobRequirements(job_count=1, ram_gb_per_job=1.0, batchable=True, batch_width=0)


# --- device resolution --------------------------------------------------------

def test_either_resolves_to_gpu_when_batchable():
    assert _resolve_device(JobRequirements(1, 1.0, "either", batchable=True)) == "gpu"


def test_either_resolves_to_cpu_when_unbatched():
    assert _resolve_device(JobRequirements(1, 1.0, "either")) == "cpu"


def test_explicit_device_is_respected():
    assert _resolve_device(JobRequirements(1, 1.0, "cpu", batchable=True)) == "cpu"
    assert _resolve_device(JobRequirements(1, 1.0, "gpu")) == "gpu"


# --- flavor choice ------------------------------------------------------------

def test_choose_cheapest_cpu_flavor_that_fits_one_job():
    # 2 GB/job: c3-4 (4 GB, cheapest CPU VM) holds one job with headroom.
    spec = choose_flavor(JobRequirements(50, 2.0, "cpu"))
    assert spec.name == "c3-4"


def test_ram_heavy_cpu_job_prefers_b3_over_c3():
    # 8 GB/job needs a >=10 GB VM; b3-16 (16 GB) is cheaper than c3-16 for the
    # same RAM because c3 spends money on vCPUs the single job cannot use.
    spec = choose_flavor(JobRequirements(10, 8.0, "cpu"))
    assert spec.name == "b3-16"
    assert spec.price_eur_hr < static_flavor_spec("c3-16").price_eur_hr


def test_choose_gpu_flavor_is_the_eu_wide_default():
    spec = choose_flavor(JobRequirements(10, 4.0, "gpu"))
    assert spec.name == "t2-le-45" and spec.kind == "gpu"


def test_ram_above_largest_flavor_fails_fast():
    with pytest.raises(RuntimeError, match="exceeds the largest cpu flavor"):
        choose_flavor(JobRequirements(1, 10_000.0, "cpu"))


# --- packing K (jobs per VM) --------------------------------------------------

def test_pack_unbatched_is_ram_bound_when_ram_tight():
    # c3-4: 2 vCPU, 4 GB * 0.8 = 3.2 GB usable; a 2 GB job fits once (not twice).
    K = jobs_per_vm(JobRequirements(10, 2.0, "cpu"), static_flavor_spec("c3-4"))
    assert K == 1


def test_pack_unbatched_is_vcpu_bound_when_ram_ample():
    # b3-32: 8 vCPU, 32 GB; small 1 GB jobs -> RAM allows 25 but vCPUs cap at 8.
    K = jobs_per_vm(JobRequirements(10, 1.0, "cpu"), static_flavor_spec("b3-32"))
    assert K == 8


def test_pack_batched_uses_preferred_width_clamped_by_ram():
    spec = static_flavor_spec("t2-le-45")   # 45 GB
    # 0.2 GB/member -> RAM holds 180; preferred width 128 wins (< RAM).
    wide = jobs_per_vm(JobRequirements(500, 0.2, "gpu", batchable=True, batch_width=128), spec)
    assert wide == 128
    # 1.5 GB/member -> RAM holds 24; the 128 request is clamped down.
    tight = jobs_per_vm(JobRequirements(500, 1.5, "gpu", batchable=True, batch_width=128), spec)
    assert tight == 24


def test_pack_batched_without_width_is_ram_filled():
    spec = static_flavor_spec("t2-le-45")
    K = jobs_per_vm(JobRequirements(500, 3.0, "gpu", batchable=True), spec)
    assert K == int(45 * 0.8 // 3.0)      # 12


def test_pack_not_even_one_fits_fails_fast():
    with pytest.raises(RuntimeError, match="not even one fits"):
        jobs_per_vm(JobRequirements(1, 40.0, "cpu"), static_flavor_spec("c3-4"))


# --- pure core ----------------------------------------------------------------

def _units(names_caps):
    return [RegionUnit(region=r, spec=static_flavor_spec(f), cap=c)
            for r, f, c in names_caps]


def test_core_single_region_waves_and_spare():
    req = JobRequirements(job_count=100, ram_gb_per_job=1.0, device="cpu")
    primary = static_flavor_spec("c3-4")     # 2 vCPU -> K=2 for 1 GB jobs
    units = _units([("GRA11", "c3-4", 10)])
    plan = plan_fleet_core(req, units, device="cpu", primary=primary)
    assert plan.vm_count == 10
    assert plan.jobs_per_vm == 2
    assert plan.slots_per_wave == 20                    # 10 VMs * K=2
    assert plan.waves == math.ceil(100 / 20) == 5
    assert plan.capacity == 100 and plan.spare_slots == 0


def test_core_spare_slots_reports_room_to_fill():
    req = JobRequirements(job_count=90, ram_gb_per_job=1.0, device="cpu")
    primary = static_flavor_spec("c3-4")
    plan = plan_fleet_core(req, _units([("GRA11", "c3-4", 10)]), device="cpu", primary=primary)
    # 10 VMs * K=2 = 20 slots/wave; 90 jobs -> 5 waves, capacity 100, 10 spare.
    assert plan.waves == 5 and plan.capacity == 100 and plan.spare_slots == 10


def test_core_worst_case_cost_sums_regions_at_their_price():
    req = JobRequirements(job_count=8, ram_gb_per_job=4.0, device="gpu",
                          minutes_per_job=60)
    primary = static_flavor_spec("t2-le-45")
    # Two GPU regions at different cards/prices; 4 GB/job, unbatched -> K=1.
    units = _units([("DE1", "t2-le-45", 2), ("BHS5", "t1-le-45", 4)])
    plan = plan_fleet_core(req, units, device="gpu", primary=primary)
    assert plan.vm_count == 6
    # K=1, 6 slots/wave, 8 jobs -> 2 waves. Every VM busy both waves:
    # DE1: 2 VM * 2 waves * 0.80 * 1h ; BHS5: 4 * 2 * 0.70 * 1h.
    assert plan.worst_case_eur == pytest.approx(2 * 2 * 0.80 + 4 * 2 * 0.70)


def test_core_budget_exceeded_raises():
    req = JobRequirements(job_count=8, ram_gb_per_job=4.0, device="gpu", minutes_per_job=60)
    primary = static_flavor_spec("t2-le-45")
    units = _units([("DE1", "t2-le-45", 2), ("BHS5", "t1-le-45", 4)])
    with pytest.raises(RuntimeError, match="exceeds budget"):
        plan_fleet_core(req, units, device="gpu", primary=primary, budget=1.0)


def test_core_no_regions_fails_fast():
    req = JobRequirements(job_count=1, ram_gb_per_job=1.0, device="gpu")
    with pytest.raises(RuntimeError, match="no region can host"):
        plan_fleet_core(req, [], device="gpu", primary=static_flavor_spec("t2-le-45"))


def test_core_max_parallel_caps_fleet_and_spreads():
    req = JobRequirements(job_count=100, ram_gb_per_job=1.0, device="cpu")
    primary = static_flavor_spec("c3-4")
    units = _units([("GRA11", "c3-4", 10), ("DE1", "c3-4", 10), ("UK1", "c3-4", 10)])
    plan = plan_fleet_core(req, units, device="cpu", primary=primary, max_parallel=5)
    assert plan.vm_count == 5
    # breadth-first split of 5 across 3 regions: 2, 2, 1.
    assert sorted(ra.vms for ra in plan.region_allocation) == [1, 2, 2]


# --- offline facade -----------------------------------------------------------

def test_plan_fleet_cpu_spreads_across_all_nine_regions():
    plan = plan_fleet(JobRequirements(200, 2.0, "cpu"))
    assert plan.device == "cpu" and plan.flavor == "c3-4"
    assert len(plan.region_allocation) == len(ALL_REGIONS)
    assert plan.vm_count == 90                       # 9 regions * 10 instance-cap
    assert isinstance(plan, FleetPlan)


def test_plan_fleet_gpu_uses_gpu_regions_and_bhs5_v100():
    plan = plan_fleet(JobRequirements(50, 4.0, "gpu"))
    assert plan.device == "gpu"
    regions = {ra.region: ra for ra in plan.region_allocation}
    assert set(regions) == set(GPU_REGIONS)
    assert regions["DE1"].flavor == "t2-le-45" and regions["DE1"].vms == 2
    assert regions["BHS5"].flavor == "t1-le-45" and regions["BHS5"].vms == 4
    assert plan.vm_count == 12                        # 4*2 + 4 (BHS5)


def test_plan_fleet_gpu_with_only_cpu_regions_fails_fast():
    with pytest.raises(RuntimeError, match="none of the regions"):
        plan_fleet(JobRequirements(10, 2.0, "gpu"), regions="SBG5,RBX-A")


def test_plan_fleet_regions_override_is_honored():
    plan = plan_fleet(JobRequirements(20, 2.0, "cpu"), regions="GRA11,DE1")
    assert {ra.region for ra in plan.region_allocation} == {"GRA11", "DE1"}
    assert plan.vm_count == 20


def test_plan_fleet_budget_guard_raises_offline():
    with pytest.raises(RuntimeError, match="exceeds budget"):
        plan_fleet(JobRequirements(1000, 2.0, "cpu", minutes_per_job=60), budget=1.0)


def test_format_plan_returns_text():
    plan = plan_fleet(JobRequirements(100, 2.0, "cpu"))
    text = format_plan(JobRequirements(100, 2.0, "cpu"), plan)
    assert "fleet plan" in text and "flavor" in text and "worst case" in text
