"""Pure-logic tests for the fleet planner. No network, no credentials."""
import math

import pytest

from flux_compute.flavors import static_flavor_spec
from flux_compute.fleet import (
    ALL_REGIONS,
    CATALOG_QUOTA_RAM_GB,
    GPU_REGIONS,
    FleetPlan,
    JobRequirements,
    RegionUnit,
    choose_flavor,
    format_plan,
    jobs_per_vm,
    plan_fleet,
    plan_fleet_core,
    _deal_counts,
    _region_cap,
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


def test_pack_batched_clamped_by_vram_then_width():
    spec = static_flavor_spec("t2-le-45")   # 45 GB host, 32 GB VRAM (25.6 usable)
    # 0.2 GB/member -> VRAM holds 128 (25.6/0.2); preferred width 128 fits.
    wide = jobs_per_vm(JobRequirements(500, 0.2, "gpu", batchable=True, batch_width=128), spec)
    assert wide == 128
    # 1.5 GB/member -> VRAM holds 17 (< host RAM's 24); the 128 request is clamped
    # down to the VRAM bound, the binding device axis for a batch.
    tight = jobs_per_vm(JobRequirements(500, 1.5, "gpu", batchable=True, batch_width=128), spec)
    assert tight == 17


def test_pack_batched_without_width_is_vram_filled():
    spec = static_flavor_spec("t2-le-45")   # 32 GB VRAM binds before 45 GB host RAM
    K = jobs_per_vm(JobRequirements(500, 3.0, "gpu", batchable=True), spec)
    assert K == int(32 * 0.8 // 3.0)      # 8 (VRAM-bound, not host-RAM's 12)


def test_pack_not_even_one_fits_fails_fast():
    with pytest.raises(RuntimeError, match="not even one fits"):
        jobs_per_vm(JobRequirements(1, 40.0, "cpu"), static_flavor_spec("c3-4"))


# --- VRAM clamp on batched GPU packing (finding #1) ---------------------------

def test_batched_gpu_is_vram_bounded_not_host_ram():
    # Example 2 re-derived: t2-le-45 has 32 GB VRAM (25.6 usable) and 45 GB host
    # RAM (36 usable). At 2 GB/member the batch is VRAM-bound at 12, NOT host-RAM
    # -bound at 18 — a naive host-RAM pack (18*2=36 GB) would OOM the 32 GB card.
    spec = static_flavor_spec("t2-le-45")
    assert spec.vram_gb == 32.0
    K = jobs_per_vm(JobRequirements(8, 2.0, "gpu", batchable=True, batch_width=128), spec)
    assert K == 12                                   # floor(32*0.8 / 2)


def test_explicit_vram_per_member_binds_the_batch():
    spec = static_flavor_spec("t2-le-45")            # 45 GB host, 32 GB VRAM
    # A member that is lean in host RAM (1 GB) but heavy in VRAM (4 GB): VRAM
    # holds 6, host RAM holds 36 -> the batch is VRAM-bound at 6.
    K = jobs_per_vm(
        JobRequirements(50, 1.0, "gpu", batchable=True, vram_gb_per_member=4.0), spec)
    assert K == 6                                    # floor(32*0.8 / 4)


def test_member_vram_above_card_fails_fast():
    spec = static_flavor_spec("t2-le-45")            # 32 GB VRAM, 25.6 usable
    with pytest.raises(RuntimeError, match="not even one member fits on the device"):
        jobs_per_vm(
            JobRequirements(1, 2.0, "gpu", batchable=True, vram_gb_per_member=40.0), spec)


def test_vram_per_member_requires_batchable():
    with pytest.raises(RuntimeError, match="only means something for batchable"):
        JobRequirements(1, 2.0, "gpu", vram_gb_per_member=4.0)


def test_batched_cpu_ignores_vram_axis():
    # CPU specs carry no VRAM, so the VRAM clamp never applies; the batch is
    # bound by host RAM as before (c3-8: 8 GB * 0.8 = 6.4 -> 6 at 1 GB/member).
    spec = static_flavor_spec("c3-8")
    assert spec.vram_gb is None
    K = jobs_per_vm(JobRequirements(50, 1.0, "cpu", batchable=True, batch_width=100), spec)
    assert K == 6


def test_plan_notes_the_conservative_vram_assumption():
    # Batched GPU with no vram_gb_per_member -> the plan says it used host RAM as
    # the VRAM footprint, so the caller knows to pass a tighter figure.
    plan = plan_fleet(JobRequirements(8, 2.0, "gpu", batchable=True, batch_width=128))
    assert any("VRAM per member not given" in n for n in plan.notes)
    # Given explicitly, no such note.
    plan2 = plan_fleet(
        JobRequirements(8, 2.0, "gpu", batchable=True, batch_width=128, vram_gb_per_member=2.0))
    assert not any("VRAM per member not given" in n for n in plan2.notes)


# --- wider GPU candidate ladder (finding #6) ----------------------------------

def test_bigger_host_ram_gpu_job_steps_up_the_family():
    # 50 GB host RAM/member no longer fits t2-le-45 (36 usable); it resolves to
    # t2-le-90 (72 usable) instead of being refused.
    spec = choose_flavor(JobRequirements(4, 50.0, "gpu"))
    assert spec.name == "t2-le-90"


def test_ram_above_largest_gpu_names_t2_le_180():
    with pytest.raises(RuntimeError, match="exceeds the largest gpu flavor: t2-le-180"):
        choose_flavor(JobRequirements(1, 200.0, "gpu"))


def test_bigger_gpu_size_propagates_to_every_region():
    # A 50 GB/member GPU job resolves end-to-end: EU regions lift to t2-le-90 and
    # BHS5 to the matching t1-le-90, not the too-small t2-le-45/t1-le-45 default.
    plan = plan_fleet(JobRequirements(4, 50.0, "gpu"))
    by_region = {ra.region: ra.flavor for ra in plan.region_allocation}
    assert by_region["DE1"] == "t2-le-90"
    assert by_region["BHS5"] == "t1-le-90"
    assert plan.flavor == "t2-le-90"


# --- region cap over RAM axis: 0, not an optimistic 1 (finding #7) ------------

def test_region_cap_is_zero_when_ram_starved():
    spec = static_flavor_spec("t2-le-45")            # 45 GB host RAM
    # RAM quota fully used -> zero VMs fit, even though cores/instances would.
    cap = _region_cap(spec, ram_used_gb=CATALOG_QUOTA_RAM_GB, ram_max_gb=CATALOG_QUOTA_RAM_GB)
    assert cap == 0
    # With headroom it is the core-bound positive cap (64 vCPU / 2 per c3-4).
    assert _region_cap(static_flavor_spec("c3-4")) == 32


def test_core_all_regions_zero_cap_fails_fast():
    req = JobRequirements(job_count=5, ram_gb_per_job=2.0, device="cpu")
    units = [RegionUnit(region="GRA11", spec=static_flavor_spec("c3-4"), cap=0),
             RegionUnit(region="DE1", spec=static_flavor_spec("c3-4"), cap=0)]
    with pytest.raises(RuntimeError, match="fits zero"):
        plan_fleet_core(req, units, device="cpu", primary=static_flavor_spec("c3-4"))


# --- budget guards the requested jobs, not the fill envelope (finding #2) ------

def test_deal_counts_sums_to_total_proportional_to_weights():
    assert _deal_counts(8, [2, 4]) == [3, 5]
    assert sum(_deal_counts(100, [10, 10, 10])) == 100
    assert _deal_counts(0, [1, 2]) == [0, 0]


def test_plan_budget_guards_jobs_not_fill():
    # 15 jobs on a fleet whose FILLED envelope (20 slots) costs more than the
    # budget, but whose 15-job cost fits: the plan is allowed, matching what
    # `sweep --budget` would guard.
    req = JobRequirements(job_count=15, ram_gb_per_job=3.0, device="cpu", minutes_per_job=60)
    primary = static_flavor_spec("c3-4")             # 3 GB -> K=1 (RAM-bound)
    units = _units([("GRA11", "c3-4", 10)])          # 10 VM, 10 slots/wave, 2 waves
    # cost_jobs = 15 * 0.0457 = 0.686; cost_filled = 10 VM * 2 waves * 0.0457 = 0.914.
    plan = plan_fleet_core(req, units, device="cpu", primary=primary, budget=0.80)
    assert plan.cost_jobs_eur == pytest.approx(15 * 0.0457)
    assert plan.cost_jobs_eur < plan.cost_filled_eur     # 15 jobs < 20-slot fill
    # A budget below the jobs cost is refused; one above it (but below fill) passes.
    with pytest.raises(RuntimeError, match="exceeds budget"):
        plan_fleet_core(req, units, device="cpu", primary=primary, budget=0.60)


def test_batched_cost_jobs_bills_vm_invocations_not_members():
    # The coordinator's live case: 128 members, K=12 on t2-le-45 across two
    # same-price regions. cost_jobs bills ceil(128/12)=11 VM-invocations
    # (4.40 EUR), NOT 128 member-instances (the old, contradictory 51.20).
    req = JobRequirements(128, 2.0, "gpu", batchable=True, batch_width=128,
                          vram_gb_per_member=2.0, minutes_per_job=30)
    primary = static_flavor_spec("t2-le-45")
    units = _units([("GRA11", "t2-le-45", 2), ("DE1", "t2-le-45", 2)])
    plan = plan_fleet_core(req, units, device="gpu", primary=primary)
    assert plan.jobs_per_vm == 12
    assert plan.cost_jobs_eur == pytest.approx(math.ceil(128 / 12) * 0.80 * 0.5)
    assert plan.cost_jobs_eur == pytest.approx(4.40)
    assert plan.cost_filled_eur == pytest.approx(4 * 3 * 0.80 * 0.5)   # 4 VM * 3 waves
    assert plan.cost_jobs_eur <= plan.cost_filled_eur


@pytest.mark.parametrize("count,width,vram", [
    (128, 128, 2.0), (50, 32, 4.0), (7, 128, 2.0), (200, 64, 1.0), (5, 2, 4.0),
    (144, 128, 2.0), (1, 128, 2.0),
])
def test_batched_cost_jobs_never_exceeds_cost_filled(count, width, vram):
    # The invariant, across the full GPU fleet (heterogeneous t2-le/t1-le K): the
    # requested-jobs cost is never above the fill-the-fleet cost.
    plan = plan_fleet(
        JobRequirements(count, 2.0, "gpu", batchable=True,
                        batch_width=width, vram_gb_per_member=vram))
    assert plan.cost_jobs_eur <= plan.cost_filled_eur


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


def test_core_two_worst_case_costs_jobs_and_filled():
    req = JobRequirements(job_count=8, ram_gb_per_job=4.0, device="gpu",
                          minutes_per_job=60)
    primary = static_flavor_spec("t2-le-45")
    # Two GPU regions at different cards/prices; 4 GB/job, unbatched -> K=1.
    units = _units([("DE1", "t2-le-45", 2), ("BHS5", "t1-le-45", 4)])
    plan = plan_fleet_core(req, units, device="gpu", primary=primary)
    assert plan.vm_count == 6
    # cost_filled: every VM busy both waves (K=1, 6 slots/wave, 8 jobs -> 2 waves).
    # DE1: 2 VM * 2 waves * 0.80 ; BHS5: 4 * 2 * 0.70.
    assert plan.cost_filled_eur == pytest.approx(2 * 2 * 0.80 + 4 * 2 * 0.70)
    # cost_jobs: the 8 requested jobs dealt across the fleet ~ vms (2:4 -> 3:5),
    # each billed one instance-hour, matching `sweep`.
    assert plan.cost_jobs_eur == pytest.approx(3 * 0.80 + 5 * 0.70)
    assert plan.cost_jobs_eur < plan.cost_filled_eur


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
    assert plan.vm_count == 288                      # 9 regions * 32 core-cap
    assert isinstance(plan, FleetPlan)


def test_plan_fleet_gpu_uses_gpu_regions_and_bhs5_v100():
    plan = plan_fleet(JobRequirements(50, 4.0, "gpu"))
    assert plan.device == "gpu"
    regions = {ra.region: ra for ra in plan.region_allocation}
    assert set(regions) == set(GPU_REGIONS)
    assert regions["DE1"].flavor == "t2-le-45" and regions["DE1"].vms == 4
    assert regions["BHS5"].flavor == "t1-le-45" and regions["BHS5"].vms == 8
    assert plan.vm_count == 24                        # 4*4 + 8 (BHS5)


def test_plan_fleet_gpu_with_only_cpu_regions_fails_fast():
    with pytest.raises(RuntimeError, match="none of the regions"):
        plan_fleet(JobRequirements(10, 2.0, "gpu"), regions="SBG5,RBX-A")


def test_plan_fleet_regions_override_is_honored():
    plan = plan_fleet(JobRequirements(20, 2.0, "cpu"), regions="GRA11,DE1")
    assert {ra.region for ra in plan.region_allocation} == {"GRA11", "DE1"}
    assert plan.vm_count == 64                       # 2 regions * 32 core-cap


def test_plan_fleet_budget_guard_raises_offline():
    with pytest.raises(RuntimeError, match="exceeds budget"):
        plan_fleet(JobRequirements(1000, 2.0, "cpu", minutes_per_job=60), budget=1.0)


def test_format_plan_returns_text():
    plan = plan_fleet(JobRequirements(100, 2.0, "cpu"))
    text = format_plan(JobRequirements(100, 2.0, "cpu"), plan)
    assert "fleet plan" in text and "flavor" in text and "worst case" in text


# --- live quota: a partial read must not describe itself as fully live -------

def test_live_region_cap_raises_when_a_USED_axis_is_missing():
    """Defaulting a missing 'used' to 0 says the region is empty and inflates the
    cap upward -- planning a fleet onto headroom that may already be occupied."""
    from types import SimpleNamespace
    import pytest
    from flux_compute.fleet import _live_region_cap, static_flavor_spec

    spec = static_flavor_spec("b3-8")
    lim = SimpleNamespace(max_total_cores=64, max_total_instances=50,
                          max_total_ram_size=496 * 1024,
                          total_instances_used=0, total_ram_used=0)   # no cores_used
    with pytest.raises(RuntimeError, match="vCPUs in use"):
        _live_region_cap(spec, lim)


def test_live_region_cap_substitutes_a_missing_MAX_and_says_which(monkeypatch):
    from types import SimpleNamespace
    from flux_compute.fleet import _live_region_cap, static_flavor_spec

    spec = static_flavor_spec("b3-8")
    lim = SimpleNamespace(total_cores_used=0, total_instances_used=0, total_ram_used=0,
                          max_total_instances=50, max_total_ram_size=496 * 1024)
    cap, substituted = _live_region_cap(spec, lim)      # no max_total_cores
    assert substituted == {"vCPU quota"}
    assert cap > 0


def test_live_region_cap_full_read_substitutes_nothing():
    from types import SimpleNamespace
    from flux_compute.fleet import _live_region_cap, static_flavor_spec

    spec = static_flavor_spec("b3-8")
    lim = SimpleNamespace(total_cores_used=8, max_total_cores=64,
                          total_instances_used=1, max_total_instances=50,
                          total_ram_used=32 * 1024, max_total_ram_size=496 * 1024)
    cap, substituted = _live_region_cap(spec, lim)
    assert substituted == set()
    # b3-8 is 2 vCPU / 8 GB, so vCPU binds: (64-8)/2 = 28, under the RAM bound
    # ((496-32)/8 = 58) and the instance bound (50-1 = 49).
    assert cap == 28


def test_offline_note_is_built_from_the_catalog_constants():
    """The note used to restate 64/50/496 as prose 450 lines from the constants."""
    from flux_compute import fleet
    from flux_compute.fleet import (CATALOG_QUOTA_CORES, CATALOG_QUOTA_INSTANCES,
                                    CATALOG_QUOTA_RAM_GB, JobRequirements, plan_fleet)
    plan = plan_fleet(JobRequirements(job_count=4, ram_gb_per_job=2, device="cpu"))
    note = " ".join(plan.notes)
    assert f"{CATALOG_QUOTA_CORES} vCPU" in note
    assert f"{CATALOG_QUOTA_INSTANCES} instances" in note
    assert f"{CATALOG_QUOTA_RAM_GB} GB" in note


def test_gpu_region_advice_is_built_from_GPU_REGIONS():
    import pytest
    from flux_compute.fleet import GPU_REGIONS, JobRequirements, plan_fleet
    with pytest.raises(RuntimeError) as exc:
        plan_fleet(JobRequirements(job_count=1, ram_gb_per_job=2, device="gpu"),
                   regions="SBG5")
    assert ", ".join(GPU_REGIONS) in str(exc.value) or list(GPU_REGIONS)[0] in str(exc.value)
