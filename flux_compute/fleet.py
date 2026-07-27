"""Resource-aware fleet planner: turn a generic job description into a launch plan.

A consumer (a simulation repo's tooling, a UI) describes what a batch of jobs
*needs* — RAM per job, whether it wants a CPU or a GPU, whether the jobs batch
onto one device, how long each takes, and how many there are — and gets back a
`FleetPlan`: which flavor, which device, how wide a fleet across which regions,
how many jobs pack onto a VM, how many waves it takes, the worst-case spend, and
how much spare capacity is left to fill. It never prints; it returns structured
data the caller renders or acts on.

**This package knows nothing about what the jobs compute.** `JobRequirements`
carries only generic resource fields — no grid sizes, no solver settings, no
consumer concepts. The planner routes on RAM, device, batchability and count
alone, so any consumer that can state those four things can size a fleet here.

The split mirrors the rest of the package: `plan_fleet_core` is pure (given
per-region caps and flavor specs it returns a plan, unit-testable with no
credentials), `plan_fleet` is the offline facade over the catalog tables, and
`plan_fleet_live` gathers real per-region quota and availability from OpenStack.
Offline plans use catalog values; a live launch re-verifies quota and flavor
availability per region and clamps to real headroom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .flavors import (
    FlavorSpec,
    DEFAULT_SIM_FLAVOR,
    live_flavor_spec,
    static_flavor_spec,
)
from .sweep import allocate_concurrency, budget_guard_shards, clamp_concurrency, parse_regions

# Leave a fraction of a VM's host RAM for the OS, GPU driver and framework
# runtime rather than packing jobs into the last byte. The RAM-fit clamp is
# `floor(ram_gb * RAM_HEADROOM / ram_gb_per_job)`.
RAM_HEADROOM = 0.8

# Cloud topology, from the flux-compute README ("Multi-region sweeps", measured
# 2026-07-21). GPU cards live only in the five GPU regions; the other four are
# CPU-only. BHS5 carries the cheaper plain V100; the other GPU regions carry the
# V100S. These are the *catalog* defaults the offline plan reasons from — a live
# plan reads each region's real flavor list and quota instead.
GPU_REGIONS = ("GRA11", "DE1", "UK1", "WAW1", "BHS5")
CPU_ONLY_REGIONS = ("SBG5", "RBX-A", "EU-WEST-PAR", "EU-SOUTH-MIL")
ALL_REGIONS = GPU_REGIONS + CPU_ONLY_REGIONS
# Each GPU region's card FAMILY. The *size* (45/90/180) is taken from the chosen
# primary, so a bigger-RAM member lifts every region to the matching size
# together: the EU regions carry the V100S (t2-le), BHS5 the cheaper V100 (t1-le).
_REGION_GPU_FAMILY = {
    "GRA11": "t2-le", "DE1": "t2-le", "UK1": "t2-le",
    "WAW1": "t2-le", "BHS5": "t1-le",
}

# Per-region compute quota, measured live 2026-07-27 in DE1/UK1/WAW1/SBG5 —
# the CS16091787 increase (granted 2026-07-19) is in effect (README "Cost
# guardrails"). The quota is per region — spreading across regions is the only
# way to widen a fleet past one region's headroom. A live plan reads the real
# numbers from the API; these bound the offline preview.
CATALOG_QUOTA_CORES = 64
CATALOG_QUOTA_INSTANCES = 50
CATALOG_QUOTA_RAM_GB = 496

# The catalog flavor ladders the offline flavor choice picks from. GPU work uses
# the EU-wide single-card V100S default (a live plan may pick a region's cheaper
# card, e.g. BHS5's V100); CPU work picks the cheapest single VM whose host RAM
# fits one job, across both the compute-optimized (c3) and general-purpose (b3)
# ladders — for a RAM-heavy job b3's 4 GB/vCPU beats c3's 2 GB/vCPU on price.
# GPU work ranks across the EU-wide V100S family: choose_flavor picks the cheapest
# that fits one member's HOST RAM, so a lean member stays on t2-le-45 (the default,
# first here) while a host-RAM-heavy member steps up to t2-le-90/180 instead of
# being refused. t1-le (V100, BHS5-only, cheaper but half the VRAM) is deliberately
# NOT a primary candidate — on price it would undercut the EU-wide default and
# misreport a BHS5-only card as the fleet's primary flavor; BHS5 still runs t1-le
# via its per-region flavor, whose smaller VRAM the per-region K already accounts for.
_GPU_CANDIDATE_FLAVORS = (DEFAULT_SIM_FLAVOR, "t2-le-90", "t2-le-180")
_CPU_CANDIDATE_FLAVORS = (
    "c3-4", "c3-8", "c3-16", "c3-32", "c3-64", "c3-128", "c3-256",
    "b3-8", "b3-16", "b3-32", "b3-64", "b3-128", "b3-256", "b3-512",
)

_DEVICES = ("cpu", "gpu", "either")


@dataclass(frozen=True)
class JobRequirements:
    """A consumer's generic description of a batch of independent jobs.

    Every field is resource-generic — nothing here names what the jobs compute.
    A later consumer-side estimator produces one of these and the planner routes
    on it alone.

    Fields:
      job_count        — how many independent jobs to run (>= 1).
      ram_gb_per_job   — peak host RAM one job (or one batched member) needs, GB.
      device           — "cpu", "gpu", or "either". "either" lets the planner
                         pick: batched work amortizes on a GPU, unbatched work
                         fans out cheaply on CPU.
      minutes_per_job  — wall-clock estimate for one job (or one batched
                         invocation); sizes waves and worst-case spend.
      batchable        — True if many jobs collapse into one device invocation
                         (e.g. one GPU absorbing many members at once). When
                         True the planner packs a VM to a batch, not to its vCPUs.
      batch_width      — preferred members per batched invocation. Only meaningful
                         when batchable; the actual batch is clamped down to what
                         a VM's RAM and (on a GPU) VRAM hold. Omitted -> the batch
                         is resource-bound.
      vram_gb_per_member — GPU device memory one batched member needs, GB. Only
                         meaningful when batchable (a batch is what co-resides on
                         the accelerator). Omitted -> the planner conservatively
                         assumes a member's VRAM footprint equals its host
                         `ram_gb_per_job`, and says so in a plan note.
    """

    job_count: int
    ram_gb_per_job: float
    device: str = "either"
    minutes_per_job: float = 30.0
    batchable: bool = False
    batch_width: int | None = None
    vram_gb_per_member: float | None = None

    def __post_init__(self):
        if self.device not in _DEVICES:
            raise RuntimeError(
                f"device must be one of {_DEVICES}, got {self.device!r}")
        if self.job_count < 1:
            raise RuntimeError(f"job_count must be >= 1, got {self.job_count}")
        if self.ram_gb_per_job <= 0:
            raise RuntimeError(
                f"ram_gb_per_job must be positive, got {self.ram_gb_per_job}")
        if self.minutes_per_job <= 0:
            raise RuntimeError(
                f"minutes_per_job must be positive, got {self.minutes_per_job}")
        if self.batch_width is not None:
            if not self.batchable:
                raise RuntimeError(
                    "batch_width was given but batchable is False; a batch width "
                    "only means something for batchable work")
            if self.batch_width < 1:
                raise RuntimeError(
                    f"batch_width must be >= 1, got {self.batch_width}")
        if self.vram_gb_per_member is not None:
            if not self.batchable:
                raise RuntimeError(
                    "vram_gb_per_member was given but batchable is False; a VRAM "
                    "footprint per member only means something for batchable work")
            if self.vram_gb_per_member <= 0:
                raise RuntimeError(
                    f"vram_gb_per_member must be positive, got {self.vram_gb_per_member}")


@dataclass(frozen=True)
class RegionUnit:
    """One region's contribution to the fleet: which flavor, and how many of it
    the region's quota can run at once. The pure planner is a function of a list
    of these; the offline and live paths differ only in how they build them."""

    region: str
    spec: FlavorSpec
    cap: int          # max concurrent VMs of `spec` this region's quota allows


@dataclass(frozen=True)
class RegionAllocation:
    """What the plan assigns one region: the VMs it runs, and their shape."""

    region: str
    flavor: str
    vms: int              # concurrent VMs granted (<= cap, <= global ceiling)
    cap: int              # what the region's quota alone could run
    jobs_per_vm: int      # packing K on this region's flavor
    price_eur_hr: float | None


@dataclass(frozen=True)
class FleetPlan:
    """A structured launch plan. No side effects; the caller renders or acts."""

    flavor: str                       # the reported (primary) flavor
    device: str                       # resolved: "cpu" or "gpu"
    gpu_model: str | None
    reason: str                       # why this flavor/device
    jobs_per_vm: int                  # packing K on the primary flavor
    vm_count: int                     # total concurrent VMs (fleet width)
    region_allocation: tuple          # (RegionAllocation, ...)
    waves: int                        # ceil(job_count / slots_per_wave)
    slots_per_wave: int               # member-slots the whole fleet runs per wave
    capacity: int                     # waves * slots_per_wave
    spare_slots: int                  # capacity - job_count (free slots to fill)
    cost_jobs_eur: float | None       # worst case for the requested job_count
                                      #   (dealt across the fleet like `sweep`;
                                      #   this is what --budget guards)
    cost_filled_eur: float | None     # worst case if every slot is filled — the
                                      #   packed capacity envelope (reported, not gated)
    notes: tuple = field(default_factory=tuple)


def _gpu_size(flavor_name: str) -> str:
    """The size suffix of a GPU flavor name: 't2-le-90' -> '90'."""
    return flavor_name.rsplit("-", 1)[1]


def _region_gpu_flavor(region: str, size: str) -> str | None:
    """The GPU flavor a region runs at a given family size, or None when the
    region carries no credit-eligible GPU family (a CPU-only region)."""
    fam = _REGION_GPU_FAMILY.get(region)
    return None if fam is None else f"{fam}-{size}"


def _resolve_device(req: JobRequirements) -> str:
    """Resolve "either" to a concrete device.

    Batched work amortizes a device invocation across many members, which is what
    a GPU is for here; unbatched work is a wide one-per-VM CPU fan-out. This is a
    generic default, not a measured crossover — a consumer that has benchmarked
    the crossover passes "cpu"/"gpu" explicitly to override it.
    """
    if req.device in ("cpu", "gpu"):
        return req.device
    return "gpu" if req.batchable else "cpu"


def _candidate_specs(device: str):
    """The catalog flavor specs the offline flavor choice ranks, for a device."""
    names = _GPU_CANDIDATE_FLAVORS if device == "gpu" else _CPU_CANDIDATE_FLAVORS
    return [static_flavor_spec(n) for n in names]


def choose_flavor(req: JobRequirements, *, ram_headroom: float = RAM_HEADROOM,
                  candidates=None) -> FlavorSpec:
    """Pick the cheapest usable flavor whose host RAM fits one job of `req`.

    `candidates` overrides the catalog ladder (the live path passes a region's
    real specs). Raises fail-fast when no candidate can hold even one job — the
    RAM-above-the-largest-flavor case, with the largest available size named.
    """
    device = _resolve_device(req)
    specs = candidates if candidates is not None else _candidate_specs(device)
    specs = [s for s in specs if s.usable_for_sim and s.kind == device]
    if not specs:
        raise RuntimeError(
            f"no usable {device} flavor is available to run these jobs")
    feasible = [s for s in specs if s.ram_gb * ram_headroom >= req.ram_gb_per_job]
    if not feasible:
        biggest = max(specs, key=lambda s: s.ram_gb)
        raise RuntimeError(
            f"per-job RAM {req.ram_gb_per_job:g} GB exceeds the largest {device} "
            f"flavor: {biggest.name} offers {biggest.ram_gb * ram_headroom:.0f} GB "
            f"usable of {biggest.ram_gb:g} GB. Reduce per-job RAM, split the job, "
            f"or run fewer batched members at once."
        )
    feasible.sort(key=lambda s: (
        s.price_eur_hr if s.price_eur_hr is not None else float("inf"),
        s.ram_gb, s.name))
    return feasible[0]


def jobs_per_vm(req: JobRequirements, spec: FlavorSpec, *,
                ram_headroom: float = RAM_HEADROOM) -> int:
    """Packing K: how many jobs/members one VM of `spec` runs at once.

    Clamped by every binding axis. Host RAM always binds
    (`K * ram_gb_per_job <= ram_gb * headroom`). For a *batched* GPU invocation the
    members co-reside on the accelerator, so GPU VRAM binds too
    (`K * vram_per_member <= vram_gb * headroom`), with `vram_per_member` the
    requirement's `vram_gb_per_member` or — conservatively, when unset — its
    `ram_gb_per_job`. Contention then depends on how the work uses the device:
      - batchable: the caller's preferred batch width (or, if unset, the
        resource fit) — many members share one device invocation.
      - unbatched on a GPU: one job holds the single accelerator, so K's
        contention bound is 1 (RAM cannot pack more usefully onto one GPU).
      - unbatched on a CPU: one serial job per vCPU (the fan-out packing).
    Raises when not even one job fits in host RAM, or one member in VRAM.
    """
    usable = spec.ram_gb * ram_headroom
    ram_fit = int(usable // req.ram_gb_per_job)
    if ram_fit < 1:
        raise RuntimeError(
            f"one job needs {req.ram_gb_per_job:g} GB but {spec.name} offers only "
            f"{usable:.0f} GB usable of {spec.ram_gb:g} GB — not even one fits.")
    resource_fit = ram_fit
    if req.batchable and spec.kind == "gpu" and spec.vram_gb is not None:
        vram_per_member = (req.vram_gb_per_member
                           if req.vram_gb_per_member is not None else req.ram_gb_per_job)
        vram_usable = spec.vram_gb * ram_headroom
        vram_fit = int(vram_usable // vram_per_member)
        if vram_fit < 1:
            raise RuntimeError(
                f"one batched member needs {vram_per_member:g} GB of VRAM but "
                f"{spec.name} has only {vram_usable:.0f} GB usable of "
                f"{spec.vram_gb:g} GB VRAM — not even one member fits on the device.")
        resource_fit = min(ram_fit, vram_fit)
    if req.batchable:
        contention = req.batch_width if req.batch_width is not None else resource_fit
    elif spec.kind == "gpu":
        contention = 1
    else:
        contention = spec.vcpus
    return max(1, min(contention, resource_fit))


def _region_cap(spec: FlavorSpec, *, cores_used=0, cores_max=CATALOG_QUOTA_CORES,
                instances_used=0, instances_max=CATALOG_QUOTA_INSTANCES,
                ram_used_gb=0.0, ram_max_gb=CATALOG_QUOTA_RAM_GB) -> int:
    """Max concurrent VMs of `spec` a region's quota allows, over all three axes.

    Cores and instances go through the shared `clamp_concurrency` (which raises
    when not even one instance fits that quota); RAM is the third axis
    (VMs * ram_gb <= ram quota). Returns the max concurrent VMs — **0** when RAM
    headroom cannot fit even one VM, matching the strictness of the core/instance
    axis rather than optimistically claiming one. The caller (`plan_fleet_core`)
    fails fast when every region resolves to 0.
    """
    # A large ceiling so clamp_concurrency returns the raw quota headroom; the
    # global --max-parallel split happens later in allocate_concurrency.
    core_inst = clamp_concurrency(
        10 ** 9, spec.vcpus, cores_used, cores_max, instances_used, instances_max)
    if (ram_max_gb or 0) < 0:               # unlimited RAM quota (-1)
        return core_inst
    ram_free = (ram_max_gb or 0) - (ram_used_gb or 0)
    ram_fit = max(0, int(ram_free // spec.ram_gb))
    return min(core_inst, ram_fit)


def _flavor_reason(req, device, primary, K) -> str:
    """A one-line 'why this flavor/device' for the plan."""
    dev_note = {
        "cpu": "CPU fan-out",
        "gpu": "GPU device",
    }[device]
    if req.device == "either":
        dev_note += (" (device 'either' -> "
                     + ("GPU: batched work amortizes the device"
                        if device == "gpu" else
                        "CPU: unbatched work fans out cheapest")
                     + ")")
    if req.batchable:
        pack = (f"batches {K} members/VM"
                + (f" (asked {req.batch_width}, RAM/VRAM-clamped)"
                   if req.batch_width and K < req.batch_width else ""))
    else:
        pack = (f"packs {K} job(s)/VM"
                + (f" across {primary.vcpus} vCPU" if K > 1 else " (RAM-bound)"))
    return (f"{dev_note}: {primary.name} "
            f"({primary.vcpus} vCPU, {primary.ram_gb:g} GB) — cheapest usable "
            f"flavor fitting {req.ram_gb_per_job:g} GB/job; {pack}")


def _deal_counts(total: int, weights) -> list:
    """Deal `total` items across positive weights, summing to exactly `total`
    (largest-remainder). Mirrors how `sweep` deals jobs in proportion to each
    region's concurrency, so the per-region job counts the budget sees match what
    a sweep would actually bill."""
    weights = list(weights)
    s = sum(weights)
    if s <= 0:
        raise RuntimeError("cannot deal jobs: no region has any allocation")
    exact = [total * w / s for w in weights]
    base = [int(math.floor(x)) for x in exact]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def plan_fleet_core(req: JobRequirements, region_units, *, device: str,
                    primary: FlavorSpec, budget=None, max_parallel=None,
                    ram_headroom: float = RAM_HEADROOM, notes=()) -> FleetPlan:
    """Pure planner: region caps + flavor specs -> FleetPlan. No network.

    `region_units` are the regions that can host this work, each with its resolved
    flavor and quota cap. `primary` is the flavor to report (for a mixed-card GPU
    fleet it is the EU-wide default; per-region flavors ride in the allocation).
    Fails fast when no region is reachable, when quota affords no VM, or when the
    worst-case spend exceeds `budget`.
    """
    units = list(region_units)
    if not units:
        raise RuntimeError(
            f"no region can host this {device} work. "
            + ("GPU work needs a GPU region (GRA11, DE1, UK1, WAW1, BHS5)."
               if device == "gpu" else
               "no region with quota headroom was supplied."))

    caps = [u.cap for u in units]
    if sum(caps) < 1:
        raise RuntimeError(
            f"no region has quota headroom for this {device} work: every region's "
            f"core/instance/RAM quota fits zero {primary.name} VMs. Free running "
            f"instances, request a quota increase, or switch region/flavor.")
    ceiling = max_parallel if max_parallel is not None else sum(caps)
    if ceiling < 1:
        raise RuntimeError(f"max_parallel must be >= 1, got {ceiling}")
    alloc = allocate_concurrency(caps, ceiling)
    live = [(u, a) for u, a in zip(units, alloc) if a > 0]
    if not live:
        raise RuntimeError(
            "no region has any quota headroom for this fleet "
            "(every region's cap is zero).")

    K_primary = jobs_per_vm(req, primary, ram_headroom=ram_headroom)
    region_alloc = []
    slots_per_wave = 0
    for u, a in live:
        K = jobs_per_vm(req, u.spec, ram_headroom=ram_headroom)
        slots_per_wave += a * K
        region_alloc.append(RegionAllocation(
            region=u.region, flavor=u.spec.name, vms=a, cap=u.cap,
            jobs_per_vm=K, price_eur_hr=u.spec.price_eur_hr))

    vm_count = sum(a for _, a in live)
    waves = math.ceil(req.job_count / slots_per_wave)
    capacity = waves * slots_per_wave
    spare = capacity - req.job_count

    # Two worst-case costs, both at each region's own price via the shared sweep
    # budget guard. `cost_jobs` is what --budget guards; it bills the REQUESTED work
    # the way it actually runs, so a `plan --budget` and the equivalent launch agree:
    #   - unbatched: one instance per job -> `job_count` VM-instances (matches how
    #     `sweep` bills, one VM per job).
    #   - batched: the members run K-at-a-time, so a wave of K members costs ONE
    #     VM-period; `job_count` members occupy `ceil(job_count / K)` VM-invocations
    #     (billing the raw member count would over-charge ~K x — the batch runs its
    #     members concurrently, not as separate VM-jobs).
    # Either way the count is dealt across regions ~ vms, exactly as `cost_filled`
    # weights them, which keeps `cost_jobs <= cost_filled` region-by-region (each
    # region's share never exceeds its `vms * waves`).
    units_needed = math.ceil(req.job_count / K_primary) if req.batchable else req.job_count
    jobs_share = _deal_counts(units_needed, [ra.vms for ra in region_alloc])
    jobs_entries = [(ra.flavor, ra.price_eur_hr, n)
                    for ra, n in zip(region_alloc, jobs_share)]
    cost_jobs = budget_guard_shards(jobs_entries, req.minutes_per_job, budget)
    # cost_filled: the packed capacity envelope (every VM busy every wave),
    # reported for the fill-the-fleet doctrine, never gated.
    filled_entries = [(ra.flavor, ra.price_eur_hr, ra.vms * waves) for ra in region_alloc]
    cost_filled = budget_guard_shards(filled_entries, req.minutes_per_job, None)

    all_notes = list(notes)
    if device == "gpu" and req.batchable and req.vram_gb_per_member is None:
        all_notes.append(
            f"VRAM per member not given: the batch is clamped assuming each "
            f"member's VRAM footprint equals its {req.ram_gb_per_job:g} GB host "
            f"footprint. Pass vram_gb_per_member for a tighter or looser device bound.")

    return FleetPlan(
        flavor=primary.name,
        device=device,
        gpu_model=primary.gpu_model,
        reason=_flavor_reason(req, device, primary, K_primary),
        jobs_per_vm=K_primary,
        vm_count=vm_count,
        region_allocation=tuple(region_alloc),
        waves=waves,
        slots_per_wave=slots_per_wave,
        capacity=capacity,
        spare_slots=spare,
        cost_jobs_eur=cost_jobs,
        cost_filled_eur=cost_filled,
        notes=tuple(all_notes),
    )


def _target_regions(device: str, regions) -> list:
    """The regions to plan across: the caller's list, or the device default."""
    if regions is None:
        return list(GPU_REGIONS if device == "gpu" else ALL_REGIONS)
    if isinstance(regions, str):
        return parse_regions(regions)
    out = [str(r).strip() for r in regions if str(r).strip()]
    if not out:
        raise RuntimeError("regions was given but named no region")
    return out


def plan_fleet(requirements: JobRequirements, budget=None, regions=None,
               max_parallel=None, *, ram_headroom: float = RAM_HEADROOM) -> FleetPlan:
    """Offline fleet plan from the catalog tables. No credentials, no network.

    Chooses the flavor for the requirement, spreads the fleet across the eligible
    regions at their catalog quota, and returns a `FleetPlan`. GPU work is planned
    across the GPU regions (each at its catalog card — BHS5's cheaper V100, the
    V100S elsewhere); CPU work spreads across all regions on one chosen CPU flavor.
    A live launch re-verifies quota and availability per region and may pick a
    cheaper regional card.
    """
    device = _resolve_device(requirements)
    primary = choose_flavor(requirements, ram_headroom=ram_headroom)
    targets = _target_regions(device, regions)
    # GPU regions run their own card family at the size the primary chose, so a
    # bigger-RAM requirement (t2-le-90/180) lifts every region together.
    size = _gpu_size(primary.name) if device == "gpu" else None

    units, dropped = [], []
    for r in targets:
        if device == "gpu":
            fname = _region_gpu_flavor(r, size)
            if fname is None:            # a CPU-only region asked for GPU work
                dropped.append(r)
                continue
            spec = static_flavor_spec(fname)
        else:
            spec = primary
        units.append(RegionUnit(region=r, spec=spec, cap=_region_cap(spec)))

    if device == "gpu" and not units:
        raise RuntimeError(
            f"GPU work was requested but none of the regions {targets} carry a "
            f"credit-eligible fp64-healthy GPU. GPU regions: {list(GPU_REGIONS)}.")

    notes = ["offline plan: catalog quota (64 vCPU / 50 instances / 496 GB per "
             "region, measured 2026-07-27) and catalog flavor shapes; a live "
             "launch re-verifies quota and availability per region."]
    if dropped:
        notes.append(f"CPU-only region(s) skipped for GPU work: {', '.join(dropped)}.")

    return plan_fleet_core(
        requirements, units, device=device, primary=primary, budget=budget,
        max_parallel=max_parallel, ram_headroom=ram_headroom, notes=notes)


def plan_fleet_live(requirements: JobRequirements, *, cloud=None, budget=None,
                    regions=None, max_parallel=None,
                    ram_headroom: float = RAM_HEADROOM) -> FleetPlan:
    """Live fleet plan: gather real per-region quota and flavor availability.

    The live counterpart to `plan_fleet`. For each target region it opens a
    connection, resolves the flavor actually available there (the requirement's
    device picks the family; `resolve_spec`/`recommended_for_sim` picks the real
    card), reads the region's live quota, and builds a `RegionUnit`. Regions that
    cannot host the work are reported together, mirroring the sweep's
    `_prepare_shards`. Requires credentials.
    """
    from .auth import connect
    from .launch import resolve_spec
    from .sweep import _flavor_vcpus

    device = _resolve_device(requirements)
    primary = choose_flavor(requirements, ram_headroom=ram_headroom)
    # CPU: pin the chosen CPU flavor everywhere. GPU: pin each region's card family
    # at the size the primary chose, so a bigger-RAM requirement resolves live the
    # same way it does offline instead of a region defaulting to too small a card.
    size = _gpu_size(primary.name) if device == "gpu" else None
    targets = _target_regions(device, regions)

    units, failures = [], []
    for region in targets:
        try:
            pin = primary.name if device == "cpu" else _region_gpu_flavor(region, size)
            conn = connect(cloud=cloud, region=region)
            reg = region
            spec_launch = resolve_spec(conn, reg, flavor=pin)
            flavor_obj = conn.compute.find_flavor(spec_launch.flavor)
            spec = live_flavor_spec(flavor_obj)
            lim = conn.get_compute_limits()
            gq = lambda k: getattr(lim, k, None)
            cap = _region_cap(
                spec,
                cores_used=gq("total_cores_used") or 0,
                cores_max=gq("max_total_cores") if gq("max_total_cores") is not None else CATALOG_QUOTA_CORES,
                instances_used=gq("total_instances_used") or 0,
                instances_max=gq("max_total_instances") if gq("max_total_instances") is not None else CATALOG_QUOTA_INSTANCES,
                ram_used_gb=(gq("total_ram_used") or 0) / 1024.0,
                ram_max_gb=(gq("max_total_ram_size") / 1024.0) if gq("max_total_ram_size") is not None else CATALOG_QUOTA_RAM_GB,
            )
            units.append(RegionUnit(region=reg, spec=spec, cap=cap))
        except Exception as exc:       # noqa: BLE001 — collected and re-raised together
            if "refused by the local clouds.yaml" in str(exc):
                raise
            failures.append(f"  {region}: {type(exc).__name__}: {str(exc)[:160]}")
    if failures and not units:
        raise RuntimeError(
            "no requested region can host this fleet:\n" + "\n".join(failures))

    notes = ["live plan: per-region quota and flavor availability read from the API."]
    if failures:
        notes.append("regions that could not host the work were skipped:\n"
                     + "\n".join(failures))
    return plan_fleet_core(
        requirements, units, device=device, primary=primary, budget=budget,
        max_parallel=max_parallel, ram_headroom=ram_headroom, notes=notes)


def format_plan(req: JobRequirements, plan: FleetPlan) -> str:
    """Render a FleetPlan as human-readable text. Pure: returns a string, never
    prints — the CLI prints it, and a consumer can reuse it in its own UI."""
    cost_jobs = (f"~EUR {plan.cost_jobs_eur:.2f}"
                 if plan.cost_jobs_eur is not None else "price n/a")
    cost_filled = (f"~EUR {plan.cost_filled_eur:.2f}"
                   if plan.cost_filled_eur is not None else "price n/a")
    lines = [
        "flux-compute fleet plan (structured; nothing launched):",
        f"  jobs        : {req.job_count} x {req.ram_gb_per_job:g} GB/job, "
        f"~{req.minutes_per_job:g} min each"
        + (f", batchable (width {req.batch_width})" if req.batchable and req.batch_width
           else ", batchable" if req.batchable else ""),
        f"  device      : {plan.device}"
        + (f"  [{plan.gpu_model}]" if plan.gpu_model else ""),
        f"  flavor      : {plan.flavor}  ({plan.reason})",
        f"  packing K   : {plan.jobs_per_vm} job(s)/VM",
        f"  fleet       : {plan.vm_count} concurrent VM(s) across "
        f"{len(plan.region_allocation)} region(s)",
    ]
    for ra in plan.region_allocation:
        price = (f"EUR {ra.price_eur_hr:.2f}/hr"
                 if ra.price_eur_hr is not None else "price n/a")
        lines.append(
            f"      {ra.region:<14} {ra.flavor:<12} {ra.vms} VM (cap {ra.cap}), "
            f"K={ra.jobs_per_vm}, {price}")
    lines += [
        f"  waves       : {plan.waves}  ({plan.slots_per_wave} job-slots/wave, "
        f"capacity {plan.capacity})",
        f"  spare slots : {plan.spare_slots}  "
        f"(round the job count up to {plan.capacity} to fill the fleet at no extra wall-clock)",
        f"  worst case  : {cost_jobs}  (the {req.job_count} requested job(s); what --budget guards)",
        f"  if filled   : {cost_filled}  (every slot full: {plan.capacity} member-slots)",
    ]
    for n in plan.notes:
        lines.append(f"  note        : {n}")
    return "\n".join(lines)
