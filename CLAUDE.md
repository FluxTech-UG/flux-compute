# flux-compute

The shared cloud-compute package for the FluxTech family: provision OVH Public
Cloud GPU instances and run the simulation repos on them. Consumers (`1DSim3`,
`LumpedSim2`, future sims) import it; it imports nothing back from them.

The family conventions (one-way dependency, git rules, the no-em-dashes /
fail-fast / plain-declarative house values) live in the parent `CLAUDE.md` and
are not restated here. This file carries only what is specific to this package.

## What this is

A control-plane package that runs on the laptop / CI side and drives OVH's
**OpenStack API** (via `openstacksdk`) to provision compute, run a job, and
fetch artifacts. It does not run inside the sims; the sims call into it to launch
cloud work.

## Critical rules

### Credentials never enter git.
OpenStack credentials (`clouds.yaml`, `openrc.sh`, application-credential
secrets, `.env`) are gitignored and must stay out of every commit. The package
reads them at runtime from clouds.yaml or the OS_* env; it never embeds or logs a
secret. Application credentials are preferred over the account password: scoped
and revocable.

### The flavor policy is enforced, not advisory.
Two independent gates govern every flavor (`flux_compute/flavors.py`):
credit-eligibility (the Startup Program covers only V100, V100S, RTX5000) and
fp64 health (the sims force x64; RTX5000 is Turing, fp64 ~1/32 fp32). A flavor
must pass both to run a sim. RTX5000 is credit-eligible but fp64-crippled, so it
is refused by default; do not add a path that launches an x64 sim on it silently.
The default sim flavor is `t2-le-45` (V100S 32GB, available across EU regions);
plain V100 (`t1-le`) is BHS5-only, so `recommended_for_sim` picks the cheapest
fp64-healthy GPU actually present in the region. When in doubt, validate a card
with 1DSim3's `scripts/gpu_check.py` before committing to it.

### Quota is PER REGION, and regions are the fleet-width lever.
Each OVH region carries its own compute quota — **64 vCPUs / 50 instances /
496 GiB** (the CS16091787 increase, granted 2026-07-19, in effect as measured
live 2026-07-27 in DE1/UK1/WAW1/SBG5). A V100S (`t2-le-45`) is 15 vCPU, so one
region fits **4 concurrent V100S**; BHS5's plain V100 (`t1-le-45`) is 8 vCPU and
fits 8. Across the five GPU regions (GRA11, DE1, UK1, WAW1, BHS5) that is **24
concurrent GPU instances**, which is what `sweep --regions` exists to reach. The
other four regions (SBG5, RBX-A, EU-WEST-PAR, EU-SOUTH-MIL) are CPU-only but
carry the same 64 vCPU each — ~576 vCPUs project-wide for CPU fan-out.

The code never trusts these numbers: preflight and sweep read live quota from
the API and clamp to real headroom, per region. Treat this section as
orientation, and let the clamp be the authority.

### Fail fast on missing credentials or no healthy GPU.
No silent defaults. Missing credentials raise with the remedy; a region exposing
no credit-eligible, fp64-healthy GPU raises rather than falling back to RTX5000 or
a blocked card. Switch region (GRA11, DE1, BHS5) instead.

### Cost guardrails are enforced by mechanism, not trust.
Every provisioned instance must have a definite teardown path; an idle GPU
quietly burns startup credits. The enforcement stack (`flux_compute/provision.py`,
`flux_compute/reap.py`): `run`/`sweep`/`bake` tear down on completion and on
error with a retried, verified server delete, and a delete that cannot be
verified prints a STRANDED INSTANCE banner with the cleanup commands and exits
nonzero. Every created server is stamped with TTL metadata at create time
(`flux_expires_at` = wall cap + a generous margin; `flux_keep=true` on `--keep`
runs), `flux-compute reap` auto-deletes only stamped instances past their
expiry (keep-flagged and unstamped name-prefix matches are report-only, taken
only via `--all` + confirmation, and unidentifiable servers are never touched),
and every command that connects surfaces strays with accrued cost before doing
its own work. `sweep --budget` is the hard spend cap and refuses unpriced
flavors. Do not add a provisioning path that bypasses the TTL stamp or the
verified teardown.

## Layout

- `flux_compute/flavors.py`: the credit + fp64 flavor policy, plus the RAM model
  (pure logic, tested). Each flavor's (vCPU, RAM, price) shape resolves two ways,
  mirroring how vCPUs are read: `static_flavor_spec(name)` derives it offline from
  the catalog (b3 = 4 GB/vCPU, c3 = 2 GB/vCPU, GPU RAM tabulated), and
  `live_flavor_spec(obj)` reads `.vcpus`/`.ram` off a live OpenStack flavor object.
  An unknown flavor is a fail-fast, never a guessed shape.
- `flux_compute/fleet.py`: the resource-aware fleet planner. A consumer states a
  generic `JobRequirements` (per-job RAM, device cpu/gpu/either, batchable +
  optional batch width, minutes/job, job count) and `plan_fleet(...)` returns a
  structured `FleetPlan` (flavor + why, device, per-region VM allocation, packing
  K, wave count, worst-case EUR, spare slots). Split like the rest of the package:
  `plan_fleet_core` is pure (region caps + specs -> plan, unit-tested), `plan_fleet`
  is the offline facade over the catalog tables, `plan_fleet_live` gathers real
  per-region quota/availability. **It contains zero simulation concepts** — it
  routes on generic resource fields only, per the family package invariant.
- `flux_compute/auth.py`: `connect()` to the OVH project from clouds.yaml / OS_* env.
  A clouds.yaml pinned to a single `region_name:` refuses every other region
  *locally*, before any request — `connect` detects that and raises the
  `regions:`-list fix, because the pin silently caps fleet width.
- `flux_compute/sweep.py`: the fan-out, including per-region sharding
  (`parse_regions` / `allocate_concurrency` / `shard_jobs` / `Shard`, all pure
  and tested). `--max-parallel` is the GLOBAL live-instance ceiling; each region
  is additionally clamped to its own headroom. The region pre-flight is
  **graceful-degrade by default**: `_prepare_shards` returns `(shards, drops)`,
  and a region that cannot fit >=1 instance is dropped with a warning (occupants
  + headroom, via `regions.occupancy_line`) and the sweep runs on the rest;
  `--strict-regions` restores refuse-on-any-unfit, and NONE-fit always refuses.
- `flux_compute/regions.py`: `flux-compute regions`, the live read-only
  per-region occupancy view (quota, running flux-compute instances via
  `reap.find_candidates`, and a flavor `fits` count over the remaining headroom).
  `build_region_status` is pure; `gather_region_status`/`_read_quota` do the live
  reads (reusing `plan_fleet_live`'s limits plumbing). `--json` (`regions_json`)
  is the shape the heat-mod-frontend region button and autonomous launchers read;
  the sweep pre-flight reuses `occupancy_line` for its drop warnings.
- `flux_compute/doctor.py`: `flux-compute doctor`, the API health check.
- `flux_compute/cli.py`: argparse entry point (`doctor`, `preflight`, `run`,
  `plan`, `sweep`, `regions`, `reap`, `bake`, `push`). `plan` prints a fleet plan for a
  generic requirement (offline from catalog tables, or `--live`). `run`/`sweep`
  take the same requirement flags (`--ram-gb`, `--device`, `--batchable`,
  `--batch-width`, `--requirements FILE`): when given and `--flavor` is absent, the
  planner chooses the flavor; an explicit `--flavor` always overrides, and with no
  requirement the behavior is unchanged.
- `examples/clouds.yaml.example`: OVH application-credential template.

## Tests

`python -m pytest tests/ -v`. The flavor-policy tests are pure logic and need no
network or credentials.

## Shared rules

@../fluxtech-meta/rules/category/package.md
