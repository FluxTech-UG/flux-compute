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
Each OVH region carries its own compute quota — **34 vCPUs / 10 instances /
420 GiB** as measured live on 2026-07-21, identical in all nine regions. A V100S
(`t2-le-45`) is 15 vCPU, so one region fits **2 concurrent V100S**; BHS5's plain
V100 (`t1-le-45`) is 8 vCPU and fits 4. Across the five GPU regions (GRA11, DE1,
UK1, WAW1, BHS5) that is **12 concurrent GPU instances**, which is what
`sweep --regions` exists to reach. The other four regions (SBG5, RBX-A,
EU-WEST-PAR, EU-SOUTH-MIL) are CPU-only but carry the same 34 vCPU each — ~306
vCPUs project-wide for CPU fan-out.

A quota increase to 50 instances / 64 vCPUs / 496 GB was granted in writing on
ticket CS16091787 (2026-07-19) but **is not in effect in any region** as of the
2026-07-21 measurement; it may land at any time, which is precisely why no
sizing decision should be taken from this paragraph. The code never trusts these
numbers: preflight and sweep read live quota from the API and clamp to real
headroom, per region. Treat this line as orientation, and let the clamp be the
authority — if it reports more headroom than stated here, the increase landed.

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

- `flux_compute/flavors.py`: the credit + fp64 flavor policy (pure logic, tested).
- `flux_compute/auth.py`: `connect()` to the OVH project from clouds.yaml / OS_* env.
  A clouds.yaml pinned to a single `region_name:` refuses every other region
  *locally*, before any request — `connect` detects that and raises the
  `regions:`-list fix, because the pin silently caps fleet width.
- `flux_compute/sweep.py`: the fan-out, including per-region sharding
  (`parse_regions` / `allocate_concurrency` / `shard_jobs` / `Shard`, all pure
  and tested). `--max-parallel` is the GLOBAL live-instance ceiling; each region
  is additionally clamped to its own headroom.
- `flux_compute/doctor.py`: `flux-compute doctor`, the API health check.
- `flux_compute/cli.py`: argparse entry point (`doctor`, `preflight`, `run`,
  `sweep`, `reap`, `bake`, `push`).
- `examples/clouds.yaml.example`: OVH application-credential template.

## Tests

`python -m pytest tests/ -v`. The flavor-policy tests are pure logic and need no
network or credentials.

## Shared rules

@../fluxtech-meta/rules/category/package.md
