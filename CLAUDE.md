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
its own work. Do not add a provisioning path that bypasses the TTL stamp or the
verified teardown. The teardown also owns the local attach state: sweep persists
`<into>/<label>/.flux_attach/` (record.json + a copy of the ephemeral SSH key)
so a killed orchestrator can `--resume`, and a clean teardown deletes it — an
orchestrator that dies and is never resumed leaves keys on disk, which
`flux-compute reap --sweep-local DIR` cleans up by removing every attach dir
whose instance is verifiably gone from a scanned region (live, unscanned-region,
and unreadable records are left alone).

**`--budget` caps the WHOLE sweep, not one job.** The guard is
`(total jobs) × (EUR/hr) × (--max-minutes)` — every job at its full wall cap — and
with `--regions` the shards' worst cases are summed against that single number.
It is therefore **independent of the region count**: regions buy wall-clock, not
spend. An unpriced flavor is refused rather than skipping the guard. Say this
plainly wherever budget is documented; a "per job" reading has propagated into a
consumer repo before.

### A job's outcome is reported, never guessed.
The remote wrapper's rc=137 is `128 + SIGKILL` and is genuinely ambiguous: the
wall cap's kill-after escalation and the kernel OOM-killer both produce it. A
sub-cap 137 triggers a kernel-log read on the still-live VM before teardown
(`provision.explain_remote_kill`) and is reported as an OOM kill, a cap timeout,
or an honest "cause unknown" — never blanket-labelled a timeout. Absence of
evidence is reported as unknown, not as innocence. Artifacts are fetched on
**every** teardown path, including the local-deadline and killed paths: partial
results are the last trace of the work and the instance is about to be deleted.

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
  and tested). `parse_jobs` owns the jobs-file format and strips comments
  (whole-line AND inline, quote-aware) from both label and params — the parse is
  the single definition of what reaches `$FLUX_JOB`, so never add a compensating
  strip in a consumer's job script. `_launch_jobs` is the shared launch path, used
  both by a fresh sweep and by `--resume --jobs`, which continues the jobs file by
  launching whatever `job_state` finds neither in flight nor collected. `--resume`
  heals SSH ingress **before the first SSH of each re-attach**
  (`_heal_ingress_before_reattach`): a fleet launched from one network is
  routinely collected from another, and a stale `/32` makes every job unreachable
  at once. The address is resolved once per resume
  (`provision.current_ingress_cidr`) and passed down. The check, the repair and
  the reporting are `provision.ensure_ssh_ingress`, shared with the steady-state
  poll loop's stuck handler so the same fault is fixed the same way and described
  in the same words whichever path notices it; it never raises, because the check
  is precautionary and the SSH attempt that follows is the authority — abandoning
  a live, billing VM over a neutron hiccup would be the worse outcome. It returns
  a status rather than a bare bool so a caller can tell "verified open" from
  "could not check", which is what the follower's fail-fast bound rests on.
  `--max-parallel` is the GLOBAL live-instance ceiling; each region
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
- `flux_compute/detach.py`: the sleep-survival machinery, all pure and unit-tested
  (`provision.py` wires it to real SSH). `launcher_script` emits the detached
  `setsid` + `timeout` launcher — which also applies the **universal glibc
  allocator tuning** (arena cap + trim threshold, opportunistic tcmalloc preload),
  so the host-RAM OOM mitigation belongs to every job rather than being re-derived
  in each consumer's job script; a job script's own `export` still overrides.
  `poll_until_done` is the reconnect-tolerant follow loop, and `on_stuck`
  escalates a sustained SSH blackout to `provision.make_stuck_handler`, which
  re-opens security-group ingress when the caller's public IP has moved — the one
  failure that breaks every job at once. The handler hands its `IngressCheck`
  back, and `classify_blackout` (pure) turns that status into the loop's decision.
  **The discriminator is who is disconnected.** An unreadable public IP means WE
  are offline (the closed-lid case the whole design exists for) and is never
  fatal, however long it lasts; a blackout where we are verifiably online AND the
  group verifiably admits us is the instance's fault and ends the follow with
  `reason="unreachable"` past `provision.UNREACHABLE_ABORT_S`, rather than letting
  a dead VM burn its whole wall cap in silence. A status that means the check
  never ran (`error`, `no-group`) never licenses that conclusion. A sleep that
  overshoots in wall-clock time is a system suspend, and a wake forces the ingress
  check on the next failure instead of after another `STUCK_AFTER_POLLS`. The
  loop's must-not-miss lines go to `on_warn`, which falls back to stderr: a
  failing self-heal reported only when a status sink happened to be wired is what
  made a four-hour fleet-wide lockout invisible.
  `AttachRecord` is written in two stages, the first **before the instance boots**,
  so a launcher killed mid-boot leaves a VM `--resume` can name and tear down.
- `flux_compute/cli.py`: argparse entry point (`doctor`, `preflight`, `run`,
  `plan`, `sweep`, `regions`, `reap`, `bake`, `push`). Sets line-buffered
  stdout/stderr at entry (`_stream_output`), so progress streams through a pipe
  and no caller needs `PYTHONUNBUFFERED`. `plan` prints a fleet plan for a
  generic requirement (offline from catalog tables, or `--live`). `run`/`sweep`
  take the same requirement flags (`--ram-gb`, `--device`, `--batchable`,
  `--batch-width`, `--requirements FILE`): when given and `--flavor` is absent, the
  planner chooses the flavor; an explicit `--flavor` always overrides, and with no
  requirement the behavior is unchanged. `sweep --detach --log FILE` runs the
  sweep as a `setsid` daemon (`_detach_into_background`) writing to `FILE`
  (`_redirect_output`, at the fd level so the rsync/ssh subprocesses land there
  too), which exists so no caller — human or launcher — needs to wrap the command
  in `nohup … > log 2>&1 &`. `--detach` without `--log` is refused: output that
  goes nowhere is indistinguishable from a run that never started.
- `examples/clouds.yaml.example`: OVH application-credential template.

## Tests

`python -m pytest tests/ -v`. The flavor-policy tests are pure logic and need no
network or credentials.

## Shared rules

@../fluxtech-meta/rules/category/package.md
