# flux-compute

Run FluxTech simulations on OVH Public Cloud GPU instances. A shared **package**
in the FluxTech family: the simulation repos (`1DSim3`, `LumpedSim2`, and future
sims) import it to provision cloud compute; it imports nothing back from them.

The family conventions (one-way dependency, git rules, the house values) live in
the parent `CLAUDE.md` and are not restated here.

## Why this exists

The sims are pure JAX, force `jax_enable_x64` (float64), and produce
config-in / artifacts-out runs with no shared state. That makes them ideal to
fan out across cloud GPUs. The payoff is parameter sweeps and large-N or
optimization jobs (many independent runs), not speeding up one small run: at
small grid sizes GPU kernel-launch latency can make a single run slower than a
laptop CPU.

## The flavor policy (the core constraint)

Two independent gates decide whether an OVH flavor may run a sim:

| Flavor                    | GPU              | Credits cover? | fp64 healthy?       | Use for sims?      |
|---------------------------|------------------|----------------|---------------------|--------------------|
| `t1-le-45/90/180`         | Tesla V100 16GB  | yes            | yes (~1/2 fp32)     | yes (BHS5 only)    |
| `t2-le-45/90/180`         | Tesla V100S 32GB | yes            | yes (~1/2 fp32)     | yes (default)      |
| `rtx5000-28/56/84`        | Quadro RTX5000   | yes            | no (~1/32 fp32)     | no                 |
| `h100/a100/l40s/l4/a10-*` | various          | no             | varies              | no                 |

The Startup Program covers only V100, V100S and RTX5000 GPUs. Of those, only the
Volta cards (V100/V100S) run double precision fast enough for the EOS-heavy
sims; RTX5000 is Turing and runs fp64 at ~1/32 of fp32, so it is covered but
refused for sims by default. **Default sim flavor: `t2-le-45`** (V100S 32GB,
available across EU regions). Plain V100 (`t1-le-*`, 16GB, slightly cheaper)
exists only in BHS5 (Canada); `recommended_for_sim` picks the cheapest
fp64-healthy GPU actually present in the target region. CPU flavors (`c3-*`,
`b3-*`) are fp64-healthy, priced from the OVH catalog, **covered by Startup
Program credits** (below), and the right choice for small runs and for a wide
one-job-per-VM fan-out.

`flux_compute.flavors.classify(name)` and `recommended_for_sim(names)` encode
this policy; it is enforced, not advisory.

### CPU credit coverage — CONFIRMED covered

**CPU instance spend (`b3-*`, `c3-*`) draws from Startup Program credits.**
Source of record: the program's product-eligibility guide (March 2026),
archived in this repo at `docs/product-eligibility-startup-program-2026-03.html`
— every General Purpose (b-series) and Compute Optimised (c-series) instance
row is marked "✓ Covered" at both program levels, and the guide's general rule
states the Public Cloud range is eligible with **GPU instances as the only
restricted family** ("only V100, V100S and RTX5000 are available with
credits"). Corroborated by OVH's public docs, which scope the voucher exclusions
to specific GPU models only (A100/H100/L4/L40S):
<https://docs.ovhcloud.com/en/guides/account-and-service-management/startup-program/available-products>.
The guide's per-flavor prices (e.g. b3-8 at EUR 0.0512/hr) match the
`_KNOWN_PRICE_EUR_HR` table, confirming both draw from the same catalog.

## Install

```bash
pip install -e .            # or: pip install -e ".[test]"
```

## Authenticate to OVH

Mint credentials in the OVH manager: **Public Cloud project > Users & Roles**.
Application credentials are preferred (scoped, revocable, no account password).
Then either:

- copy `examples/clouds.yaml.example` to `./clouds.yaml` (gitignored) and pass
  `--cloud <name>`, or
- `source` an OVH `openrc.sh`, or export the application-credential `OS_*` vars.

GPU flavors are region-specific. Verified availability: V100S (`t2-le`) in
`GRA11`, `DE1`, `UK1`, `WAW1`, `BHS5`; plain V100 (`t1-le`) only in `BHS5`. For a
German entity, `DE1` (Frankfurt) or `GRA11` are the natural choices. Note the
legacy short code `DE` has no compute endpoint; use `DE1`.

## Verify the API works

```bash
flux-compute doctor --cloud flux-ovh
```

Authenticates, lists visible flavors and images, and reports which GPUs are
credit-eligible and fp64-healthy plus the recommended default. This is the
end-to-end "is the API working?" check.

## Commands (all verified live on OVH)

- **`doctor` / `preflight`** — API health and launch-readiness.
- **`plan --ram-gb GB --device D --count N`** — size a fleet for a generic job
  requirement and print the flavor, per-region VM spread, jobs-per-VM packing,
  wave count and worst-case cost, without launching (see *Fleet planning* below).
- **`run --upload --script --fetch`** — provision a V100S, upload repos, run a job
  script, fetch artifacts, tear down (`--smoke` for a GPU check; `--plan` for a dry run).
- **`sweep --jobs FILE --max-parallel K --budget EUR`** — fan out one instance per
  job, with a pre-flight worst-case cost guard and a per-job wall-clock cap. Each
  job runs **detached** on its VM and is followed by a reconnect-tolerant poll
  loop, so a laptop sleep does not kill it; `sweep --resume` re-attaches to an
  interrupted run (see *Surviving laptop sleep* below). Add
  **`--regions A,B,C`** to shard one sweep across several regions at once (see
  *Multi-region sweeps* below) — the way to run a fleet wider than one region's
  quota.
- **`reap [--yes] [--all]`** — list every flux-compute instance with age, hourly
  price and accrued cost; delete the ones past their stamped TTL (see Cost
  guardrails below).
- **`push DIR CONTAINER`** — durable artifact copies to OVH Object Storage (Swift).

### Fleet planning (size the experiment to the machine)

A consumer describes a batch of jobs generically — RAM per job, whether it wants
a CPU or a GPU, whether the jobs batch onto one device, minutes per job, and how
many — and the planner returns which flavor, how wide a fleet across which
regions, how many jobs pack onto a VM, how many waves it takes, the worst-case
spend, and how much spare capacity is left to fill. The package knows nothing
about what the jobs compute: it routes on those generic resource fields alone.

```bash
# 100 batched GPU jobs, 2 GB each, capped at EUR 50 worst-case:
flux-compute plan --count 100 --ram-gb 2 --device gpu --batchable --batch-width 128 --budget 50

# A wide CPU fan-out from a requirements file (flags override the file):
flux-compute plan --requirements jobreq.json --regions GRA11,DE1,UK1
```

The plan is **structured data** (`flux_compute.fleet.plan_fleet` returns a
`FleetPlan`; the CLI renders it). Consumers call `plan_fleet(JobRequirements(...))`
directly. `JobRequirements` fields: `job_count`, `ram_gb_per_job`, `device`
(`cpu` / `gpu` / `either`), `minutes_per_job`, `batchable`, `batch_width`.
`device: either` lets the planner pick — batched work amortizes on a GPU,
unbatched work fans out cheapest on CPU; pass `cpu`/`gpu` to force it.

Packing K (jobs per VM) is clamped by **both** the VM's vCPUs and its RAM
(`K × ram_gb_per_job ≤ host RAM × headroom`), so a RAM-hungry job packs fewer per
VM than a lean one regardless of core count. The plan reports the **spare slots**
left in the fleet — the sizing doctrine in action: round the job count up to fill
them, because the marginal slot is close to free (a wave runs whether or not it is
full).

Offline plans (the default) use catalog values — the per-region quota (34 vCPU /
10 instances / 420 GB, measured 2026-07-21) and the catalog flavor shapes.
`flux-compute plan --live` reads the real per-region quota and flavor availability
from the API instead; a live **launch** always re-verifies and clamps to real
headroom per region. The planner reuses the sweep's per-region sharding
(`allocate_concurrency`), so the region spread it shows mirrors how a sweep fans
across the same regions. The plan is a **sizing envelope**, not a literal sweep
transcript: it reports the full quota fleet and a packing K, while `sweep` itself
launches one instance per job — it does not pack K, and runs only as many VMs as
the jobs need. Use the plan to pick the flavor and size the batch; the K figure is
the target for a consumer's own batched launcher, not something `sweep` enforces.

`run` and `sweep` accept the same requirement flags: give `--ram-gb`/`--device`
(and optionally `--batchable`/`--batch-width`/`--requirements`) and omit
`--flavor`, and the planner chooses the flavor. An explicit `--flavor` always
overrides, and with no requirement the behavior is unchanged.

### Multi-region sweeps (the fleet-width lever)

**OVH compute quota is per region, not per project.** Every region carries its
own 34 vCPUs / 10 instances / 420 GiB (measured live 2026-07-21, identical in all
nine). A V100S (`t2-le-45`, 15 vCPU) therefore fits **2 per region** — so a
single-region sweep tops out at 2 GPUs no matter how high `--max-parallel` goes.
Spreading the same sweep across regions is what widens the fleet:

| Region | GPU flavor chosen | vCPU | Concurrent |
|---|---|---|---|
| GRA11, DE1, UK1, WAW1 | `t2-le-45` (V100S 32GB) | 15 | 2 each |
| BHS5 | `t1-le-45` (V100 16GB, cheaper) | 8 | 4 |
| SBG5, RBX-A, EU-WEST-PAR, EU-SOUTH-MIL | CPU only | — | — |

**12 concurrent GPU instances** across the five GPU regions, versus 2 in one.
For CPU fan-out the nine regions total ~306 vCPUs.

```bash
flux-compute sweep --cloud flux-ovh \
    --regions GRA11,DE1,UK1,WAW1,BHS5 \
    --jobs jobs.txt --script job.sh --fetch out \
    --max-parallel 12 --budget 40
```

`--max-parallel` stays what it always was — the total instances alive at once
**across the whole sweep** — and each region is additionally clamped to its own
quota headroom, so the fleet can never outrun either bound. Jobs are dealt to
regions in proportion to the concurrency each was granted, so the shards finish
together rather than one region idling while another drains. Flavor and price are
resolved **per region** (BHS5 picking the cheaper V100 above is that at work), and
the budget guard sums the shards' worst cases against the one `--budget`.
`--plan` prints the whole allocation table without launching anything.

A region that cannot run the sweep — no credit-eligible fp64-healthy GPU, no
quota headroom, no compute endpoint — raises rather than being skipped: silently
dropping a region you asked for would quietly halve a fleet you believe is
running. All failing regions are reported together, so it is one fix-up round.

**Your `clouds.yaml` must not pin one region.** An entry with a single
`region_name:` makes every other region fail *locally*, before any request is
sent — and so silently caps fleet width at one region. Use a `regions:` list
instead (see `examples/clouds.yaml.example`); `connect` detects the pin and
prints this fix.

`sweep --resume` works across regions unchanged: each job's attach record stores
its own region, so re-attach reconnects to the right one.

### Surviving laptop sleep (detached jobs + resume)

A foreground `ssh host 'job'` binds the remote job to that one TCP session: when
the operator's laptop sleeps (a closed-lid commute), the session dies and sshd
HUPs the remote job minutes later, killing the in-flight run. `run` and `sweep`
avoid this in two halves (`flux_compute/detach.py`):

1. **The job runs detached.** A generated launcher starts it under `setsid` in a
   new session with every descriptor redirected to files on the VM, so sshd
   closes the launching channel immediately and a later disconnect cannot HUP the
   job. `nohup`/`setsid` is chosen over `tmux` (not guaranteed installed) and
   `systemd-run` (needs a user D-Bus session or root over SSH): it depends on
   nothing beyond coreutils, needs no privilege, and leaves plain pollable files
   (`~/job.out`, `~/job.pid`, `~/job.rc`). A `timeout` wrapper on the VM is the
   laptop-independent runaway backstop; the integer return code lands in
   `~/job.rc` only when the job finishes.

2. **A reconnect-tolerant poll loop follows it.** A fresh short SSH every ~15 s
   reads `~/job.rc` (done?) and incrementally tails `~/job.out` (live log). A
   failed poll — the laptop just woke, a network flap — is retried with
   exponential backoff (5 s → 60 s cap) and is **never fatal**; only the local
   wall-clock deadline (the remote cap + a 2-minute grace) aborts. On the rc
   appearing, the full `~/job.out` is pulled into `<into>/<label>/job.log`, the
   artifacts are fetched, and the VM is torn down — the same success path as
   before.

For the harder case — the process is fully killed (a sleep long enough to be
terminated, a closed terminal) — each sweep job persists an **attach record** and
a copy of its ephemeral key under `<into>/<label>/.flux_attach/`. Re-attach and
finish with:

```bash
flux-compute sweep --resume --into cloud-sweep --cloud flux-ovh
```

`--resume` scans `<into>` for in-flight jobs, re-establishes the poll loop against
each still-running VM, and on completion collects the log + artifacts and tears
the VM down. A VM already gone (reaped past its TTL, or torn down) is recorded as
lost. The `.flux_attach/` dir is removed on clean teardown, so its presence is
exactly the set of jobs still needing collection.

### Tested and rejected on OVH: baked images

`bake` / `run --image` work, but booting from an OVH custom snapshot takes ~12 min
(image staging) — slower than the stock image + ~5 min install it replaces. The
code is kept (correct and cloud-general) but is **not recommended on OVH**; prefer
the stock image + per-job install.

## Cost guardrails

"Every provisioned instance tears down" is enforced by mechanism, not trust:

- **Hard spend cap**: `sweep --budget EUR` refuses to start when worst-case
  spend (jobs x price x wall cap) exceeds the cap, and refuses outright when the
  flavor has no known price. With `--regions`, the shards' worst cases are summed
  against the one budget. Concurrency is clamped to compute-quota headroom, read
  live from the API **per region** (34 vCPUs / 10 instances each as measured
  2026-07-21 — 2 concurrent V100S, or 4 of BHS5's 8-vCPU V100).
- **Wall caps**: the remote job runs under a `timeout` wrapper on the VM, so a
  hung job is killed at its cap independently of the laptop (the local poll loop
  and `flux-compute reap`'s TTL stamp are the two further backstops).
- **Verified teardown**: the per-run server delete is retried and verified gone
  (`wait_for_delete`); a delete that cannot be verified prints a multi-line
  STRANDED INSTANCE banner with the exact cleanup commands and exits nonzero.
- **TTL metadata**: every created server is stamped `flux_created_by` and
  `flux_expires_at` (wall cap + max(30 min, 25% of the cap) — generous on
  purpose: reap must never fire early). `--keep` runs also stamp
  `flux_keep=true`.
- **`flux-compute reap`**: auto-deletes only instances that are positively
  metadata-stamped AND past their stamped expiry (`--yes` for non-interactive
  use), removing the same-named keypair and security group with them. Everything
  else it only reports, annotated with the decision basis: keep-flagged and
  unstamped-legacy (name-prefix, no stamp) instances need `--all` plus an
  interactive confirmation; servers it cannot positively identify as
  flux-compute-created are never listed or touched. Exits nonzero while strays
  remain.
- **Stray visibility**: every command that connects (`doctor`, `preflight`,
  `run`, `sweep`, `bake`, `push`) first surfaces any stranded or kept instance
  with its accrued cost and points at `reap` — advisory only; no command other
  than `reap` ever deletes.

## Tests

```bash
python -m pytest tests/ -v
```

The flavor-policy tests are pure logic and need no network or credentials.
