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
The guide's `b3-8`..`b3-64` prices match the `_KNOWN_PRICE_EUR_HR` table exactly
(e.g. b3-8 at EUR 0.0512/hr); its `c3-*` and `b3-128+` rows run ~10% lower than
the table's 2026-07-04 DE order-catalog reads. The DE order catalog is the
account's billing basis, and the gap is conservative for budgeting (worst-case
spend is over-, never under-, estimated) — see the cross-check note in
`flavors.py`.

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
  loop, so a laptop sleep does not kill it; `sweep --resume` continues an
  interrupted run (see *Surviving laptop sleep* below). Add
  **`--regions A,B,C`** to shard one sweep across several regions at once (see
  *Multi-region sweeps* below) — the way to run a fleet wider than one region's
  quota.

### The jobs file

One job per line, `LABEL = PARAMS`. The label names the per-job artifact
subdirectory; `PARAMS` reaches the job script verbatim as `$FLUX_JOB` (and the
label as `$FLUX_LABEL`). A line with no `=` is both label and params.

```
# a whole-line comment
anchor      = --select anchor
heavy_nx128 = --select nx128 --resume    # inline comments are stripped too
```

Comments — whole-line and inline — and surrounding whitespace are stripped from
**both** the label and the params, so what the remote receives is exactly the
parameters. A `#` only opens a comment when it is unquoted and at the start of a
line or preceded by whitespace, so `--tag run#3` and `--note "a # b"` keep their
hashes, exactly as a shell would read them.

### Uploads

`--upload DIR` rsyncs `DIR` to `~/<basename>` on each instance.
`--upload SRC:DEST` lands `SRC` at `~/DEST` instead, for when the remote name
must differ from the local one — a git worktree being the usual case:

```bash
--upload /path/to/1DSim3-experiment:1DSim3     # arrives as ~/1DSim3
```

Uploads always exclude the heavy and hazardous trees: VCS and cache dirs, and
any `.flux_attach` records or the sweep's own `--into` results dir when it lives
inside an upload source (a live fleet writes and deletes those records while the
upload runs).
- **`regions [--regions A,B,C] [--flavor NAME] [--json]`** — live, read-only
  per-region occupancy: quota (vCPU / instances / RAM used vs total), the running
  flux-compute instances occupying each region (name, flavor, age, TTL bucket) and
  a count of foreign servers, and how many of a flavor (`--flavor`, default
  `b3-32`) still fit the remaining headroom. `--json` is the machine-readable
  shape for the frontend region-status button and for launchers that check
  occupancy before fanning out. Safe against live fleets (read-only throughout).
- **`reap [--yes] [--all] [--force]`** — list every flux-compute instance with
  age, hourly price and accrued cost; delete the ones past their stamped TTL
  (see Cost guardrails below). `--all --force` is the non-interactive way to
  take instances still inside their TTL, for stopping a runaway fleet from a
  script or a session with no tty.
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
(`cpu` / `gpu` / `either`), `minutes_per_job`, `batchable`, `batch_width`,
`vram_gb_per_member` (GPU device memory per batched member; batchable-only).
`device: either` lets the planner pick — batched work amortizes on a GPU,
unbatched work fans out cheapest on CPU; pass `cpu`/`gpu` to force it.

Packing K (jobs per VM) is clamped by **every binding axis**: host RAM
(`K × ram_gb_per_job ≤ host RAM × headroom`) and vCPUs for CPU fan-out, and — for a
**batched GPU** invocation — the card's **VRAM** (`K × vram_gb_per_member ≤ VRAM ×
headroom`), since the batched members co-reside on the accelerator (host RAM does
not bound them). Pass `vram_gb_per_member` for batched GPU work; omitted, the
planner conservatively assumes a member's VRAM footprint equals its host
`ram_gb_per_job` and says so in a plan note. The plan reports the **spare slots**
left in the fleet — the sizing doctrine in action: round the job count up to fill
them, because the marginal slot is close to free (a wave runs whether or not it is
full).

The plan shows **two worst-case costs**: `plan --budget` guards the **requested
jobs** cost (the N jobs dealt across the fleet, exactly as `sweep --budget` bills
one instance per job), and the **fill-the-fleet** cost (every spare slot used) is
reported beside it — so both the cost of your jobs and the cost if you round up to
fill are visible.

Offline plans (the default) use catalog values — the per-region quota (64 vCPU /
50 instances / 496 GB, measured 2026-07-27) and the catalog flavor shapes.
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
own 64 vCPUs / 50 instances / 496 GiB (measured live 2026-07-27; the CS16091787
increase is in effect). A V100S (`t2-le-45`, 15 vCPU) therefore fits **4 per
region** — so a single-region sweep tops out at 4 GPUs no matter how high
`--max-parallel` goes.
Spreading the same sweep across regions is what widens the fleet:

| Region | GPU flavor chosen | vCPU | Concurrent |
|---|---|---|---|
| GRA11, DE1, UK1, WAW1 | `t2-le-45` (V100S 32GB) | 15 | 4 each |
| BHS5 | `t1-le-45` (V100 16GB, cheaper) | 8 | 8 |
| SBG5, RBX-A, EU-WEST-PAR, EU-SOUTH-MIL | CPU only | — | — |

**24 concurrent GPU instances** across the five GPU regions, versus 4 in one.
For CPU fan-out the nine regions total ~576 vCPUs.

```bash
flux-compute sweep --cloud flux-ovh \
    --regions GRA11,DE1,UK1,WAW1,BHS5 \
    --jobs jobs.txt --script job.sh --fetch out \
    --max-parallel 24 --budget 40
```

`--max-parallel` stays what it always was — the total instances alive at once
**across the whole sweep** — and each region is additionally clamped to its own
quota headroom, so the fleet can never outrun either bound. Jobs are dealt to
regions in proportion to the concurrency each was granted, so the shards finish
together rather than one region idling while another drains. Flavor and price are
resolved **per region** (BHS5 picking the cheaper V100 above is that at work), and
the budget guard sums the shards' worst cases against the one `--budget`.
`--plan` prints the whole allocation table without launching anything.

A region that cannot fit at least one instance of the chosen flavor — no
credit-eligible fp64-healthy GPU, no quota headroom (another fleet is living
there), no compute endpoint — is **dropped with a warning** naming its occupants
and headroom, and the wave allocation is recomputed over the regions that do fit.
The sweep proceeds as long as one region fits; it refuses only when **none** do.
This turns a partial-capacity situation into a running sweep on the free regions
instead of an all-or-nothing failure. Pass **`--strict-regions`** to restore
refuse-on-any-unfit (exact-width mode, for a caller that needs the full width or
nothing); the refusal then lists every unfit region with its reason and
occupancy, so it is one fix-up round. Run `flux-compute regions` first to see the
live occupancy before launching.

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

If SSH stops working for every job at once — the usual cause being the operator's
**public IP moving** out of the one address each instance's security group allows
— the poll loop says so instead of retrying in silence: after a few consecutive
transport failures it prints `SSH unreachable since <time>`, re-reads the public
IP, and opens ingress for the new one, after which polling recovers on its own.
(A blackout used to be indistinguishable from a healthy long job, since both
produce no output.)

For the harder case — the process is fully killed (a sleep long enough to be
terminated, a closed terminal) — each sweep job persists an **attach record** and
a copy of its ephemeral key under `<into>/<label>/.flux_attach/`. Continue with:

```bash
# re-attach to what is still running:
flux-compute sweep --resume --into cloud-sweep --cloud flux-ovh

# ... and also launch the jobs of the file that never started:
flux-compute sweep --resume --into cloud-sweep --cloud flux-ovh \
    --jobs jobs.txt --script job.sh --fetch out
```

`--resume` scans `<into>` for in-flight jobs, re-establishes the poll loop against
each still-running VM, and on completion collects the log + artifacts and tears
the VM down. A VM already gone (reaped past its TTL, or torn down) is recorded as
lost. The `.flux_attach/` dir is removed on clean teardown, so its presence is
exactly the set of jobs still needing collection.

Given `--jobs` (with `--script`/`--fetch`) it then **continues the jobs file**:
every job that is neither in flight nor already collected is launched now, on the
normal sweep path with the same quota clamp and budget guard. Jobs already
collected — a `job.log` was pulled, whatever the outcome — are skipped, so
`--resume` continues a sweep rather than retrying its failures. Without `--jobs`
it only re-attaches, unchanged.

The record is written **before** the instance boots (with the instance name,
which is generated locally), so a launcher killed mid-boot leaves a VM that
`--resume` can still find and tear down instead of an unnamed orphan billing in
the console. Such a VM can be killed but not collected — its ephemeral key was
never persisted — and `--resume` reports exactly that.

### When a job dies: what the status line means

`sweep` records one line per job, and the return code is explained rather than
guessed:

| rc | reported as |
|---|---|
| 0 | `ok` |
| 124 | `job timed out (remote cap)` — `timeout` TERM'd it at `--max-minutes` |
| 137 at ~its cap | `job timed out (remote cap; SIGKILL after TERM)` |
| 137 far short of its cap, kernel log confirms | `OOM-killed (rc=137, kernel oom-killer confirmed ...)` |
| 137 far short of its cap, no evidence | `killed (rc=137, SIGKILL ...) - cause unknown` |
| other nonzero | `job nonzero` |

137 is `128 + SIGKILL` and is genuinely ambiguous: the wall cap's kill-after
escalation and the kernel OOM-killer both produce it. Reading every 137 as a
timeout sent an OOM investigation chasing phantom slow jobs, so a sub-cap 137
now triggers a kernel-log read on the still-live VM (`dmesg`/`journalctl -k`)
before teardown, and is never called a timeout. A log that cannot be read is
reported as unknown, never as innocence.

**Artifacts are fetched on every path**, not only after a clean exit — a job
killed by its cap, by the OOM-killer, or abandoned at the local deadline has
still written checkpoints and partial results, and the instance is about to be
deleted. Those partial fetches are best-effort and are labelled `PARTIAL`.

Every job also runs with the glibc allocator capped (`MALLOC_ARENA_MAX=2`,
a lowered trim threshold, and tcmalloc preloaded when the image already has it).
A thread pool otherwise spawns up to `8 × ncpu` malloc arenas and fragments large
transient buffers across them until RSS ratchets into the OOM-killer on a VM
whose real working set fits. A job script's own `export` still wins, so
consumer-side settings keep working.

### Tested and rejected on OVH: baked images

`bake` / `run --image` work, but booting from an OVH custom snapshot takes ~12 min
(image staging) — slower than the stock image + ~5 min install it replaces. The
code is kept (correct and cloud-general) but is **not recommended on OVH**; prefer
the stock image + per-job install.

## Cost guardrails

"Every provisioned instance tears down" is enforced by mechanism, not trust:

- **Hard spend cap**: `sweep --budget EUR` caps **the whole sweep's** worst case,
  not one job's. The guard computes `(total jobs) × (EUR/hr) × (--max-minutes)` —
  every job running to its full wall cap — and refuses to start above the number;
  it refuses outright when the flavor has no known price, since a money cap that
  cannot see the price is not a cap. With `--regions` the per-region shards' worst
  cases are **summed against that one budget**, so the cap is *independent of how
  many regions the sweep spans*: regions buy wall-clock, not spend. Concurrency is
  clamped to compute-quota headroom, read live from the API **per region** (64
  vCPUs / 50 instances each as measured 2026-07-27 — 4 concurrent V100S, or 8 of
  BHS5's 8-vCPU V100).
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
  else it only reports, annotated with the decision basis: keep-flagged,
  within-TTL and unstamped-legacy (name-prefix, no stamp) instances need `--all`
  plus a confirmation — interactive by default, or an explicit `--force` for a
  script or a session with no tty (`--force` alone is refused; it means something
  only with `--all`). Two flags rather than one widened `--yes`, so "skip the
  routine prompt" and "kill running work" can never be the same keystroke — and
  so nobody has to reach for `yes | flux-compute reap --all`, which answers every
  prompt in the command blind, including ones added later. Servers it cannot
  positively identify as flux-compute-created are never listed or touched. Exits
  nonzero while strays remain.
- **Stray visibility**: every command that connects (`doctor`, `preflight`,
  `run`, `sweep`, `bake`, `push`) first surfaces any stranded or kept instance
  with its accrued cost and points at `reap` — advisory only; no command other
  than `reap` ever deletes.

## Tests

```bash
python -m pytest tests/ -v
```

The flavor-policy tests are pure logic and need no network or credentials.
