# flux-compute

The shared cloud-compute package for the FluxTech family: provision OVH Public
Cloud GPU instances and run the simulation repos on them. Consumers (`1DSim3`,
`LumpedSim2`, future sims) import it; it imports nothing back from them.

The family conventions (one-way dependency, git rules, the no-em-dashes /
fail-fast / plain-declarative house values) live in the parent `CLAUDE.md` and
are not restated here. This file carries only what is specific to this package.

<!-- fluxtech-meta:cross-repo-access BEGIN (generated — run `make sync` in fluxtech-meta) -->

## Cross-Repo Access

These repos are interconnected — packages (`co2-eos`, `fluxstyle`), the consumers
built on them, and `Physics Spec` as the shared physics source of truth (the family
`CLAUDE.md` at the repo-family root has the full map). When a task here consumes
from or feeds another repo, **reading that repo's files directly is natural and
encouraged** — work from the real source, not a remembered or copied version.
**Editing a file outside this repo is forbidden without explicit permission for that
change.** Read freely across the family; write only here.

<!-- fluxtech-meta:cross-repo-access END -->

<!-- fluxtech-meta:fail-fast BEGIN (generated — run `make sync` in fluxtech-meta) -->

## Fail Fast

No silent defaults; no caught exceptions that substitute a fallback value. A missing
config field, an unphysical state, a failed inversion, a conservation or consistency
violation beyond tolerance — raise immediately, with context. When a real value
should exist, never paper over its absence with a default.

<!-- fluxtech-meta:fail-fast END -->

<!-- fluxtech-meta:living-documents BEGIN (generated — run `make sync` in fluxtech-meta) -->

## Living Documents — No Archaeology

Every file states only the current state. When an approach changes, rewrite the
affected passage with the new result and delete what it replaced — git holds the
history. This holds for instructions and framing as much as for outputs: say what
the architecture *is*, not what it replaced. Never leave negative framing ("unlike
the previous approach", "no longer …"): a retired claim left in the text plants a
competing attractor a later reader may draw from. The one exception is a prior
approach that is the model's likely default from training — a standard pattern it
would reach for unprompted; there, an explicit override is worth stating.

<!-- fluxtech-meta:living-documents END -->

<!-- fluxtech-meta:no-redundant-cd BEGIN (generated — run `make sync` in fluxtech-meta) -->

## No Redundant cd

Commands run from the repo root; you are already there. Run them directly
(`python scripts/run.py`), never `cd /path && …`. A leading `cd` into the repo you
are already in is noise and can trip the permission prompt.

<!-- fluxtech-meta:no-redundant-cd END -->

<!-- fluxtech-meta:collaboration BEGIN (generated — run `make sync` in fluxtech-meta) -->

## Collaboration Workflow

**Pull `main` before doing anything else in a repo.** The first action of any session in a repo is `git pull` on `main` — before reading deeply, branching, or editing — so every change starts from the current remote state and not a stale local base. This is the non-negotiable first step every time, not a thing to do later: starting on a stale base is what turns ordinary commits into merge conflicts and divergence.

Branches and pull requests are for review by a collaborator, not a solo ritual. Working alone in a repo, commit and push to `main` directly — do not open a PR to yourself; reserve a feature branch and PR for changes that need another person's review. Never force-push a shared branch, and push before ending the session so work is never stranded locally.

<!-- fluxtech-meta:collaboration END -->

<!-- fluxtech-meta:being-a-package BEGIN (generated — run `make sync` in fluxtech-meta) -->

## Being a Package

This repo is a package: consumers import it; it imports nothing back from them. Keep
the public surface stable and single-sourced — canonical values live in exactly one
place and every other site derives from it, never a hand-typed copy. A consumer
concern that tries to reach back into this package means the boundary is drawn in the
wrong place; resolve it on the consumer side.

<!-- fluxtech-meta:being-a-package END -->

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

### Fail fast on missing credentials or no healthy GPU.
No silent defaults. Missing credentials raise with the remedy; a region exposing
no credit-eligible, fp64-healthy GPU raises rather than falling back to RTX5000 or
a blocked card. Switch region (GRA11, DE1, BHS5) instead.

### Cost guardrails are mandatory once `run` exists (Phase 1+).
Every provisioned instance must have a definite teardown path; an idle GPU
quietly burns startup credits. Phase 1's `run` tears down on completion and on
error; Phase 2 adds a hard spend cap.

## Layout

- `flux_compute/flavors.py`: the credit + fp64 flavor policy (pure logic, tested).
- `flux_compute/auth.py`: `connect()` to the OVH project from clouds.yaml / OS_* env.
- `flux_compute/doctor.py`: `flux-compute doctor`, the API health check.
- `flux_compute/cli.py`: argparse entry point (`doctor`; `run` is a Phase 1 stub).
- `examples/clouds.yaml.example`: OVH application-credential template.

## Tests

`python -m pytest tests/ -v`. The flavor-policy tests are pure logic and need no
network or credentials.
