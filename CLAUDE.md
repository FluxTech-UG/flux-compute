# flux-compute

The shared cloud-compute package for the FluxTech family: provision OVH Public
Cloud GPU instances and run the simulation repos on them. Consumers (`1DSim3`,
`LumpedSim2`, future sims) import it; it imports nothing back from them.

The family conventions (one-way dependency, git rules, the no-em-dashes /
fail-fast / plain-declarative house values) live in the parent `CLAUDE.md` and
are not restated here. This file carries only what is specific to this package.

> **Onboarding to fluxtech-meta is pending.** This repo is not yet a target in
> `fluxtech-meta/registry.json`, so the shared CLAUDE.md blocks (Cross-Repo
> Access, Fail Fast, Living Documents, No Redundant cd, Collaboration, Being a
> Package) are not yet synced in here. To bring it under sync: add a target with
> `"categories": ["package"]` in `fluxtech-meta/registry.json` and run
> `make sync`. That is a write into another repo, so it needs explicit scope.

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
The default sim flavor is `t1-le-45` (cheapest fp64-healthy covered GPU). When in
doubt, validate a card with 1DSim3's `scripts/gpu_check.py` before committing to
it.

### Fail fast on missing credentials or no healthy GPU.
No silent defaults. Missing credentials raise with the remedy; a region exposing
no credit-eligible, fp64-healthy GPU raises rather than falling back to RTX5000 or
a blocked card. Switch region (GRA9, GRA11, BHS5) instead.

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
