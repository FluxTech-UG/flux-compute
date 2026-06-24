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
`b3-*`) are always covered and fp64-healthy, and are the right choice for small
runs.

`flux_compute.flavors.classify(name)` and `recommended_for_sim(names)` encode
this policy; it is enforced, not advisory.

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
- **`run --upload --script --fetch`** — provision a V100S, upload repos, run a job
  script, fetch artifacts, tear down (`--smoke` for a GPU check; `--plan` for a dry run).
- **`sweep --jobs FILE --max-parallel K --budget EUR`** — fan out one instance per
  job, with a pre-flight worst-case cost guard and a per-job wall-clock cap.
- **`push DIR CONTAINER`** — durable artifact copies to OVH Object Storage (Swift).

### Tested and rejected on OVH: baked images

`bake` / `run --image` work, but booting from an OVH custom snapshot takes ~12 min
(image staging) — slower than the stock image + ~5 min install it replaces. The
code is kept (correct and cloud-general) but is **not recommended on OVH**; prefer
the stock image + per-job install.

## Tests

```bash
python -m pytest tests/ -v
```

The flavor-policy tests are pure logic and need no network or credentials.
