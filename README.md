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
| `t1-le-45/90/180`         | Tesla V100 16GB  | yes            | yes (~1/2 fp32)     | yes (default)      |
| `t2-le-45/90/180`         | Tesla V100S 32GB | yes            | yes (~1/2 fp32)     | yes (more VRAM)    |
| `rtx5000-28/56/84`        | Quadro RTX5000   | yes            | no (~1/32 fp32)     | no                 |
| `h100/a100/l40s/l4/a10-*` | various          | no             | varies              | no                 |

The Startup Program covers only V100, V100S and RTX5000 GPUs. Of those, only the
Volta cards (V100/V100S) run double precision fast enough for the EOS-heavy
sims; RTX5000 is Turing and runs fp64 at ~1/32 of fp32, so it is covered but
refused for sims by default. **Default sim flavor: `t1-le-45`** (cheapest
fp64-healthy covered GPU). CPU flavors (`c3-*`, `b3-*`) are always covered and
fp64-healthy, and are the right choice for small runs.

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

GPU flavors are region-specific; use a GPU region (`GRA9`, `GRA11`, `BHS5`).

## Verify the API works

```bash
flux-compute doctor --cloud flux-ovh
```

Authenticates, lists visible flavors and images, and reports which GPUs are
credit-eligible and fp64-healthy plus the recommended default. This is the
end-to-end "is the API working?" check.

## Roadmap

- **Phase 0 (here):** flavor policy + `doctor` API health check.
- **Phase 1:** `flux-compute run`: provision a V100, bootstrap (CUDA jaxlib plus
  the consumer repo and `co2-eos`), run one config, pull artifacts, tear down.
- **Phase 2:** sweep / fan-out across instances, artifacts to Object Storage
  (S3), hard cost cap and auto-teardown guardrails.

## Tests

```bash
python -m pytest tests/ -v
```

The flavor-policy tests are pure logic and need no network or credentials.
