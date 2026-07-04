# CPU batch readiness — make a wide CPU sweep safe

Cold prompt for a fresh session in this repo. Do the tasks in order; commit
after each. Depth: refactor along the way — the flavor/image/quota seams you
touch are young, leave them cleaner than you found them.

## Context

A consumer repo is about to fan ~200 independent CPU simulation runs through
`sweep` (sequential adaptive solves; the measured V100S is ~2× slower than a
laptop on that workload, so CPU flavors one-job-per-VM are the right device).
The orchestration primitive already fits — one ephemeral VM per job, arbitrary
upload + script, wall-cap, unconditional teardown — and CPU flavors already
pass `classify()`. Three gaps make a wide CPU batch unsafe today; they are all
consumer-agnostic and stay that way (this package never references any sim
repo — the consumer carries its own job scripts).

---

## Task 1 — CPU prices, and a budget guard that refuses instead of going blind

**What/Why.** `_KNOWN_PRICE_EUR_HR` (flavors.py) carries only t1/t2/rtx5000,
so for a CPU flavor `worst_case_eur` is None and the `--budget` guard in
`run_sweep` is silently skipped ("price n/a"). That is a silent default —
exactly what the family's fail-fast rule forbids on the one command that can
spend real money N-wide. Two changes: (a) add prices for the CPU flavors we
would actually use (the b3-* and c3-* families), following the same
cited-price-list convention the GPU entries use; (b) when `--budget` is set
and the resolved flavor has no known price, **refuse to start** with a clear
error naming the flavor, instead of skipping the guard.

**Reference data** (pricelist.ovh, read 2026-07-04 — indicative; verify
against the account's current price list before committing, and extend the
flavors.py citation comment accordingly): b3-8 ≈ €0.0605/h, b3-16 ≈ €0.1208/h,
c3-8 ≈ €0.1078/h. Fill sibling sizes from the same source.

**Done when.** A `sweep --flavor b3-8 --budget … --plan`-style dry-run prints
a real worst-case euro figure; an unpriced flavor plus `--budget` errors out
before any launch; unit tests cover both paths (the tests/ suite already fakes
flavor lists).

---

## Task 2 — CPU-aware image selection and smoke

**What/Why.** `select_gpu_image` (launch.py) demands an NVIDIA Ubuntu image,
so a CPU flavor boots a GPU-driver image unless `--image` is passed — wasteful
and surprising. Make image resolution flavor-aware: a CPU-classified flavor
defaults to a plain Ubuntu LTS image; GPU behavior and the `--image` override
are unchanged. While you are in there: `run --smoke` asserts the GPU via
`nvidia-smi`, which can never pass on CPU — give the smoke a CPU path that
verifies boot + remote exec (e.g. a trivial python check) instead, so a
one-VM CPU smoke is a meaningful preflight for a fleet.

**Done when.** `run --plan --flavor c3-8` resolves a non-NVIDIA image with no
`--image` flag; GPU plan output is byte-identical to before; unit tests cover
the CPU/GPU image split; smoke's CPU path exists and is unit-tested at the
spec level (a live b3-8 smoke costs ~a cent — run it if credentials are at
hand, otherwise leave the command in the commit message for John).

---

## Task 3 — quota-aware fan-out

**What/Why.** `run_sweep` submits one future per job with no quota check:
past the project's instance/core quota, `create_server` fails and those jobs
land as `rc=-1` failures rather than queuing. Read the compute quota at sweep
start (preflight already knows how) and clamp the effective concurrency to
what quota headroom allows for the resolved flavor's cores/instances, keeping
`--max-parallel` as the user ceiling; log the clamp explicitly. If even one
instance cannot fit, fail fast with the preflight-style message instead of
launching a doomed fleet. A create-failure that still slips through should
read as "quota/capacity" in the job record, not as a generic failed run.

**Done when.** With a mocked quota of K instances, a 3K-job sweep runs at
concurrency ≤ K and completes all jobs; the clamp is visible in output; unit
tests cover clamp, fail-fast, and the no-quota-headroom error.

---

## Task 4 — record CPU credit eligibility

**What/Why.** README/flavors.py document Startup Program eligibility for GPUs
only (V100/V100S in; RTX5000 refused for fp64; H100-class blocked). The
allowlist treats CPU prefixes as usable, but whether CPU instance spend is
covered by the program credits is not recorded anywhere. Check the account's
March-2026 eligibility document this repo already cites; if it does not
answer the question, write the open question into the README eligibility
section addressed to John rather than guessing — do not mark CPU as
credit-covered without a source.

**Done when.** README + flavors.py docstring state CPU credit status with its
source (or the explicit open question); committed.
