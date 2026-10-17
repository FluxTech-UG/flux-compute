"""Fan out a parameter sweep across ephemeral instances, with a hard cost ceiling.

Each job runs on its own ephemeral instance (provision -> upload -> run the
consumer's script with the job's params in $FLUX_LABEL/$FLUX_JOB -> fetch
artifacts -> teardown), up to --max-parallel at once. Three guards bound spend
and blast radius: a pre-flight worst-case check (jobs x price x per-job wall cap)
refuses to start above the budget; the effective concurrency is clamped to the
compute-quota headroom for the flavor, so the fleet cannot outrun the quota into
create failures; and each job's remote exec is killed at --max-minutes so a hung
job cannot run up the bill. Teardown is per-job and unconditional.

**Compute quota is per region, so regions multiply the fleet.** `--regions A,B,C`
shards one sweep across several regions at once: each region is resolved, quota-
clamped and priced on its own (flavor availability differs by region), and the
per-region fleets run concurrently under one global `--max-parallel` ceiling and
one `--budget`. A single `--region` is the one-shard case of the same path.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from . import detach, regions
from .auth import connect
from .flavors import classify, dearest_default_flavor
from .launch import resolve_spec
from .provision import (
    LOCAL_GRACE_S, POLL_INTERVAL_SWEEP_S, RC_SIGKILL, RESUME_MIN_DEADLINE_S,
    TeardownStrandError, _gpu_instance, _launch_detached, _print_plan, _region,
    _rsync_down, _rsync_up, _scp_up, _server_by_name_or_id, classify_exit,
    current_ingress_cidr, ensure_ssh_ingress, follow_detached_job,
    looks_like_cap_kill, make_stuck_handler, parse_upload_spec, probe_oom_kill,
    pull_job_log, rsync_down_best_effort, teardown_by_name, ttl_minutes_for,
)
from .reap import warn_strays

# Per-label attach state, persisted under <into>/<label>/.flux_attach/ so a hard
# kill of the orchestrator can re-attach to the still-running VM. The dir's
# presence marks a job as not-yet-finalized; it is removed on clean teardown.
ATTACH_DIR = ".flux_attach"
ATTACH_RECORD = "record.json"
ATTACH_KEY = "id_key"
# The pulled remote job log. Its presence is what marks a job as collected, so
# `sweep --resume --jobs ...` can tell an already-run job from a never-started one.
JOB_LOG = "job.log"
# The job's end record (`JobOutcome`): label, the region it ended in, rc, status,
# and whether it failed before its script started. Written on every path that
# ends a job, BEFORE the attach record is dropped, so a reader that sees the
# record gone always finds the outcome. `--rerun-failed-start` and `--wait-next`
# read it; neither reads a log.
OUTCOME = "outcome.json"
# The steps a job can fail in before its script runs on the instance. A failure in
# any of them leaves nothing to collect (the instance is torn down at once), so
# `--rerun-failed-start` may launch the job again into the same folder.
PRESTART_PHASES = ("boot", "upload", "launch")
# How often `--wait-next` re-reads the campaign's local state (a directory scan,
# no network).
WAIT_POLL_S = 5.0


@dataclass(frozen=True)
class JobOutcome:
    """How one job ended, as `<into>/<label>/outcome.json`.

    `region` is the region the job last ran in (a retried upload moves it).
    `prestart_phase` names the step it failed in when it failed before its
    script started (one of `PRESTART_PHASES`), and is None for every job whose
    script ran, and for every job where that is not known for certain.
    """

    label: str
    region: str | None
    rc: int
    status: str
    ended_epoch: float
    prestart_phase: str | None = None

    @property
    def failed_before_start(self) -> bool:
        return self.prestart_phase is not None

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        return cls(**json.loads(text))


def _outcome(label, region, rc, status, prestart_phase=None):
    if prestart_phase is not None and prestart_phase not in PRESTART_PHASES:
        raise ValueError(f"unknown pre-start phase {prestart_phase!r}")
    return JobOutcome(label=label, region=region, rc=rc, status=status,
                      ended_epoch=time.time(), prestart_phase=prestart_phase)


def _record_outcome(dest, outcome):
    """Write a job's outcome record atomically (temp file + rename)."""
    os.makedirs(dest, exist_ok=True)
    tmp = os.path.join(dest, OUTCOME + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(outcome.to_json())
    os.replace(tmp, os.path.join(dest, OUTCOME))
    return outcome


def read_outcome(dest):
    """The job's `JobOutcome`, or None when it has not ended. An outcome file
    that does not parse raises with its path: it is the record two commands
    decide on, and a guess in its place would rerun or report the wrong job."""
    path = os.path.join(dest, OUTCOME)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        text = fh.read()
    try:
        return JobOutcome.from_json(text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"unreadable outcome record {path}: {exc}") from exc


def strip_inline_comment(line):
    """Return `line` with any trailing ``#`` comment removed, quote-aware.

    A jobs file is an operator-edited document, so its lines get annotated the way
    every other config line does::

        heavy_nx128 = --select nx128 --resume    # rerun: OOM'd on b3-32

    Everything after ``=`` reaches the job script verbatim as ``$FLUX_JOB``, so
    an unstripped comment was passed through to the remote as part of the
    parameters, matched nothing, and every VM in the fleet did no work. Stripping
    belongs here, at the parse, and not in each consumer's job script -- a
    defensive strip downstream cannot fix a jobs file the launcher already
    mis-read, and only one of them can be the definition.

    A ``#`` opens a comment only when it is **unquoted** and **at the start of the
    line or preceded by whitespace**, so genuine parameter values survive:
    ``--tag run#3`` keeps its hash (no preceding space) and ``--note "a # b"``
    keeps its (quoted). That is the shell's own reading of the same text, which
    is the intuition an operator writing a params line already has.
    """
    out = []
    quote = None
    at_boundary = True        # start-of-line counts as a whitespace boundary
    for ch in str(line):
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            at_boundary = False
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            at_boundary = False
            continue
        if ch == "#" and at_boundary:
            break
        out.append(ch)
        at_boundary = ch.isspace()
    return "".join(out).strip()


def parse_jobs(text):
    """Parse a jobs file: each non-blank, non-comment line is 'LABEL = PARAMS'.

    Without '=', the whole line is both label and params. Labels must be unique
    and filesystem-safe (they name the per-job artifact subdir). PARAMS reaches
    the job script as $FLUX_JOB.

    Comments -- whole-line and inline -- are stripped from BOTH the label and the
    params (`strip_inline_comment`), as is surrounding whitespace, so what the
    remote receives is exactly the parameters and nothing else.
    """
    jobs = []
    seen = set()
    for raw in text.splitlines():
        line = strip_inline_comment(raw)
        if not line:
            continue
        if "=" in line:
            label, params = (s.strip() for s in line.split("=", 1))
        else:
            label = params = line
        if not label or "/" in label or label in seen:
            raise RuntimeError(f"bad or duplicate job label: {label!r}")
        seen.add(label)
        jobs.append((label, params))
    if not jobs:
        raise RuntimeError("no jobs found in jobs file")
    return jobs


def worst_case_eur(n_jobs, price_eur_hr, max_minutes):
    """Total worst-case cost: every job runs to its wall cap. Concurrency does
    not change the total, only the wall-clock."""
    if price_eur_hr is None:
        return None
    return n_jobs * price_eur_hr * (max_minutes / 60.0)


def budget_guard_shards(entries, max_minutes, budget_eur):
    """Enforce --budget against the worst-case spend of every shard, or raise.

    `entries` is [(flavor, price_eur_hr, n_jobs), ...] -- one per region shard,
    since flavor and therefore price can differ by region. The budget is a single
    cap on the whole sweep, so the shards' worst cases are summed.

    Returns the total worst-case EUR (None when any flavor has no known price and
    no budget is set). With a budget set, an unpriced flavor is refused rather
    than silently skipping the guard: a money cap that cannot see the price is
    not a cap. Add the flavor's price to _KNOWN_PRICE_EUR_HR (flavors.py).
    """
    unpriced = sorted({str(f) for f, price, _ in entries if price is None})
    if unpriced:
        if budget_eur is not None:
            raise RuntimeError(
                f"--budget EUR {budget_eur:.2f} was set, but flavor(s) "
                f"{', '.join(repr(f) for f in unpriced)} have no known price, so "
                f"worst-case spend cannot be bounded. Add the price to "
                f"_KNOWN_PRICE_EUR_HR in flavors.py, or drop --budget to run unguarded."
            )
        return None
    total = sum(worst_case_eur(n, price, max_minutes) for _, price, n in entries)
    if budget_eur is not None and total > budget_eur:
        raise RuntimeError(
            f"worst-case ~EUR {total:.2f} exceeds budget EUR {budget_eur:.2f}; "
            "lower --max-minutes, run fewer jobs, or raise --budget."
        )
    return total


def budget_guard(flavor, price_eur_hr, n_jobs, max_minutes, budget_eur):
    """Single-shard budget guard: the one-region case of budget_guard_shards."""
    return budget_guard_shards([(flavor, price_eur_hr, n_jobs)], max_minutes, budget_eur)


def clamp_concurrency(max_parallel, vcpus_per_instance, cores_used, cores_max,
                      instances_used, instances_max):
    """Clamp requested parallelism to what compute-quota headroom allows.

    One instance per job, so the number that can run at once is bounded by both
    the core quota (headroom // vCPU-per-instance) and the instance quota
    (headroom). Returns the effective concurrency (>=1, never above
    max_parallel). Raises (fail-fast) when the headroom cannot fit even one
    instance, rather than launching a fleet doomed to rc=-1 create failures.
    """
    if vcpus_per_instance <= 0:
        raise RuntimeError(f"flavor vCPU count {vcpus_per_instance!r} is not positive")
    # OpenStack reports an unlimited quota as -1; treat it as no bound rather
    # than letting the subtraction go negative and falsely refuse. The
    # arithmetic stays integral throughout: float("inf") // n is nan, and nan
    # poisons min() into passing max_parallel through unclamped.
    if (cores_max or 0) < 0:
        fit_cores, cores_desc = max(max_parallel, 1), "unlimited cores"
    else:
        cores_free = (cores_max or 0) - (cores_used or 0)
        fit_cores = cores_free // vcpus_per_instance
        cores_desc = f"{cores_free} cores"
    if (instances_max or 0) < 0:
        fit_inst, inst_desc = max(max_parallel, 1), "unlimited instances"
    else:
        fit_inst = (instances_max or 0) - (instances_used or 0)
        inst_desc = f"{fit_inst} instances"
    fit = min(fit_cores, fit_inst)
    if fit < 1:
        raise RuntimeError(
            f"compute quota cannot fit even one instance: headroom is "
            f"{cores_desc} / {inst_desc}, but each instance needs "
            f"{vcpus_per_instance} vCPU. Free running instances, request a quota "
            f"increase, or switch region."
        )
    return min(max_parallel, fit)


def parse_regions(text):
    """Parse `--regions A,B,C` into an ordered, de-duplicated region list.

    Fails fast on an empty or blank-only value rather than silently falling back
    to the single-region path: `--regions ''` is a mistake, not a default.
    """
    names = [s.strip() for s in str(text).split(",")]
    names = [n for n in names if n]
    if not names:
        raise RuntimeError("--regions was given but named no region")
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def allocate_concurrency(caps, max_parallel):
    """Split a global live-instance ceiling across regions, never past each cap.

    `caps[i]` is what region i's own quota headroom allows. `--max-parallel`
    stays what it has always been -- the total number of instances alive at once
    across the whole sweep -- so adding regions widens the fleet only up to that
    one number, and the blast radius and budget intuition carry over unchanged.

    Filled breadth-first (one slot per region per pass), so a ceiling smaller
    than the region count spreads across distinct regions rather than saturating
    the first. Returns per-region allocations summing to min(max_parallel, sum(caps)).
    """
    if max_parallel < 1:
        raise RuntimeError(f"--max-parallel must be at least 1, got {max_parallel}")
    alloc = [0] * len(caps)
    remaining = min(max_parallel, sum(caps))
    while remaining > 0:
        progressed = False
        for i, cap in enumerate(caps):
            if remaining == 0:
                break
            if alloc[i] < cap:
                alloc[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:      # every region is at its cap
            break
    return alloc


def shard_jobs(jobs, weights):
    """Deal jobs to regions in proportion to each region's allocated concurrency.

    A region that may run 4 instances at once should carry ~4x the jobs of a
    region allowed 1, so the shards finish together instead of one region idling
    while another drains. Dealt round-robin over a weight-expanded region cycle,
    which both proportions the load and keeps consecutive jobs spread across
    regions. Zero-weight regions (no allocation) get nothing.
    """
    cycle = [i for i, w in enumerate(weights) for _ in range(max(0, w))]
    if not cycle:
        raise RuntimeError("no region has any concurrency allocation")
    shards = [[] for _ in weights]
    for n, job in enumerate(jobs):
        shards[cycle[n % len(cycle)]].append(job)
    return shards


@dataclass(frozen=True)
class Shard:
    """One region's slice of a sweep: where to launch, how wide, and which jobs."""
    region: str
    spec: object            # LaunchSpec, resolved in this region
    vcpus: int
    cap: int                # what this region's quota headroom allows
    concurrency: int = 0    # what the global ceiling actually granted it
    jobs: tuple = ()

    def with_plan(self, concurrency, jobs):
        return Shard(region=self.region, spec=self.spec, vcpus=self.vcpus,
                     cap=self.cap, concurrency=concurrency, jobs=tuple(jobs))


def _prepare_shard(cloud, region, flavor, image, max_parallel):
    """Resolve one region: connect, pick the flavor/image, read its own quota.

    Every region is resolved independently because availability genuinely differs
    -- V100S exists in some OVH regions and not others, and the cheapest healthy
    GPU (and its price) is a per-region answer.
    """
    conn = connect(cloud=cloud, region=region)
    warn_strays(conn)
    reg = _region(conn, region)
    spec = resolve_spec(conn, reg, flavor=flavor, image=image)
    lim = conn.get_compute_limits()
    gq = lambda k: getattr(lim, k, None)
    vcpus = _flavor_vcpus(conn, spec.flavor)
    cap = clamp_concurrency(
        max_parallel, vcpus,
        gq("total_cores_used"), gq("max_total_cores"),
        gq("total_instances_used"), gq("max_total_instances"),
    )
    return Shard(region=reg, spec=spec, vcpus=vcpus, cap=cap)


@dataclass(frozen=True)
class RegionDrop:
    """A requested region that cannot host the sweep, and why. The raw `region`
    (possibly None for the clouds.yaml default) is kept so an occupancy read can
    re-target it; `label` is what to print."""

    region: str | None
    reason: str

    @property
    def label(self) -> str:
        return self.region or "(default region)"


def _prepare_shards(cloud, regions, flavor, image, max_parallel):
    """Resolve every requested region; return (shards, drops).

    A region that cannot host the sweep (no credit-eligible fp64-healthy GPU, no
    quota headroom for even one instance, no compute endpoint) becomes a
    `RegionDrop` carrying the reason instead of a shard, rather than raising. The
    caller decides between graceful-degrade (drop unfit regions with a warning
    and run on the rest, the default) and refuse-the-whole-sweep
    (`--strict-regions`). A clouds.yaml region pin is one global config fault
    (fixing it fixes every region), so it still surfaces whole and at once.
    """
    shards, drops = [], []
    for region in regions:
        try:
            shards.append(_prepare_shard(cloud, region, flavor, image, max_parallel))
        except Exception as exc:
            if "refused by the local clouds.yaml" in str(exc):
                raise
            drops.append(RegionDrop(
                region=region, reason=f"{type(exc).__name__}: {str(exc)[:160]}"))
    return shards, drops


_REFUSAL_TAIL = (
    "Drop them from --regions, or fix the cause above (a region with no "
    "credit-eligible fp64-healthy GPU cannot host a sim; check `flux-compute "
    "doctor --region <name>`). Run `flux-compute regions` to see live occupancy."
)


def _region_refusal(cloud, drops, *, all_dropped):
    """The informative refusal message when the sweep cannot proceed: every
    dropped region with its reason and (best-effort) who is occupying it."""
    lines = []
    for d in drops:
        occ = regions.occupancy_line(cloud, d.region)
        line = f"  {d.label}: {d.reason}"
        if occ:
            line += f"\n      occupied by: {occ}"
        lines.append(line)
    head = ("no requested region can run this sweep:" if all_dropped
            else "these requested regions cannot run this sweep:")
    return head + "\n" + "\n".join(lines) + "\n\n" + _REFUSAL_TAIL


def _warn_region_drops(cloud, drops, surviving):
    """Graceful-degrade: print a warning per dropped region (reason + who occupies
    it) and note that the sweep proceeds on the survivors."""
    for d in drops:
        print(f"WARNING: dropping region {d.label} (cannot fit >=1 instance): {d.reason}")
        occ = regions.occupancy_line(cloud, d.region)
        if occ:
            print(f"         occupied by: {occ}")
    print(f"proceeding on {surviving} of {surviving + len(drops)} requested region(s); "
          "re-run with --strict-regions to refuse the whole sweep instead.")


def _flavor_vcpus(conn, flavor_name):
    """Read the vCPU count for a flavor from the compute API, or raise."""
    fl = conn.compute.find_flavor(flavor_name)
    vcpus = getattr(fl, "vcpus", None) if fl is not None else None
    if vcpus is None:
        raise RuntimeError(
            f"could not read the vCPU count for flavor {flavor_name!r} from the compute API"
        )
    return vcpus


def _failure_status(exc):
    """Label a per-job failure for the job record. A teardown strand reads as
    'STRANDED' (the instance may still be billing) and a create-time
    quota/capacity rejection as 'quota/capacity', rather than a generic error,
    so the two costly failure modes are diagnosable at a glance."""
    name = type(exc).__name__
    msg = str(exc)
    hay = f"{name} {msg}".lower()
    if "strand" in hay:
        return f"STRANDED: {name}: {msg[:100]}"
    if any(k in hay for k in ("quota", "no valid host", "over quota", "toomanyrequests")):
        return f"quota/capacity: {name}: {msg[:100]}"
    return f"error: {name}: {msg[:100]}"


def _fan_out(jobs, run_one, max_workers, on_result=None):
    """Run run_one(job) across up to max_workers threads (one instance per job);
    return the results in completion order. A clamped max_workers keeps the live
    fleet within quota. on_result, if given, is called with each result as it
    lands (for streaming progress)."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if on_result is not None:
                on_result(r)
            results.append(r)
    return results


def _attach_dir(dest):
    return os.path.join(dest, ATTACH_DIR)


def _persist_record(dest, rec):
    """Write an attach record atomically (temp file + rename), so a process killed
    mid-write leaves the previous record intact rather than a truncated JSON file
    that no later --resume can parse."""
    adir = _attach_dir(dest)
    os.makedirs(adir, exist_ok=True)
    tmp = os.path.join(adir, ATTACH_RECORD + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(rec.to_json())
    os.replace(tmp, os.path.join(adir, ATTACH_RECORD))
    return rec


def _write_pending_record(dest, *, label, cloud, region, name, remote_script,
                          fetch, into, cap_seconds):
    """Record the job BEFORE its instance boots.

    The window between `create_server` and the post-launch record write is short
    but real, and a launcher killed inside it left a booted, billing VM that no
    later --resume could even name: an orphan findable only by hand in the OVH
    console. The instance name is generated locally, so it can be persisted
    before anything is created -- and the name is exactly what teardown needs.
    The record is upgraded in place once the instance is up.
    """
    return _persist_record(dest, detach.AttachRecord(
        label=label, cloud=cloud, region=region, name=name,
        remote_script=remote_script, fetch=fetch, into=into,
        cap_seconds=cap_seconds, launch_epoch=time.time()))


def _write_attach_record(dest, *, label, cloud, region, name, server_id, ip,
                         keyfile, remote_script, fetch, into, cap_seconds):
    """Upgrade the pending record to a full one: the boot-time facts plus a
    durable copy of the ephemeral private key under <dest>/.flux_attach/, so a
    hard kill of this process can re-attach to the still-running VM. The key copy
    is 0600 and is removed with the record on clean teardown."""
    adir = _attach_dir(dest)
    os.makedirs(adir, exist_ok=True)
    key_copy = os.path.join(adir, ATTACH_KEY)
    shutil.copyfile(keyfile, key_copy)
    os.chmod(key_copy, 0o600)
    return _persist_record(dest, detach.AttachRecord(
        label=label, cloud=cloud, region=region, name=name, server_id=server_id,
        ip=ip, keyfile=key_copy, remote_script=remote_script, fetch=fetch,
        into=into, cap_seconds=cap_seconds, launch_epoch=time.time()))


def _close_job(dest, outcome):
    """End a job: write its outcome, THEN drop its attach record. The order is
    the contract `--wait-next` reads by: a job whose record is gone has its
    outcome on disk."""
    _record_outcome(dest, outcome)
    _clear_attach_record(dest)


def _close_unrecoverable(dest, outcome):
    """Record a job that ended badly (its VM lost, orphaned, or its launch
    failed), then drop its attach record.

    The status is written as the local job.log FIRST, so `job_state` reads the
    job as ended and a later --resume (or the detached child after a pre-flight)
    never counts it as pending and launches it a second time. A job.log
    `_finalize` already pulled is the authoritative record and is kept."""
    os.makedirs(dest, exist_ok=True)
    log = os.path.join(dest, JOB_LOG)
    if not os.path.isfile(log):
        with open(log, "w") as fh:
            fh.write(f"flux-compute: {outcome.status}\n")
    _close_job(dest, outcome)


def _clear_previous_attempt(dest):
    """Before a new attempt, remove what an earlier attempt that failed before
    its script started left: its one-line status job.log (which
    `_close_unrecoverable` never overwrites, so it would outlive the new
    attempt) and its outcome. A job.log pulled from a run whose script ran is
    real output and is left for the new run to replace. Only a job with no
    attach record is ever launched, so this never touches a live job's state."""
    prior = read_outcome(dest)
    if prior is None or not prior.failed_before_start:
        return
    for name in (JOB_LOG, OUTCOME):
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            os.remove(path)


def _clear_attach_record(dest):
    """Drop the attach dir (record + key copy) after a clean finalize + teardown;
    its absence means the job is done."""
    shutil.rmtree(_attach_dir(dest), ignore_errors=True)


def _load_attach_records(into):
    """Every persisted attach record under <into> (in-flight or interrupted jobs)."""
    recs = []
    if not os.path.isdir(into):
        return recs
    for label in sorted(os.listdir(into)):
        path = os.path.join(into, label, ATTACH_DIR, ATTACH_RECORD)
        if os.path.isfile(path):
            with open(path) as fh:
                recs.append(detach.AttachRecord.from_json(fh.read()))
    return recs


def job_state(into, label):
    """Where one job of the jobs file stands, from what is on local disk:

      "in_flight"    : an attach record exists -- launched, not yet collected.
      "failed_start" : it ended before its script started (its outcome names
                       the boot, upload or launch step). `--resume` skips it;
                       `--resume --rerun-failed-start` launches it again.
      "collected"    : it ended after its script started, or its end is
                       recorded without that distinction (a job.log or an
                       outcome). `--resume` never launches it again, whatever
                       the outcome was: a resume continues a sweep, it does not
                       retry jobs that ran.
      "pending"      : none of these -- never started, so --resume may launch it.
    """
    dest = os.path.join(into, label)
    if os.path.isfile(os.path.join(dest, ATTACH_DIR, ATTACH_RECORD)):
        return "in_flight"
    outcome = read_outcome(dest)
    if outcome is not None and outcome.failed_before_start:
        return "failed_start"
    if outcome is not None or os.path.isfile(os.path.join(dest, JOB_LOG)):
        return "collected"
    return "pending"


def _status_for_outcome(outcome, *, elapsed_s=None, cap_seconds=None, oom=None):
    """Map a follow outcome to the sweep's (rc, status) record fields.

    The remote return code is explained by `classify_exit`, so an ambiguous
    rc=137 is reported for what it is (see provision.py): only a kill that
    actually reached its wall cap is called a timeout.
    """
    if outcome.reason == "deadline":
        return -1, ("LOCAL DEADLINE (job unfinished; server torn down; "
                    "partial artifacts fetched)")
    if outcome.reason == "unreachable":
        return -1, ("UNREACHABLE (SSH stayed dead while this machine was online "
                    "and the security group admitted it, so the instance is the "
                    "cause; server torn down, partial artifacts fetched if any)")
    rc = outcome.rc
    return rc, classify_exit(rc, elapsed_s=elapsed_s, cap_seconds=cap_seconds, oom=oom)


def _finalize(ip, keyfile, dest, fetch, outcome, *, elapsed_s=None, cap_seconds=None):
    """Collect a followed job: pull the authoritative job.log, then the artifacts.
    Returns (rc, status). Teardown is the caller's.

    Artifacts are fetched on EVERY path, not only on a clean finish. A job killed
    by its cap, by the OOM-killer, or abandoned at the local deadline has still
    written checkpoints and partial results, and the instance is about to be
    deleted -- so a clean-exit-only fetch turned every imperfect run into a total
    loss. A failed clean-run fetch still raises (that is a real error); the
    partial fetches are best-effort and never mask the job's own outcome.
    """
    pull_job_log(ip, keyfile, os.path.join(dest, JOB_LOG))
    clean = outcome.reason == "done" and outcome.rc == 0
    if clean:
        _rsync_down(ip, keyfile, fetch, dest)
    else:
        rsync_down_best_effort(ip, keyfile, fetch, dest)
    oom = None
    if outcome.reason == "done" and outcome.rc == RC_SIGKILL:
        # Probe the kernel log while the instance is still alive: the evidence
        # that distinguishes an OOM kill from a cap kill dies with the VM.
        if looks_like_cap_kill(elapsed_s, cap_seconds) is not True:
            oom = probe_oom_kill(ip, keyfile)
    return _status_for_outcome(outcome, elapsed_s=elapsed_s,
                               cap_seconds=cap_seconds, oom=oom)


def _job_warn(label):
    """A labelled stderr sink for the poll loop's must-not-miss lines.

    A sweep fans many jobs across threads, so an unlabelled "woke after 4h" or
    "giving up" line is unattributable in the merged log. These are warnings only
    -- the per-poll progress chatter stays off, because 24 jobs x one line every
    15s is a log nobody reads, and an unread log is how a fleet-wide lockout went
    unnoticed for four hours in the first place.
    """
    def _warn(msg):
        print(f"  [{label}] {msg}", file=sys.stderr)
    return _warn


def _heal_ingress_before_reattach(conn, rec, cidr, emit=print):
    """Make sure this VM's security group still admits us, BEFORE the first SSH.

    A resume is the moment the caller is most likely to be somewhere else: the
    fleet was launched from one network and is being collected from another
    (overnight, a commute, a VPN). Each instance's group admits exactly the /32
    that launched it, so a moved address locks the operator out of every job at
    once -- and a re-attach that just starts polling reads as a healthy long job,
    since silence is what both look like. Checking up front turns hours of silent
    unreachability into one printed line and a first poll that works.

    The check, the repair and the reporting are `ensure_ssh_ingress`, shared with
    the steady-state poll loop's stuck handler so a re-attach and a mid-flight
    blackout repair the same fault the same way and say so in the same words.
    """
    return ensure_ssh_ingress(conn, rec.name, cidr=cidr, label=rec.label, emit=emit)


def _reattach_records(cloud, region, into, max_parallel) -> int:
    """Re-attach to every persisted in-flight job under <into>, collect and tear
    down. Returns 0 when all of them ended cleanly."""
    records = _load_attach_records(into)
    if not records:
        print(f"resume: no in-flight jobs found under {into}/ (nothing to re-attach).")
        return 0
    print(f"resume: {len(records)} in-flight job(s) under {into}/; "
          f"re-attaching (up to {max_parallel} at once).")
    try:
        warn_strays(connect(cloud=cloud, region=region))
    except Exception as exc:
        print(f"note: stray check skipped ({type(exc).__name__}: {str(exc)[:80]})")

    # Resolved ONCE for the whole fleet: every job's group is compared against
    # the same address, and 16 re-attaches do not make 16 identical lookups.
    # None (the read failed) still flows down, so each job says so and continues.
    cidr = current_ingress_cidr()
    if cidr is None:
        print("note: could not read this machine's public IP; SSH ingress cannot "
              "be re-checked. If every job below goes unreachable, that is the "
              "first thing to suspect.")

    def _resume_one(rec):
        dest = os.path.join(rec.into, rec.label)
        try:
            conn = connect(cloud=rec.cloud or cloud, region=rec.region or region)
            server = _server_by_name_or_id(conn, rec.name, rec.server_id or None)
            if server is None:
                status = "lost: VM gone (reaped?); cannot re-attach"
                _close_unrecoverable(dest, _outcome(rec.label, rec.region, -1, status))
                return (rec.label, -1, status)
            if not rec.attachable:
                # A pending (pre-boot) record whose instance DID come up: the
                # launcher died before it could persist the address and the
                # ephemeral key, so this VM can never be logged into again. It can
                # still be named, and therefore killed -- which is the whole reason
                # the record is written before the boot. Stop the billing and say
                # plainly that the results are gone.
                teardown_by_name(conn, rec.name, rec.server_id or None)
                # Whether its script started is unknown (the key is recorded
                # after the launch), so it is not marked as a pre-start failure.
                status = ("orphan: instance booted but the launcher died before "
                          "recording its key; torn down (results unrecoverable)")
                _close_unrecoverable(dest, _outcome(rec.label, rec.region, -1, status))
                return (rec.label, -1, status)
            # Before the first SSH: re-open ingress if we are on a new network.
            _heal_ingress_before_reattach(conn, rec, cidr)
            # The original deadline may already have elapsed while the orchestrator
            # was down, yet the job (or its remote-cap kill) may have finished and
            # just needs collecting -- so floor the follow at RESUME_MIN_DEADLINE_S.
            remaining = rec.launch_epoch + rec.cap_seconds + LOCAL_GRACE_S - time.time()
            deadline = max(remaining, RESUME_MIN_DEADLINE_S)
            outcome = follow_detached_job(
                rec.ip, rec.keyfile, rec.cap_seconds,
                deadline_s=deadline, poll_interval=POLL_INTERVAL_SWEEP_S,
                on_warn=_job_warn(rec.label),
                on_stuck=make_stuck_handler(conn, rec.name, label=rec.label))
            rc, status = _finalize(
                rec.ip, rec.keyfile, dest, rec.fetch, outcome,
                elapsed_s=time.time() - rec.launch_epoch, cap_seconds=rec.cap_seconds)
            teardown_by_name(conn, rec.name, rec.server_id or None)
            _close_job(dest, _outcome(rec.label, rec.region, rc, status))
            return (rec.label, rc, status)
        except Exception as exc:
            # No outcome: the job may still be running on its VM, and its
            # record stays for the next --resume.
            return (rec.label, -1, _failure_status(exc))

    def _log(r):
        label, rc, status = r
        print(f"  [{label}] rc={rc} {status}")

    results = _fan_out(records, _resume_one, max(1, max_parallel), on_result=_log)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"resume done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1



@dataclass(frozen=True)
class ResumeWork:
    """What `--resume --jobs FILE` finds on disk for the jobs of FILE.

    `never_started` are the pending jobs, `failed_start` the jobs that ended
    before their script started, `collected` the count of jobs that ended after
    (or without that distinction). `launch` is what the resume launches: the
    never-started jobs, plus the failed-start ones under `--rerun-failed-start`,
    in jobs-file order."""

    jobs: list
    never_started: list
    failed_start: list
    collected: int
    launch: list


def _resume_pending(into, jobs_file, script, fetch, rerun_failed_start=False):
    """The jobs of `jobs_file` that `--resume` would launch, as a `ResumeWork`.

    Read from local disk only (`job_state`), so the answer is the same before and
    after the re-attach half runs: every in-flight job leaves it as ended, never
    as pending, because a finalized job has its pulled job.log and outcome and a
    lost or orphaned one has the status job.log and outcome
    `_close_unrecoverable` writes. Jobs to launch without --script/--fetch are
    refused.
    """
    with open(jobs_file) as fh:
        jobs = parse_jobs(fh.read())
    states = {label: job_state(into, label) for label, _ in jobs}
    never = [j for j in jobs if states[j[0]] == "pending"]
    failed = [j for j in jobs if states[j[0]] == "failed_start"]
    collected = sum(1 for s in states.values() if s == "collected")
    launch = [j for j in jobs if j in never or (rerun_failed_start and j in failed)]
    if launch and (not script or not fetch):
        raise RuntimeError(
            f"resume: {len(launch)} job(s) in {jobs_file} are to be launched, but "
            "launching them needs --script and --fetch (as a normal sweep does). "
            "Re-run with those, or drop --jobs to only re-attach.")
    return ResumeWork(jobs=jobs, never_started=never, failed_start=failed,
                      collected=collected, launch=launch)


def resume_sweep(cloud=None, region=None, regions=None, into="cloud-sweep",
                 max_parallel=4, jobs_file=None, uploads=(), script=None,
                 fetch=None, max_minutes=30, budget_eur=None, flavor=None,
                 image=None, strict_regions=False, rerun_failed_start=False) -> int:
    """Continue an interrupted sweep: re-attach what is running, then launch what
    never started.

    Two halves:

    1. **Re-attach.** Scan <into>/*/.flux_attach/ for in-flight jobs, re-establish
       the poll loop against each recorded VM, and on completion pull the log +
       artifacts and tear the VM down.
    2. **Continue the jobs file.** Given --jobs (with --script/--fetch), every job
       that is neither in flight nor ended (`job_state`) is launched now, on the
       normal sweep path. A killed orchestrator therefore resumes the whole sweep
       rather than only the VMs that happened to be alive when it died. With
       `rerun_failed_start` the jobs that ended before their script started
       (boot, upload or launch failure) are launched again too, into the same
       per-job folders, so one campaign keeps one results tree. Jobs whose script
       ran are never launched again, whatever their outcome.

    Without --jobs it does step 1 only.
    """
    rc = _reattach_records(cloud, region, into, max_parallel)
    if not jobs_file:
        return rc

    work = _resume_pending(into, jobs_file, script, fetch, rerun_failed_start)
    n_failed = len(work.failed_start)
    if work.failed_start and not rerun_failed_start:
        print(f"resume: {n_failed} job(s) failed before their script started and "
              "are skipped; add --rerun-failed-start to launch them again: "
              + ", ".join(label for label, _ in work.failed_start))
    if not work.launch:
        print(f"resume: jobs file fully accounted for ({work.collected} collected, "
              f"{n_failed} failed before start, "
              f"{len(work.jobs) - work.collected - n_failed} attached above); "
              "nothing left to launch.")
        return rc

    if rerun_failed_start:
        for label, _ in work.failed_start:
            prior = read_outcome(os.path.join(into, label))
            print(f"resume: rerunning [{label}], which failed before start: "
                  f"{prior.status}")
    print(f"resume: launching {len(work.launch)} of {len(work.jobs)} job(s) "
          f"({len(work.never_started)} never started, "
          f"{n_failed if rerun_failed_start else 0} failed before start; "
          f"{work.collected} already collected).")
    rc |= _launch_jobs(
        cloud=cloud, region=region, regions=regions, flavor=flavor, image=image,
        jobs=work.launch, uploads=uploads, script=script, fetch=fetch, into=into,
        max_parallel=max_parallel, max_minutes=max_minutes,
        budget_eur=budget_eur, plan_only=False, strict_regions=strict_regions)
    return rc


def _upload_excludes(src, into):
    """Exclude the sweep's own results dir when it lives inside an upload source.

    <into> holds the fetched artifacts AND the live .flux_attach records of the
    running fleet, so uploading a repo that contains it both re-ships results the
    cloud already has and races records that are being created and deleted
    mid-transfer. `--into` is a caller's choice, so the exclusion has to be
    derived from it rather than assumed to be the default name.
    """
    rel = os.path.relpath(os.path.abspath(into), os.path.abspath(src))
    if rel == os.curdir or rel.startswith(os.pardir) or os.path.isabs(rel):
        return ()          # outside the upload, or IS the upload (nothing to carve out)
    return ("/" + rel.rstrip("/"),)     # a leading '/' anchors it at the transfer root


class UploadFailed(RuntimeError):
    """An attempt's upload to its instance failed; the instance is torn down
    (verified) and the job's pending record names that dead instance. Raised
    only to the retry logic in `_make_run_one`, which either retries the job in
    another region or closes it as a pre-start failure."""

    def __init__(self, region, status):
        super().__init__(status)
        self.region = region
        self.status = status


def _prestart_status(phase, region, status):
    where = region or "(default region)"
    return f"failed before start ({phase}, {where}): {status}"


def _region_fits_one(cloud, shard):
    """Raise unless `shard`'s region has live quota headroom for one more
    instance of its flavor right now (a fresh read: the plan-time cap is stale
    by the time an upload fails)."""
    conn = connect(cloud=cloud, region=shard.region)
    lim = conn.get_compute_limits()
    gq = lambda k: getattr(lim, k, None)
    clamp_concurrency(1, shard.vcpus,
                      gq("total_cores_used"), gq("max_total_cores"),
                      gq("total_instances_used"), gq("max_total_instances"))


def pick_retry_shard(cloud, failed, alternates, budget_eur, _fits=None):
    """Choose the region a job whose upload failed in `failed` is retried in.

    The candidates are the sweep's own resolved region shards (`alternates`),
    other than the failed one, in order: the retry never leaves the regions the
    operator asked for. Under a `--budget`, a candidate whose flavor is dearer
    than the failed shard's (or unpriced) is refused, because the job was priced
    at the failed shard's rate and the guard must still bound it. A candidate is
    taken only if a live quota read fits one more instance.

    Returns (shard, reasons): the chosen shard or None, and one line per refused
    candidate saying why, which becomes the job's status when nothing fits.
    """
    fits = _fits or _region_fits_one
    reasons = []
    price = failed.spec.est_cost_eur_hr
    for s in alternates:
        if s.region == failed.region:
            continue
        alt_price = s.spec.est_cost_eur_hr
        if budget_eur is not None and (alt_price is None or price is None
                                       or alt_price > price):
            reasons.append(f"{s.region}: {s.spec.flavor} at "
                           f"{'an unknown price' if alt_price is None else f'EUR {alt_price:.2f}/hr'}"
                           f" is outside the budget this job was priced at")
            continue
        try:
            fits(cloud, s)
        except Exception as exc:
            reasons.append(f"{s.region}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        return s, reasons
    if not reasons:
        reasons.append("no other region in this sweep")
    return None, reasons


def _make_run_one(cloud, shard, upload_pairs, script, fetch, into, max_minutes,
                  alternates=(), budget_eur=None):
    """Build the per-job runner bound to one region shard (provision -> upload ->
    detached run -> follow -> fetch -> teardown). `upload_pairs` are already-parsed
    (local_src, remote_dest) pairs (`parse_upload_spec`).

    **An upload failure is retried once, in another region.** The upload is the
    step most exposed to a region's network (a booted VM whose port 22 stops
    answering mid-rsync), and it fails before the script starts, so nothing is
    lost by starting over elsewhere. The failed instance is torn down first; the
    retry region is `pick_retry_shard` over `alternates` (the sweep's resolved
    shards). When none fits, or the retry fails too, the job ends as a pre-start
    failure whose status says why, and `sweep --resume --rerun-failed-start`
    launches it again into the same folder.
    """
    cap_seconds = max_minutes * 60
    remote = os.path.basename(script)

    def _attempt(job, s, *, may_retry, note=""):
        label, params = job
        region = s.region
        name = f"flux-compute-sweep-{uuid.uuid4().hex[:8]}"
        dest = os.path.join(into, label)

        # BEFORE the instance boots: record the job under its (locally generated)
        # instance name, so a launcher killed during boot leaves a VM that
        # `--resume` can still find and tear down instead of an unnamed orphan.
        os.makedirs(dest, exist_ok=True)
        _clear_previous_attempt(dest)
        _write_pending_record(
            dest, label=label, cloud=cloud, region=region, name=name,
            remote_script=remote, fetch=fetch, into=into, cap_seconds=cap_seconds)
        phase = "boot"          # the step in flight; None once the script runs
        try:
            conn = connect(cloud=cloud, region=region)
            with _gpu_instance(conn, s.spec, name,
                               ttl_minutes=ttl_minutes_for(max_minutes)) as (server, ip, keyfile):
                phase = "upload"
                for local, remote_dest in upload_pairs:
                    _rsync_up(local, ip, keyfile, remote_dest,
                              extra_excludes=_upload_excludes(local, into))
                _scp_up(script, ip, keyfile, remote)

                env = f"FLUX_LABEL={shlex.quote(label)} FLUX_JOB={shlex.quote(params)}"

                # Launch the job detached so a laptop sleep cannot HUP it, then
                # upgrade the record with the boot-time facts so even a hard kill
                # of THIS process can re-attach to the still-running VM
                # (`flux-compute sweep --resume`). The record carries THIS
                # attempt's region, so --resume reconnects to the right one.
                # A launch that raises is a pre-start failure: the instance is
                # torn down at once, so nothing the script might have begun can
                # be collected.
                phase = "launch"
                started = time.time()
                _launch_detached(ip, keyfile, remote, cap_seconds, env_prefix=env)
                phase = None
                _write_attach_record(
                    dest, label=label, cloud=cloud, region=region, name=name,
                    server_id=server.id, ip=ip, keyfile=keyfile, remote_script=remote,
                    fetch=fetch, into=into, cap_seconds=cap_seconds)

                # Follow to completion, tolerant of reconnection (sleep/wake): a
                # failed poll is retried with backoff, only the local deadline
                # (remote cap + grace) aborts. A sustained SSH blackout escalates
                # through the stuck handler instead of retrying silently.
                outcome = follow_detached_job(
                    ip, keyfile, cap_seconds,
                    deadline_s=cap_seconds + LOCAL_GRACE_S,
                    poll_interval=POLL_INTERVAL_SWEEP_S,
                    on_warn=_job_warn(label),
                    on_stuck=make_stuck_handler(conn, name, label=label))

                rc, status = _finalize(ip, keyfile, dest, fetch, outcome,
                                       elapsed_s=time.time() - started,
                                       cap_seconds=cap_seconds)
                status += note
                _close_job(dest, _outcome(label, region, rc, status))
                return (label, rc, status)
        except TeardownStrandError as exc:
            # The instance could not be verified gone, so its record MUST survive:
            # it is the handle `--resume` uses to finish the teardown. The
            # outcome is still written: this attempt has ended.
            status = _failure_status(exc) + note
            _record_outcome(dest, _outcome(label, region, -1, status, phase))
            return (label, -1, status)
        except Exception as exc:
            # Teardown ran (the context manager's finally): the attempt is
            # finished, badly.
            status = _failure_status(exc)
            if phase == "upload" and may_retry:
                raise UploadFailed(region, status) from exc
            if phase is not None:
                status = _prestart_status(phase, region, status)
            status += note
            # Drop the record so --resume neither re-attaches to a dead VM nor
            # counts the job as still in flight; the outcome says whether
            # --rerun-failed-start may launch it again.
            _close_unrecoverable(dest, _outcome(label, region, -1, status, phase))
            return (label, -1, status)

    def _one(job):
        label = job[0]
        try:
            return _attempt(job, shard, may_retry=True)
        except UploadFailed as first:
            alt, reasons = pick_retry_shard(cloud, shard, alternates, budget_eur)
            if alt is None:
                status = (_prestart_status("upload", first.region, first.status)
                          + "; not retried: " + "; ".join(reasons))
                _close_unrecoverable(os.path.join(into, label),
                                     _outcome(label, first.region, -1, status, "upload"))
                return (label, -1, status)
            print(f"  [{first.region}/{label}] upload failed ({first.status}); "
                  f"instance torn down; retrying once in {alt.region}")
            return _attempt(job, alt, may_retry=False,
                            note=f" [retried in {alt.region} after an upload "
                                 f"failure in {first.region}]")

    return _one


@dataclass(frozen=True)
class LaunchPlan:
    """Everything a launch is decided on, resolved before any instance exists:
    the parsed uploads, every region shard, the shards that will run (with their
    jobs dealt), and the whole sweep's worst-case EUR (None when unpriced and no
    budget is set)."""

    upload_pairs: list
    shards: tuple
    live: tuple
    worst_case_eur: float | None


def _plan_launch(*, cloud, region, regions, flavor, image, jobs, uploads,
                 max_parallel, max_minutes, budget_eur, strict_regions) -> LaunchPlan:
    """The refusing half of a launch: validate uploads, resolve and clamp the
    region shards, deal the jobs and run the budget guard. Raises on every refusal
    a sweep can meet before it provisions, and launches nothing, so it is safe to
    run in the foreground ahead of `--detach` (`preflight_sweep`)."""
    if regions and region:
        raise RuntimeError(
            "pass either --region (one region) or --regions (several), not both")
    # Parse (and validate) uploads BEFORE anything is provisioned: a mistyped
    # --upload should cost nothing, not a booted instance.
    upload_pairs = [parse_upload_spec(u) for u in uploads]

    # One shard per region; a lone --region (or the clouds.yaml default) is the
    # single-shard case of the same path.
    targets = parse_regions(regions) if regions else [region]
    shards, drops = _prepare_shards(cloud, targets, flavor, image, max_parallel)

    # Graceful-degrade pre-flight (the default): a region that cannot fit >=1
    # instance of the chosen flavor is dropped with a warning naming its
    # occupants and headroom, and the wave allocation is recomputed over the
    # survivors. The sweep refuses only when NONE fit, or under --strict-regions
    # (exact-width mode). This turns a partial-capacity situation into a running
    # sweep instead of an all-or-nothing failure.
    if drops and (strict_regions or not shards):
        raise RuntimeError(_region_refusal(cloud, drops, all_dropped=not shards))
    if drops:
        _warn_region_drops(cloud, drops, surviving=len(shards))

    # --max-parallel is the GLOBAL live-instance ceiling; each region is
    # additionally bounded by its own quota headroom.
    alloc = allocate_concurrency([s.cap for s in shards], max_parallel)
    live = [s.with_plan(a, []) for s, a in zip(shards, alloc) if a > 0]
    if not live:
        raise RuntimeError("no region has any quota headroom for this sweep")
    dealt = shard_jobs(jobs, [s.concurrency for s in live])
    live = [s.with_plan(s.concurrency, j) for s, j in zip(live, dealt)]

    wc = budget_guard_shards(
        [(s.spec.flavor, s.spec.est_cost_eur_hr, len(s.jobs)) for s in live],
        max_minutes, budget_eur)
    return LaunchPlan(upload_pairs=upload_pairs, shards=tuple(shards),
                      live=tuple(live), worst_case_eur=wc)


def _launch_jobs(*, cloud, region, regions, flavor, image, jobs, uploads, script,
                 fetch, into, max_parallel, max_minutes, budget_eur, plan_only,
                 strict_regions) -> int:
    """Plan and run a list of jobs: shard across regions, guard the budget, fan
    out. Shared by a fresh `sweep` and by `sweep --resume --jobs`, so a resumed
    sweep launches its remaining jobs through exactly the same quota clamp,
    budget guard and region sharding as the original."""
    plan = _plan_launch(
        cloud=cloud, region=region, regions=regions, flavor=flavor, image=image,
        jobs=jobs, uploads=uploads, max_parallel=max_parallel,
        max_minutes=max_minutes, budget_eur=budget_eur,
        strict_regions=strict_regions)
    upload_pairs, shards, live, wc = (
        plan.upload_pairs, plan.shards, plan.live, plan.worst_case_eur)
    total_conc = sum(s.concurrency for s in live)
    tail = f"worst-case ~EUR {wc:.2f}" if wc is not None else "price n/a"

    if len(live) == 1 and len(shards) == 1:
        _print_plan(live[0].spec)
        s = live[0]
        if s.concurrency < max_parallel:
            print(f"quota clamp: concurrency {max_parallel} -> {s.concurrency} "
                  f"({s.spec.flavor} = {s.vcpus} vCPU each)")
    else:
        print(f"multi-region sweep across {len(live)} of {len(shards)} region(s) "
              f"(quota is per region, so regions widen the fleet):")
        for s in live:
            cost = (f"EUR {s.spec.est_cost_eur_hr:.2f}/hr"
                    if s.spec.est_cost_eur_hr is not None else "price n/a")
            print(f"  {s.region:<14} {s.spec.flavor:<12} ({s.vcpus} vCPU, {cost})  "
                  f"{s.concurrency} parallel (cap {s.cap})  {len(s.jobs)} jobs")
        idle = [s.region for s in shards if s.region not in {l.region for l in live}]
        if idle:
            print(f"  unused (global --max-parallel {max_parallel} reached): {', '.join(idle)}")

    print(f"sweep: {len(jobs)} jobs, up to {total_conc} parallel, "
          f"per-job cap {max_minutes} min; {tail}")

    if plan_only:
        print("plan only: no instances launched.")
        return 0

    os.makedirs(into, exist_ok=True)

    def _log_for(region):
        def _log(r):
            label, rc, status = r
            tag = f"{region}/" if len(live) > 1 else ""
            print(f"  [{tag}{label}] rc={rc} {status}")
        return _log

    def _run_shard(s):
        return _fan_out(list(s.jobs),
                        _make_run_one(cloud, s, upload_pairs, script, fetch, into,
                                      max_minutes, alternates=shards,
                                      budget_eur=budget_eur),
                        s.concurrency, on_result=_log_for(s.region))

    # Shards run concurrently; each stays within its own region's allocation, so
    # the live fleet never exceeds the global ceiling.
    if len(live) == 1:
        results = _run_shard(live[0])
    else:
        results = []
        with ThreadPoolExecutor(max_workers=len(live)) as ex:
            for fut in as_completed([ex.submit(_run_shard, s) for s in live]):
                results.extend(fut.result())

    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"sweep done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1


def run_sweep(cloud=None, region=None, regions=None, flavor=None, uploads=(), script=None,
              jobs_file=None, fetch=None, into="cloud-sweep",
              max_parallel=4, max_minutes=30, budget_eur=None, image=None,
              plan_only=False, resume=False, strict_regions=False,
              rerun_failed_start=False) -> int:
    if resume:
        # --resume re-attaches in-flight jobs and, when a jobs file is given,
        # launches the ones that never started (and, with rerun_failed_start,
        # the ones that failed before their script started).
        return resume_sweep(
            cloud=cloud, region=region, regions=regions, into=into,
            max_parallel=max_parallel, jobs_file=jobs_file, uploads=uploads,
            script=script, fetch=fetch, max_minutes=max_minutes,
            budget_eur=budget_eur, flavor=flavor, image=image,
            strict_regions=strict_regions, rerun_failed_start=rerun_failed_start)
    if rerun_failed_start:
        raise RuntimeError("--rerun-failed-start needs --resume (it continues a sweep)")
    jobs = _fresh_jobs(jobs_file, script, fetch, plan_only)
    return _launch_jobs(
        cloud=cloud, region=region, regions=regions, flavor=flavor, image=image,
        jobs=jobs, uploads=uploads, script=script, fetch=fetch, into=into,
        max_parallel=max_parallel, max_minutes=max_minutes, budget_eur=budget_eur,
        plan_only=plan_only, strict_regions=strict_regions)


def _fresh_jobs(jobs_file, script, fetch, plan_only):
    """Validate a fresh sweep's spec and return its parsed jobs, or raise."""
    if not jobs_file:
        raise RuntimeError("sweep needs --jobs (a jobs file)")
    if not plan_only:
        if not script:
            raise RuntimeError("sweep needs --script (the per-job job script)")
        if not fetch:
            raise RuntimeError("sweep needs --fetch (home-relative artifact dir per job)")
    with open(jobs_file) as fh:
        return parse_jobs(fh.read())


@dataclass(frozen=True)
class Preflight:
    """What a sweep that passed its pre-flight is about to launch."""

    n_jobs: int
    worst_case_eur: float | None


def preflight_sweep(cloud=None, region=None, regions=None, flavor=None, uploads=(),
                    script=None, jobs_file=None, fetch=None, into="cloud-sweep",
                    max_parallel=4, max_minutes=30, budget_eur=None, image=None,
                    plan_only=False, resume=False, strict_regions=False,
                    rerun_failed_start=False):
    """Run every refusal `run_sweep` would meet before launching, and launch nothing.

    Takes `run_sweep`'s arguments. Raises RuntimeError on a bad spec, an unfit
    region set or a budget refusal; returns a `Preflight` for the jobs that would
    launch, or None when nothing would (a `--resume` with no jobs file, or one
    whose jobs are all accounted for). The CLI runs it before `--detach` forks,
    so a refused sweep is refused on the terminal, not inside the log. The
    detached child still runs the full sweep, and with it the same checks.

    A `--resume` runs the spec checks, `_resume_pending` (counting the
    failed-start reruns when asked) and the budget guard on
    static prices only (`_resume_static_plan`). Its region fit is left to the
    child: the in-flight fleet still holds the quota it frees on collection, so
    a live quota read here would refuse exactly the jobs the child can launch.
    """
    if resume:
        if not jobs_file:
            return None
        jobs = _resume_pending(into, jobs_file, script, fetch,
                               rerun_failed_start).launch
        if not jobs:
            return None
        return _resume_static_plan(
            region=region, regions=regions, flavor=flavor, jobs=jobs,
            uploads=uploads, max_minutes=max_minutes, budget_eur=budget_eur)
    jobs = _fresh_jobs(jobs_file, script, fetch, plan_only)
    plan = _plan_launch(
        cloud=cloud, region=region, regions=regions, flavor=flavor, image=image,
        jobs=jobs, uploads=uploads, max_parallel=max_parallel,
        max_minutes=max_minutes, budget_eur=budget_eur,
        strict_regions=strict_regions)
    return Preflight(n_jobs=len(jobs), worst_case_eur=plan.worst_case_eur)


def _resume_static_plan(*, region, regions, flavor, jobs, uploads, max_minutes,
                        budget_eur):
    """The resume pre-flight: spec checks and the budget guard on static prices,
    with no connection and no quota read.

    The price is the static table's (`classify`). With --flavor that is the
    flavor the child launches. Without it the child picks the cheapest healthy
    GPU each region offers, unknown offline, so every job is charged at the
    dearest priced flavor that default can pick: an upper bound, since a budget
    guard that can under-estimate is not a cap."""
    if regions and region:
        raise RuntimeError(
            "pass either --region (one region) or --regions (several), not both")
    for u in uploads:
        parse_upload_spec(u)
    if flavor:
        label, price = flavor, classify(flavor).price_eur_hr
    else:
        label, price = dearest_default_flavor()
    try:
        wc = budget_guard_shards([(label, price, len(jobs))], max_minutes, budget_eur)
    except RuntimeError as exc:
        if flavor:
            raise
        raise RuntimeError(
            f"{exc} (no --flavor: every job is priced at the dearest allowed "
            f"flavor, {label}; pass --flavor to bound it by the one you launch)"
        ) from exc
    return Preflight(n_jobs=len(jobs), worst_case_eur=wc)


# --- wait for the next job of a campaign to end --------------------------------

def _campaign_state(into):
    """{label: (attach record or None, outcome or None)} for every job folder
    under <into> that holds either. The record is read BEFORE the outcome: a job
    writes its outcome before it drops its record (`_close_job`), so a read in
    this order never sees a job with neither that has in fact ended."""
    state = {}
    for label in sorted(os.listdir(into)):
        dest = os.path.join(into, label)
        if not os.path.isdir(dest):
            continue
        rec_path = os.path.join(dest, ATTACH_DIR, ATTACH_RECORD)
        rec = None
        if os.path.isfile(rec_path):
            with open(rec_path) as fh:
                rec = detach.AttachRecord.from_json(fh.read())
        out = read_outcome(dest)
        if rec is not None or out is not None:
            state[label] = (rec, out)
    return state


def _running(state):
    """Labels whose job is running: an attach record with no outcome written
    since it launched (a teardown strand keeps its record AND writes its
    outcome, and has ended)."""
    return {label for label, (rec, out) in state.items()
            if rec is not None and (out is None or out.ended_epoch < rec.launch_epoch)}


def format_outcome_line(outcome):
    """The one line `--wait-next` prints: label, region, rc, status."""
    region = outcome.region or "(default region)"
    return f"{outcome.label} region={region} rc={outcome.rc} {outcome.status}"


def wait_next_job(into, *, poll_interval=WAIT_POLL_S, _sleep=time.sleep, emit=print):
    """Block until the next job of the campaign under <into> ends; print its line.

    "Next" is relative to this call: the outcomes already on disk when it starts
    are the baseline, and the first job whose outcome appears or changes after
    it (a job running now, or one the sweep launches while this waits) is the
    one reported. Read from the campaign's local state only (attach records and
    outcome records), never from a log, so no earlier failure is replayed.

    Returns 0 once a job has ended and its line is printed, whatever that job's
    own rc (the line carries it). Raises RuntimeError, which the CLI turns into a
    nonzero exit, when the campaign has no running job at the start, and when a
    running job's record vanishes with no outcome (removed by hand or by
    `reap --sweep-local`).
    """
    if not os.path.isdir(into):
        raise RuntimeError(f"--wait-next: no campaign folder {into}/")
    state = _campaign_state(into)
    baseline = {label: out.ended_epoch for label, (_, out) in state.items()
                if out is not None}
    running = _running(state)
    if not running:
        raise RuntimeError(
            f"--wait-next: no running jobs under {into}/ (no attach record without "
            "an outcome); nothing to wait for")
    while True:
        _sleep(poll_interval)
        state = _campaign_state(into)
        ended = [out for label, (_, out) in state.items()
                 if out is not None and baseline.get(label) != out.ended_epoch]
        if ended:
            emit(format_outcome_line(min(ended, key=lambda o: o.ended_epoch)))
            return 0
        vanished = sorted(label for label in running
                          if label not in state or state[label][0] is None)
        if vanished:
            raise RuntimeError(
                f"--wait-next: job(s) {', '.join(vanished)} under {into}/ lost their "
                "attach record without writing an outcome (removed by hand or by "
                "`flux-compute reap --sweep-local`?); their end is not known")
        running = _running(state)
