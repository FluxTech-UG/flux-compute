"""Fan out a parameter sweep across ephemeral instances, with a hard cost ceiling.

Each job runs on its own ephemeral instance (provision -> upload -> run the
consumer's script with the job's params in $FLUX_LABEL/$FLUX_JOB -> fetch
artifacts -> teardown), up to --max-parallel at once. Three guards bound spend
and blast radius: a pre-flight worst-case check (jobs x price x per-job wall cap)
refuses to start above the budget; the effective concurrency is clamped to the
compute-quota headroom for the flavor, so the fleet cannot outrun the quota into
create failures; and each job's remote exec is killed at --max-minutes so a hung
job cannot run up the bill. Teardown is per-job and unconditional.
"""
from __future__ import annotations

import os
import shlex
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import detach
from .auth import connect
from .launch import resolve_spec
from .provision import (
    LOCAL_GRACE_S, POLL_INTERVAL_SWEEP_S, RESUME_MIN_DEADLINE_S,
    _gpu_instance, _launch_detached, _print_plan, _region, _rsync_down,
    _rsync_up, _scp_up, _server_by_name_or_id, follow_detached_job, pull_job_log,
    teardown_by_name, ttl_minutes_for,
)
from .reap import warn_strays

# Per-label attach state, persisted under <into>/<label>/.flux_attach/ so a hard
# kill of the orchestrator can re-attach to the still-running VM. The dir's
# presence marks a job as not-yet-finalized; it is removed on clean teardown.
ATTACH_DIR = ".flux_attach"
ATTACH_RECORD = "record.json"
ATTACH_KEY = "id_key"


def parse_jobs(text):
    """Parse a jobs file: each non-blank, non-# line is 'LABEL = PARAMS'.

    Without '=', the whole line is both label and params. Labels must be unique
    and filesystem-safe (they name the per-job artifact subdir). PARAMS reaches
    the job script as $FLUX_JOB.
    """
    jobs = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
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


def budget_guard(flavor, price_eur_hr, n_jobs, max_minutes, budget_eur):
    """Enforce --budget against the worst-case sweep spend, or raise (fail-fast).

    Returns the worst-case EUR (None when the flavor has no known price and no
    budget is set). With a budget set, an unpriced flavor is refused rather than
    silently skipping the guard: a money cap that cannot see the price is not a
    cap. Add the flavor's price to _KNOWN_PRICE_EUR_HR (flavors.py) to price it.
    """
    wc = worst_case_eur(n_jobs, price_eur_hr, max_minutes)
    if budget_eur is not None:
        if price_eur_hr is None:
            raise RuntimeError(
                f"--budget EUR {budget_eur:.2f} was set, but flavor {flavor!r} has no "
                f"known price, so worst-case spend cannot be bounded. Add its price to "
                f"_KNOWN_PRICE_EUR_HR in flavors.py, or drop --budget to run unguarded."
            )
        if wc > budget_eur:
            raise RuntimeError(
                f"worst-case ~EUR {wc:.2f} exceeds budget EUR {budget_eur:.2f}; "
                "lower --max-minutes, run fewer jobs, or raise --budget."
            )
    return wc


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


def _write_attach_record(dest, *, label, cloud, region, name, server_id, ip,
                         keyfile, remote_script, fetch, into, cap_seconds):
    """Persist the attach record plus a durable copy of the ephemeral private key
    under <dest>/.flux_attach/, so a hard kill of this process can re-attach to
    the still-running VM. The key copy is 0600 and is removed with the record on
    clean teardown."""
    adir = _attach_dir(dest)
    os.makedirs(adir, exist_ok=True)
    key_copy = os.path.join(adir, ATTACH_KEY)
    shutil.copyfile(keyfile, key_copy)
    os.chmod(key_copy, 0o600)
    rec = detach.AttachRecord(
        label=label, cloud=cloud, region=region, name=name, server_id=server_id,
        ip=ip, keyfile=key_copy, remote_script=remote_script, fetch=fetch,
        into=into, cap_seconds=cap_seconds, launch_epoch=time.time())
    with open(os.path.join(adir, ATTACH_RECORD), "w") as fh:
        fh.write(rec.to_json())
    return rec


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


def _status_for_outcome(outcome):
    """Map a follow outcome to the sweep's (rc, status) record fields."""
    if outcome.reason == "deadline":
        return -1, "LOCAL DEADLINE (job unfinished; server torn down)"
    rc = outcome.rc
    if rc == 0:
        return 0, "ok"
    if rc in (124, 137):
        return rc, "job timed out (remote cap)"
    return rc, "job nonzero"


def _finalize(ip, keyfile, dest, fetch, outcome):
    """Collect a followed job: pull the authoritative job.log always, and the
    artifact dir on completion. Returns (rc, status). Teardown is the caller's."""
    pull_job_log(ip, keyfile, os.path.join(dest, "job.log"))
    if outcome.reason == "done":
        _rsync_down(ip, keyfile, fetch, dest)
    return _status_for_outcome(outcome)


def resume_sweep(cloud=None, region=None, into="cloud-sweep", max_parallel=4) -> int:
    """Re-attach to still-running detached jobs after a full orchestrator restart.

    Scans <into>/*/.flux_attach/ for in-flight jobs, re-establishes the poll loop
    against each recorded VM, and on completion pulls the log + artifacts and tears
    the VM down -- the recovery the sleep incident actually needed. A VM already
    gone (reaped/torn down) is recorded as lost and its record cleared.
    """
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

    def _resume_one(rec):
        dest = os.path.join(rec.into, rec.label)
        try:
            conn = connect(cloud=rec.cloud or cloud, region=rec.region or region)
            server = _server_by_name_or_id(conn, rec.name, rec.server_id)
            if server is None:
                _clear_attach_record(dest)
                return (rec.label, -1, "lost: VM gone (reaped?); cannot re-attach")
            # The original deadline may already have elapsed while the orchestrator
            # was down, yet the job (or its remote-cap kill) may have finished and
            # just needs collecting -- so floor the follow at RESUME_MIN_DEADLINE_S.
            remaining = rec.launch_epoch + rec.cap_seconds + LOCAL_GRACE_S - time.time()
            deadline = max(remaining, RESUME_MIN_DEADLINE_S)
            outcome = follow_detached_job(
                rec.ip, rec.keyfile, rec.cap_seconds,
                deadline_s=deadline, poll_interval=POLL_INTERVAL_SWEEP_S)
            rc, status = _finalize(rec.ip, rec.keyfile, dest, rec.fetch, outcome)
            teardown_by_name(conn, rec.name, rec.server_id)
            _clear_attach_record(dest)
            return (rec.label, rc, status)
        except Exception as exc:
            return (rec.label, -1, _failure_status(exc))

    def _log(r):
        label, rc, status = r
        print(f"  [{label}] rc={rc} {status}")

    results = _fan_out(records, _resume_one, max(1, max_parallel), on_result=_log)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"resume done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1


def run_sweep(cloud=None, region=None, flavor=None, uploads=(), script=None,
              jobs_file=None, fetch=None, into="cloud-sweep",
              max_parallel=4, max_minutes=30, budget_eur=None, image=None,
              plan_only=False, resume=False) -> int:
    if resume:
        return resume_sweep(cloud=cloud, region=region, into=into,
                            max_parallel=max_parallel)
    if not jobs_file:
        raise RuntimeError("sweep needs --jobs (a jobs file)")
    if not plan_only:
        if not script:
            raise RuntimeError("sweep needs --script (the per-job job script)")
        if not fetch:
            raise RuntimeError("sweep needs --fetch (home-relative artifact dir per job)")

    with open(jobs_file) as fh:
        jobs = parse_jobs(fh.read())

    conn0 = connect(cloud=cloud, region=region)
    warn_strays(conn0)
    spec = resolve_spec(conn0, _region(conn0, region), flavor=flavor, image=image)
    _print_plan(spec)

    wc = budget_guard(spec.flavor, spec.est_cost_eur_hr, len(jobs), max_minutes, budget_eur)
    tail = f"worst-case ~EUR {wc:.2f}" if wc is not None else "price n/a"

    # Clamp concurrency to compute-quota headroom for this flavor; --max-parallel
    # stays the user ceiling. Fails fast if not even one instance fits.
    lim = conn0.get_compute_limits()
    gq = lambda k: getattr(lim, k, None)
    vcpus = _flavor_vcpus(conn0, spec.flavor)
    cores_free = (gq("max_total_cores") or 0) - (gq("total_cores_used") or 0)
    inst_free = (gq("max_total_instances") or 0) - (gq("total_instances_used") or 0)
    effective = clamp_concurrency(
        max_parallel, vcpus,
        gq("total_cores_used"), gq("max_total_cores"),
        gq("total_instances_used"), gq("max_total_instances"),
    )
    if effective < max_parallel:
        print(f"quota clamp: concurrency {max_parallel} -> {effective} "
              f"(headroom {cores_free} cores / {inst_free} instances; "
              f"{spec.flavor} = {vcpus} vCPU each)")
    print(f"sweep: {len(jobs)} jobs, up to {effective} parallel, "
          f"per-job cap {max_minutes} min; {tail}")

    if plan_only:
        print("plan only: no instances launched.")
        return 0

    os.makedirs(into, exist_ok=True)

    def _one(job):
        label, params = job
        conn = connect(cloud=cloud, region=region)
        name = f"flux-compute-sweep-{uuid.uuid4().hex[:8]}"
        try:
            with _gpu_instance(conn, spec, name,
                               ttl_minutes=ttl_minutes_for(max_minutes)) as (server, ip, keyfile):
                for local in uploads:
                    base = os.path.basename(os.path.abspath(local.rstrip("/")))
                    _rsync_up(local, ip, keyfile, base)
                remote = os.path.basename(script)
                _scp_up(script, ip, keyfile, remote)

                dest = os.path.join(into, label)
                os.makedirs(dest, exist_ok=True)
                cap_seconds = max_minutes * 60
                env = f"FLUX_LABEL={shlex.quote(label)} FLUX_JOB={shlex.quote(params)}"

                # Launch the job detached so a laptop sleep cannot HUP it, then
                # persist an attach record so even a hard kill of THIS process can
                # re-attach to the still-running VM (`flux-compute sweep --resume`).
                _launch_detached(ip, keyfile, remote, cap_seconds, env_prefix=env)
                _write_attach_record(
                    dest, label=label, cloud=cloud, region=region, name=name,
                    server_id=server.id, ip=ip, keyfile=keyfile, remote_script=remote,
                    fetch=fetch, into=into, cap_seconds=cap_seconds)

                # Follow to completion, tolerant of reconnection (sleep/wake): a
                # failed poll is retried with backoff, only the local deadline
                # (remote cap + grace) aborts.
                outcome = follow_detached_job(
                    ip, keyfile, cap_seconds,
                    deadline_s=cap_seconds + LOCAL_GRACE_S,
                    poll_interval=POLL_INTERVAL_SWEEP_S)

                rc, status = _finalize(ip, keyfile, dest, fetch, outcome)
                _clear_attach_record(dest)   # clean finalize: drop record + key
                return (label, rc, status)
        except Exception as exc:
            return (label, -1, _failure_status(exc))

    def _log(r):
        label, rc, status = r
        print(f"  [{label}] rc={rc} {status}")

    results = _fan_out(jobs, _one, effective, on_result=_log)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"sweep done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1
