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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .auth import connect
from .launch import resolve_spec
from .provision import (
    _gpu_instance, _print_plan, _region, _rsync_down, _rsync_up, _scp_up, _ssh,
    ttl_minutes_for,
)
from .reap import warn_strays


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


def run_sweep(cloud=None, region=None, flavor=None, uploads=(), script=None,
              jobs_file=None, fetch=None, into="cloud-sweep",
              max_parallel=4, max_minutes=30, budget_eur=None, image=None,
              plan_only=False) -> int:
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
        try:
            with _gpu_instance(conn, spec, f"flux-compute-sweep-{uuid.uuid4().hex[:8]}",
                               ttl_minutes=ttl_minutes_for(max_minutes)) as (_server, ip, keyfile):
                for local in uploads:
                    base = os.path.basename(os.path.abspath(local.rstrip("/")))
                    _rsync_up(local, ip, keyfile, base)
                remote = os.path.basename(script)
                _scp_up(script, ip, keyfile, remote)
                env = f"FLUX_LABEL={shlex.quote(label)} FLUX_JOB={shlex.quote(params)}"
                res = _ssh(ip, keyfile, f"chmod +x ~/{remote} && {env} bash -lc '~/{remote}'",
                           timeout=max_minutes * 60, capture=True)
                dest = os.path.join(into, label)
                os.makedirs(dest, exist_ok=True)
                with open(os.path.join(dest, "job.log"), "w") as lf:
                    lf.write(res.stdout or "")
                    if res.stderr:
                        lf.write("\n--- stderr ---\n" + res.stderr)
                _rsync_down(ip, keyfile, fetch, dest)
                return (label, res.returncode, "ok" if res.returncode == 0 else "job nonzero")
        except Exception as exc:
            return (label, -1, _failure_status(exc))

    def _log(r):
        label, rc, status = r
        print(f"  [{label}] rc={rc} {status}")

    results = _fan_out(jobs, _one, effective, on_result=_log)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"sweep done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1
