"""Fan out a parameter sweep across GPU instances, with a hard cost ceiling.

Each job runs on its own ephemeral instance (provision -> upload -> run the
consumer's script with the job's params in $FLUX_LABEL/$FLUX_JOB -> fetch
artifacts -> teardown), up to --max-parallel at once. Two guards bound spend: a
pre-flight worst-case check (jobs x price x per-job wall cap) refuses to start
above the budget, and each job's remote exec is killed at --max-minutes so a
hung job cannot run up the bill. Teardown is per-job and unconditional.
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
)


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


def run_sweep(cloud=None, region=None, flavor=None, uploads=(), script=None,
              jobs_file=None, fetch=None, into="cloud-sweep",
              max_parallel=4, max_minutes=30, budget_eur=None, image=None) -> int:
    if not script:
        raise RuntimeError("sweep needs --script (the per-job job script)")
    if not jobs_file:
        raise RuntimeError("sweep needs --jobs (a jobs file)")
    if not fetch:
        raise RuntimeError("sweep needs --fetch (home-relative artifact dir per job)")

    with open(jobs_file) as fh:
        jobs = parse_jobs(fh.read())

    conn0 = connect(cloud=cloud, region=region)
    spec = resolve_spec(conn0, _region(conn0, region), flavor=flavor, image=image)
    _print_plan(spec)

    wc = worst_case_eur(len(jobs), spec.est_cost_eur_hr, max_minutes)
    tail = f"worst-case ~EUR {wc:.2f}" if wc is not None else "price n/a"
    print(f"sweep: {len(jobs)} jobs, up to {max_parallel} parallel, "
          f"per-job cap {max_minutes} min; {tail}")
    if budget_eur is not None and wc is not None and wc > budget_eur:
        raise RuntimeError(
            f"worst-case ~EUR {wc:.2f} exceeds budget EUR {budget_eur:.2f}; "
            "lower --max-minutes, run fewer jobs, or raise --budget.")

    os.makedirs(into, exist_ok=True)

    def _one(job):
        label, params = job
        conn = connect(cloud=cloud, region=region)
        try:
            with _gpu_instance(conn, spec, f"flux-compute-sweep-{uuid.uuid4().hex[:8]}") as (ip, keyfile):
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
            return (label, -1, f"error: {type(exc).__name__}: {str(exc)[:100]}")

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for fut in as_completed(futures):
            label, rc, status = fut.result()
            print(f"  [{label}] rc={rc} {status}")
            results.append((label, rc, status))

    ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"sweep done: {ok}/{len(results)} ok; artifacts under {into}/")
    return 0 if ok == len(results) else 1
