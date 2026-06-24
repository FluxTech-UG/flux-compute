"""flux-compute command-line entry point."""
from __future__ import annotations

import argparse
import sys


def _add_target_args(p):
    p.add_argument("--cloud", default=None,
                   help="clouds.yaml entry name (else OS_* env vars are used).")
    p.add_argument("--region", default=None,
                   help="Region override (else OS_REGION_NAME / clouds.yaml).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="flux-compute",
        description="Run FluxTech simulations on OVH Public Cloud GPU instances.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor",
        help="Verify OVH OpenStack API access; list credit-eligible, fp64-healthy GPU flavors.",
    )
    _add_target_args(doctor)

    pre = sub.add_parser(
        "preflight",
        help="Read-only launch-readiness check (quota, network, keypair, NVIDIA image).",
    )
    _add_target_args(pre)

    run = sub.add_parser(
        "run",
        help="Provision a GPU instance, run a job, fetch artifacts, tear down.",
    )
    _add_target_args(run)
    run.add_argument("--plan", action="store_true",
                     help="Resolve and print the launch spec without launching (dry run).")
    run.add_argument("--smoke", action="store_true",
                     help="Provision, confirm the GPU is visible (nvidia-smi), and tear down. Billable.")
    run.add_argument("--upload", action="append", default=[], metavar="DIR",
                     help="Local dir to rsync to ~/<name> on the instance (repeatable).")
    run.add_argument("--script", default=None, metavar="FILE",
                     help="Local bash script uploaded and run on the instance (your setup + job).")
    run.add_argument("--fetch", action="append", default=[], metavar="REMOTE:LOCAL",
                     help="Copy REMOTE (home-relative dir) back to LOCAL after the job (repeatable).")
    run.add_argument("--keep", action="store_true",
                     help="Leave the instance running after the job for debugging (you must tear it down).")
    run.add_argument("--flavor", default=None,
                     help="Override the flavor (else the cheapest fp64-healthy GPU available).")

    sweep = sub.add_parser(
        "sweep",
        help="Fan out a parameter sweep across GPU instances with a hard cost cap.",
    )
    _add_target_args(sweep)
    sweep.add_argument("--upload", action="append", default=[], metavar="DIR",
                       help="Local dir to rsync to ~/<name> on each instance (repeatable).")
    sweep.add_argument("--script", default=None, metavar="FILE",
                       help="Per-job bash script; reads $FLUX_LABEL and $FLUX_JOB.")
    sweep.add_argument("--jobs", default=None, metavar="FILE",
                       help="Jobs file: each line 'LABEL = PARAMS' (PARAMS -> $FLUX_JOB).")
    sweep.add_argument("--fetch", default=None, metavar="REMOTE",
                       help="Home-relative artifact dir pulled per job into <into>/<label>/.")
    sweep.add_argument("--into", default="cloud-sweep", metavar="DIR",
                       help="Local base dir for fetched artifacts (default: cloud-sweep).")
    sweep.add_argument("--flavor", default=None,
                       help="Override the flavor (else the cheapest fp64-healthy GPU available).")
    sweep.add_argument("--max-parallel", type=int, default=4,
                       help="Max instances alive at once (default 4).")
    sweep.add_argument("--max-minutes", type=int, default=30,
                       help="Per-job remote wall-clock cap; kills a hung job (default 30).")
    sweep.add_argument("--budget", type=float, default=None, metavar="EUR",
                       help="Refuse to start if worst-case cost (jobs x price x cap) exceeds this.")

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            from .doctor import run_doctor
            return run_doctor(cloud=args.cloud, region=args.region)

        if args.command == "preflight":
            from .preflight import run_preflight
            return run_preflight(cloud=args.cloud, region=args.region)

        if args.command == "run":
            if args.plan:
                from .launch import plan
                return plan(cloud=args.cloud, region=args.region, flavor=args.flavor)
            if args.smoke:
                from .provision import smoke_test
                return smoke_test(cloud=args.cloud, region=args.region, flavor=args.flavor)
            if args.script or args.upload:
                from .provision import run_job
                return run_job(cloud=args.cloud, region=args.region, flavor=args.flavor,
                               uploads=args.upload, script=args.script, fetch=args.fetch,
                               keep=args.keep)
            raise SystemExit(
                "Specify a mode: `--plan` (free dry run), `--smoke` (GPU check + teardown), "
                "or `--upload/--script/--fetch` (provision, run your job, fetch artifacts, teardown)."
            )

        if args.command == "sweep":
            from .sweep import run_sweep
            return run_sweep(cloud=args.cloud, region=args.region, flavor=args.flavor,
                             uploads=args.upload, script=args.script, jobs_file=args.jobs,
                             fetch=args.fetch, into=args.into, max_parallel=args.max_parallel,
                             max_minutes=args.max_minutes, budget_eur=args.budget)
    except RuntimeError as exc:
        print(f"flux-compute {args.command}: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
