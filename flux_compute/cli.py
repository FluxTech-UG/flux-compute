"""flux-compute command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys


def _add_target_args(p):
    p.add_argument("--cloud", default=None,
                   help="clouds.yaml entry name (else OS_* env vars are used).")
    p.add_argument("--region", default=None,
                   help="Region override (else OS_REGION_NAME / clouds.yaml).")


def _add_requirement_args(p, *, with_count):
    """Generic fleet-requirement flags. `with_count` adds --count (the `plan`
    command needs a job count; sweep gets it from the jobs file)."""
    if with_count:
        p.add_argument("--count", type=int, default=None, metavar="N",
                       help="Number of independent jobs.")
    p.add_argument("--ram-gb", type=float, default=None, metavar="GB",
                   help="Peak host RAM one job (or one batched member) needs.")
    p.add_argument("--device", default=None, choices=("cpu", "gpu", "either"),
                   help="Device requirement: cpu, gpu, or either (default either).")
    p.add_argument("--minutes", type=float, default=None, metavar="MIN",
                   help="Wall-clock estimate per job (default 30).")
    p.add_argument("--batchable", action="store_true",
                   help="Many jobs collapse into one device invocation (GPU vmap-style).")
    p.add_argument("--batch-width", type=int, default=None, metavar="B",
                   help="Preferred members per batched invocation (only with --batchable).")
    p.add_argument("--vram-gb", type=float, default=None, metavar="GB",
                   help="GPU device memory one batched member needs (only with --batchable). "
                        "Omitted, the planner assumes it equals --ram-gb.")
    p.add_argument("--requirements", default=None, metavar="FILE",
                   help="JSON file of requirement fields; explicit flags override it.")


def _build_requirements(args, *, default_count=None):
    """Assemble JobRequirements from --requirements JSON overlaid by flags.

    Returns None when the caller supplied no requirement at all (so sweep/run keep
    their current behavior). `default_count` fills job_count when the command has
    no --count flag (sweep counts its jobs file separately)."""
    from .fleet import JobRequirements

    data = {}
    if getattr(args, "requirements", None):
        with open(args.requirements) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise SystemExit("--requirements JSON must be an object of fields")
        # Fail fast on an unrecognized key rather than silently ignoring it: a
        # typo'd field (e.g. "minutes" for "minutes_per_job", "batch" for
        # "batch_width") would otherwise fall back to a default and mis-size a
        # paid fleet with no warning.
        allowed = {"job_count", "ram_gb_per_job", "device", "minutes_per_job",
                   "batchable", "batch_width", "vram_gb_per_member"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SystemExit(
                f"--requirements JSON has unrecognized field(s): {', '.join(unknown)}. "
                f"Valid fields: {', '.join(sorted(allowed))}.")

    def pick(flag, key, default=None):
        val = getattr(args, flag, None)
        if val is not None:
            return val
        return data.get(key, default)

    ram = pick("ram_gb", "ram_gb_per_job")
    raw_count = pick("count", "job_count")   # no default: it must not mask "nothing requested"
    batchable = bool(getattr(args, "batchable", False) or data.get("batchable", False))
    # Nothing requested at all -> no requirement (current behavior).
    if ram is None and raw_count is None and not data:
        return None
    if ram is None:
        raise SystemExit("a fleet requirement needs --ram-gb (or ram_gb_per_job in --requirements)")
    count = raw_count if raw_count is not None else default_count
    return JobRequirements(
        job_count=int(count) if count is not None else 1,
        ram_gb_per_job=float(ram),
        device=pick("device", "device", "either"),
        minutes_per_job=float(pick("minutes", "minutes_per_job", 30.0)),
        batchable=batchable,
        batch_width=pick("batch_width", "batch_width"),
        vram_gb_per_member=pick("vram_gb", "vram_gb_per_member"),
    )


def _flavor_from_requirements(args, *, default_count):
    """Resolve the flavor for a run/sweep: an explicit --flavor always wins;
    otherwise, if a fleet requirement was given, the planner chooses it. Returns
    None (current behavior) when neither is present."""
    if getattr(args, "flavor", None):
        return args.flavor
    req = _build_requirements(args, default_count=default_count)
    if req is None:
        return None
    from .fleet import choose_flavor
    chosen = choose_flavor(req)
    print(f"flux-compute: requirement -> flavor {chosen.name} "
          f"({chosen.kind}, {chosen.vcpus} vCPU, {chosen.ram_gb:g} GB); override with --flavor.")
    return chosen.name


def _stream_output():
    """Make progress output appear as it happens, even through a pipe.

    Python line-buffers stdout only when it is a TTY; redirected or piped
    (``flux-compute sweep ... | tee run.log``, or any launcher capturing output)
    it switches to 4 KiB block buffering, so a multi-hour fleet writes NOTHING to
    the log until the buffer fills or the process exits. An empty log is then
    indistinguishable from a hung launcher, and chasing that phantom cost a
    session. Setting line buffering at the entry point fixes every print in the
    package at once, and requires nothing of the caller (no PYTHONUNBUFFERED, no
    ``stdbuf``) -- which is the point: the caller cannot be relied on to remember.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass          # already unbuffered, detached, or a non-TextIO stand-in


def main(argv=None) -> int:
    _stream_output()
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
    _add_requirement_args(run, with_count=False)
    run.add_argument("--plan", action="store_true",
                     help="Resolve and print the launch spec without launching (dry run).")
    run.add_argument("--smoke", action="store_true",
                     help="Provision, verify the device (nvidia-smi on a GPU flavor, a boot + "
                          "remote-exec check on a CPU flavor), and tear down. Billable.")
    run.add_argument("--upload", action="append", default=[], metavar="DIR[:DEST]",
                     help="Local dir to rsync up (repeatable). 'DIR' lands at ~/<basename>; "
                          "'SRC:DEST' lands SRC at ~/DEST, so the remote name need not match "
                          "the local one (e.g. a worktree: --upload /path/1DSim3-wt:1DSim3).")
    run.add_argument("--script", default=None, metavar="FILE",
                     help="Local bash script uploaded and run on the instance (your setup + job).")
    run.add_argument("--fetch", action="append", default=[], metavar="REMOTE:LOCAL",
                     help="Copy REMOTE (home-relative dir) back to LOCAL after the job (repeatable).")
    run.add_argument("--keep", action="store_true",
                     help="Leave the instance running after the job for debugging (you must tear it down).")
    run.add_argument("--flavor", default=None,
                     help="Override the flavor (else the cheapest fp64-healthy GPU available).")
    run.add_argument("--image", default=None,
                     help="Boot from this image name instead of the flavor-aware default "
                          "(NVIDIA image for a GPU flavor, plain Ubuntu for a CPU flavor; e.g. a baked image).")

    plan = sub.add_parser(
        "plan",
        help="Size a fleet for a generic job requirement (RAM/device/count) and print "
             "the flavor, region spread, packing and worst-case cost. No launch.",
    )
    _add_target_args(plan)
    _add_requirement_args(plan, with_count=True)
    plan.add_argument("--regions", default=None, metavar="A,B,C",
                      help="Plan across these regions (default: the device's eligible regions).")
    plan.add_argument("--max-parallel", type=int, default=None, metavar="N",
                      help="Global cap on concurrent VMs (default: the full quota fleet).")
    plan.add_argument("--budget", type=float, default=None, metavar="EUR",
                      help="Refuse the plan if the WHOLE batch's worst-case spend "
                           "(all --count jobs x price x --minutes) exceeds this. One cap for "
                           "the batch, not per job, and independent of the region spread.")
    plan.add_argument("--live", action="store_true",
                      help="Read real per-region quota and availability from the API "
                           "(needs credentials) instead of the offline catalog tables.")

    sweep = sub.add_parser(
        "sweep",
        help="Fan out a parameter sweep across GPU instances with a hard cost cap.",
    )
    _add_target_args(sweep)
    _add_requirement_args(sweep, with_count=False)
    sweep.add_argument("--upload", action="append", default=[], metavar="DIR[:DEST]",
                       help="Local dir to rsync up to each instance (repeatable). 'DIR' lands "
                            "at ~/<basename>; 'SRC:DEST' lands SRC at ~/DEST, so the remote "
                            "name need not match the local one (e.g. a worktree: "
                            "--upload /path/1DSim3-wt:1DSim3).")
    sweep.add_argument("--script", default=None, metavar="FILE",
                       help="Per-job bash script; reads $FLUX_LABEL and $FLUX_JOB.")
    sweep.add_argument("--jobs", default=None, metavar="FILE",
                       help="Jobs file: each line 'LABEL = PARAMS' (PARAMS -> $FLUX_JOB). "
                            "Blank lines, whole-line '#' comments and inline '# ...' comments "
                            "are stripped from both label and params.")
    sweep.add_argument("--fetch", default=None, metavar="REMOTE",
                       help="Home-relative artifact dir pulled per job into <into>/<label>/.")
    sweep.add_argument("--into", default="cloud-sweep", metavar="DIR",
                       help="Local base dir for fetched artifacts (default: cloud-sweep).")
    sweep.add_argument("--flavor", default=None,
                       help="Override the flavor (else the cheapest fp64-healthy GPU available).")
    sweep.add_argument("--regions", default=None, metavar="A,B,C",
                       help="Shard the sweep across several regions at once (comma-separated). "
                            "Compute quota is PER REGION, so this is how a fleet grows past one "
                            "region's headroom. Mutually exclusive with --region. Needs a "
                            "clouds.yaml that is not pinned to one region (use `regions:`).")
    sweep.add_argument("--max-parallel", type=int, default=4,
                       help="Max instances alive at once ACROSS ALL regions (default 4). "
                            "Each region is additionally clamped to its own quota headroom.")
    sweep.add_argument("--max-minutes", type=int, default=30,
                       help="Per-job remote wall-clock cap; kills a hung job (default 30).")
    sweep.add_argument("--budget", type=float, default=None, metavar="EUR",
                       help="Hard cap on the WHOLE SWEEP's worst-case spend, not a per-job cap: "
                            "the guard computes (total jobs) x (flavor EUR/hr) x (--max-minutes) "
                            "-- every job running to its full wall cap -- and refuses to start "
                            "above this. With --regions the per-region shards' worst cases are "
                            "SUMMED against this one number, so the cap is independent of how "
                            "many regions the sweep spans (regions buy wall-clock, not spend). "
                            "An unpriced flavor is refused rather than skipping the guard.")
    sweep.add_argument("--plan", action="store_true",
                       help="Resolve the spec and print the cost/budget (and quota) preview without launching (dry run).")
    sweep.add_argument("--image", default=None,
                       help="Boot from this image name instead of the auto-selected image.")
    sweep.add_argument("--resume", action="store_true",
                       help="Continue an interrupted sweep: re-attach to its still-running "
                            "detached jobs (reads <into>/*/.flux_attach/), collect and tear them "
                            "down. Needs only --into (and --cloud/--region). Pass --jobs (with "
                            "--script/--fetch) as well to ALSO launch the jobs of that file that "
                            "never started -- jobs already collected are skipped.")
    sweep.add_argument("--strict-regions", action="store_true",
                       help="Refuse the whole sweep if ANY requested region cannot fit >=1 "
                            "instance of the chosen flavor (exact-width mode). Default: drop "
                            "unfit regions with a warning (naming their occupants and headroom) "
                            "and run the sweep on the regions that do fit.")

    bake = sub.add_parser(
        "bake",
        help="Build a reusable GPU image (preinstall the heavy stack) to skip per-job installs.",
    )
    _add_target_args(bake)
    bake.add_argument("--name", required=True, help="Name of the image to create.")
    bake.add_argument("--script", required=True,
                      help="Setup script baked into the image (installs the common stack, e.g. into ~/venv).")
    bake.add_argument("--upload", action="append", default=[], metavar="DIR",
                      help="Local dir to rsync up before running setup (repeatable).")
    bake.add_argument("--flavor", default=None, help="Override the flavor.")
    bake.add_argument("--replace", action="store_true",
                      help="Delete existing same-name images after the new one is built.")

    regions_cmd = sub.add_parser(
        "regions",
        help="Live, read-only per-region occupancy: quota used/total, running "
             "flux-compute instances, and how many of a flavor fit the headroom.",
    )
    _add_target_args(regions_cmd)
    regions_cmd.add_argument("--regions", default=None, metavar="A,B,C",
                             help="Show these regions (comma-separated). Default: every region "
                                  "the cloud entry is configured for.")
    regions_cmd.add_argument("--flavor", default=None, metavar="NAME",
                             help="Flavor for the 'fits' column — how many fit each region's "
                                  "remaining headroom (default b3-32).")
    regions_cmd.add_argument("--json", action="store_true", dest="as_json",
                             help="Emit machine-readable JSON (for the frontend button / agents).")

    reap = sub.add_parser(
        "reap",
        help="List flux-compute instances (age/cost/bucket) and delete the ones past "
             "their stamped TTL, with their keypair and security group.",
    )
    _add_target_args(reap)
    reap.add_argument("--regions", default=None, metavar="A,B,C",
                      help="Scan these regions (comma-separated). Default: EVERY region the "
                           "cloud entry is configured for, since servers and quota are both "
                           "per region and a multi-region sweep strands them per region.")
    reap.add_argument("--yes", action="store_true",
                      help="Skip confirmation for expired-stamped instances (non-interactive reap).")
    reap.add_argument("--all", action="store_true", dest="take_all",
                      help="Also take keep-flagged / within-TTL / unstamped-legacy instances. "
                           "Interactive confirmation is required unless --force is also given.")
    reap.add_argument("--force", action="store_true",
                      help="Non-interactive confirmation for EVERYTHING --all selects, including "
                           "instances still inside their TTL (a live fleet). This kills running "
                           "work: it exists so a runaway fleet can be stopped from a script or a "
                           "non-tty session without piping `yes`, which is indiscriminate and "
                           "answers prompts nobody read. Implies --yes.")

    push = sub.add_parser(
        "push",
        help="Upload a local artifact directory to an OVH Object Storage container.",
    )
    _add_target_args(push)
    push.add_argument("dir", help="Local directory to upload.")
    push.add_argument("container", help="Object Storage container name (created if missing).")
    push.add_argument("--prefix", default="", help="Optional object-name prefix within the container.")

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            from .doctor import run_doctor
            return run_doctor(cloud=args.cloud, region=args.region)

        if args.command == "preflight":
            from .preflight import run_preflight
            return run_preflight(cloud=args.cloud, region=args.region)

        if args.command == "plan":
            from .fleet import format_plan, plan_fleet, plan_fleet_live
            req = _build_requirements(args)
            if req is None:
                raise SystemExit(
                    "plan needs a requirement: --count and --ram-gb (or a --requirements JSON file).")
            if args.live:
                fleet = plan_fleet_live(req, cloud=args.cloud, budget=args.budget,
                                        regions=args.regions, max_parallel=args.max_parallel)
            else:
                fleet = plan_fleet(req, budget=args.budget, regions=args.regions,
                                   max_parallel=args.max_parallel)
            print(format_plan(req, fleet))
            return 0

        if args.command == "run":
            flavor = _flavor_from_requirements(args, default_count=1)
            if args.plan:
                from .launch import plan
                return plan(cloud=args.cloud, region=args.region, flavor=flavor)
            if args.smoke:
                from .provision import smoke_test
                return smoke_test(cloud=args.cloud, region=args.region, flavor=flavor)
            if args.script or args.upload:
                from .provision import run_job
                return run_job(cloud=args.cloud, region=args.region, flavor=flavor,
                               uploads=args.upload, script=args.script, fetch=args.fetch,
                               keep=args.keep, image=args.image)
            raise SystemExit(
                "Specify a mode: `--plan` (free dry run), `--smoke` (GPU check + teardown), "
                "or `--upload/--script/--fetch` (provision, run your job, fetch artifacts, teardown)."
            )

        if args.command == "sweep":
            from .sweep import run_sweep
            flavor = _flavor_from_requirements(args, default_count=1)
            return run_sweep(cloud=args.cloud, region=args.region, regions=args.regions,
                             flavor=flavor,
                             uploads=args.upload, script=args.script, jobs_file=args.jobs,
                             fetch=args.fetch, into=args.into, max_parallel=args.max_parallel,
                             max_minutes=args.max_minutes, budget_eur=args.budget, image=args.image,
                             plan_only=args.plan, resume=args.resume,
                             strict_regions=args.strict_regions)

        if args.command == "regions":
            from .regions import DEFAULT_FITS_FLAVOR, run_regions
            return run_regions(cloud=args.cloud, region=args.region, regions=args.regions,
                               flavor=args.flavor or DEFAULT_FITS_FLAVOR, as_json=args.as_json)

        if args.command == "bake":
            from .image import bake
            return bake(cloud=args.cloud, region=args.region, name=args.name,
                        script=args.script, flavor=args.flavor, uploads=args.upload,
                        replace=args.replace)

        if args.command == "reap":
            from .reap import run_reap
            return run_reap(cloud=args.cloud, region=args.region, regions=args.regions,
                            yes=args.yes, take_all=args.take_all, force=args.force)

        if args.command == "push":
            from .objstore import run_push
            return run_push(cloud=args.cloud, region=args.region,
                            local_dir=args.dir, container=args.container, prefix=args.prefix)
    except RuntimeError as exc:
        print(f"flux-compute {args.command}: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
