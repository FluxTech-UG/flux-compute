"""flux-compute command-line entry point."""
from __future__ import annotations

import argparse


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
    doctor.add_argument("--cloud", default=None,
                        help="clouds.yaml entry name (else OS_* env vars are used).")
    doctor.add_argument("--region", default=None,
                        help="Region override (else OS_REGION_NAME / clouds.yaml).")

    sub.add_parser(
        "run",
        help="(Phase 1, not yet implemented) Provision a GPU instance, run one job, "
             "fetch artifacts, tear down.",
    )

    args = parser.parse_args(argv)

    if args.command == "doctor":
        from .doctor import run_doctor
        return run_doctor(cloud=args.cloud, region=args.region)

    if args.command == "run":
        raise SystemExit(
            "`flux-compute run` is Phase 1 and not implemented yet.\n"
            "Run `flux-compute doctor` first to confirm API access."
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
