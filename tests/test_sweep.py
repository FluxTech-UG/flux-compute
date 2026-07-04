"""Pure-logic tests for the sweep helpers. No network, no credentials."""
import threading
import time

import pytest

from flux_compute.sweep import (
    _failure_status,
    _fan_out,
    budget_guard,
    clamp_concurrency,
    parse_jobs,
    worst_case_eur,
)


def test_parse_label_equals_params():
    jobs = parse_jobs("alpha = N_x=128\nbeta = N_x=256\n")
    assert jobs == [("alpha", "N_x=128"), ("beta", "N_x=256")]


def test_parse_skips_blanks_and_comments():
    jobs = parse_jobs("# header\n\nonly = x\n  # indented comment\n")
    assert jobs == [("only", "x")]


def test_parse_line_without_equals_is_label_and_params():
    assert parse_jobs("spec_operating_point\n") == [("spec_operating_point", "spec_operating_point")]


def test_duplicate_label_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("a = 1\na = 2\n")


def test_label_with_slash_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("bad/label = 1\n")


def test_empty_jobs_raises():
    with pytest.raises(RuntimeError):
        parse_jobs("# only comments\n\n")


def test_worst_case_cost():
    assert worst_case_eur(3, 0.80, 30) == pytest.approx(1.20)   # 3 * 0.80 * 0.5
    assert worst_case_eur(10, 0.80, 6) == pytest.approx(0.80)   # 10 * 0.80 * 0.1


def test_worst_case_price_unknown_is_none():
    assert worst_case_eur(5, None, 30) is None


def test_budget_guard_priced_under_budget_returns_worst_case():
    # 200 b3-8 jobs at 0.0512/hr, 30-min cap -> 200 * 0.0512 * 0.5 = 5.12.
    wc = budget_guard("b3-8", 0.0512, 200, 30, budget_eur=10.0)
    assert wc == pytest.approx(5.12)


def test_budget_guard_priced_over_budget_raises():
    with pytest.raises(RuntimeError, match="exceeds budget"):
        budget_guard("b3-8", 0.0512, 200, 30, budget_eur=4.0)


def test_budget_guard_unpriced_with_budget_refuses_and_names_flavor():
    with pytest.raises(RuntimeError) as exc:
        budget_guard("b3-8-flex", None, 200, 30, budget_eur=10.0)
    msg = str(exc.value)
    assert "b3-8-flex" in msg          # names the offending flavor
    assert "no known price" in msg     # and says why it refuses


def test_budget_guard_unpriced_without_budget_returns_none():
    # No budget set: an unknown price is not fatal, the guard just cannot bound it.
    assert budget_guard("b3-8-flex", None, 200, 30, budget_eur=None) is None


# --- quota-aware concurrency clamp -------------------------------------------

def test_clamp_concurrency_bounded_by_core_quota():
    # 8 cores free, 2 vCPU each -> 4 instances fit; well under the 10 ceiling.
    assert clamp_concurrency(10, 2, cores_used=0, cores_max=8,
                             instances_used=0, instances_max=100) == 4


def test_clamp_concurrency_bounded_by_instance_quota():
    # Cores allow 50, but only 3 instance slots free.
    assert clamp_concurrency(10, 2, cores_used=0, cores_max=100,
                             instances_used=0, instances_max=3) == 3


def test_clamp_concurrency_user_ceiling_wins_when_quota_is_ample():
    assert clamp_concurrency(4, 2, cores_used=0, cores_max=1000,
                             instances_used=0, instances_max=1000) == 4


def test_clamp_concurrency_no_core_headroom_raises():
    # 5 cores free but the flavor needs 15 -> not even one fits.
    with pytest.raises(RuntimeError, match="cannot fit even one"):
        clamp_concurrency(4, 15, cores_used=10, cores_max=15,
                          instances_used=0, instances_max=100)


def test_clamp_concurrency_no_instance_headroom_raises():
    with pytest.raises(RuntimeError, match="cannot fit even one"):
        clamp_concurrency(4, 2, cores_used=0, cores_max=100,
                          instances_used=5, instances_max=5)


# --- fan-out respects the clamp and completes every job ----------------------

def test_fan_out_bounds_concurrency_and_runs_every_job():
    K = 4
    jobs = list(range(3 * K))          # a 3K-job sweep
    lock = threading.Lock()
    state = {"live": 0, "peak": 0}

    def run_one(job):
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        time.sleep(0.02)
        with lock:
            state["live"] -= 1
        return (job, 0, "ok")

    results = _fan_out(jobs, run_one, K)
    assert len(results) == len(jobs)   # all jobs completed
    assert state["peak"] <= K          # never exceeded the clamped concurrency


# --- create-failure labelling ------------------------------------------------

def test_failure_status_flags_quota_and_capacity():
    assert _failure_status(Exception("Quota exceeded for instances")).startswith("quota/capacity")
    assert _failure_status(Exception("No valid host was found")).startswith("quota/capacity")


def test_failure_status_generic_error_stays_generic():
    assert _failure_status(RuntimeError("SSH never opened within timeout")).startswith("error")
