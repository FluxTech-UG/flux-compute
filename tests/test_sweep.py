"""Pure-logic tests for the sweep helpers. No network, no credentials."""
import pytest

from flux_compute.sweep import parse_jobs, worst_case_eur


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
