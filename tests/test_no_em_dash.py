"""Guard: tracked prose and source carry no em dash (U+2014).

The FluxTech family writes no em dashes; a sentence takes the punctuation it
needs (period, comma, colon, parentheses). An en dash in a numeric range
(64–576 vCPUs) is a different character and is allowed. A markdown table cell
holding only the em dash is an empty-cell data marker, not prose, and is
allowed. `docs/MAP.md` is scanned: it is generated from the docstrings, so a
clean source keeps it clean.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".html", ".css", ".js", ".sh"}
EXCLUDED = {
    # Archived copy of OVH's own eligibility guide: vendor text kept verbatim as the source of record.
    "docs/product-eligibility-startup-program-2026-03.html",
}
# Standalone data tokens stripped from a line before it is tested.
ALLOWED_TOKENS = ("| " + EM_DASH + " |",)


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    return [f for f in out if f and Path(f).suffix in SUFFIXES and f not in EXCLUDED]


def _strip_allowed(line):
    # A table row can hold adjacent empty cells (two dash cells in a row), which share a pipe,
    # so strip repeatedly until no token is left.
    prev = None
    while prev != line:
        prev = line
        for token in ALLOWED_TOKENS:
            line = line.replace(token, "|  |")
    return line


def test_no_em_dash_in_tracked_text():
    hits = []
    for rel in _tracked_files():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            if EM_DASH in _strip_allowed(line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    assert not hits, "em dash (U+2014) found; use a period, comma, colon or parentheses:\n" + "\n".join(hits)
