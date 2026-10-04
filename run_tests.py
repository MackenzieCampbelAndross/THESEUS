#!/usr/bin/env python3
"""Run every Python test suite in the THESIUS repository and print a summary.

Each suite runs in its own pytest process from its own directory, so the
subsystems' package layouts never collide.

Usage:
    python run_tests.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES = [
    ("Member 2 - Tracking, estimation and control", ROOT / "astra-link-member2"),
    ("Member 3 - Security and reliable transport", ROOT / "astra_link_member3"),
    ("Integration - Tracking to security bridge", ROOT / "astra_link_integration"),
]


def main() -> int:
    results = []
    for name, path in SUITES:
        print(f"\n=== {name} ({path.name}) ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        print(tail or proc.stderr.strip()[-2000:])
        passed = re.search(r"(\d+) passed", tail)
        results.append((name, proc.returncode, int(passed.group(1)) if passed else 0))

    print("\n" + "=" * 64)
    print(f"{'Suite':<50}{'Passed':>8}  Status")
    print("-" * 64)
    for name, code, n in results:
        print(f"{name:<50}{n:>8}  {'OK' if code == 0 else 'FAILED'}")
    print("-" * 64)
    total = sum(n for _, _, n in results)
    ok = all(code == 0 for _, code, _ in results)
    print(f"{'Total':<50}{total:>8}  {'ALL PASSED' if ok else 'FAILURES PRESENT'}")
    print("Frontend (TypeScript) tests are run separately with: npm test")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
