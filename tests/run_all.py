#!/usr/bin/env python3
"""Master test runner for the Turbo OCR test suite.

Usage:
    python tests/run_all.py                    # Run all tests
    python tests/run_all.py --suite unit       # Run only unit tests
    python tests/run_all.py --suite integration
    python tests/run_all.py --suite regression
    python tests/run_all.py --suite benchmark
    python tests/run_all.py --server-url http://myhost:8000
    python tests/run_all.py --suite unit --suite integration  # Multiple suites

Equivalent pytest commands:
    pytest tests/unit/ -v
    pytest tests/integration/ -v
    pytest tests/regression/ -v
    pytest tests/benchmark/ -v -s
"""

import argparse
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SUITES = {
    "unit": TESTS_DIR / "unit",
    "integration": TESTS_DIR / "integration",
    "regression": TESTS_DIR / "regression",
    "benchmark": TESTS_DIR / "benchmark",
}


def main():
    parser = argparse.ArgumentParser(
        description="Turbo OCR test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=list(SUITES.keys()) + ["all"],
        help="Test suite(s) to run (default: all). Can specify multiple.",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"),
        help="OCR server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--grpc-target",
        default=os.environ.get("OCR_GRPC_TARGET", "localhost:50051"),
        help="gRPC target (default: localhost:50051)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Stop on first failure",
    )
    parser.add_argument(
        "-k",
        default=None,
        help="Only run tests matching expression (passed to pytest -k)",
    )
    args = parser.parse_args()

    # Determine which suites to run
    suites = args.suite or ["all"]
    if "all" in suites:
        dirs = list(SUITES.values())
    else:
        dirs = [SUITES[s] for s in suites]

    # Build pytest args
    pytest_args = [str(d) for d in dirs]
    pytest_args.extend([
        f"--server-url={args.server_url}",
        f"--grpc-target={args.grpc_target}",
        f"--rootdir={TESTS_DIR}",
    ])

    if args.verbose or "benchmark" in (args.suite or []):
        pytest_args.append("-v")

    if "benchmark" in (args.suite or []):
        pytest_args.append("-s")  # Show print output for benchmarks

    if args.exitfirst:
        pytest_args.append("-x")

    if args.k:
        pytest_args.extend(["-k", args.k])

    # Import and run pytest
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: pip install -r tests/requirements.txt")
        sys.exit(1)

    print(f"Running suites: {', '.join(suites)}")
    print(f"Server: {args.server_url}")
    print(f"gRPC: {args.grpc_target}")
    print()

    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
