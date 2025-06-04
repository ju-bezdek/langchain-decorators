#!/usr/bin/env python3
"""
Test runner script for langchain-decorators

This script provides convenient commands to run different test suites:
- All tests
- Unit tests only
- Integration tests only
- Specific test files
- Tests with coverage reporting
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        print("Please ensure pytest is installed: pip install pytest")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run langchain-decorators tests")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "coverage", "fast", "slow"],
        nargs="?",
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file (e.g., test_prompt_decorator.py)"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function (e.g., test_simple_prompt_decorator)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    # Build command based on test type
    if args.file:
        cmd = base_cmd + [f"tests/{args.file}"]
        description = f"Running tests in {args.file}"
    elif args.function:
        cmd = base_cmd + ["-k", args.function]
        description = f"Running test function: {args.function}"
    elif args.test_type == "unit":
        cmd = base_cmd + ["-m", "unit"]
        description = "Running unit tests only"
    elif args.test_type == "integration":
        cmd = base_cmd + ["-m", "integration"]
        description = "Running integration tests only"
    elif args.test_type == "fast":
        cmd = base_cmd + ["-m", "not slow"]
        description = "Running fast tests only"
    elif args.test_type == "slow":
        cmd = base_cmd + ["-m", "slow"]
        description = "Running slow tests only"
    elif args.test_type == "coverage":
        if args.no_cov:
            cmd = base_cmd + ["tests/"]
        else:
            cmd = base_cmd + [
                "--cov=src/langchain_decorators",
                "--cov-report=html",
                "--cov-report=term-missing",
                "tests/"
            ]
        description = "Running all tests with coverage"
    else:  # all
        cmd = base_cmd + ["tests/"]
        description = "Running all tests"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if success:
        print(f"\n✅ Tests completed successfully!")
        if args.test_type == "coverage" and not args.no_cov:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()