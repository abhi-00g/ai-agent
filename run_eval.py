"""
Run the ATLAS evaluation suite.

Usage:
    python run_eval.py

This runs all test cases against the agent and generates a report
with pass rates, per-category breakdowns, and failure details.

Note: This makes real API calls to Gemini and Tavily, so it will
use your free tier credits. A full run of ~28 tests takes about
2-3 minutes (with rate limit delays between tests).
"""

import os
import sys

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(__file__))

from eval.runner import run_eval
from eval.report import generate_report, save_report


def main():
    print("=" * 64)
    print("  ATLAS — Running Evaluation Suite")
    print("=" * 64)
    print()

    # Clean up any leftover eval memory data from previous runs
    # so memory tests start fresh every time
    memory_file = "memory/user_memory.json"
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print("  Cleared previous memory data for clean eval run.")
        print()

    # Run all test cases
    print("  Running test cases...\n")
    results = run_eval()

    # Generate and display the report
    report = generate_report(results)
    print(report)

    # Save the report to a file
    save_report(report)


if __name__ == "__main__":
    main()
