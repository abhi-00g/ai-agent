"""
Run the ATLAS evaluation suite.

Usage:
    python run_eval.py

Uses a separate memory directory (memory_eval/) so eval runs
never wipe the user's personal saved memories in memory/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Override the memory path BEFORE importing agent modules.
# This ensures the eval uses memory_eval/ instead of memory/.
import agent.tools.memory as memory_module
memory_module.MEMORY_DIR = "memory_eval"
memory_module.MEMORY_FILE = os.path.join("memory_eval", "user_memory.json")

from eval.runner import run_eval
from eval.report import generate_report, save_report


def main():
    print("=" * 64)
    print("  ATLAS — Running Evaluation Suite")
    print("=" * 64)
    print()

    # Clean up eval memory (NOT the user's memory/)
    memory_file = "memory_eval/user_memory.json"
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print("  Cleared eval memory data for clean run.")
        print()

    print("  Running test cases...\n")
    results = run_eval()

    report = generate_report(results)
    print(report)

    save_report(report)

    # Clean up eval memory after run
    if os.path.exists(memory_file):
        os.remove(memory_file)
    if os.path.exists("memory_eval") and not os.listdir("memory_eval"):
        os.rmdir("memory_eval")


if __name__ == "__main__":
    main()
