"""
Evaluation Runner

Runs the ATLAS agent against a set of test cases and collects results.
Each test case is evaluated on three criteria:

1. Answer correctness — does the answer contain expected keywords?
2. Tool selection — did the agent use the expected tools?
3. Step efficiency — did it stay within the allowed step count?

Design decisions:
- Test cases are YAML, not Python — non-developers can add tests without
  touching code. The same pattern used by the guardrails config.
- Sequential groups let memory tests run in order (save first, then recall)
  without resetting the agent between them.
- A delay between test cases prevents hitting Gemini's rate limit.
- Results are collected as dicts, then passed to the report generator.
  The runner doesn't format output — separation of concerns.

In an interview: "I built a custom evaluation framework that tests the agent
across 9 categories. Each test measures answer accuracy, tool selection
correctness, and step efficiency. The framework runs 28 test cases and
generates a metrics report with per-category breakdowns."
"""

import os
import sys
import time
import yaml

# Add project root to path so we can import agent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.core import Agent, is_safety_refusal, ATLAS_SAFETY_MESSAGE


# Path to test cases file
TEST_CASES_FILE = os.path.join(os.path.dirname(__file__), "test_cases.yaml")

# Delay between test cases (seconds) to avoid rate limiting.
# Gemini 3.6 Flash free tier has rate limits per minute.
# 3 seconds between tests keeps us well under the limit.
DELAY_BETWEEN_TESTS = 5


def load_test_cases() -> list[dict]:
    """Load test cases from YAML file."""
    with open(TEST_CASES_FILE, "r") as f:
        return yaml.safe_load(f)


def run_single_test(agent: Agent, test_case: dict) -> dict:
    """
    Run a single test case against the agent and return the result.

    Returns a dict with:
    - id, category, question (from the test case)
    - answer: the agent's actual response
    - tools_used: list of tools the agent called
    - steps: number of tool calls made
    - answer_correct: bool — did the answer contain expected keywords?
    - tools_correct: bool — did the agent use the expected tools?
    - steps_ok: bool — did it stay within max_steps?
    - passed: bool — all three checks passed
    - blocked_correctly: bool — for guardrail tests, was it blocked?
    """
    test_id = test_case["id"]
    question = test_case["question"]
    expected_tools = test_case.get("expected_tools", [])
    expected_contains = test_case.get("expected_contains", [])
    max_steps = test_case.get("max_steps", 5)
    should_be_blocked = test_case.get("blocked", False)

    # Track which tools the agent uses by monitoring the conversation
    # history before and after the chat call
    history_before = len(agent.conversation_history)

    # Run the agent
    answer = agent.chat(question)

    # Extract tools used from conversation history
    tools_used = []
    for msg in agent.conversation_history[history_before:]:
        if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULT"):
            # Extract tool name from "TOOL_RESULT (tool_name): ..."
            tool_name = msg["content"].split("(")[1].split(")")[0]
            tools_used.append(tool_name)

    steps = len(tools_used)

    # --- Evaluate ---

    # For guardrail tests: check if the message was blocked
    if should_be_blocked:
        blocked_correctly = (
            "Abhishek" in answer and
            len(tools_used) == 0
        )
        return {
            "id": test_id,
            "category": test_case["category"],
            "question": question,
            "answer": answer[:100] + "..." if len(answer) > 100 else answer,
            "tools_used": tools_used,
            "steps": steps,
            "answer_correct": blocked_correctly,
            "tools_correct": True,
            "steps_ok": True,
            "passed": blocked_correctly,
            "blocked_correctly": blocked_correctly,
        }

    # Check 1: Answer correctness
    # At least one expected keyword must appear in the answer (case-insensitive)
    # If no expected keywords are specified, this check passes automatically
    if expected_contains:
        answer_lower = answer.lower().replace(",", "")
        answer_correct = any(
            keyword.lower() in answer_lower for keyword in expected_contains
        )
    else:
        answer_correct = True

    # Check 2: Tool selection
    # The agent should have used all expected tools (order doesn't matter
    # for this check — the agent might reasonably reorder steps)
    if expected_tools:
        tools_correct = all(tool in tools_used for tool in expected_tools)
    else:
        # If no tools expected, check that no tools were used
        # (for identity and casual conversation tests)
        tools_correct = len(tools_used) == 0

    # Check 3: Step efficiency
    # The agent should not exceed max_steps
    if max_steps > 0:
        steps_ok = steps <= max_steps
    else:
        steps_ok = True

    # Overall pass: all three checks must pass
    passed = answer_correct and tools_correct and steps_ok

    return {
        "id": test_id,
        "category": test_case["category"],
        "question": question,
        "answer": answer[:100] + "..." if len(answer) > 100 else answer,
        "tools_used": tools_used,
        "steps": steps,
        "expected_tools": expected_tools,
        "answer_correct": answer_correct,
        "tools_correct": tools_correct,
        "steps_ok": steps_ok,
        "passed": passed,
    }


def run_eval() -> list[dict]:
    """
    Run all test cases and return results.

    Handles sequential groups: test cases with the same sequential_group
    share an agent instance (so memory save → recall works). All other
    test cases get a fresh agent to avoid cross-contamination.
    """
    test_cases = load_test_cases()
    results = []

    # Group sequential test cases
    sequential_groups = {}
    standalone_tests = []

    for tc in test_cases:
        group = tc.get("sequential_group")
        if group:
            if group not in sequential_groups:
                sequential_groups[group] = []
            sequential_groups[group].append(tc)
        else:
            standalone_tests.append(tc)

    total = len(test_cases)
    completed = 0

    # Run standalone tests (fresh agent for each)
    for tc in standalone_tests:
        completed += 1
        print(f"  [{completed}/{total}] {tc['id']}: {tc['question'][:50]}...")

        agent = Agent()
        result = run_single_test(agent, tc)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"           → {status}")

        # Delay to avoid rate limiting (skip for guardrail tests —
        # they don't call the LLM so no delay needed)
        if not tc.get("blocked", False):
            time.sleep(DELAY_BETWEEN_TESTS)

    # Run sequential groups (shared agent within each group)
    for group_name, group_tests in sequential_groups.items():
        agent = Agent()  # One agent for the whole group

        for tc in group_tests:
            completed += 1
            print(f"  [{completed}/{total}] {tc['id']}: {tc['question'][:50]}...")

            result = run_single_test(agent, tc)
            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"
            print(f"           → {status}")

            time.sleep(DELAY_BETWEEN_TESTS)

    return results
