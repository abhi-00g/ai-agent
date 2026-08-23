"""
Evaluation Report Generator

Takes raw results from the eval runner and produces a formatted report
with overall metrics and per-category breakdowns.

Metrics tracked:
- Overall pass rate (total passed / total tests)
- Answer accuracy (answers containing expected keywords)
- Tool selection accuracy (correct tools used)
- Step efficiency (stayed within step limits)
- Per-category pass rates
- Average steps per category
- List of failed test cases with details

Why separate the report from the runner?
Same reason we separate routes from services in the backend: the runner
collects data, the report formats it. If you want to add a JSON export
or a dashboard visualization later, you only change this file.
"""

from datetime import datetime, timezone
from collections import defaultdict


def generate_report(results: list[dict]) -> str:
    """
    Generate a formatted evaluation report from test results.

    Args:
        results: list of result dicts from the eval runner.

    Returns:
        A formatted string report ready to print or save to a file.
    """
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    # Per-criteria counts
    answer_correct = sum(1 for r in results if r["answer_correct"])
    tools_correct = sum(1 for r in results if r["tools_correct"])
    steps_ok = sum(1 for r in results if r["steps_ok"])

    # Per-category stats
    categories = defaultdict(lambda: {"total": 0, "passed": 0, "steps": []})
    for r in results:
        cat = r["category"]
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        categories[cat]["steps"].append(r["steps"])

    # Failed tests
    failures = [r for r in results if not r["passed"]]

    # --- Build the report ---
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("  ATLAS — Evaluation Report")
    lines.append(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 64)

    # Overall summary
    lines.append("")
    lines.append("  OVERALL RESULTS")
    lines.append("  " + "-" * 40)
    pass_rate = (passed / total * 100) if total > 0 else 0
    lines.append(f"  Total test cases:      {total}")
    lines.append(f"  Passed:                {passed}")
    lines.append(f"  Failed:                {failed}")
    lines.append(f"  Pass rate:             {pass_rate:.1f}%")

    # Per-criteria breakdown
    lines.append("")
    lines.append("  CRITERIA BREAKDOWN")
    lines.append("  " + "-" * 40)
    lines.append(f"  Answer accuracy:       {answer_correct}/{total} ({answer_correct/total*100:.1f}%)")
    lines.append(f"  Tool selection:        {tools_correct}/{total} ({tools_correct/total*100:.1f}%)")
    lines.append(f"  Step efficiency:       {steps_ok}/{total} ({steps_ok/total*100:.1f}%)")

    # Per-category breakdown
    lines.append("")
    lines.append("  PER-CATEGORY RESULTS")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Category':<20} {'Pass Rate':<15} {'Avg Steps'}")
    lines.append("  " + "-" * 40)

    for cat_name in sorted(categories.keys()):
        cat = categories[cat_name]
        cat_pass_rate = (cat["passed"] / cat["total"] * 100) if cat["total"] > 0 else 0
        avg_steps = sum(cat["steps"]) / len(cat["steps"]) if cat["steps"] else 0
        lines.append(
            f"  {cat_name:<20} {cat['passed']}/{cat['total']} ({cat_pass_rate:>5.1f}%)    {avg_steps:.1f}"
        )

    # Failed tests details
    if failures:
        lines.append("")
        lines.append("  FAILED TESTS")
        lines.append("  " + "-" * 40)
        for f in failures:
            lines.append(f"")
            lines.append(f"  [{f['id']}] {f['question'][:60]}")
            lines.append(f"    Answer correct: {f['answer_correct']}")
            lines.append(f"    Tools correct:  {f['tools_correct']}")
            lines.append(f"    Steps OK:       {f['steps_ok']}")
            lines.append(f"    Tools used:     {f.get('tools_used', [])}")
            lines.append(f"    Expected tools: {f.get('expected_tools', [])}")
            lines.append(f"    Answer:         {f['answer'][:80]}")
    else:
        lines.append("")
        lines.append("  All tests passed!")

    lines.append("")
    lines.append("=" * 64)
    lines.append("")

    return "\n".join(lines)


def save_report(report: str, filepath: str = "eval/eval_report.txt"):
    """Save the report to a file for reference."""
    with open(filepath, "w") as f:
        f.write(report)
    print(f"  Report saved to: {filepath}")
