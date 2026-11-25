"""
Short, public-safe run report: counts and admission progress only, no post or note text.
Written to the GitHub Actions job summary; also handy locally.

uv run src/run_report.py
"""

from collections import Counter

import ledger
import outcomes
from check_scores import admission_report


def report() -> str:
    recs = ledger.records_since(24)
    runs = [r for r in recs if r.get("decision") == "run_summary"]
    decisions = Counter(r["decision"] for r in recs if r.get("decision") != "run_summary")
    last = runs[-1] if runs else {}
    lines = [
        "## Note writer",
        "",
        f"Last run: researched {last.get('processed', 0)}, submitted {last.get('submitted', 0)}, "
        f"{last.get('llm_calls', 0)} Grok calls, ${last.get('llm_usd', 0)}"
        + (f" ({last['reason']})" if last.get("reason") else ""),
        "",
        f"Last 24h: {len(runs)} runs, ${ledger.usd_last_24h():.2f} xAI spend",
        "",
        "| decision | posts |",
        "|---|---|",
        *[f"| {d} | {n} |" for d, n in decisions.most_common()],
        "",
    ]
    test_notes = [n for n in outcomes.load() if n.get("test_mode", True)]
    if test_notes:
        lines += ["```", *admission_report(test_notes), "```"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
