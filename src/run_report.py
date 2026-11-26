"""
Short, public-safe run report: counts and admission progress only, no post or note text.
Written to the GitHub Actions job summary; also handy locally.

uv run src/run_report.py
"""

from collections import Counter

import benchmark
import ledger
import outcomes
from check_scores import admission_report


def report() -> str:
    recs = ledger.records_since(24)
    runs = [r for r in recs if r.get("decision") == "run_summary"]
    decisions = Counter(r["decision"] for r in recs if r.get("decision") not in ("run_summary", "seen"))
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
    lines += feedback()
    return "\n".join(lines)


def feedback() -> list:
    """Feed posts we saw vs what people's notes achieved on them (public data, about two days behind).
    Rows: what we did with the post. Columns: did anyone's note on it get rated helpful."""
    meta = benchmark.meta()
    if not meta:
        return []
    last = {}  # post id -> our last decision ("seen" only if we never researched it)
    feats = {}
    for r in ledger.all_records():
        pid, d = r.get("post_id"), r.get("decision")
        if d == "seen":
            feats[pid] = r
            last.setdefault(pid, "not researched")
        elif d != "run_summary" and pid != "-":
            last[pid] = d
    table = Counter()
    for pid, d in last.items():
        out = benchmark.post_outcome(pid)
        if out is not None:
            table[(d, "helpful note exists" if out[1] else "noted, none helpful")] += 1
    lines = ["", f"### Feed posts vs public data ({meta.get('data_date')})", ""]
    if not table:
        return lines + ["No overlap yet between posts we saw and the public data (it runs about two days behind)."]
    lines += ["| what we did | helpful note exists | noted, none helpful |", "|---|---|---|"]
    for d in sorted({k[0] for k in table}):
        lines.append(f"| {d} | {table[(d, 'helpful note exists')]} | {table[(d, 'noted, none helpful')]} |")
    # which ranking features separate posts that ended up with a helpful note
    rows = [(feats[p], benchmark.post_outcome(p)[1]) for p in feats if benchmark.post_outcome(p) is not None]
    if len(rows) >= 30:
        lines += ["", "Mean ranking feature, posts with a helpful note vs without:", "", "| feature | with | without |", "|---|---|---|"]
        for k in ("score", "fresh", "spread", "demand", "media", "topic", "checkable"):
            a = [float(r.get(k) or 0) for r, h in rows if h]
            b = [float(r.get(k) or 0) for r, h in rows if not h]
            if a and b:
                lines.append(f"| {k} | {sum(a) / len(a):.2f} (n={len(a)}) | {sum(b) / len(b):.2f} (n={len(b)}) |")
    return lines


if __name__ == "__main__":
    print(report())
