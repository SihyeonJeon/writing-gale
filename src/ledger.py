"""
Append-only JSONL ledger of every post the writer has looked at.

- Skips posts already handled (one note per post; avoids paying the LLM twice for the same post).
- Keeps decisions, gate problems and note text so later runs can be analysed against
  evaluator scores / CRH outcomes.

Location: data/ledger.jsonl (gitignored). Override with LEDGER_PATH.
"""

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

_lock = threading.Lock()

# Decisions that mean "don't look at this post again". Errors are retried on later runs.
FINAL_DECISIONS = {"prefilter_skip", "critic_skip", "submitted", "duplicate", "NO NOTE NEEDED", "NOT ENOUGH EVIDENCE", "gate_failed", "dry_run"}


def ledger_path() -> Path:
    return Path(os.getenv("LEDGER_PATH", Path(__file__).resolve().parent.parent / "data" / "ledger.jsonl"))


def handled_post_ids(include_dry_run: bool = False) -> Set[str]:
    path = ledger_path()
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = rec.get("decision")
            if decision == "dry_run" and not include_dry_run:
                continue
            if decision in FINAL_DECISIONS:
                done.add(str(rec.get("post_id")))
    return done


def record(post_id: str, decision: str, **fields: Any) -> None:
    rec: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "post_id": str(post_id),
        "decision": decision,
        **fields,
    }
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def records_since(hours: float = 24) -> List[Dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if datetime.fromisoformat(rec["ts"]) >= since:
                    out.append(rec)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def submitted_last_24h(test_mode: bool) -> int:
    """Notes submitted in the last 24 hours in this mode (for the daily cap)."""
    return sum(
        1 for r in records_since(24)
        if r.get("decision") == "submitted" and bool(r.get("test_mode", True)) == test_mode
    )


def usd_last_24h() -> float:
    """xAI spend over the last 24 hours, from the per-run summaries."""
    return sum(float(r.get("llm_usd") or 0) for r in records_since(24) if r.get("decision") == "run_summary")
