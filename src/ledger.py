"""
Append-only JSONL ledger of every post the writer has looked at.

- Skips posts already handled (one note per post; avoids paying the LLM twice for the same post).
- Keeps decisions, gate problems and note text so later runs can be analysed against
  evaluator scores / CRH outcomes.

Location: data/ledger.jsonl (gitignored). Override with LEDGER_PATH.
"""

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

_lock = threading.Lock()

# Decisions that mean "don't look at this post again". Errors are retried on later runs.
FINAL_DECISIONS = {"prefilter_skip", "evaluator_skip", "critic_skip", "submitted", "duplicate", "NO NOTE NEEDED", "NOT ENOUGH EVIDENCE", "gate_failed", "dry_run"}


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
            if decision == "evaluator_skip" and rec.get("test_mode", True):
                score = re.search(r"scored this note (-?\d+(?:\.\d+)?)", rec.get("error", ""))
                minimum = max(-1.9, min(0.0, float(os.getenv("TEST_CLAIM_MIN_SCORE") or 0)))
                if score and minimum < 0 and float(score[1]) >= minimum:
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


def usd_last_hours(hours: float) -> float:
    """xAI spend over the last `hours`, from the per-run summaries."""
    return sum(float(r.get("llm_usd") or 0) for r in records_since(hours) if r.get("decision") == "run_summary")


def first_seen() -> Dict[str, datetime]:
    """When each post first showed up in our feed (the time of its `seen` record)."""
    out: Dict[str, datetime] = {}
    for rec in all_records():
        if rec.get("decision") == "seen":
            try:
                out.setdefault(str(rec.get("post_id")), datetime.fromisoformat(rec["ts"]))
            except (KeyError, ValueError):
                continue
    return out


def latest_feed_observations(now: datetime | None = None) -> Dict[str, Dict[str, Any]]:
    """Latest usable prior metric snapshot for each candidate, at least a minute apart."""
    now = now or datetime.now(timezone.utc)
    out = {}
    for rec in records_since(1):
        if rec.get("decision") not in {"seen", "feed_observed"} or not rec.get("post_public_metrics"):
            continue
        try:
            elapsed = (now - datetime.fromisoformat(rec["ts"])).total_seconds()
            pid = str(rec["post_id"])
            if 60 <= elapsed <= 3600 and (pid not in out or rec["ts"] > out[pid]["ts"]):
                out[pid] = rec
        except (KeyError, ValueError, TypeError):
            continue
    return out


def seen_post_ids() -> Set[str]:
    path = ledger_path()
    if not path.exists():
        return set()
    out: Set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("decision") == "seen":
                out.add(str(rec.get("post_id")))
    return out


def all_records() -> List[Dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
