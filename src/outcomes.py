"""
Learn from how our own submitted notes were scored.

`uv run src/outcomes.py` pulls every note this writer has written (test mode, and live once admitted) from
GET /2/notes/search/notes_written and caches it in data/outcomes.json. No LLM calls; one X API call per
100 notes.

The writer then puts a few of our best-scored notes in its prompt as style examples, and a couple of
poorly-scored ones as patterns to avoid. Good = rated helpful by people (live), or high on the admission
evaluators (test mode). Only our own note text is used, never post text.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_GOOD = 3
MAX_BAD = 2


def outcomes_path() -> Path:
    return Path(os.getenv("OUTCOMES_PATH", DATA_DIR / "outcomes.json"))


def _fetch(test_mode: bool) -> List[Dict]:
    from cnapi.xurl_util import run_xurl

    notes: List[Dict] = []
    token = None
    while True:
        path = (
            f"/2/notes/search/notes_written?test_mode={'true' if test_mode else 'false'}"
            "&max_results=100&note.fields=id,info,status,test_result"
        )
        if token:
            path += f"&pagination_token={token}"
        resp = run_xurl(["xurl", path])
        for n in resp.get("data", []):
            n["test_mode"] = test_mode
            notes.append(n)
        token = (resp.get("meta") or {}).get("next_token")
        if not token:
            return notes


def refresh(include_live: bool) -> List[Dict]:
    notes = _fetch(test_mode=True)
    if include_live:
        notes += _fetch(test_mode=False)
    path = outcomes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, ensure_ascii=False))
    return notes


def load() -> List[Dict]:
    path = outcomes_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def scores(note: Dict) -> Dict[str, str]:
    outcomes = (note.get("test_result") or {}).get("evaluation_outcome") or []
    return {o.get("evaluator_type"): o.get("evaluator_score_bucket") for o in outcomes}


def classify(note: Dict) -> str:
    """'good', 'bad' or 'unknown'."""
    status = str(note.get("status") or "").lower()
    if "not_helpful" in status:
        return "bad"
    if "helpful" in status:  # currently_rated_helpful
        return "good"
    s = scores(note)
    if s.get("ClaimOpinion") == "low" or s.get("UrlValidity") == "low" or s.get("HarassmentAbuse") == "low":
        return "bad"
    if s.get("ClaimOpinion") == "high" and s.get("UrlValidity") == "high":
        return "good"
    return "unknown"


def _text(note: Dict) -> str:
    return ((note.get("info") or {}).get("text") or "").strip()


def examples_for_prompt(notes: List[Dict] | None = None) -> str:
    notes = load() if notes is None else notes
    # Live helpful notes are the strongest signal, so they sort first; then newest first (ids grow).
    def key(n: Dict):
        return (not n.get("test_mode", True), int(n.get("id") or 0))

    good = [_text(n) for n in sorted(notes, key=key, reverse=True) if classify(n) == "good" and _text(n)][:MAX_GOOD]
    bad = [_text(n) for n in sorted(notes, key=key, reverse=True) if classify(n) == "bad" and _text(n)][:MAX_BAD]
    if not good and not bad:
        return ""
    out = "\nNotes this writer submitted earlier, for style only (different posts; do not reuse their facts):\n"
    if good:
        out += "Scored well:\n" + "".join(f"- {t}\n" for t in good)
    if bad:
        out += "Scored poorly (avoid what made them opinionated, unsupported or broken):\n" + "".join(f"- {t}\n" for t in bad)
    return out


if __name__ == "__main__":
    import argparse

    import dotenv

    dotenv.load_dotenv()
    ap = argparse.ArgumentParser(description="Refresh data/outcomes.json from notes_written.")
    ap.add_argument("--live", action="store_true", help="Also fetch live (non-test) notes")
    args = ap.parse_args()
    ns = refresh(include_live=args.live or os.getenv("TEST_MODE", "true").lower() == "false")
    counts: Dict[str, int] = {}
    for n in ns:
        counts[classify(n)] = counts.get(classify(n), 0) + 1
    print(f"{len(ns)} notes cached in {outcomes_path()}: {counts}")
