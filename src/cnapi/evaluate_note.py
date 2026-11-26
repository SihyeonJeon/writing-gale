"""
Ask X how its ClaimOpinion evaluator scores a note before submitting it.

POST /2/notes/evaluate returns claim_opinion_score for a (note text, post) pair without creating a
note. It is the same kind of score the admission criteria bucket into high / medium / low, costs no
LLM credits, and counts against the X API rate limit only.

Observed on our first test notes: notes later bucketed High scored 0.14 to 0.85, a Medium -1.86,
Lows -2.59 and -3.32.
"""

import json
import subprocess
from typing import Optional

from .xurl_util import run_xurl


def claim_opinion_score(note_text: str, post_id: str) -> Optional[float]:
    """The score, or None if the endpoint could not be reached (never blocks a note by itself)."""
    cmd = ["xurl", "-X", "POST", "/2/notes/evaluate", "-d", json.dumps({"note_text": note_text, "post_id": post_id})]
    try:
        data = run_xurl(cmd).get("data") or {}
        score = data.get("claim_opinion_score")
        return float(score) if score is not None else None
    except (subprocess.CalledProcessError, json.JSONDecodeError, TypeError, ValueError, OSError):
        return None
