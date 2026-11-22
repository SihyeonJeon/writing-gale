"""
Rank eligible posts so the limited daily note quota goes where a note is most likely to reach
Currently Rated Helpful (CRH).

Signals, from the public Community Notes data (2025-06 .. 2026-11, ~900k English notes):
- freshness: notes written within ~1h of the first note on a post reach CRH ~20% of the time,
  vs ~4% after a day. Older posts are worth much less.
- media: media notes reach CRH ~17% vs ~11% overall (recycled / AI-generated / edited media is
  checkable and raters agree on it).
- demand: posts that several people write notes on reach CRH 2-3x as often as single-note posts.
  We can't see other notes in real time, so the number of note requests / suggested sources is
  used as a stand-in.
- spread: engagement per hour (likes, reposts, quotes, replies) - a note on a post that is still
  spreading gets seen and rated.

The weights are a starting guess and should be re-fit against our own CRH outcomes.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from data_models import PostWithContext

FRESH_HALF_LIFE_H = 6.0
MEDIA_BOOST = 1.5


def _age_hours(pwc: PostWithContext, now: datetime) -> float:
    return max(0.0, (now - pwc.post.created_at).total_seconds() / 3600)


def score_post(pwc: PostWithContext, now: datetime | None = None) -> Tuple[float, Dict[str, float]]:
    now = now or datetime.now(timezone.utc)
    age = _age_hours(pwc, now)
    m = pwc.post.public_metrics or {}
    engagement = (
        m.get("like_count", 0)
        + 2 * m.get("retweet_count", 0)
        + 2 * m.get("quote_count", 0)
        + m.get("reply_count", 0)
    )
    velocity = engagement / max(age, 0.5)
    has_media = bool(pwc.post.media) or bool(pwc.quoted_post and pwc.quoted_post.media)
    requests = len(pwc.note_request_suggestions) + sum(
        int(link.get("count", 1) or 1) for link in pwc.suggested_source_links_with_counts
    )

    parts = {
        "fresh": 0.5 ** (age / FRESH_HALF_LIFE_H),
        "spread": math.log10(1 + velocity),
        "demand": math.log2(1 + requests),
        "media": MEDIA_BOOST if has_media else 1.0,
    }
    score = parts["media"] * parts["fresh"] * (1 + parts["spread"] + 0.5 * parts["demand"])
    parts["age_h"] = round(age, 2)
    parts["velocity"] = round(velocity, 1)
    return score, parts


def rank_posts(posts: List[PostWithContext], skip_sensitive: bool = True) -> List[Tuple[PostWithContext, float, Dict[str, float]]]:
    now = datetime.now(timezone.utc)
    ranked = []
    for p in posts:
        if skip_sensitive and p.post.possibly_sensitive:
            continue
        s, parts = score_post(p, now)
        ranked.append((p, s, parts))
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked
