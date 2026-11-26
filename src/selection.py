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
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from data_models import PostWithContext

FRESH_HALF_LIFE_H = 6.0
# Posts reach the eligible feed hours after they were posted (measured 2026-11-26: youngest 5h, median
# 15h), so post age separates them poorly. What we control is how soon we act once a post shows up:
# a post new to our feed counts double, fading to no boost with a 1-hour half-life. A first guess.
ARRIVAL_HALF_LIFE_H = 1.0
MEDIA_BOOST = 1.5

# Topic multipliers from public-data CRH rates of notes mentioning each topic (base ~11.4%):
# AI-generated 20.3%, satire/parody 15.7%; Israel/Gaza 6.8%, elections 7.1%, immigration 8.0%.
TOPIC_WEIGHTS = [
    (re.compile(r"\bAI\b|\bdeepfake|\bgenerated\b|\bsatir|\bparody", re.I), 1.4),
    (re.compile(r"\bIsrael|\bGaza|\bPalestin|\bHamas\b|\bZionis|\bIDF\b", re.I), 0.6),
    (re.compile(r"\belection|\bballot|\bvoter fraud|\brigged\b", re.I), 0.6),
    (re.compile(r"\bimmigra|\bmigrant|\bdeport|\billegals?\b", re.I), 0.7),
    # Historical atrocities: notes here read as denial unless they fix a concrete detail; first test
    # note on such a post scored ClaimOpinion low.
    (re.compile(r"\bgenocid|\bmassacre|\bholocaust|\bslavery\b|\bcoloni[sz]|\bwar crimes?\b|\bethnic cleansing", re.I), 0.5),
]


_STRIP_RE = re.compile(r"https?://\S+|@\w+|#\w+")


def prefilter_reason(pwc: PostWithContext) -> str:
    """Cheap rule-based skip before any LLM call; '' means research the post.
    Deliberately conservative: a wrongly skipped post costs a possible helpful note."""
    texts = [pwc.post.text or ""]
    if pwc.quoted_post:
        texts.append(pwc.quoted_post.text or "")
    stripped = _STRIP_RE.sub(" ", " ".join(texts))
    has_media = bool(pwc.post.media) or bool(pwc.quoted_post and pwc.quoted_post.media)
    # Character count too, so languages written without spaces (Japanese, Chinese) aren't skipped.
    if not has_media and len(stripped.split()) < 4 and len("".join(stripped.split())) < 20:
        return "no media and almost no text, nothing to fact-check"
    return ""


# First guesses, to be re-fit on our own results: replies and first-person stories are mostly opinion or
# unverifiable; figures, attributions and "breaking"-style framing are what notes usually correct.
_ANECDOTE_RE = re.compile(r"\b(I|my|me|we|our)\b")
_CLAIM_RE = re.compile(
    r"\d{2,}|%|\bbreaking\b|\bconfirmed\b|\bannounced?\b|\breport(?:s|ed|edly)?\b|\bleaked\b|\bofficial"
    r"|\bstudy\b|\bdata\b|\bvideo\b|\bphoto\b|\bfootage\b|\bshows?\b|\bsaid\b|\baccording to\b",
    re.I,
)


def checkable_weight(pwc: PostWithContext) -> float:
    text = pwc.post.text or ""
    w = 1.0
    if _CLAIM_RE.search(text):
        w *= 1.3
    if pwc.in_reply_to_post is not None or text.lstrip().startswith("@"):
        w *= 0.7
    if len(_ANECDOTE_RE.findall(text)) >= 3 and not pwc.post.media:
        w *= 0.6
    return w


def topic_weight(text: str) -> float:
    w = 1.0
    for rx, mult in TOPIC_WEIGHTS:
        if rx.search(text):
            w *= mult
    return w


def _age_hours(pwc: PostWithContext, now: datetime) -> float:
    return max(0.0, (now - pwc.post.created_at).total_seconds() / 3600)


def score_post(
    pwc: PostWithContext, now: datetime | None = None, first_seen: datetime | None = None
) -> Tuple[float, Dict[str, float]]:
    """`first_seen` is when the post first appeared in our feed; None means it is new right now."""
    now = now or datetime.now(timezone.utc)
    age = _age_hours(pwc, now)
    in_feed_h = max(0.0, (now - first_seen).total_seconds() / 3600) if first_seen else 0.0
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
        "topic": topic_weight(" ".join(t for t in [pwc.post.text, pwc.quoted_post.text if pwc.quoted_post else ""] if t)),
    }
    parts["checkable"] = checkable_weight(pwc)
    parts["arrival"] = 1 + 0.5 ** (in_feed_h / ARRIVAL_HALF_LIFE_H)
    # Note requests weigh more than raw spread: a viral opinion or anecdote gets no note, while a post
    # people asked to have checked usually has something checkable in it.
    score = (
        parts["topic"] * parts["media"] * parts["fresh"] * parts["checkable"] * parts["arrival"]
        * (1 + 0.5 * parts["spread"] + 1.0 * parts["demand"])
    )
    parts["age_h"] = round(age, 2)
    parts["in_feed_h"] = round(in_feed_h, 2)
    parts["velocity"] = round(velocity, 1)
    return score, parts


def rank_posts(
    posts: List[PostWithContext], skip_sensitive: bool = True, first_seen: Dict[str, datetime] | None = None
) -> List[Tuple[PostWithContext, float, Dict[str, float]]]:
    now = datetime.now(timezone.utc)
    ranked = []
    for p in posts:
        if skip_sensitive and p.post.possibly_sensitive:
            continue
        s, parts = score_post(p, now, (first_seen or {}).get(str(p.post.post_id)))
        ranked.append((p, s, parts))
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked
