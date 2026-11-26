"""
What the public Community Notes data says, at hand while writing.

Files come from tools/build_benchmark.py (data/benchmark/). Everything here degrades to "no data":
the writer still runs without the files.

- similar_helpful_notes(post text): notes people rated helpful whose wording overlaps this post, found
  with BM25. Recurring hoaxes, recycled videos and repeated talking points come back often, and the
  note that worked last time shows both the approach and the sources that raters accepted.
- distrusted_domain(url): domains whose notes are rated Not Helpful far more often than average.
- post_outcome(post id): whether anyone's note on a post was rated helpful (for the feedback report).
"""

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

DIR = Path(os.getenv("BENCHMARK_DIR", Path(__file__).resolve().parent.parent / "data" / "benchmark"))
URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[^\W\d_]{3,}|\d{4}", re.UNICODE)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
K1, B = 1.5, 0.75
NOT_HELPFUL_FACTOR = 2.5  # a domain is distrusted when its Not Helpful share is this many times the average
MIN_SHARED_TERMS = 3


def tokens(text: str) -> List[str]:
    text = URL_RE.sub(" ", text.lower())
    out = [w for w in WORD_RE.findall(text) if not CJK_RE.fullmatch(w)]
    for run in CJK_RE.findall(text):  # no spaces between words: use character pairs
        out += [run[i:i + 2] for i in range(len(run) - 1)]
    return out


@lru_cache(maxsize=1)
def _index():
    path = DIR / "exemplars.jsonl"
    if not path.exists():
        return None
    docs, postings, lengths = [], {}, []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            tf = Counter(tokens(rec["text"]))
            if not tf:
                continue
            i = len(docs)
            docs.append(rec)
            lengths.append(sum(tf.values()))
            for t, c in tf.items():
                postings.setdefault(t, []).append((i, c))
    avg = sum(lengths) / max(len(lengths), 1)
    return docs, postings, lengths, avg


def similar_helpful_notes(post_text: str, post_id: str = "", k: int = 3) -> List[Dict]:
    """Top-k helpful notes by BM25 against the post's wording. Notes on this same post are left out:
    the point is to learn from similar cases, not to copy someone's note."""
    idx = _index()
    if idx is None:
        return []
    docs, postings, lengths, avg = idx
    n = len(docs)
    scores: Dict[int, float] = {}
    shared: Dict[int, int] = Counter()
    for t in set(tokens(post_text)):
        plist = postings.get(t)
        if not plist or len(plist) > max(n * 0.2, 5):  # unseen or too common to mean anything
            continue
        idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
        for i, c in plist:
            scores[i] = scores.get(i, 0.0) + idf * c * (K1 + 1) / (c + K1 * (1 - B + B * lengths[i] / avg))
            shared[i] += 1
    ranked = sorted((i for i in scores if shared[i] >= MIN_SHARED_TERMS and docs[i]["post_id"] != str(post_id)), key=lambda i: -scores[i])
    out, seen = [], set()
    for i in ranked:
        key = URL_RE.sub("", docs[i]["text"])[:80]
        if key in seen:  # the same note pasted on many copies of a post
            continue
        seen.add(key)
        out.append({**docs[i], "score": round(scores[i], 1)})
        if len(out) == k:
            break
    return out


@lru_cache(maxsize=1)
def _domains() -> Tuple[Dict[str, Dict], Dict[str, float]]:
    path = DIR / "domains.json"
    if not path.exists():
        return {}, {}
    d = json.loads(path.read_text())
    return d["domains"], d["base"]


def _host(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    for p in ("www.", "m.", "mobile.", "amp."):
        h = h.removeprefix(p)
    return h


def distrusted_domain(url: str) -> Optional[str]:
    """A reason string if notes citing this domain are rated Not Helpful far more often than average."""
    domains, base = _domains()
    d = domains.get(_host(url))
    if not d or not base or d["not_helpful"] < NOT_HELPFUL_FACTOR * base["not_helpful"]:
        return None
    return (
        f"{_host(url)}: {d['not_helpful']:.1%} of {d['notes']} notes citing it were rated Not Helpful "
        f"(average {base['not_helpful']:.1%})"
    )


@lru_cache(maxsize=1)
def _posts() -> Dict[str, List[int]]:
    path = DIR / "posts.json"
    return json.loads(path.read_text()) if path.exists() else {}


def post_outcome(post_id: str) -> Optional[Tuple[int, bool]]:
    """(number of notes people wrote on the post, any rated helpful), or None if not in the data."""
    v = _posts().get(str(post_id))
    return (v[0], bool(v[1])) if v else None


def meta() -> Dict:
    path = DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}
