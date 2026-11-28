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

from note_patterns import HELPFUL, script_group

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
    """Legacy aggregate diagnostic retained for callers; source checks use the cited page's evidence."""
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


@lru_cache(maxsize=1)
def _cards() -> Dict:
    path = DIR / "cards.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cards_for_prompt(has_media: bool, max_cards: int = 10) -> str:
    """Observed opening frequencies retained for older benchmark caches."""
    data = _cards()
    cards = [c for c in data.get("cards", []) if c.get("examples")]
    if not cards:
        return ""
    cards.sort(key=lambda c: (c["media"] != has_media, -c["helpful"]))
    base = data["base_helpful"]
    out = (
        f"\nObserved openings in {data['notes']:,} public notes over {data['days']} days "
        f"(currently Helpful fraction {base:.1%}, including pending notes in the denominator). "
        "Reproduce visible form using this post's verified facts. These are observational frequencies. "
        "Keep the examples' historical facts and links confined to the reference examples.\n"
    )
    for c in cards[:max_cards]:
        out += f'- Opens "{c["opening"]} ...": helpful {c["helpful"]:.1%} of {c["notes"]:,} notes by {c.get("authors", "?")} writers.\n'
        out += "".join(f"    real note: {e[:300]}\n" for e in c["examples"][:1])
    for opening in data.get("avoid", [])[:8]:
        out += (
            f'- Observed opening "{opening["opening"]}": currently Helpful '
            f'{opening["helpful"]:.1%} of {opening["notes"]:,} notes.\n'
        )
    return out


@lru_cache(maxsize=1)
def _patterns() -> dict:
    path = DIR / "observed_patterns.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def observed_patterns_for_prompt(post_text: str, post_id: str = "") -> str:
    data = _patterns()
    if data.get("schema_version") != 1:
        return ""
    script = script_group(post_text)
    cohort = data.get("cohorts", {}).get(script)
    if not cohort or not cohort.get("forms"):
        return ""
    helpful = cohort["status_counts"].get(HELPFUL, 0)
    lines = [
        "\nObserved public-note forms",
        f'Snapshot {data["data_date"]}; window {data["days"]} days; script group {script} (mixed languages).',
        f'Cohort {cohort["notes"]:,} notes; currently Helpful {helpful:,}.',
        f'Measurement: {data["measurement"]}.',
        "For a justified note, reproduce a compatible observed form with the current verified facts.",
        "Choose the post's language. Keep each reference's historical facts and URLs within that reference.",
        "Frequency tables describe observations. Keep explanations of raters' motives out of the drafting process.",
    ]
    comparison = data["comparison"]
    if comparison["pairs"]:
        lines.append(
            f'Same-post observations across {comparison["posts"]:,} posts and {comparison["pairs"]:,} pairs. '
            'Each post has one currently Helpful note. Peer statuses include pending ratings. '
            'Timing, author and language differences remain unadjusted.'
        )
        for name, count in comparison["features"].items():
            lines.append(
                f'- {name}: Helpful-side {count["helpful"]:,}/{comparison["pairs"]:,}; '
                f'peer-side {count["peer"]:,}/{comparison["pairs"]:,}.'
            )
    source_data = data.get("domain_observations")
    if source_data:
        global_count = data["cohorts"]["all"]["status_counts"].get(HELPFUL, 0)
        lines.append(
            f'Cited-host counts in all {global_count:,} currently Helpful notes across scripts. '
            'A note may cite multiple hosts. These counts measure observed use.'
        )
        for host, observation in sorted(
            source_data["domains"].items(), key=lambda item: -item[1]["helpful_notes"],
        )[:8]:
            lines.append(
                f'- {host}: cited by {observation["helpful_notes"]:,} Helpful notes; '
                f'{observation["notes"]:,} citing notes across all statuses.'
            )
    shown = 0
    shown_posts = {str(post_id)}
    for form in cohort["forms"]:
        # The writer API requires a source URL and the existing 280-character text limit.
        if form["url_count"] == "0" or form["body_char_band"] in ("0", "281+"):
            continue
        examples = [example for example in form["examples"] if example["post_id"] not in shown_posts]
        if not examples:
            continue
        lines.append(
            f'Form: body characters {form["body_char_band"]}, URL count {form["url_count"]}, '
            f'emoji present {form["emoji"]}, exclamation present {form["exclamation"]}; '
            f'observed in {form["notes"]:,}/{helpful:,} Helpful notes, '
            f'{form["posts"]:,} posts, {form["authors"]:,} authors.'
        )
        example = examples[0]
        lines.append(f'Observed note {example["note_id"]}, post {example["post_id"]}: {example["text"]}')
        shown_posts.add(example["post_id"])
        shown += 1
        if shown == 3:
            break
    return "\n".join(lines) + "\n" if shown else ""


def source_observations_for_prompt(urls: List[str]) -> str:
    snapshot = _patterns().get("domain_observations")
    domains, base = (snapshot["domains"], snapshot["base"]) if snapshot else _domains()
    if not domains:
        return ""
    lines = []
    for host in sorted({_host(url) for url in urls}):
        observation = domains.get(host)
        if not observation:
            continue
        if "helpful_notes" in observation:
            lines.append(
                f'- {host}: {observation["notes"]:,} citing notes; '
                f'currently Helpful {observation["helpful_notes"]:,}; '
                f'currently Not Helpful {observation["not_helpful_notes"]:,}; '
                f'pending {observation["needs_more_ratings"]:,}.'
            )
        else:
            lines.append(
                f'- {host}: {observation["notes"]:,} citing notes; '
                f'currently Helpful {observation["helpful"]:.2%}; '
                f'currently Not Helpful {observation["not_helpful"]:.2%}.'
            )
    if not lines:
        return ""
    return (
        "\nObserved citation-domain frequencies in the public-note benchmark. "
        f'Corpus currently Helpful fraction {base.get("helpful", 0):.2%}. '
        "Denominators include pending notes. Topic, language and exposure are unadjusted. "
        "Verify this particular page's facts and accessibility.\n" + "\n".join(lines) + "\n"
    )


def meta() -> Dict:
    path = DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}
