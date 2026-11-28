"""Literal text measurements and public-note frequency tables. No inferred rating reasons."""

import re
from collections import Counter, defaultdict

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
EMOJI_RE = re.compile("[\U0001f1e6-\U0001f1ff\U0001f300-\U0001faff\u2600-\u27bf]|[0-9#*]\ufe0f?\u20e3")
SCRIPT_RE = {
    "latin": re.compile(r"[A-Za-z\u00c0-\u024f]"),
    "cjk": re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"),
    "cyrillic": re.compile(r"[\u0400-\u052f]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
}
HELPFUL = "CURRENTLY_RATED_HELPFUL"
NOT_HELPFUL = "CURRENTLY_RATED_NOT_HELPFUL"


def script_group(text: str) -> str:
    text = URL_RE.sub("", text)
    counts = {name: len(pattern.findall(text)) for name, pattern in SCRIPT_RE.items()}
    return max(counts, key=counts.get) if any(counts.values()) else "other"


def text_features(text: str) -> dict:
    body = re.sub(r"\s+", " ", URL_RE.sub("", text)).strip()
    length = len(body)
    bands = ((0, "0"), (80, "1-80"), (160, "81-160"), (240, "161-240"), (280, "241-280"))
    band = next((label for bound, label in bands if length <= bound), "281+")
    return {
        "body_chars": length,
        "body_char_band": band,
        "urls": len(URL_RE.findall(text)),
        "emoji": bool(EMOJI_RE.search(body)),
        "exclamation": "!" in body,
    }


def build_patterns(records, data_date: str, days: int) -> dict:
    """Records are newest-first, with the columns used by build_benchmark.py."""
    cohorts = {}
    posts = defaultdict(list)
    first_time = None
    last_time = None
    for row in records:
        text = row["summary"]
        feature = text_features(text)
        script = script_group(text)
        status = row["currentStatus"]
        timestamp = int(row["t"])
        first_time = min(first_time, timestamp) if first_time is not None else timestamp
        last_time = max(last_time, timestamp) if last_time is not None else timestamp
        form_key = (feature["body_char_band"], min(feature["urls"], 3), feature["emoji"], feature["exclamation"])
        for name in ("all", script):
            cohort = cohorts.setdefault(name, {"statuses": Counter(), "forms": {}, "lengths": []})
            cohort["statuses"][status] += 1
            if status != HELPFUL:
                continue
            cohort["lengths"].append(feature["body_chars"])
            form = cohort["forms"].setdefault(form_key, {
                "notes": 0, "posts": set(), "authors": set(), "examples": [], "example_authors": set(),
            })
            form["notes"] += 1
            form["posts"].add(row["tweetId"])
            form["authors"].add(row["noteAuthorParticipantId"])
            if (len(form["examples"]) < 3 and row["noteAuthorParticipantId"] not in form["example_authors"]
                    and row["tweetId"] not in {example["post_id"] for example in form["examples"]}):
                form["examples"].append({"note_id": row["noteId"], "post_id": row["tweetId"], "text": text})
                form["example_authors"].add(row["noteAuthorParticipantId"])
        posts[row["tweetId"]].append((status, {
            "two_or_more_urls": feature["urls"] >= 2,
            "body_81_160_chars": 80 < feature["body_chars"] <= 160,
            "emoji": feature["emoji"], "exclamation": feature["exclamation"],
        }))

    comparison = {"posts": 0, "pairs": 0, "peer_statuses": Counter(), "features": {}}
    for notes in posts.values():
        winners = [note for note in notes if note[0] == HELPFUL]
        peers = [note for note in notes if note[0] != HELPFUL]
        if len(winners) != 1 or not peers:
            continue
        comparison["posts"] += 1
        for status, peer in peers:
            comparison["pairs"] += 1
            comparison["peer_statuses"][status] += 1
            for name, value in winners[0][1].items():
                counts = comparison["features"].setdefault(name, {"helpful": 0, "peer": 0})
                counts["helpful"] += value
                counts["peer"] += peer[name]

    output = {}
    for name, cohort in cohorts.items():
        lengths = sorted(cohort["lengths"])
        forms = []
        for key, form in sorted(cohort["forms"].items(), key=lambda item: -item[1]["notes"])[:5]:
            forms.append({
                "body_char_band": key[0], "url_count": "3+" if key[1] == 3 else str(key[1]),
                "emoji": key[2], "exclamation": key[3], "notes": form["notes"],
                "posts": len(form["posts"]), "authors": len(form["authors"]), "examples": form["examples"],
            })
        output[name] = {
            "notes": sum(cohort["statuses"].values()), "status_counts": dict(cohort["statuses"]),
            "helpful_body_chars": {
                label: lengths[round((len(lengths) - 1) * q)]
                for label, q in (("p10", .1), ("median", .5), ("p90", .9))
            } if lengths else {},
            "forms": forms,
        }
    return {
        "schema_version": 1, "data_date": data_date, "days": days,
        "created_from_millis": first_time, "created_to_millis": last_time,
        "measurement": "Unicode body characters after removing URLs and collapsing whitespace; URL occurrences; emoji-range and exclamation presence",
        "script_grouping": "Largest count among Latin, CJK, Cyrillic and Arabic ranges; mixed languages within each group",
        "comparison": comparison, "cohorts": output,
    }
