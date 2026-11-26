"""
Build the writer's benchmark files from the public Community Notes data, so its choices rest on what
raters actually rated helpful instead of on hand-set rules.

  uv run --with pandas tools/build_benchmark.py                 # download the latest daily files
  uv run --with pandas tools/build_benchmark.py NOTES.zip STATUS.zip   # or use local copies

Writes to data/benchmark/ (gitignored; note text written by other people is not committed):
  domains.json    per cited domain: notes, share rated helpful, share rated not helpful (last 180 days)
  exemplars.jsonl notes rated helpful in the last 60 days: {"id", "post_id", "text"}
  posts.json      for posts noted in the last 21 days: number of notes and whether any was rated helpful
  meta.json       data date and counts

Source: https://communitynotes.x.com/guide/en/under-the-hood/download-data
"""

import io
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "benchmark"
BASE = "https://ton.twimg.com/birdwatch-public-data/{d:%Y/%m/%d}/{kind}/{kind}-00000.zip"
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
CRH, CRNH = "CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"
DOMAIN_DAYS, EXEMPLAR_DAYS, POST_DAYS = 180, 60, 21
MIN_DOMAIN_NOTES = 200


def _download(kind: str) -> tuple[bytes, str]:
    for back in range(0, 5):
        d = datetime.now(timezone.utc) - timedelta(days=back)
        r = requests.get(BASE.format(d=d, kind=kind), timeout=600)
        if r.status_code == 200:
            return r.content, f"{d:%Y-%m-%d}"
    sys.exit(f"no public {kind} file found for the last 5 days")


def _read(src, cols) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(src) if isinstance(src, bytes) else src)
    with z.open(z.namelist()[0]) as f:
        return pd.read_csv(f, sep="\t", usecols=cols, quoting=3, dtype=str, on_bad_lines="skip")


def _host(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    for p in ("www.", "m.", "mobile.", "amp."):
        h = h.removeprefix(p)
    return h


def main() -> None:
    if len(sys.argv) == 3:
        notes_src, status_src, date = sys.argv[1], sys.argv[2], "local"
    else:
        notes_src, date = _download("notes")
        status_src, _ = _download("noteStatusHistory")
    notes = _read(notes_src, ["noteId", "createdAtMillis", "tweetId", "classification", "summary"])
    status = _read(status_src, ["noteId", "currentStatus"])
    notes = notes[notes.classification == "MISINFORMED_OR_POTENTIALLY_MISLEADING"]
    notes["t"] = pd.to_numeric(notes.createdAtMillis, errors="coerce")
    newest = notes.t.max()
    d = notes[notes.t > newest - DOMAIN_DAYS * 86400e3].merge(status, on="noteId")
    d["summary"] = d.summary.fillna("")

    # domains
    n, good, bad = Counter(), Counter(), Counter()
    for text, st in zip(d.summary, d.currentStatus):
        for h in {_host(u) for u in URL_RE.findall(text)} - {""}:
            n[h] += 1
            good[h] += st == CRH
            bad[h] += st == CRNH
    base = {"notes": len(d), "helpful": float((d.currentStatus == CRH).mean()), "not_helpful": float((d.currentStatus == CRNH).mean())}
    domains = {
        h: {"notes": c, "helpful": round(good[h] / c, 4), "not_helpful": round(bad[h] / c, 4)}
        for h, c in n.most_common() if c >= MIN_DOMAIN_NOTES
    }

    # exemplars: recent helpful notes, text flattened (the public files turn line breaks into spaces)
    ex = d[(d.currentStatus == CRH) & (d.t > newest - EXEMPLAR_DAYS * 86400e3)]
    # posts: did anyone's note get rated helpful
    recent = d[d.t > newest - POST_DAYS * 86400e3]
    posts = recent.groupby("tweetId").currentStatus.agg(notes="size", helpful=lambda s: int((s == CRH).any()))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "domains.json").write_text(json.dumps({"base": base, "domains": domains}))
    with (OUT / "exemplars.jsonl").open("w") as f:
        for nid, tid, text in zip(ex.noteId, ex.tweetId, ex.summary):
            f.write(json.dumps({"id": nid, "post_id": tid, "text": re.sub(r"\s+", " ", text).strip()}, ensure_ascii=False) + "\n")
    (OUT / "posts.json").write_text(json.dumps({t: [int(r.notes), int(r.helpful)] for t, r in posts.iterrows()}))
    meta = {"data_date": date, "built": datetime.now(timezone.utc).isoformat(), "notes": len(d), "domains": len(domains), "exemplars": len(ex), "posts": len(posts), **base}
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
