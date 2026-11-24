"""
Build an offline evaluation set from the public Community Notes data download
(https://communitynotes.x.com/guide/en/under-the-hood/download-data).

For recent English posts it picks
  - "write" posts: at least one misleading note reached CURRENTLY_RATED_HELPFUL
  - "skip"  posts: notes were written, every decided note was CURRENTLY_RATED_NOT_HELPFUL
and writes eval/cases.jsonl with the post id, label, and the reference notes.
Post text is fetched separately (fetch_posts.py) because the download has ids only.

usage: uv run --with pandas python eval/build_eval_set.py <dir with notes-*.zip and noteStatusHistory-*.zip> [n_per_label] [days]
"""

import json
import sys
from pathlib import Path

import pandas as pd

data = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
days = int(sys.argv[3]) if len(sys.argv) > 3 else 21

notes = pd.read_csv(next(data.glob("notes-*.zip")), sep="\t", dtype=str, on_bad_lines="skip", quoting=3,
                    usecols=["noteId", "createdAtMillis", "tweetId", "classification", "summary"])
status = pd.read_csv(next(data.glob("noteStatusHistory-*.zip")), sep="\t", dtype=str, on_bad_lines="skip",
                     usecols=["noteId", "currentStatus"])
d = notes.merge(status, on="noteId")
d["created"] = pd.to_datetime(d.createdAtMillis.astype("int64"), unit="ms")
d = d[d.created >= d.created.max() - pd.Timedelta(days=days)]
d["summary"] = d.summary.fillna("")
d = d[d.summary.map(lambda s: sum(c.isascii() for c in s) / max(1, len(s))) > 0.97]
mis = d[d.classification == "MISINFORMED_OR_POTENTIALLY_MISLEADING"]

per_post = mis.groupby("tweetId").currentStatus.agg(list)
write_ids = per_post[per_post.map(lambda s: "CURRENTLY_RATED_HELPFUL" in s)].index
skip_ids = per_post[per_post.map(lambda s: "CURRENTLY_RATED_NOT_HELPFUL" in s and "CURRENTLY_RATED_HELPFUL" not in s)].index

rng = 7
pick_w = pd.Series(write_ids).sample(min(n, len(write_ids)), random_state=rng)
pick_s = pd.Series(skip_ids).sample(min(n, len(skip_ids)), random_state=rng)

out = Path(__file__).parent / "cases.jsonl"
with out.open("w") as f:
    for label, ids in (("write", pick_w), ("skip", pick_s)):
        for tid in ids:
            refs = mis[mis.tweetId == tid][["currentStatus", "summary"]].to_dict("records")
            f.write(json.dumps({"post_id": tid, "label": label, "reference_notes": refs}, ensure_ascii=False) + "\n")
print(f"write candidates={len(write_ids)} skip candidates={len(skip_ids)}; wrote {len(pick_w)}+{len(pick_s)} cases to {out}")
