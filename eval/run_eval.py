"""Run the note writer (no submission) on the eval set and score its decisions.

- write posts: a CRH note exists -> we want WRITE NOTE (recall)
- skip  posts: every decided note was rated Not Helpful -> we want no note (precision proxy)

Caveat: posts are days/weeks old, so the model's search may find the live Community Note.
Results go to eval/results-<tag>.jsonl.

usage: uv run python eval/run_eval.py [tag] [concurrency]
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dotenv  # noqa: E402

from cnapi.get_api_eligible_posts import _parse_posts_eligible_response  # noqa: E402
from note_writer.write_note import research_post_and_write_note  # noqa: E402

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")
here = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 4

cases = {c["post_id"]: c for c in (json.loads(line) for line in (here / "cases.jsonl").open())}
posts = []
for resp in json.loads((here / "posts.json").read_text()):
    posts += _parse_posts_eligible_response(resp)
posts = [p for p in posts if p.post.post_id in cases]


def run(p):
    r = research_post_and_write_note(p)
    c = cases[p.post.post_id]
    outcome = "write" if r.note else "critic_skip" if (r.refusal or "").startswith("CRITIC_SKIP") else ("gate_failed" if (r.error or "").startswith("GATE_FAILED") else ("error" if r.error else "skip"))
    return {
        "post_id": p.post.post_id, "label": c["label"], "outcome": outcome, "refusal": r.refusal,
        "error": r.error, "note": r.note.note_text if r.note else None,
        "tags": [t.value for t in r.note.misleading_tags] if r.note else None,
        "warnings": r.gate_warnings, "post_text": p.post.text, "has_media": bool(p.post.media),
        "reference_notes": c["reference_notes"],
    }


with ThreadPoolExecutor(conc) as ex:
    results = list(ex.map(run, posts))

out = here / f"results-{tag}.jsonl"
with out.open("w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

tab = Counter((r["label"], r["outcome"]) for r in results)
for label in ("write", "skip"):
    tot = sum(v for (lab, _), v in tab.items() if lab == label)
    print(f"{label:5s} posts={tot}: " + ", ".join(f"{o}={tab[(label, o)]}" for o in ("write", "critic_skip", "skip", "gate_failed", "error")))
w = [r for r in results if r["label"] == "write"]
s = [r for r in results if r["label"] == "skip"]
print(f"recall on CRH posts: {sum(r['outcome'] == 'write' for r in w) / max(1, len(w)):.2f}")
print(f"note rate on all-Not-Helpful posts: {sum(r['outcome'] == 'write' for r in s) / max(1, len(s)):.2f}")
print(f"saved {out}")
