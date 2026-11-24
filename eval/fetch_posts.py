"""Fetch post text/media/context for eval/cases.jsonl via the X API (xurl), in batches of 100.
Writes eval/posts.json (raw API responses). Deleted/protected posts are reported and dropped."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cnapi.xurl_util import run_xurl  # noqa: E402

here = Path(__file__).parent
cases = [json.loads(line) for line in (here / "cases.jsonl").open()]
ids = [c["post_id"] for c in cases]
responses = []
for i in range(0, len(ids), 100):
    path = (
        f"/2/tweets?ids={','.join(ids[i:i + 100])}"
        "&tweet.fields=author_id,created_at,referenced_tweets,note_tweet,public_metrics,possibly_sensitive,lang"
        "&expansions=attachments.media_keys,referenced_tweets.id,referenced_tweets.id.attachments.media_keys"
        "&media.fields=duration_ms,height,media_key,preview_image_url,type,url,width"
    )
    responses.append(run_xurl(["xurl", path]))
(here / "posts.json").write_text(json.dumps(responses, ensure_ascii=False))
got = sum(len(r.get("data", [])) for r in responses)
errs = sum(len(r.get("errors", [])) for r in responses)
print(f"fetched {got} posts, {errs} unavailable")
