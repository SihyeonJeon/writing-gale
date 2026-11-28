# Running the writer

The writer runs on GitHub Actions (`.github/workflows/community_note_writer.yaml`), every 5 minutes. Each run reads the feed (X API, no LLM
cost) and researches at most one post, the highest-ranked one not handled yet.

GitHub's own `schedule` trigger proved unreliable (3 runs in 19 hours for a `7,37 * * * *` cron), so an
outside timer starts the workflow through the `workflow_dispatch` API every 5 minutes; the cron stays as a
fallback. Any always-on machine works: one `curl -X POST .../actions/workflows/community_note_writer.yaml/dispatches`
with a token that has Actions write access to this repo only. The script and systemd units are in
`tools/trigger/`. Everything that selects posts and writes notes runs from this repository on Actions; the
timer only says "run now".

## Switch

| `WRITER_MODE` variable | What scheduled runs do |
|---|---|
| unset / `off` | exit at once, no API calls |
| `dry_run` | research and write notes, don't submit |
| `live` | submit notes (test mode until `TEST_MODE=false`) |

```sh
gh variable set WRITER_MODE --body dry_run   # or live / off
```

The switch pauses or resumes the whole writer. Notes are never approved one by one: the contest
requires a fully automated writer.

## Limits (repository variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `NUM_POSTS` | 1 | posts researched per run |
| `DAILY_NOTE_CAP` | 10 | notes submitted per rolling 24h |
| `MAX_USD_PER_RUN` | 0.5 | xAI spend per run (checked before each post) |
| `DAILY_USD_CAP` | 5 | xAI spend per rolling 24h |
| `HOURLY_USD_CAP` | 0.4 | xAI spend per rolling hour, spreads the daily budget over the day |
| `TEST_MODE` | true | set to `false` only after admission |

## Secrets

`X_API_KEY`, `X_API_KEY_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`, `XAI_API_KEY`.

Optional: `GOOGLE_VISION_API_KEY` turns on reverse image search for posts with an image (Google Cloud
Vision web detection, capped at 30 calls a day and 900 per 30 days).

## Monitoring

- Each run's summary page shows counts, spend and admission progress (no post text).
- `uv run src/check_scores.py` lists every test note with its evaluator scores and the admission check.
- State (ledger of handled posts, spend, cached scores) lives in `data/` in the Actions cache.

## Pipeline per post

1. Rank eligible posts by freshness, spread, note requests, media and topic (`src/selection.py`).
   A post that just entered the feed counts double, fading over the next hours.
2. Skip posts with nothing to check before any LLM call.
3. For a post with video: download the smallest MP4 the API lists, take 4 frames and transcribe the audio
   locally (Whisper). Nothing is kept after the run.
   The prompt also carries observed forms built from public Community Notes: joint distributions of
   length, URL count and punctuation, with real Helpful examples, sample counts and source note IDs.
   Frequencies supply examples; they do not supply explanations of raters' motives.
4. Research and draft with Grok (web + X search, post images, video frames, transcript). The model lists
   the claim, each fact with its page and an exact supporting quotation, and any assumptions, before
   writing the note. Visual identifications need diagnostic features in the supplied media; comments
   interpreting a video do not establish its cause.
5. Deterministic gates: length, URL present and live (a deep link that lands on a front page counts as
   dead), cited pages must come from the search results, banned source types, style. Then the cited pages
   are opened and the note's numbers, names and evidence quotations are checked against readable text.
   One repair round, then stop that candidate.
   If the writer names a cited page as the source of the post's image, and that page is neither official
   nor one the reverse image search matched, the page's lead image is compared with the post's image by
   the vision model; a different picture sends the note back.
6. X's own ClaimOpinion evaluator scores the note (`POST /2/notes/evaluate`, no LLM cost). Below 0, the
   note gets the repair round or is dropped.
7. The critic receives the same images/video frames, evidence ledger and checked sources. It verifies
   factual support and alignment with the post's actual claim. Both checks must pass with no issues.
   Each assertion has a source-role check. Causes, intent and assertions of absence require a primary
   record; another user's interpretation does not establish them.

## Author observations and validation traces

The eligible-feed call expands `author_id` and requests `user.fields=id,name,username,verified,public_metrics`.
`Post.author` holds the author snapshot and its observation time, separately from post engagement.
First-seen ledger entries record author ID, username, author metrics, post metrics and post creation time.
Missing author data stays absent. Follower count refers to its observation time. Author popularity is
recorded for analysis; it is not an instruction to the note writer or an added ranking weight.

Each research result includes `validation_trace`: assembled-prompt SHA256, drafts and evidence quotes,
gate failures, numeric ClaimOpinion scores, and the critic verdict. The worker saves it as a
`research_completed` ledger event, including rejected candidates. These records connect an unexpected
output to its input and the check that accepted or rejected it.

For a repeatable preview on cached posts, without creating notes:

```sh
uv run tools/preview_notes.py --input data/review-context.json --output data/preview.json --max-usd 1
```

The preview runs source checks, X's evaluation endpoint and the critic. It limits each model request
to six tool calls and 4,000 output tokens, uses no retries, and stops starting calls with $0.20 remaining
in the requested preview budget. Inspect the saved drafts and verification trace, update general
inputs, and run the same cached cases again. Preview results do not enter the submission ledger.

Contest submissions continue to use the assigned contest feed. Author observations can inform
prioritization within that feed; following an author's account does not add its posts to that feed.

GitHub disables scheduled workflows in public repos after 60 days without commits; push something
before that.
