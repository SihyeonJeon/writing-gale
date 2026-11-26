# Running the writer

The writer runs on GitHub Actions (`.github/workflows/community_note_writer.yaml`), every 30 minutes.

GitHub's own `schedule` trigger proved unreliable (3 runs in 19 hours for a `7,37 * * * *` cron), so an
outside timer starts the workflow through the `workflow_dispatch` API every 30 minutes; the cron stays as a
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
| `NUM_POSTS` | 3 | posts researched per run |
| `DAILY_NOTE_CAP` | 10 | notes submitted per rolling 24h |
| `MAX_USD_PER_RUN` | 0.5 | xAI spend per run (checked before each post) |
| `DAILY_USD_CAP` | 5 | xAI spend per rolling 24h |
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
2. Skip posts with nothing to check before any LLM call.
3. For a post with video: download the smallest MP4 the API lists, take 4 frames and transcribe the audio
   locally (Whisper). Nothing is kept after the run.
   The prompt also carries template cards built from the public Community Notes data: the note openings
   with the highest measured helpful rates (at least 400 notes by 100+ writers each), each with a real
   helpful note, and the openings with below-average rates to avoid.
4. Research and draft with Grok (web + X search, post images, video frames, transcript). The model lists
   the claim, each fact with the page it came from, and any assumptions, before writing the note.
5. Deterministic gates: length, URL present and live (a deep link that lands on a front page counts as
   dead), cited pages must come from the search results, banned source types, style. Then the cited pages
   are opened and the note's numbers and names must be on them. One repair round, then give up.
6. X's own ClaimOpinion evaluator scores the note (`POST /2/notes/evaluate`, no LLM cost). Below 0, the
   note gets the repair round or is dropped.
7. Critic pass rates the note as raters from different perspectives would; only clear passes are submitted.

GitHub disables scheduled workflows in public repos after 60 days without commits; push something
before that.
