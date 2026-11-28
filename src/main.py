import argparse
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

from cnapi.get_api_eligible_posts import get_posts_eligible_for_notes
from cnapi.submit_note import submit_note
from data_models import NoteResult, Post, PostWithContext
import dotenv
import ledger
import outcomes
from selection import rank_posts
from note_writer.llm_util import USAGE
from note_writer.write_note import research_post_and_write_note
from selection import prefilter_reason

# Public CI logs: set LOG_POST_TEXT=0 to keep post text out of them (post id, decision and note still print).
VERBOSE = os.getenv("LOG_POST_TEXT", "1") != "0"


def _xurl_error_details(exc: Exception) -> str:
    """xurl prints the X API's error response to stdout; surface it instead of a bare exit code."""
    if isinstance(exc, subprocess.CalledProcessError):
        return ((exc.stdout or "") + (exc.stderr or "")).strip() or str(exc)
    return str(exc)


def _check_setup() -> None:
    """Fail early, with a clear message, on the most common setup problems."""
    if not os.getenv("XAI_API_KEY"):
        sys.exit("XAI_API_KEY is not set. Copy .env.example to .env and fill in your xAI API key.")
    if shutil.which("xurl") is None:
        sys.exit("xurl is not installed, or its folder is not on your PATH. See step A.1 of the README.")


def _worker(
    post_with_context: PostWithContext,
    dry_run: bool = False,
    test_mode: bool = True,
    submit_guard: Callable[[], bool] | None = None,
):
    """
    Fetch and try to write and submit a note for one post.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    """
    reason = prefilter_reason(post_with_context)
    if reason:
        ledger.record(post_with_context.post.post_id, "prefilter_skip", reason=reason, test_mode=test_mode)
        print(f"Post {post_with_context.post.post_id}: skipped before research ({reason})")
        return
    note_result: NoteResult = research_post_and_write_note(post_with_context)
    if note_result.validation_trace:
        ledger.record(
            post_with_context.post.post_id, "research_completed", test_mode=test_mode,
            validation_trace=note_result.validation_trace,
        )

    log_strings: List[str] = ["-" * 20, f"Post: {post_with_context.post.post_id}", "-" * 20]
    if note_result.context_description is not None and VERBOSE:
        log_strings.append(
            f"\n*POST TEXT AND ANY QUOTED OR REPLIED-TO POST:*\n{note_result.context_description}\n"
        )
    if note_result.error is not None:
        log_strings.append(f"\n*ERROR:* {note_result.error}")
    if note_result.refusal:
        log_strings.append(f"\n*REFUSAL:* {note_result.refusal}")
    if note_result.note:
        log_strings.append(f"\n*NOTE:*\n  {note_result.note.note_text}\n")
        log_strings.append(
            f"\n*MISLEADING TAGS:*\n  {[tag.value for tag in note_result.note.misleading_tags]}\n"
        )

    post_id = post_with_context.post.post_id
    note_text = note_result.note.note_text if note_result.note else None
    tags = [t.value for t in note_result.note.misleading_tags] if note_result.note else None
    if note_result.refusal and note_result.refusal.startswith("CRITIC_SKIP"):
        ledger.record(post_id, "critic_skip", detail=note_result.refusal[:2000], test_mode=test_mode)
    elif note_result.refusal:
        ledger.record(post_id, note_result.refusal, test_mode=test_mode)
    elif note_result.error:
        decision = "gate_failed" if note_result.error.startswith("GATE_FAILED") else "error"
        if "ClaimOpinion evaluator scored" in note_result.error:
            decision = "evaluator_skip"
        ledger.record(post_id, decision, error=note_result.error[:2000], test_mode=test_mode)
    elif note_result.note is not None and dry_run:
        ledger.record(post_id, "dry_run", note_text=note_text, misleading_tags=tags, gate_warnings=note_result.gate_warnings, test_mode=test_mode)

    if note_result.note is not None and not dry_run:
        if submit_guard is not None and not submit_guard():
            ledger.record(post_id, "target_skip", note_text=note_text, test_mode=test_mode)
            print("".join(log_strings) + "\nTest note target reserved by another worker; skipping submission.")
            return
        try:
            try:
                response = submit_note(note=note_result.note, test_mode=test_mode, verbose_if_failed=False)
            except Exception as first:
                # If the API objects to the line breaks before the links, the research is already paid
                # for: send the same note on one line instead of losing it.
                if "already created a note" in _xurl_error_details(first) or "\n" not in note_result.note.note_text:
                    raise
                log_strings.append(f"\n*RETRYING ON ONE LINE* after: {_xurl_error_details(first)[:300]}\n")
                note_result.note.note_text = " ".join(note_result.note.note_text.split())
                note_text = note_result.note.note_text
                response = submit_note(note=note_result.note, test_mode=test_mode, verbose_if_failed=False)
            data = response.get("data", {})
            note_id = data.get("id") or data.get("note_id")
            log_strings.append(f"\n*SUCCESSFULLY SUBMITTED NOTE*\nNote ID: {note_id}\n")
            ledger.record(post_id, "submitted", note_id=note_id, note_text=note_text, misleading_tags=tags, gate_warnings=note_result.gate_warnings, test_mode=test_mode)
        except Exception as e:
            details = _xurl_error_details(e)
            log_strings.append(
                "\n*ERROR SUBMITTING NOTE* (if it says a note already exists, we already wrote a note "
                f"on this post; moving on):\n{details}\n"
            )
            decision = "duplicate" if "already created a note" in details else "submit_error"
            ledger.record(post_id, decision, error=details[:2000], note_text=note_text, misleading_tags=tags, test_mode=test_mode)
    print("".join(log_strings) + "\n")


def main(
    num_posts: int = 10,
    fetch: int = 50,
    dry_run: bool = False,
    concurrency: int = 1,
    test_mode: bool = True,
    max_usd: float = 0.0,
    daily_cap: int = 0,
    daily_usd: float = 0.0,
    hourly_usd: float = 0.0,
):
    """
    Get up to `num_posts` recent posts eligible for notes and write notes for them.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    `max_usd` stops starting new posts once this run's xAI spend reaches it (0 = no limit).
    `daily_cap` stops once this many notes were submitted in the last 24h (0 = no limit).
    `daily_usd` caps xAI spend over the last 24h across runs (0 = no limit).
    `hourly_usd` caps xAI spend over the last hour, so frequent runs spread the daily budget over the
    day instead of using it up in a few hours (0 = no limit).
    """
    test_target = int(os.getenv("TEST_NOTE_TARGET") or 0)
    remaining_target = None
    if test_mode and test_target and not dry_run:
        # Refresh from X before spending, and include local submissions while X catches up.
        notes = outcomes.refresh(include_live=False)
        submitted_ids = {str(n["id"]) for n in notes}
        submitted_ids.update(
            str(r["note_id"]) for r in ledger.all_records()
            if r.get("decision") == "submitted" and r.get("test_mode", True) and r.get("note_id")
        )
        remaining_target = max(test_target - len(submitted_ids), 0)
        print(f"Test note target {test_target}: {len(submitted_ids)} submitted, {remaining_target} left")
        if not remaining_target:
            ledger.record("-", "run_summary", processed=0, reason="test note target reached", test_mode=True)
            print("Test note target reached. Waiting for admission review.")
            return

    # A reached limit stops research, not the feed read: the feed is still fetched and logged (no LLM
    # cost) so we keep knowing when each post first appeared.
    pause = ""
    if hourly_usd:
        spent_1h = ledger.usd_last_hours(1)
        print(f"Hourly budget ${hourly_usd:.2f}: ${spent_1h:.2f} spent in the last hour")
        if spent_1h >= hourly_usd:
            pause = "hourly budget reached"
    if daily_usd:
        spent = ledger.usd_last_24h()
        print(f"Daily budget ${daily_usd:.2f}: ${spent:.2f} spent in the last 24h")
        if spent >= daily_usd:
            pause = "daily budget reached"
        else:
            max_usd = min(max_usd, daily_usd - spent) if max_usd else daily_usd - spent
    if daily_cap and not dry_run:
        done_24h = ledger.submitted_last_24h(test_mode=test_mode)
        remaining = daily_cap - done_24h
        print(f"Daily cap {daily_cap}: {done_24h} submitted in the last 24h, {max(remaining, 0)} left")
        if remaining <= 0:
            pause = "daily cap reached"
        # Most researched posts end without a note, so research a few more than the notes left.
        num_posts = min(num_posts, max(remaining, 0) * 3)
    if pause:
        num_posts = 0

    print(f"Getting up to {num_posts} recent posts eligible for notes (test_mode={test_mode})")
    try:
        eligible_posts: List[PostWithContext] = get_posts_eligible_for_notes(
            max_results=max(fetch, num_posts, 1), test_mode=test_mode,
            max_pages=max(1, min(int(os.getenv("FEED_PAGES") or 1), 10)),
        )
    except subprocess.CalledProcessError as e:
        sys.exit(
            "Could not fetch posts eligible for notes. The X API said:\n"
            f"{_xurl_error_details(e)}\n\n"
            "Check that xurl is logged in (`xurl auth status`, README step A.1), that your account is "
            "enrolled as an AI Note Writer (README step 2.2), and that your app has Read and write "
            "permissions (README step 2.1)."
        )
    print(f"Found {len(eligible_posts)} recent posts eligible for notes")
    # In dry-run mode, posts already dry-run are skipped too, so a scheduled dry run doesn't pay twice.
    handled = ledger.handled_post_ids(include_dry_run=dry_run)
    skipped = [p for p in eligible_posts if p.post.post_id in handled]
    eligible_posts = [p for p in eligible_posts if p.post.post_id not in handled]
    # Rank by freshness, spread, demand and media (see selection.py) and keep the top num_posts.
    ranked = rank_posts(eligible_posts, first_seen=ledger.first_seen())
    # Every feed post is logged once with its ranking features (numbers only, no text). Joined later with
    # the public data, this shows which kinds of posts end up with a note people rate helpful.
    already_seen = ledger.seen_post_ids()
    for p, sc, parts in ranked:
        if p.post.post_id not in already_seen:
            author = p.post.author
            ledger.record(
                p.post.post_id, "seen", score=round(sc, 3),
                author_id=p.post.author_id,
                author_username=author.username if author else None,
                author_public_metrics=author.public_metrics if author else None,
                author_observed_at=author.observed_at.isoformat() if author else None,
                post_created_at=p.post.created_at.isoformat(),
                post_public_metrics=p.post.public_metrics,
                **{k: round(v, 3) for k, v in parts.items()},
            )
    for p, sc, parts in ranked[: num_posts + 5]:
        print(f"  {p.post.post_id} score={sc:.2f} {parts}")
    eligible_posts = [p for p, _, _ in ranked[:num_posts]]
    if skipped:
        print(f"  Skipping {len(skipped)} posts already handled (see data/ledger.jsonl)")
    print(
        f"  Eligible Post IDs: {', '.join([str(post_with_context.post.post_id) for post_with_context in eligible_posts])}\n"
    )
    if pause:
        print(f"Not researching this run: {pause}.")
        ledger.record("-", "run_summary", processed=0, reason=pause, test_mode=test_mode)
        return
    if len(eligible_posts) == 0:
        print("No posts to process.")
        ledger.record("-", "run_summary", processed=0, test_mode=test_mode)
        return

    submitted_before = ledger.submitted_last_24h(test_mode=test_mode)
    state = {"started": 0, "reserved": 0}
    lock = threading.Lock()

    def reserve_submission() -> bool:
        # Keep the target exact while researching multiple candidates for the last open slot.
        # A failed API call consumes this run's reservation. The next run refreshes X and retries.
        with lock:
            if state["reserved"] >= remaining_target:
                return False
            state["reserved"] += 1
            return True

    def run_one(post: PostWithContext) -> None:
        with lock:
            if remaining_target is not None and state["reserved"] >= remaining_target:
                return
            if max_usd and USAGE["usd"] >= max_usd:
                return
            if daily_cap and not dry_run and ledger.submitted_last_24h(test_mode=test_mode) >= daily_cap:
                return
            state["started"] += 1
        _worker(
            post, dry_run, test_mode,
            submit_guard=reserve_submission if remaining_target is not None else None,
        )

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for future in [executor.submit(run_one, post) for post in eligible_posts]:
                future.result()
    else:
        for post in eligible_posts:
            run_one(post)

    stopped = len(eligible_posts) - state["started"]
    submitted = ledger.submitted_last_24h(test_mode=test_mode) - submitted_before
    ledger.record(
        "-", "run_summary", processed=state["started"], not_started=stopped, submitted=submitted,
        llm_calls=USAGE["calls"], llm_usd=round(USAGE["usd"], 4), dry_run=dry_run, test_mode=test_mode,
    )
    print(
        f"Done. Researched {state['started']} posts, submitted {submitted} notes. "
        f"Grok: {USAGE['calls']} calls, ${USAGE['usd']:.3f}."
        + (f" Stopped early (budget/cap), {stopped} posts left for the next run." if stopped else "")
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Run note-writing bot once.")
    parser.add_argument("--num-posts", type=int, default=10, help="Number of posts to process")
    parser.add_argument("--fetch", type=int, default=50, help="How many eligible posts to fetch and rank (max 100)")
    parser.add_argument("--dry-run", action="store_true", help="Do not submit notes to the API, just print them to the console")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent tasks to run")
    parser.add_argument(
        "--max-usd", type=float, default=float(os.getenv("MAX_USD_PER_RUN") or 0),
        help="Stop starting new posts once this run's xAI spend reaches this (0 = no limit)",
    )
    parser.add_argument(
        "--daily-cap", type=int, default=int(os.getenv("DAILY_NOTE_CAP") or 0),
        help="Stop once this many notes were submitted in the last 24h (0 = no limit)",
    )
    parser.add_argument(
        "--hourly-usd", type=float, default=float(os.getenv("HOURLY_USD_CAP") or 0),
        help="Cap xAI spend over the last hour across runs (0 = no limit)",
    )
    parser.add_argument(
        "--daily-usd", type=float, default=float(os.getenv("DAILY_USD_CAP") or 0),
        help="Cap xAI spend over the last 24h across runs (0 = no limit)",
    )
    args = parser.parse_args()
    _check_setup()
    main(
        num_posts=args.num_posts,
        fetch=min(args.fetch, 100),
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        test_mode=os.getenv("TEST_MODE", "true").strip().lower() != "false",
        max_usd=args.max_usd,
        daily_cap=args.daily_cap,
        daily_usd=args.daily_usd,
        hourly_usd=args.hourly_usd,
    )
