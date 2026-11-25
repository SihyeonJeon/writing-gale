"""
Offline checks: no network, no API keys, no LLM calls. CI runs these before the writer so a broken push
fails here instead of spending credits.

uv run python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ledger  # noqa: E402
import outcomes  # noqa: E402
from check_scores import admission_report  # noqa: E402
from data_models import Media, Post, PostWithContext  # noqa: E402
from note_writer import critic  # noqa: E402
from note_writer.gates import check_note, normalize_note, unseen_urls  # noqa: E402
from note_writer.write_note import NOTE_RESPONSE_SCHEMA, _get_prompt_for_note_writing  # noqa: E402
from selection import prefilter_reason, rank_posts  # noqa: E402


def _post(text: str, hours_ago: float = 1.0, media: bool = False, likes: int = 0, pid: str = "1") -> PostWithContext:
    return PostWithContext(
        post=Post(
            post_id=pid,
            author_id="a",
            created_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            text=text,
            media=[Media(media_key="m", media_type="photo", url="https://pbs.twimg.com/x.jpg")] if media else [],
            public_metrics={"like_count": likes},
        )
    )


def _all_required(schema: dict) -> bool:
    """xAI strict json_schema: every object lists all its properties as required."""
    if schema.get("type") == "object":
        if set(schema.get("required", [])) != set(schema.get("properties", {})):
            return False
        return all(_all_required(v) for v in schema["properties"].values())
    if schema.get("type") == "array":
        return _all_required(schema["items"])
    return True


class Gates(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(
            normalize_note("Community Note: From 2019 — not 2024 https://a.com/x–y"),
            "From 2019, not 2024 https://a.com/x–y",
        )
        self.assertEqual(normalize_note("Rose 3–5% https://b.org"), "Rose 3-5% https://b.org")

    def test_unseen(self):
        seen = ["https://www.reuters.com/world/a-b/", "https://apnews.com/article/x"]
        self.assertEqual(unseen_urls(["https://reuters.com/world/a-b"], seen), [])
        self.assertEqual(unseen_urls(["https://apnews.com/other"], seen), [])  # same site is fine
        self.assertEqual(unseen_urls(["https://made-up.org/p"], seen), ["https://made-up.org/p"])

    def test_check_note(self):
        good = "The video is from a 2019 protest in Santiago, Chile. https://apnews.com/article/x"
        self.assertTrue(check_note(good, check_urls=False).ok)
        self.assertFalse(check_note(good, check_urls=False, seen_urls=["https://bbc.com/a"]).ok)
        self.assertFalse(check_note("No source here", check_urls=False).ok)
        self.assertFalse(check_note("Obviously false https://x.org", check_urls=False).ok)
        self.assertFalse(check_note("See https://grok.com/share/abc", check_urls=False).ok)


class Selection(unittest.TestCase):
    def test_prefilter(self):
        self.assertTrue(prefilter_reason(_post("lol")))
        self.assertEqual(prefilter_reason(_post("lol", media=True)), "")
        self.assertEqual(prefilter_reason(_post("This photo shows the 2024 flood in Valencia")), "")
        self.assertEqual(prefilter_reason(_post("東京で昨日大きな地震があり多数の建物が倒壊した")), "")

    def test_rank_prefers_fresh(self):
        ranked = rank_posts([_post("old claim here", 30, likes=90000, pid="old"), _post("new claim here", 1, likes=6900, pid="new")])
        self.assertEqual(ranked[0][0].post.post_id, "new")


class Ledger(unittest.TestCase):
    def test_daily_count_and_handled(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["LEDGER_PATH"] = str(Path(d) / "l.jsonl")
            try:
                ledger.record("1", "submitted", test_mode=True)
                ledger.record("2", "submitted", test_mode=False)
                ledger.record("3", "dry_run", test_mode=True)
                ledger.record("4", "error", test_mode=True)
                old = {"ts": (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(), "post_id": "5", "decision": "submitted", "test_mode": True}
                with open(os.environ["LEDGER_PATH"], "a") as f:
                    f.write(json.dumps(old) + "\n")
                ledger.record("-", "run_summary", llm_usd=0.25)
                ledger.record("-", "run_summary", llm_usd=0.5)
                self.assertAlmostEqual(ledger.usd_last_24h(), 0.75)
                self.assertEqual(ledger.submitted_last_24h(test_mode=True), 1)
                self.assertEqual(ledger.handled_post_ids(), {"1", "2", "5"})
                self.assertEqual(ledger.handled_post_ids(include_dry_run=True), {"1", "2", "3", "5"})
            finally:
                del os.environ["LEDGER_PATH"]


def _note(i, text, status="", co=None, uv="high"):
    outs = [{"evaluator_type": "UrlValidity", "evaluator_score_bucket": uv}, {"evaluator_type": "HarassmentAbuse", "evaluator_score_bucket": "high"}]
    if co:
        outs.append({"evaluator_type": "ClaimOpinion", "evaluator_score_bucket": co})
    return {"id": str(i), "info": {"text": text}, "status": status, "test_result": {"evaluation_outcome": outs}, "test_mode": True}


class Outcomes(unittest.TestCase):
    def test_examples(self):
        notes = [_note(1, "good one", co="high"), _note(2, "bad one", co="low"), _note(3, "meh", co="medium")]
        ex = outcomes.examples_for_prompt(notes)
        self.assertIn("good one", ex)
        self.assertIn("bad one", ex)
        self.assertNotIn("meh", ex)
        self.assertEqual(outcomes.examples_for_prompt([]), "")

    def test_admission(self):
        notes = [_note(i, "n", co="high") for i in range(50)]
        self.assertIn("meets all criteria", admission_report(notes)[-1])
        notes[0] = _note(0, "n", co="high", uv="low")
        notes[1] = _note(1, "n", co="high", uv="low")
        notes[2] = _note(2, "n", co="high", uv="low")
        self.assertIn("not there yet", admission_report(notes)[-1])


class Prompts(unittest.TestCase):
    def test_schemas_strict(self):
        self.assertTrue(_all_required(NOTE_RESPONSE_SCHEMA))
        self.assertTrue(_all_required(critic.CRITIC_SCHEMA))

    def test_prompt_renders(self):
        p = _get_prompt_for_note_writing("Post ID: 1", "\nEXAMPLES\n")
        self.assertIn("Post ID: 1", p)
        self.assertIn("EXAMPLES", p)
        self.assertIn("NOT ENOUGH EVIDENCE", p)
        self.assertIn("relies on", critic._prompt("post", "note", ["a guess"]))


if __name__ == "__main__":
    unittest.main()
