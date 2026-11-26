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
from note_writer.gates import _landed_elsewhere, check_note, normalize_note, unseen_urls  # noqa: E402
from note_writer.support import specifics, unsupported_specifics  # noqa: E402
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
            normalize_note("Community Note: From 2019 \u2014 not 2024 https://a.com/x\u2013y"),
            "From 2019, not 2024\n\nhttps://a.com/x\u2013y",
        )
        self.assertEqual(normalize_note("Rose 3\u20135% https://b.org"), "Rose 3-5%\n\nhttps://b.org")
        # a link in the middle moves to the end, each link on its own line, no leftover punctuation
        self.assertEqual(
            normalize_note("From a 1981 book. https://a.org/b No other record exists. https://c.org/d."),
            "From a 1981 book. No other record exists.\n\nhttps://a.org/b\nhttps://c.org/d",
        )
        once = normalize_note("Point one. https://a.org/b https://c.org/d")
        self.assertEqual(normalize_note(once), once)

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
        self.assertFalse(check_note("So, the photo is from 2019. https://a.org/b", check_urls=False).ok)
        self.assertFalse(check_note("The clip is a game-changer for the case. https://a.org/b", check_urls=False).ok)
        self.assertEqual(normalize_note("The photo is from 2019! https://a.org/b"), "The photo is from 2019.\n\nhttps://a.org/b")


class Support(unittest.TestCase):
    NOTE = "The Army describes a detonation of lead azide at the Iowa plant affecting 148 square miles. https://army.mil/a"

    def test_specifics(self):
        nums, names = specifics(self.NOTE)
        self.assertEqual(nums, ["148"])
        self.assertEqual(names, ["Army", "Iowa"])

    def test_dates_and_possessives(self):
        nums, names = specifics("Per Janine Roberts' 1981 book, the blast on Nov. 25, 2026 and 3 March 2024 hit Texas. https://a.org")
        self.assertEqual(nums, ["1981", "2024", "2026"])
        self.assertEqual(names, ["Janine", "Roberts", "Texas"])

    def test_missing_and_present(self):
        page = ("the army announced a controlled detonation of lead azide in iowa. " * 40)
        self.assertEqual(unsupported_specifics(self.NOTE, {"https://army.mil/a": page}), ["148"])
        self.assertEqual(unsupported_specifics(self.NOTE, {"https://army.mil/a": page + " 148 square miles"}), [])
        # unreadable or thin pages: can't check, so no complaint
        self.assertEqual(unsupported_specifics(self.NOTE, {"https://army.mil/a": None}), [])

    def test_soft_redirect(self):
        self.assertTrue(_landed_elsewhere("https://a.com/news/x", "https://a.com/"))
        self.assertFalse(_landed_elsewhere("https://a.com/news/x", "https://a.com/news/x-new"))
        self.assertFalse(_landed_elsewhere("https://a.com/", "https://www.a.com/"))


class Benchmark(unittest.TestCase):
    def test_lookup(self):
        import benchmark as b
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "exemplars.jsonl").write_text("\n".join(json.dumps(x) for x in [
                {"id": "1", "post_id": "10", "text": "The video shows a fireworks factory explosion in Malta in June, not Ukraine. https://bbc.com/a"},
                {"id": "2", "post_id": "11", "text": "The bridge photo is from a 2019 flood in Venice. https://reuters.com/b"},
                {"id": "3", "post_id": "12", "text": "Senate rules require sixty votes for cloture. https://senate.gov/c"},
            ]))
            (Path(d) / "domains.json").write_text(json.dumps({"base": {"notes": 1000, "helpful": 0.1, "not_helpful": 0.02},
                "domains": {"bad.example": {"notes": 300, "helpful": 0.05, "not_helpful": 0.06}, "ok.example": {"notes": 300, "helpful": 0.1, "not_helpful": 0.02}}}))
            (Path(d) / "posts.json").write_text(json.dumps({"10": [3, 1], "11": [1, 0]}))
            (Path(d) / "cards.json").write_text(json.dumps({"base_helpful": 0.11, "notes": 1000, "days": 365,
                "cards": [{"opening": "this is not a", "notes": 500, "authors": 200, "helpful": 0.16, "media": False, "examples": ["This is not a law. https://a.org"]},
                          {"opening": "this video is ai", "notes": 900, "authors": 300, "helpful": 0.23, "media": True, "examples": ["This video is AI-generated. https://b.org"]}],
                "avoid": [{"opening": "there is no evidence", "notes": 2000, "authors": 1000, "helpful": 0.07}]}))
            old = b.DIR
            b.DIR = Path(d)
            for f in (b._index, b._domains, b._posts, b._cards):
                f.cache_clear()
            try:
                media_first = b.cards_for_prompt(True)
                self.assertLess(media_first.index("this video is ai"), media_first.index("this is not a"))
                text_first = b.cards_for_prompt(False)
                self.assertLess(text_first.index("this is not a"), text_first.index("this video is ai"))
                self.assertIn("there is no evidence", text_first)
                post = "Huge explosion at a fireworks factory in Malta caught on video"
                self.assertEqual([h["id"] for h in b.similar_helpful_notes(post)], ["1"])
                self.assertEqual(b.similar_helpful_notes(post, post_id="10"), [])  # never the same post
                self.assertIn("bad.example", b.distrusted_domain("https://www.bad.example/x"))
                self.assertIsNone(b.distrusted_domain("https://ok.example/x"))
                self.assertEqual(b.post_outcome("10"), (3, True))
                self.assertIsNone(b.post_outcome("99"))
            finally:
                b.DIR = old
                for f in (b._index, b._domains, b._posts, b._cards):
                    f.cache_clear()

    def test_cjk_tokens(self):
        import benchmark as b
        self.assertIn("地震", b.tokens("東京で地震があった"))


class ReverseImage(unittest.TestCase):
    def test_summarise_and_describe(self):
        from note_writer import reverse_image as ri
        web = {"bestGuessLabels": [{"label": "venice flood 2019"}], "webEntities": [{"description": "Venice"}, {"score": 1}],
               "pagesWithMatchingImages": [
                   {"url": "https://b.example/crop", "pageTitle": "Crop", "partialMatchingImages": [{"url": "x"}]},
                   {"url": "https://a.example/orig", "pageTitle": "Acqua alta", "fullMatchingImages": [{"url": "y"}]}]}
        found = ri.summarise(web)
        self.assertEqual(found["pages"][0]["url"], "https://a.example/orig")  # full matches first
        text = ri.describe_for_prompt(found)
        self.assertIn("venice flood 2019", text)
        self.assertIn("Same image on: https://a.example/orig", text)
        self.assertEqual(ri.describe_for_prompt(None), "")
        self.assertEqual(ri.page_urls(found), ["https://a.example/orig", "https://b.example/crop"])

    def test_no_key_no_call(self):
        from note_writer import reverse_image as ri
        os.environ.pop("GOOGLE_VISION_API_KEY", None)
        self.assertIsNone(ri.search(_post("photo of a flood", media=True)))


class Selection(unittest.TestCase):
    def test_prefilter(self):
        self.assertTrue(prefilter_reason(_post("lol")))
        self.assertEqual(prefilter_reason(_post("lol", media=True)), "")
        self.assertEqual(prefilter_reason(_post("This photo shows the 2024 flood in Valencia")), "")
        self.assertEqual(prefilter_reason(_post("東京で昨日大きな地震があり多数の建物が倒壊した")), "")

    def test_checkable(self):
        from selection import checkable_weight
        story = _post("I got in my car and I found my keys, we were shocked")
        claim = _post("BREAKING: officials confirmed 40% of ballots were destroyed")
        self.assertLess(checkable_weight(story), 1.0)
        self.assertGreater(checkable_weight(claim), 1.0)

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
    outs = [{"evaluator_type": "UrlValidity", "evaluator_score_bucket": uv.capitalize()}, {"evaluator_type": "HarassmentAbuse", "evaluator_score_bucket": "High"}]
    if co:
        outs.append({"evaluator_type": "ClaimOpinion", "evaluator_score_bucket": co.capitalize()})  # API sends "High"
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
