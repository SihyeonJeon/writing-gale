"""Regression checks for observed-form inputs and evidence-only verification."""

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import benchmark
from data_models import Post, PostWithContext
from note_patterns import HELPFUL, NOT_HELPFUL, build_patterns, script_group, text_features
from note_writer import critic
from note_writer.gates import banned_url, check_note, normalize_note
from note_writer.write_note import _get_prompt_for_note_writing, research_post_and_write_note


def sample_records():
    rows = [
        ("1", "p1", "a1", HELPFUL, "The photo is from 2020. 🙂 https://source.example/1"),
        ("2", "p1", "a2", "NEEDS_MORE_RATINGS", "The photo is from 2020. https://source.example/1 https://source.example/2"),
        ("3", "p2", "a1", HELPFUL, "The photo is from 2021. 🙂 https://source.example/3"),
        ("4", "p3", "a3", HELPFUL, "The photo is from 2022. 🙂 https://source.example/4"),
        ("5", "p4", "a4", NOT_HELPFUL, "The photo is from 2023. https://source.example/5"),
        ("6", "p5", "a5", HELPFUL, "이 사진은 다른 장소에서 촬영됐습니다. https://source.example/long-english-address"),
    ]
    return [
        {"noteId": nid, "tweetId": pid, "noteAuthorParticipantId": author, "currentStatus": status,
         "summary": text, "t": i}
        for i, (nid, pid, author, status, text) in reversed(list(enumerate(rows)))
    ]


class ObservedPatterns(unittest.TestCase):
    def test_literal_measurements_and_script_do_not_count_url_letters(self):
        self.assertEqual(script_group("이 사진 https://english.example/a-long-address"), "cjk")
        feature = text_features("A 🙂! https://example.org")
        self.assertEqual(feature["body_chars"], 4)
        self.assertEqual(feature["urls"], 1)
        self.assertTrue(feature["emoji"])
        self.assertTrue(feature["exclamation"])
        self.assertTrue(text_features("The flag is 🇵🇭 https://example.org")["emoji"])
        self.assertTrue(text_features("Item 1️⃣ https://example.org")["emoji"])
        self.assertEqual(text_features("A   photo.  https://example.org")["body_chars"], 8)

    def test_counts_keep_pending_peers_and_example_provenance(self):
        data = build_patterns(sample_records(), "2026-11-20", 180)
        self.assertEqual(data["cohorts"]["all"]["notes"], 6)
        latin = data["cohorts"]["latin"]
        self.assertEqual(latin["status_counts"][HELPFUL], 3)
        form = latin["forms"][0]
        self.assertEqual((form["notes"], form["posts"], form["authors"]), (3, 3, 2))
        self.assertTrue(form["emoji"])
        self.assertEqual(len(form["examples"]), 2)
        self.assertTrue(all("note_id" in example and "post_id" in example for example in form["examples"]))
        comparison = data["comparison"]
        self.assertEqual(comparison["posts"], 1)
        self.assertEqual(comparison["pairs"], 1)
        self.assertEqual(comparison["peer_statuses"], {"NEEDS_MORE_RATINGS": 1})
        self.assertEqual(comparison["features"]["emoji"], {"helpful": 1, "peer": 0})

    def test_input_reports_counts_and_excludes_this_posts_reference(self):
        data = build_patterns(sample_records(), "2026-11-20", 180)
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "DIR", Path(directory)):
            (Path(directory) / "observed_patterns.json").write_text(json.dumps(data))
            benchmark._patterns.cache_clear()
            try:
                context = benchmark.observed_patterns_for_prompt("The photo was taken elsewhere", "p3")
            finally:
                benchmark._patterns.cache_clear()
        self.assertIn("Snapshot 2026-11-20", context)
        self.assertIn("3/3 Helpful notes", context)
        self.assertIn("2 authors", context)
        self.assertIn("emoji present True", context)
        self.assertIn("Peer statuses include pending", context)
        self.assertIn("Observed note 3, post p2", context)
        self.assertNotIn("Observed note 4, post p3", context)

    def test_no_data_adds_no_synthetic_form(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "DIR", Path(directory)):
            benchmark._patterns.cache_clear()
            try:
                self.assertEqual(benchmark.observed_patterns_for_prompt("A post"), "")
            finally:
                benchmark._patterns.cache_clear()

    def test_exact_domain_counts_are_diagnostic_input(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark, "DIR", Path(directory)):
            (Path(directory) / "domains.json").write_text(json.dumps({
                "base": {"helpful": .1}, "domains": {
                    "source.example": {"notes": 1000, "helpful": .1, "not_helpful": .01,
                                       "helpful_notes": 100, "not_helpful_notes": 10, "needs_more_ratings": 890}
                },
            }))
            benchmark._domains.cache_clear()
            benchmark._patterns.cache_clear()
            try:
                context = benchmark.source_observations_for_prompt(["https://source.example/article"])
            finally:
                benchmark._domains.cache_clear()
                benchmark._patterns.cache_clear()
        self.assertIn("currently Helpful 100", context)
        self.assertIn("pending 890", context)
        self.assertIn("unadjusted", context)
        self.assertTrue(check_note("The record gives the date. https://source.example/article", check_urls=False).ok)

    def test_research_entry_passes_observed_forms_to_writer(self):
        post = PostWithContext(post=Post(post_id="10", author_id="a", created_at="2026-11-20T00:00:00Z",
                                        text="The photo is from another city.", media=[]))
        with ExitStack() as stack:
            for name, result in {
                "inspect_video": None, "_similar_notes_for_prompt": "", "reverse_image.search": None,
                "observed_patterns_for_prompt": "Snapshot 2026-11-20; observed note 123; emoji present True",
            }.items():
                stack.enter_context(patch("note_writer.write_note." + name, return_value=result))
            model = stack.enter_context(patch(
                "note_writer.write_note.get_grok_search_json_response",
                return_value={"decision": "NO NOTE NEEDED"},
            ))
            research_post_and_write_note(post)
        prompt = model.call_args.args[0]
        self.assertIn("observed note 123", prompt)
        self.assertIn("emoji present True", prompt)
        self.assertNotIn("Scored well", prompt)
        self.assertNotIn("roughly 60 to 200", prompt)

    def test_missing_public_reference_stops_before_paid_research(self):
        post = PostWithContext(post=Post(post_id="10", author_id="a", created_at="2026-11-20T00:00:00Z",
                                        text="A factual claim.", media=[]))
        with patch("note_writer.write_note.observed_patterns_for_prompt", return_value=""), \
             patch("note_writer.write_note.cards_for_prompt", return_value=""), \
             patch("note_writer.write_note.get_grok_search_json_response") as model, \
             patch("note_writer.write_note.reverse_image.search") as vision:
            result = research_post_and_write_note(post)
        self.assertIn("PUBLIC_REFERENCE_DATA_UNAVAILABLE", result.error)
        model.assert_not_called()
        vision.assert_not_called()

    def test_style_marks_are_preserved_and_ai_source_alias_is_blocked(self):
        text = "The photo shows the original event — here! 🙂 https://source.example/1"
        self.assertIn("— here! 🙂", normalize_note(text))
        self.assertTrue(check_note(text, check_urls=False).ok)
        self.assertTrue(banned_url("https://x.com/grok/status/123"))
        self.assertTrue(banned_url("https://mobile.twitter.com/Grok/status/123"))
        self.assertFalse(banned_url("https://x.com/deadpoolmovie/status/123"))


class EvidenceVerification(unittest.TestCase):
    def test_submit_requires_both_evidence_checks_and_no_issues(self):
        for supported, addressed, issues, expected in [
            (True, True, [], True),
            (False, True, [], False),
            (True, False, [], False),
            (True, True, ["unsupported_assumption"], False),
        ]:
            with self.subTest(supported=supported, addressed=addressed, issues=issues):
                with patch("note_writer.critic.get_grok_search_json_response", return_value={
                    "evidence_supported": supported, "claim_addressed": addressed,
                    "claim_checks": [{"claim": "The date", "kind": "other", "support_role": "primary_record",
                                      "support_quote": "The source states the date.", "established": True}],
                    "issues": issues, "reason": "The source states the date.",
                }):
                    self.assertEqual(critic.critique(None, "post", "note")["submit"], expected)

    def test_prompts_use_observed_form_and_evidence_requirements(self):
        prompt = _get_prompt_for_note_writing("post", "observed counts")
        self.assertIn("observed counts", prompt)
        self.assertIn("current research", prompt)
        self.assertNotIn("people on both sides", prompt)
        self.assertNotIn("no hashtags, emojis", prompt)
        self.assertNotIn("helpful_across_perspectives", json.dumps(critic.CRITIC_SCHEMA))
        self.assertIn("evidence_supported", critic._prompt("post", "note"))
