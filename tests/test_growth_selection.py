"""Prefer observed new spread over stale cumulative popularity."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import ledger
from data_models import Post, PostWithContext
from selection import recent_growth, score_post


def metrics(likes, views=10000):
    return {'like_count': likes, 'retweet_count': 10, 'quote_count': 2,
            'reply_count': 5, 'impression_count': views}


def post(pid, values, now):
    return PostWithContext(post=Post(post_id=pid, author_id='a', created_at=now-timedelta(hours=3),
                                    text='The record states 2026.', media=[], public_metrics=values))


class GrowthSelection(unittest.TestCase):
    def test_growing_post_beats_large_stalled_post(self):
        now = datetime.now(timezone.utc)
        before = (now-timedelta(minutes=5)).isoformat()
        stalled = post('1', metrics(10000), now)
        growing = post('2', metrics(500), now)
        old_stalled = {'ts': before, 'post_public_metrics': metrics(10000)}
        old_growing = {'ts': before, 'post_public_metrics': metrics(400, 8000)}
        self.assertGreater(score_post(stalled, now)[0], score_post(growing, now)[0])
        slow, slow_parts = score_post(stalled, now, previous_observation=old_stalled)
        fast, fast_parts = score_post(growing, now, previous_observation=old_growing)
        self.assertGreater(fast, slow)
        self.assertEqual(slow_parts['recent_engagement_per_h'], 0)
        self.assertEqual(fast_parts['recent_like_per_h'], 1200)
        self.assertEqual(fast_parts['recent_impression_per_h'], 24000)

    def test_missing_reset_and_stale_snapshots_use_lifetime_fallback(self):
        now = datetime.now(timezone.utc)
        values = metrics(100)
        valid = {'ts': (now-timedelta(minutes=5)).isoformat(), 'post_public_metrics': metrics(90)}
        self.assertTrue(recent_growth(values, valid, now))
        for previous in [None, {**valid, 'ts': (now-timedelta(seconds=10)).isoformat()},
                         {**valid, 'ts': (now-timedelta(hours=2)).isoformat()},
                         {**valid, 'post_public_metrics': metrics(110)},
                         {**valid, 'post_public_metrics': {'like_count': 90}}]:
            with self.subTest(previous=previous):
                self.assertEqual(recent_growth(values, previous, now), {})
                self.assertEqual(score_post(post('1', values, now), now,
                                            previous_observation=previous)[1]['recent_velocity_used'], 0)

    def test_ledger_uses_latest_snapshot_without_collapsing_observation_interval(self):
        now = datetime.now(timezone.utc)
        def rec(minutes, decision='feed_observed'):
            return {'ts': (now-timedelta(minutes=minutes)).isoformat(), 'post_id': '1',
                    'decision': decision, 'post_public_metrics': metrics(100)}
        snapshots = [rec(20, 'seen'), rec(5), rec(.1), rec(1, 'research_completed')]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'LEDGER_PATH': str(Path(folder)/'ledger.jsonl')}):
            Path(ledger.ledger_path()).write_text('\n'.join(json.dumps(r) for r in snapshots))
            self.assertEqual(ledger.latest_feed_observations(now)['1']['ts'], rec(5)['ts'])


if __name__ == '__main__':
    unittest.main()
