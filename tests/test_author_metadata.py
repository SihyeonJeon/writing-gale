"""Keep author snapshots distinct from post engagement and absent author data."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cnapi.get_api_eligible_posts import _fetch_posts_eligible_for_notes, _parse_posts_eligible_response


class AuthorMetadata(unittest.TestCase):
    def test_request_expands_authors_in_existing_feed_call(self):
        with patch('cnapi.get_api_eligible_posts.run_xurl', return_value={}) as run:
            _fetch_posts_eligible_for_notes(10, True)
        path = run.call_args.args[0][1]
        self.assertIn('expansions=author_id,', path)
        self.assertIn('user.fields=id,name,username,verified,public_metrics', path)
        self.assertIn('test_mode=true', path)

    def test_author_and_post_metrics_are_separate_and_missing_stays_missing(self):
        post = {'id': '1', 'author_id': 'a', 'created_at': '2026-11-28T00:00:00Z',
                'text': 'A claim', 'public_metrics': {'like_count': 91},
                'referenced_tweets': [{'id': '2', 'type': 'quoted'}]}
        quoted = {'id': '2', 'author_id': 'b', 'created_at': '2026-11-27T00:00:00Z', 'text': 'Source'}
        resp = {'data': [post], 'includes': {'users': [
            {'id': 'a', 'username': 'author', 'name': 'Author', 'public_metrics': {'followers_count': 1234}}
        ], 'tweets': [quoted]}}
        before = datetime.now(timezone.utc)
        result = _parse_posts_eligible_response(resp)[0]
        self.assertEqual(result.post.public_metrics['like_count'], 91)
        self.assertEqual(result.post.author.public_metrics['followers_count'], 1234)
        self.assertEqual(result.post.author.user_id, result.post.author_id)
        self.assertGreaterEqual(result.post.author.observed_at, before)
        self.assertIsNone(result.quoted_post.author)
        self.assertEqual(result.quoted_post.public_metrics, {})

    def test_older_cached_posts_need_no_author_snapshot(self):
        result = _parse_posts_eligible_response({'data': [
            {'id': '1', 'author_id': 'a', 'created_at': '2026-11-28T00:00:00Z', 'text': 'A claim'}
        ]})[0]
        self.assertIsNone(result.post.author)


if __name__ == '__main__':
    unittest.main()
