"""Source quotations and visual evidence must reach the final verifier."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from note_writer.support import evidence_quote_problems, source_excerpt
from note_writer.critic import critique
from note_writer.write_note import research_post_and_write_note
from data_models import Post, PostWithContext

class EvidenceQuotes(unittest.TestCase):
    def test_semantic_failure_revises_input_and_rechecks_the_repair(self):
        post = PostWithContext(post=Post(post_id='1', author_id='a', created_at='2026-11-28T00:00:00Z',
                                        text='A factual claim', media=[]))
        draft = {'decision': 'WRITE NOTE', 'note_text': 'The blackout was staged. https://example.org/record',
                 'misleading_tags': ['factual_error'], 'evidence': [], 'assumptions': [],
                 '_cited_urls': ['https://example.org/record']}
        repaired = {**draft, 'note_text': 'The original recording dates to 2020. https://example.org/record'}
        failed = {'submit': False, 'reason': 'A comment does not establish staging.',
                  'claim_checks': [{'claim': 'The blackout was staged.', 'kind': 'cause_or_intent',
                                    'established': False, 'support_role': 'third_party_interpretation'}]}
        with ExitStack() as stack:
            for name, value in {
                'observed_patterns_for_prompt': 'Observed reference', 'inspect_video': None,
                '_similar_notes_for_prompt': '', 'reverse_image.search': None,
                '_source_context': 'Checked source', 'url_status': 200, 'page_text': 'Source text',
                'evidence_quote_problems': [], 'unsupported_specifics': [], 'claim_opinion_score': 1.0,
                'image_check.check': ([], []),
            }.items():
                stack.enter_context(patch('note_writer.write_note.' + name, return_value=value))
            stack.enter_context(patch('note_writer.gates.url_status', return_value=200))
            model = stack.enter_context(patch('note_writer.write_note.get_grok_search_json_response',
                                               side_effect=[draft, repaired]))
            stack.enter_context(patch('note_writer.write_note.critique', side_effect=[failed, {'submit': True}]))
            result = research_post_and_write_note(post)
        self.assertIsNotNone(result.note)
        self.assertIn('2020', result.note.note_text)
        self.assertIn('The blackout was staged.', model.call_args_list[1].args[0])
        self.assertIn('Start a fresh draft', model.call_args_list[1].args[0])
        self.assertNotIn('history', model.call_args_list[1].kwargs)
        self.assertEqual(len([t for t in result.validation_trace if t['stage'] == 'critic']), 2)

    def test_source_excerpt_reaches_article_beyond_navigation(self):
        text = 'menu shopping entertainment sign in. ' * 120
        text += 'The Toronto recording was filmed in 2010 at the G20 protest. ' * 6
        text += 'footer subscribe privacy policy. ' * 120
        excerpt = source_excerpt(text, 'Toronto recording G20 protest')
        self.assertIn('The Toronto recording was filmed in 2010', excerpt)
        self.assertLessEqual(len(excerpt), 2400)

    def test_fabricated_quote_is_rejected_and_exact_quote_passes(self):
        url = 'https://example.org/record'
        source = 'The event was held in Toronto in 2010. ' + 'Archived official record. ' * 30
        quote = {'url': url, 'fact': 'The event was in Toronto.', 'quote': 'The event was held in Toronto in 2010.'}
        self.assertEqual(evidence_quote_problems([quote], {url: source}), [])
        fabricated = {**quote, 'quote': 'This video was filmed in London in 2026.'}
        self.assertIn('absent', evidence_quote_problems([fabricated], {url: source})[0])
        self.assertTrue(evidence_quote_problems([], {}))
        self.assertTrue(evidence_quote_problems([{**quote, 'quote': ''}], {url: source}))
        linked = {**quote, 'quote': 'The CBS video (archived here) contains the statement.'}
        page = 'The CBS video (archived here ) contains the statement. ' + source
        self.assertEqual(evidence_quote_problems([linked], {url: page}), [])

    def test_critic_gets_actual_frames_and_evidence_ledger(self):
        evidence = [{'fact': 'Recorded in Toronto.', 'url': 'https://example.org',
                     'quote': 'The event was held in Toronto.'}]
        with patch('note_writer.critic.get_grok_search_json_response', return_value={
            'evidence_supported': True, 'claim_addressed': True, 'issues': [], 'reason': 'Direct record',
            'claim_checks': [{'claim': 'Recorded in Toronto.', 'kind': 'other',
                              'support_role': 'primary_record', 'support_quote': 'Recorded in Toronto.',
                              'established': True}],
        }) as model:
            result = critique(None, 'The post', 'The correction', evidence=evidence,
                              image_urls=['data:image/jpeg;base64,frame'])
        self.assertTrue(result['submit'])
        self.assertEqual(model.call_args.kwargs['image_urls'], ['data:image/jpeg;base64,frame'])
        self.assertIn('The event was held in Toronto.', model.call_args.args[0])

    def test_comments_cannot_establish_cause_or_absence(self):
        for kind in ['cause_or_intent', 'absence']:
            with self.subTest(kind=kind), patch('note_writer.critic.get_grok_search_json_response', return_value={
                'evidence_supported': True, 'claim_addressed': True, 'issues': [], 'reason': 'A comment says so',
                'claim_checks': [{'claim': 'The power was cut.', 'kind': kind,
                                  'support_role': 'third_party_interpretation', 'support_quote': 'Power was cut',
                                  'established': True}],
            }):
                result = critique(None, 'post', 'note')
                self.assertFalse(result['submit'])
                self.assertIn('unsupported_assumption', result['issues'])

if __name__ == '__main__':
    unittest.main()
