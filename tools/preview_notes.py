"""Run a local writing regression on cached post contexts, without submitting notes.

Example: uv run tools/preview_notes.py --input data/review-context.json --output data/preview.json
Input is a list of PostWithContext objects, or review records containing `post_context`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from data_models import PostWithContext
from note_patterns import text_features
from note_writer import llm_util
from note_writer.write_note import research_post_and_write_note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, action='append', help='Default: all cached cases')
    parser.add_argument('--max-usd', type=float, default=1.0, help='Stop new calls with $0.20 remaining; usage is reported after each request')
    args = parser.parse_args()
    if args.max_usd < .20:
        parser.error('--max-usd must be at least 0.20')
    load_dotenv()
    records = json.loads(args.input.read_text())
    indices = args.case_index if args.case_index is not None else list(range(len(records)))
    # Validate every selected input before making the first paid request.
    posts = [(i, PostWithContext.model_validate(records[i].get('post_context', records[i]))) for i in indices]
    request = llm_util._make_request
    start_usd = llm_util.USAGE['usd']

    def bounded_request(payload, max_attempts=1):
        if llm_util.USAGE['usd'] - start_usd >= args.max_usd - .20:
            raise RuntimeError('PREVIEW_BUDGET_STOP')
        return request({**payload, 'max_output_tokens': 4000, 'max_tool_calls': 6}, max_attempts=1)

    llm_util._make_request = bounded_request
    cases = []
    try:
        for index, post in posts:
            if llm_util.USAGE['usd'] - start_usd >= args.max_usd - .20:
                break
            start = time.monotonic()
            before = llm_util.USAGE['usd']
            result = research_post_and_write_note(post)
            cases.append({
                'case_index': index, 'post_id': post.post.post_id,
                'result': result.model_dump(mode='json'),
                'features': text_features(result.note.note_text) if result.note else None,
                'seconds': round(time.monotonic() - start, 2), 'usd': llm_util.USAGE['usd'] - before,
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                'submitted': 0, 'usd': llm_util.USAGE['usd'] - start_usd,
                'selected_cases': len(posts), 'cases': cases,
            }, ensure_ascii=False, indent=2))
            print(f"case={index} note={bool(result.note)} refusal={result.refusal} error={result.error}", flush=True)
    finally:
        llm_util._make_request = request


if __name__ == '__main__':
    main()
