"""
Second-opinion pass on a finished note, before it spends daily quota.

The rubric mirrors the reasons Community Notes raters can pick when rating a note helpful or not
helpful. Notes rated Not Helpful shrink the writer's daily limit, so a skipped borderline note is
cheaper than a submitted one.
"""

from data_models import PostWithContext
from note_writer.llm_util import get_grok_search_json_response

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "note_not_needed",
                    "opinion_or_speculation",
                    "argumentative_or_biased_language",
                    "sources_missing_or_unreliable",
                    "sources_do_not_support_note",
                    "misses_key_points_or_irrelevant",
                    "incorrect_information",
                    "hard_to_understand",
                    "corrects_framing_not_fact",
                    "only_absence_of_evidence",
                ],
            },
        },
        "helpful_across_perspectives": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["issues", "helpful_across_perspectives", "reason"],
    "additionalProperties": False,
}

SUBMIT_MIN_SCORE = 4
BLOCKING_ISSUES = {
    "note_not_needed",
    "incorrect_information",
    "sources_do_not_support_note",
    "corrects_framing_not_fact",
    "opinion_or_speculation",
}


def _prompt(post_description: str, note_text: str, assumptions: list[str] | None = None) -> str:
    extra = ""
    if assumptions:
        extra = "\nThe writer listed these points the note relies on that no source states directly:\n" + "".join(
            f"- {a}\n" for a in assumptions
        ) + "If the note depends on any of them, flag opinion_or_speculation.\n"
    return f"""You review a proposed Community Note before it is submitted on X. Community Notes only show a note \
when raters who usually disagree with each other both rate it helpful.

Judge the note strictly, as a skeptical rater from each side of the relevant debate would.

Raters call a note HELPFUL when it: cites high-quality sources; is easy to understand; directly addresses \
the post's claim; provides important context; uses neutral or unbiased language.
Raters call a note NOT HELPFUL when it: contains opinion or speculation; uses argumentative or biased \
language; has missing, unreliable or irrelevant sources; is incorrect; misses key points; is hard to \
understand; or when the post is not misleading and no note is needed.

Also flag:
- corrects_framing_not_fact: the note disputes interpretation, framing or opinion, not a checkable fact
- only_absence_of_evidence: the note's only content is "there is no evidence / no credible reports"

Use search to check that the cited URLs say what the note claims and that the post really makes the claim.

Rate helpful_across_perspectives 1-5 (5 = raters on both sides would very likely rate it helpful). \
List every issue that applies. Keep reason to one sentence.

Post:
{post_description}

Proposed note:
{note_text}
{extra}"""


def critique(post_with_context: PostWithContext, post_description: str, note_text: str, assumptions: list[str] | None = None) -> dict:
    verdict = get_grok_search_json_response(_prompt(post_description, note_text, assumptions), CRITIC_SCHEMA)
    blocking = sorted(set(verdict["issues"]) & BLOCKING_ISSUES)
    verdict["submit"] = verdict["helpful_across_perspectives"] >= SUBMIT_MIN_SCORE and not blocking
    verdict["blocking"] = blocking
    return verdict
