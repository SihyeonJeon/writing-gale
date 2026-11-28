"""Verify current source support and claim alignment before submission."""

from data_models import PostWithContext
from note_writer.llm_util import get_grok_search_json_response

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_supported": {"type": "boolean"},
        "claim_addressed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "post_has_no_checkable_claim",
                    "claim_not_addressed",
                    "sources_missing",
                    "sources_do_not_support_note",
                    "unsupported_assumption",
                    "incorrect_information",
                    "unverified_media_identity",
                ],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["evidence_supported", "claim_addressed", "issues", "reason"],
    "additionalProperties": False,
}


def _prompt(post_description: str, note_text: str, assumptions: list[str] | None = None) -> str:
    extra = ""
    if assumptions:
        extra = "\nThe draft lists these points it relies on without direct source support:\n" + "".join(
            f"- {assumption}\n" for assumption in assumptions
        )
    return f"""Verify the proposed Community Note against the post and its cited evidence.

Set evidence_supported=true only when each factual statement is directly established by the cited
pages and checked-source excerpts. Preserve their uncertainty, dates and numbers. For image or video
identity, require the actual matching recording or other direct evidence of the identification.
Set claim_addressed=true only when the source-supported correction addresses a specific factual
claim made in the target post. Use quoted and replied-to posts as context.

Use search to open the cited pages when needed. Reference notes and observed frequency tables supply
format examples. Use the current cited evidence for verification. A site's citation frequency does
not establish that a particular page supports a particular assertion.

List the specific failed verification checks. In reason, cite the relevant source statement or the
specific missing support. Keep reason to one sentence. Keep predictions of human reactions, inferred
rating motives and imagined explanations for observed frequencies out of this verification.

Post and checked sources:
{post_description}

Proposed note:
{note_text}
{extra}"""


def critique(post_with_context: PostWithContext, post_description: str, note_text: str, assumptions: list[str] | None = None) -> dict:
    verdict = get_grok_search_json_response(_prompt(post_description, note_text, assumptions), CRITIC_SCHEMA)
    verdict["submit"] = verdict["evidence_supported"] and verdict["claim_addressed"] and not verdict["issues"]
    return verdict
