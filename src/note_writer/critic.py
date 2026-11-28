"""Verify current source support and claim alignment before submission."""

import json

from data_models import PostWithContext
from note_writer.llm_util import get_grok_search_json_response

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_supported": {"type": "boolean"},
        "claim_addressed": {"type": "boolean"},
        "claim_checks": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {"type": "string", "enum": ["cause_or_intent", "absence", "identity", "other"]},
                    "support_role": {"type": "string", "enum": ["primary_record", "attributed_statement", "third_party_interpretation", "missing"]},
                    "support_quote": {"type": "string"},
                    "established": {"type": "boolean"},
                },
                "required": ["claim", "kind", "support_role", "support_quote", "established"],
                "additionalProperties": False,
            },
        },
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
    "required": ["evidence_supported", "claim_addressed", "claim_checks", "issues", "reason"],
    "additionalProperties": False,
}


def _prompt(post_description: str, note_text: str, assumptions: list[str] | None = None, evidence: list[dict] | None = None) -> str:
    extra = ""
    if assumptions:
        extra = "\nThe draft lists these points it relies on without direct source support:\n" + "".join(
            f"- {assumption}\n" for assumption in assumptions
        )
    if evidence:
        extra += "\nDraft evidence ledger (verify each quotation and its scope):\n" + json.dumps(evidence, ensure_ascii=False)
    return f"""Verify the proposed Community Note against the post and its cited evidence.

Set evidence_supported=true only when each factual statement is directly established by the cited
pages and checked-source excerpts. Preserve their uncertainty, dates and numbers. For image or video
identity, require the actual matching recording or other direct evidence of the identification.
Inspect the supplied images and video frames yourself. For species or object identification, require
visible diagnostic features matching the source. General descriptions alone cannot identify the
particular image. Distinguish a visible sequence from its cause; comments asserting a power cut do
not prove it. Verify dates for the particular recording, rather than a different event's schedule.
Apply this distinction literally: "the screen and power indicator turn off" is an observation;
"someone cut power", "the crash was faked" and "the controller is unpaired" assert a cause,
intent or device state. A comment making that assertion only documents the commenter's interpretation.
It does not independently establish the assertion. Without a directly documented cause/device state,
set evidence_supported=false and include unsupported_assumption, even when multiple comments agree.
For "no recording exists" or "has not confirmed", require direct evidence establishing that exact
scope. Another user's assertion and an unsuccessful search do not prove absence. Check an account's
parody label as a self-description; it alone does not prove every factual claim false.
Set claim_addressed=true only when the source-supported correction addresses a specific factual
claim made in the target post. Use quoted and replied-to posts as context.

Fill claim_checks for every assertion in the note. Classify its kind, name the source's role and
quote the passage establishing it. A reply interpreting the target video is third_party_interpretation,
even when the reply states its conclusion confidently. For cause_or_intent and absence, established
requires a primary_record explicitly documenting that cause or exact scope of absence. The target
video is a primary record of visible events, not of a hidden cause or intent. A statement explicitly
attributed in the note can use attributed_statement; an unqualified assertion cannot borrow that role.

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


def critique(
    post_with_context: PostWithContext, post_description: str, note_text: str,
    assumptions: list[str] | None = None, evidence: list[dict] | None = None,
    image_urls: list[str] | None = None,
) -> dict:
    verdict = get_grok_search_json_response(
        _prompt(post_description, note_text, assumptions, evidence), CRITIC_SCHEMA, image_urls=image_urls,
    )
    checks = verdict.get("claim_checks") or []
    unsupported = [
        c for c in checks
        if not c["established"] or (
            c["kind"] in {"cause_or_intent", "absence"} and c["support_role"] != "primary_record"
        )
    ]
    if unsupported:
        verdict["evidence_supported"] = False
        if "unsupported_assumption" not in verdict["issues"]:
            verdict["issues"].append("unsupported_assumption")
    verdict["submit"] = bool(checks) and verdict["evidence_supported"] and verdict["claim_addressed"] and not verdict["issues"]
    return verdict
