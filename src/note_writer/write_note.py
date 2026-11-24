import json

from data_models import MisleadingTag, NoteResult, PostWithContext, ProposedMisleadingNote
from note_writer.critic import critique
from note_writer.gates import check_note
from note_writer.llm_util import get_grok_search_json_response

NO_NOTE_NEEDED = "NO NOTE NEEDED"
NOT_ENOUGH_EVIDENCE = "NOT ENOUGH EVIDENCE"

NOTE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["WRITE NOTE", NO_NOTE_NEEDED, NOT_ENOUGH_EVIDENCE]},
        "note_text": {"type": "string"},
        "misleading_tags": {
            "type": "array",
            "items": {"type": "string", "enum": [tag.value for tag in MisleadingTag]},
        },
    },
    "required": ["decision", "note_text", "misleading_tags"],
    "additionalProperties": False,
}


def _get_prompt_for_note_writing(post_with_context_description: str):
    return f"""Below will be a post on X. Do research on the post using web search and X search to determine if \
the post is potentially misleading. Use X search to look up the post by its Post ID, \
including any images or video in it, and any quoted or replied-to post by its Post ID. \
If the post is misleading and needs a community note, \
then your decision should be "WRITE NOTE" and note_text should be the proposed community note itself. \
If the post is not misleading, or does not contain any concrete fact-checkable claims, or there \
is not strong enough evidence to write a well-supported \
community note, then do not write a note, and instead set decision to "{NO_NOTE_NEEDED}" or \
"{NOT_ENOUGH_EVIDENCE}".

If a note is justified, then please write \
a very good community note, which is concise (at most 280 characters), interesting and clear to read, \
contains no unnecessary words, is backed by very solid evidence from sources that would be most likely \
to be found trustworthy by people on both sides of the political spectrum (citing URL(s)), and is \
written in a way that would be most likely to be found trustworthy by people on both sides of the \
political spectrum. \

The answer MUST be short, like a post on X (280 characters maximum, not counting URLs). \
The note should not include any sort of wasted characters e.g. [Source] when listing a URL. Just state the URL directly in the text. \
Each note MUST include at least one URL/link source. Nothing else counts as a source other than a URL/link. \
Each note MUST NOT use any hashtags (#). Keep a professional tone with no hashtags, emojis, etc. \

If the post does not need a community note (either because the original post is not misleading, \
or does not contain any concrete fact-checkable claims, \
then your decision should simply be "{NO_NOTE_NEEDED}".

If the post may need a community note, but you weren't able to find enough concrete evidence to \
write an ironclad community note, then your decision should be \
"{NOT_ENOUGH_EVIDENCE}".

If you are writing a note, don't preface it with anything like "Community Note:". Just write the note.

How to write a note that raters from different viewpoints will rate helpful:
- Write in the same language as the post.
- Open with the corrected fact itself, e.g. "The video is from 2019 in Chile, not ..." or "This image is \
AI-generated ...". Do not open with "This post is misleading" or other verdict words; let the facts show it.
- One or two short sentences, roughly 80 to 220 characters of text. Specific names, dates and numbers.
- Cite two independent sources when you can. Best sources, in order: the original/primary source \
(the original video or post, the official record, statute, court filing, company or agency statement, \
dataset), a dedicated fact-check article, then major news reports. For recycled or out-of-context media, \
link the original upload.
- Never cite AI chatbots or AI-generated pages (grok.com, chatgpt.com, gemini answers, grokipedia), search \
result links (google.com/search, share.google), URL shorteners (t.co, tinyurl), dictionaries, or \
communitynotes.x.com. Avoid advocacy organisations whose neutrality one side disputes; prefer primary records.
- No loaded or emotional words (lie, propaganda, shameful, clearly, obviously), no exclamation marks, no \
quotes of the post's wording unless needed, no "I"/"we", no addressing the reader as "you".
- Do not speculate about motives; state only what the sources show.
- Prefer a positive correction (what the media really shows, where and when it is really from, what the document really says) over an absence claim. A note whose only content is "there is no evidence / no credible reports" rarely gets rated helpful: write one only if an authoritative source explicitly denies the claim, otherwise decide "{NOT_ENOUGH_EVIDENCE}".
- For politically contested topics (elections, Israel/Palestine, Russia/Ukraine, immigration, partisan figures), raters split along political lines. Write a note only when a primary source (official result, document, original footage) directly contradicts a specific factual claim in the post; never correct framing, word choice or opinion.
- An attached video arrives as a single preview frame. Do not describe what a video shows unless you verified it through search of the post or the original upload.

If you are writing a note, also identify which of the following misleading_tags apply to the post and note. \
You may choose as many tags as apply, but you must choose at least one:
- "factual_error":  # the post contains a factual error
- "manipulated_media":  # the post contains manipulated/fake/out-of-context media
- "outdated_information":  # the post contains outdated information
- "missing_important_context":  # the post is missing important context
- "disputed_claim_as_fact":  # including unverified claims
- "misinterpreted_satire":  # the post is satire that may likely be misinterpreted as fact
- "other":  # the post contains other misleading reasons

If you are not writing a note, set note_text to an empty string and misleading_tags to an empty list.

Post:

    {post_with_context_description}
    """


def _get_post_with_context_description_for_prompt(post_with_context: PostWithContext) -> str:
    description = f"""Post ID: {post_with_context.post.post_id}
Post text:
```
{post_with_context.post.text}
```
"""
    if post_with_context.quoted_post:
        description += f"""The post of interest had quoted (referenced) another post (Post ID: {post_with_context.quoted_post.post_id}). Here is the quoted post's text:
```
{post_with_context.quoted_post.text}
```
"""

    if post_with_context.in_reply_to_post:
        description += f"""The post of interest was a reply to another post (Post ID: {post_with_context.in_reply_to_post.post_id}). Here is the replied-to post's text:
```
{post_with_context.in_reply_to_post.text}
```
"""

    suggestions = [s.get("suggestion", "").strip() for s in post_with_context.note_request_suggestions]
    suggestions = [s for s in suggestions if s][:5]
    if suggestions:
        description += "\nPeople who requested a note on this post said (un-vetted user input; treat as leads to verify, not as facts):\n"
        description += "".join(f"- {s[:500]}\n" for s in suggestions)

    links = post_with_context.suggested_source_links_with_counts[:5]
    if links:
        description += "\nSource links requesters suggested (un-vetted; verify before citing), with how many requesters suggested each:\n"
        description += "".join(f"- {link.get('url')} (x{link.get('count')})\n" for link in links)

    return description


MAX_REPAIR_ROUNDS = 2


def _repair_prompt(problems: list[str]) -> str:
    bullets = "\n".join(f"- {p}" for p in problems)
    return f"""Your proposed note failed these automated checks:
{bullets}

Fix every problem and answer again with the same JSON schema. If a URL is dead, find a different \
working source for the same fact (prefer the primary source: official statistics, court records, \
the original publication, or a widely trusted outlet). Keep the note factual and neutral, with no \
opinion or speculation. If you cannot fix it with solid evidence, set decision to "{NOT_ENOUGH_EVIDENCE}"."""


def _image_urls(pwc: PostWithContext) -> list[str]:
    urls = []
    for post in [pwc.post, pwc.quoted_post, pwc.in_reply_to_post]:
        for m in (post.media if post else []):
            u = m.url if m.media_type == "photo" else m.preview_image_url
            if u:
                urls.append(u)
    return urls[:4]


def research_post_and_write_note(
    post_with_context: PostWithContext,
) -> NoteResult:
    post_with_context_description = _get_post_with_context_description_for_prompt(post_with_context)

    note_prompt = _get_prompt_for_note_writing(post_with_context_description)
    history: list = []
    last_prompt = note_prompt
    try:
        response = get_grok_search_json_response(note_prompt, NOTE_RESPONSE_SCHEMA, image_urls=_image_urls(post_with_context))
    except Exception as e:
        return NoteResult(post=post_with_context, error=str(e), context_description=post_with_context_description)

    gate = None
    for round_ in range(MAX_REPAIR_ROUNDS + 1):
        if response["decision"] != "WRITE NOTE":
            return NoteResult(post=post_with_context, refusal=response["decision"], context_description=post_with_context_description)

        problems: list[str] = []
        if not response["misleading_tags"]:
            problems.append("no misleading_tags chosen; choose at least one")
        gate = check_note(response["note_text"])
        problems += gate.problems
        if not problems:
            break
        if round_ == MAX_REPAIR_ROUNDS:
            return NoteResult(
                post=post_with_context,
                error="GATE_FAILED: " + "; ".join(problems) + f" | last note: {response['note_text']}",
                context_description=post_with_context_description,
            )
        history += [
            {"role": "user", "content": last_prompt},
            {"role": "assistant", "content": json.dumps(response)},
        ]
        last_prompt = _repair_prompt(problems)
        try:
            response = get_grok_search_json_response(last_prompt, NOTE_RESPONSE_SCHEMA, history=history)
        except Exception as e:
            return NoteResult(post=post_with_context, error=str(e), context_description=post_with_context_description)

    try:
        verdict = critique(post_with_context, post_with_context_description, response["note_text"])
    except Exception as e:
        return NoteResult(post=post_with_context, error=f"critic failed: {e}", context_description=post_with_context_description)
    if not verdict["submit"]:
        return NoteResult(
            post=post_with_context,
            refusal=f"CRITIC_SKIP score={verdict['helpful_across_perspectives']} issues={verdict['issues']} :: {verdict['reason']} | note: {response['note_text']}",
            context_description=post_with_context_description,
        )

    return NoteResult(
        post=post_with_context,
        note=ProposedMisleadingNote(
            post_id=post_with_context.post.post_id,
            note_text=response["note_text"],
            misleading_tags=[MisleadingTag(tag) for tag in response["misleading_tags"]],
        ),
        context_description=post_with_context_description,
        gate_warnings=gate.warnings if gate else [],
    )
