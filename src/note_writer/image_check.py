"""
Before a media note goes out: does the page it links really carry the post's image?

A note that says "this photo is from X" and links a page is only as good as that page. The link is
accepted without a further look when
  - it is an official source (government, military, university, international body), or
  - the reverse image search already listed that page as carrying the same image.
That applies to one link only: the page the writer itself names as the image's source
(`image_source_url`). Ordinary supporting sources, such as a fact-check article with its own
thumbnail, are not compared. Otherwise the page's lead image (og:image / twitter:image) and the post's image are shown to the
vision model side by side, with search off, and it says whether they are the same picture.

Only a clear "different" blocks the note. A page with no lead image, or one we cannot read, is
reported as "not checked" rather than guessed at.
"""

import html
import re
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

from note_writer.gates import UA, _host
from note_writer.llm_util import get_grok_search_json_response

OFFICIAL_SUFFIXES = (".gov", ".mil", ".edu", ".int", ".gov.uk", ".gouv.fr", ".go.jp", ".go.kr", ".gc.ca", ".gov.au")
OFFICIAL_HOSTS = {"europa.eu", "un.org", "who.int", "nasa.gov"}
META_RE = re.compile(
    r"""<meta[^>]+(?:property|name)=["'](?:og:image(?::secure_url)?|twitter:image(?::src)?)["'][^>]*content=["']([^"']+)["']"""
    r"""|<meta[^>]+content=["']([^"']+)["'][^>]+(?:property|name)=["'](?:og:image(?::secure_url)?|twitter:image(?::src)?)["']""",
    re.IGNORECASE,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": ["same_image", "same_scene_or_event", "different", "cannot_tell"]},
        "reason": {"type": "string"},
    },
    "required": ["relation", "reason"],
    "additionalProperties": False,
}


def is_official(url: str) -> bool:
    h = _host(url)
    return h in OFFICIAL_HOSTS or any(h == s.lstrip(".") or h.endswith(s) for s in OFFICIAL_SUFFIXES) or any(
        h.endswith("." + o) for o in OFFICIAL_HOSTS
    )


def lead_image(url: str, timeout: float = 15.0) -> Optional[str]:
    """Address of the page's lead image, or None if the page can't be read or declares none."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "text/html,*/*"}, stream=True)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return None
        head = r.raw.read(400_000, decode_content=True).decode(r.encoding or "utf-8", errors="replace")
        r.close()
    except requests.RequestException:
        return None
    m = META_RE.search(head)
    if not m:
        return None
    src = html.unescape(m.group(1) or m.group(2)).strip()
    full = urljoin(url, src)
    return full if urlparse(full).scheme in ("http", "https") else None


def needs_check(source_url: str, note_urls: List[str], verified_pages: List[str], has_image: bool) -> bool:
    """True when the writer named a cited page as the image's source and nothing has vouched for it yet."""
    if not source_url or not has_image:
        return False
    u = source_url.rstrip("/")
    if u not in {n.rstrip("/") for n in note_urls}:  # only pages the note actually links
        return False
    return not is_official(u) and u not in {v.rstrip("/") for v in verified_pages}


def compare(post_image_url: str, page_image_url: str) -> Tuple[str, str]:
    prompt = (
        "Two images follow. The first is attached to a social media post. The second is the lead image of a "
        "web page cited as showing where the first image comes from. Compare what they show, ignoring crops, "
        "captions, watermarks, resizing and compression. Answer same_image (the same picture or frame), "
        "same_scene_or_event (clearly the same scene, event or subject from another angle or moment), different, "
        "or cannot_tell (too little to go on, or the second is a logo or generic site image). Reason in one sentence."
    )
    out = get_grok_search_json_response(prompt, SCHEMA, image_urls=[post_image_url, page_image_url], search=False)
    return out["relation"], out["reason"]


def check(source_url: str, note_urls: List[str], verified_pages: List[str], post_image_url: Optional[str]) -> Tuple[List[str], List[str]]:
    """(problems, notes for the log). Problems send the note back for repair."""
    if not needs_check(source_url, note_urls, verified_pages, bool(post_image_url)):
        return [], []
    img = lead_image(source_url)
    if not img:
        return [], [f"image not checked (no readable lead image): {source_url}"]
    try:
        relation, reason = compare(post_image_url, img)
    except Exception as e:
        return [], [f"image not checked ({type(e).__name__}): {source_url}"]
    if relation != "different":
        return [], [f"image check {relation}: {source_url}"]
    return [
        f"the page named as the image's source shows a different picture ({reason}): {source_url}. "
        "Cite a page that carries the same image, or do not say where the image is from"
    ], [f"image check different: {source_url}"]
