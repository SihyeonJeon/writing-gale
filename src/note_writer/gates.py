"""
Deterministic pre-submit checks. Each one maps to a hard rule or an admission evaluator:

- length:      1-280 chars, URLs not counted (contest rule). Whether the API counts URLs is
               unverified; a rejection shows up as a submit error in the ledger.
- has_url:     the API rejects notes without an http(s) URL
- url_live:    UrlValidity evaluator wants >= 95% of notes with every link returning HTTP 200
- style:       no hashtags / emojis / "Community Note:" prefixes (starter guidance, ClaimOpinion hygiene)
- opinion:     cheap lexical screen for hedging/opinion words that tend to drag ClaimOpinion down
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
TRAILING_PUNCT = ".,;:!?"
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")
OPINION_RE = re.compile(
    r"\b(i think|i believe|in my opinion|clearly|obviously|probably|likely|seems|appears to|"
    r"we should|shameful|disgusting|ridiculous|lies|liar|propaganda|"
    r"uncorroborated|baseless|debunked|hoax|conspiracy theor(?:y|ies)|so-called)\b",
    re.IGNORECASE,
)
PREFIX_RE = re.compile(r"^\s*(community note|note|context)\s*:", re.IGNORECASE)

MAX_CHARS = 280

# Domains whose citations do badly in public Community Notes data (low CRH, high CRNH) or that
# aren't real sources: AI chatbots, search-result links, shorteners, dictionaries.
BANNED_DOMAINS = {
    "grok.com", "chatgpt.com", "grokipedia.com", "gemini.google.com", "perplexity.ai",
    "share.google", "t.co", "tinyurl.com", "bit.ly",
    "communitynotes.x.com", "merriam-webster.com", "dictionary.com", "dictionary.cambridge.org",
}


def _host(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return h.removeprefix("www.").removeprefix("m.")


def banned_url(url: str) -> bool:
    h = _host(url)
    if h == "google.com" and "/search" in url:
        return True
    return any(h == d or h.endswith("." + d) for d in BANNED_DOMAINS)
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

_url_cache: Dict[str, Optional[int]] = {}


def extract_urls(text: str) -> List[str]:
    return [u.rstrip(TRAILING_PUNCT) for u in URL_RE.findall(text)]


def text_length_without_urls(text: str) -> int:
    return len(re.sub(r"\s+", " ", URL_RE.sub("", text)).strip())


SOFT_REDIRECT = 310  # our own code: the link answered 200 but landed on the site's front page


def _landed_elsewhere(requested: str, final: str) -> bool:
    """A deep link that ends on the bare front page is a removed article, not a source."""
    try:
        a, b = urlparse(requested), urlparse(final)
    except ValueError:
        return False
    return len(a.path.strip("/")) > 0 and b.path.strip("/") == "" and not b.query


def _fetch_status(url: str, method: str, timeout: float, headers: dict) -> int:
    r = requests.request(method, url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
    status = r.status_code
    if status == 200 and _landed_elsewhere(url, r.url):
        status = SOFT_REDIRECT
    r.close()
    return status


def url_status(url: str, timeout: float = 10.0) -> Optional[int]:
    """Final HTTP status after redirects, or None on network error. Cached per process."""
    if url in _url_cache:
        return _url_cache[url]
    status: Optional[int] = None
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*"}
    try:
        status = _fetch_status(url, "HEAD", timeout, headers)
        # Many sites reject HEAD (403/405) but serve GET fine; the evaluator cares about GET.
        if status != 200:
            status = _fetch_status(url, "GET", timeout, headers)
    except requests.RequestException:
        status = None
    if status is None:  # one retry for transient network errors / slow hosts
        try:
            status = _fetch_status(url, "GET", 20, headers)
        except requests.RequestException:
            status = None
    _url_cache[url] = status
    return status


# Statuses that usually mean "this client was bot-blocked", not "the page is gone". Major trusted
# sources (Reuters 401, CDC 403 seen from a laptop) do this. Treated as warnings until real
# UrlValidity scores tell us whether X's checker is blocked too.
BLOCKED_STATUSES = {401, 403, 429, 999}


def normalize_note(text: str) -> str:
    """Fix style and layout without another LLM call: dashes, label prefixes, and links.
    Layout follows how people write notes that get rated helpful: the point first, then a blank
    line, then each source link on its own line. URLs themselves are left untouched."""
    parts = re.split(f"({URL_RE.pattern})", text)
    body, urls = [], []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # URL; sentence punctuation stuck to it goes back to the text
            url = part.rstrip(TRAILING_PUNCT)
            if url not in urls:
                urls.append(url)
            body.append(part[len(url):])
            continue
        part = re.sub(r"(\d)\s*[\u2013\u2014]\s*(\d)", r"\1-\2", part)  # number ranges
        part = re.sub(r"\s*[\u2014\u2013]\s*", ", ", part)
        body.append(part)
    prose = re.sub(r"\s+", " ", "".join(body))
    prose = re.sub(r"\s+([.,;:!?])", r"\1", prose)
    prose = re.sub(r"([.!?])[.,;:]+", r"\1", prose)  # "sentence. ." left where a link was lifted out
    prose = PREFIX_RE.sub("", prose).strip(" ,;:")
    if not urls:
        return prose
    return (prose + "\n\n" if prose else "") + "\n".join(urls)


def _norm_url(url: str) -> str:
    try:
        u = urlparse(url.strip())
    except ValueError:
        return url
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    return f"{host}{u.path.rstrip('/')}" + (f"?{u.query}" if u.query else "")


def unseen_urls(note_urls: List[str], seen_urls: List[str]) -> List[str]:
    """Note URLs that the research step's search tools never returned, by exact page or by site.
    A model citing a page it never opened is how hallucinated or non-supporting links get in."""
    pages = {_norm_url(u) for u in seen_urls}
    hosts = {_host(u) for u in seen_urls}
    return [u for u in note_urls if _norm_url(u) not in pages and _host(u) not in hosts]


@dataclass
class GateResult:
    ok: bool
    problems: List[str] = field(default_factory=list)
    dead_urls: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_note(text: str, check_urls: bool = True, seen_urls: Optional[List[str]] = None) -> GateResult:
    problems: List[str] = []
    urls = extract_urls(text)

    if not urls:
        problems.append("note has no http(s) URL source")
    n = text_length_without_urls(text)
    if n == 0:
        problems.append("note has no text besides URLs")
    if n > MAX_CHARS:
        problems.append(f"note is {n} chars without URLs (max {MAX_CHARS})")
    if "#" in URL_RE.sub("", text):
        problems.append("note contains a hashtag")
    if EMOJI_RE.search(text):
        problems.append("note contains an emoji")
    if PREFIX_RE.search(text):
        problems.append('note starts with a label like "Community Note:"')
    body = URL_RE.sub("", text)
    if re.search("[\u2014\u2013]", body):
        problems.append("note uses an em/en dash; use a period or comma (dashes read as machine-written and score slightly lower)")
    if re.search(r"\?\s*(?:yes|no)\b", body, re.IGNORECASE):
        problems.append("note asks a question and answers it; state the fact directly")
    m = OPINION_RE.search(body)
    if m:
        problems.append(f'note contains opinion/hedging wording: "{m.group(0)}"')

    for u in urls:
        if banned_url(u):
            problems.append(f"cites a source raters distrust or that is not a real source (AI chatbot, search link, shortener, dictionary): {u}")

    # Only enforced when the research step returned citations at all (it can answer from memory).
    if seen_urls:
        for u in unseen_urls(urls, seen_urls):
            problems.append(f"cites a page your search never returned; cite only pages you actually found: {u}")

    dead: List[str] = []
    if check_urls and urls:
        with ThreadPoolExecutor(max_workers=min(4, len(urls))) as ex:
            statuses = list(ex.map(url_status, urls))
    warnings: List[str] = []
    if check_urls and urls:
        for u, s in zip(urls, statuses):
            if s == 200:
                continue
            if s in BLOCKED_STATUSES:
                warnings.append(f"URL bot-blocked from here (HTTP {s}), may still be fine: {u}")
            else:
                dead.append(u)
                why = "redirects to the site's front page, so the article is gone" if s == SOFT_REDIRECT else f"got {s}"
                problems.append(f"URL does not return HTTP 200 ({why}): {u}")

    return GateResult(ok=not problems, problems=problems, dead_urls=dead, warnings=warnings)
