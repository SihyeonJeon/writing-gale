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

import requests

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
TRAILING_PUNCT = ".,;:!?"
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")
OPINION_RE = re.compile(
    r"\b(i think|i believe|in my opinion|clearly|obviously|probably|likely|seems|appears to|"
    r"we should|shameful|disgusting|ridiculous|lies|liar|propaganda)\b",
    re.IGNORECASE,
)
PREFIX_RE = re.compile(r"^\s*(community note|note|context)\s*:", re.IGNORECASE)

MAX_CHARS = 280
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

_url_cache: Dict[str, Optional[int]] = {}


def extract_urls(text: str) -> List[str]:
    return [u.rstrip(TRAILING_PUNCT) for u in URL_RE.findall(text)]


def text_length_without_urls(text: str) -> int:
    return len(URL_RE.sub("", text).strip())


def url_status(url: str, timeout: float = 10.0) -> Optional[int]:
    """Final HTTP status after redirects, or None on network error. Cached per process."""
    if url in _url_cache:
        return _url_cache[url]
    status: Optional[int] = None
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*"}
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        status = r.status_code
        # Many sites reject HEAD (403/405) but serve GET fine; the evaluator cares about GET.
        if status != 200:
            r = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
            status = r.status_code
            r.close()
    except requests.RequestException:
        status = None
    _url_cache[url] = status
    return status


# Statuses that usually mean "this client was bot-blocked", not "the page is gone". Major trusted
# sources (Reuters 401, CDC 403 seen from a laptop) do this. Treated as warnings until real
# UrlValidity scores tell us whether X's checker is blocked too.
BLOCKED_STATUSES = {401, 403, 429, 999}


@dataclass
class GateResult:
    ok: bool
    problems: List[str] = field(default_factory=list)
    dead_urls: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_note(text: str, check_urls: bool = True) -> GateResult:
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
    m = OPINION_RE.search(URL_RE.sub("", text))
    if m:
        problems.append(f'note contains opinion/hedging wording: "{m.group(0)}"')

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
                problems.append(f"URL does not return HTTP 200 (got {s}): {u}")

    return GateResult(ok=not problems, problems=problems, dead_urls=dead, warnings=warnings)
