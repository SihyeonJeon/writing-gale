"""
Check, without an LLM, that the specifics in a note are on the pages it cites.

The writer model says its sources support the note; this opens the cited pages and looks. Every number
(2+ digits) and every proper name in the note should appear somewhere in the text of the cited pages.
A miss sends the note back for one repair round ("these specifics are not on your cited pages").

Limits, on purpose:
- Pages we can't read (bot-blocked, JavaScript-only, X posts, PDFs) are skipped, and if too little text was
  readable overall the check doesn't run. It can only catch, never approve.
- It checks presence, not meaning. The critic pass still judges whether the source supports the claim.
"""

import html
import re
from typing import Dict, List, Optional, Tuple

import requests

from note_writer.gates import UA, URL_RE, _host, extract_urls

MAX_BYTES = 3_000_000
MIN_TEXT_CHARS = 1500
UNREADABLE_HOSTS = {"x.com", "twitter.com", "youtube.com", "youtu.be", "instagram.com", "tiktok.com", "facebook.com"}

_page_cache: Dict[str, Optional[str]] = {}

# Capitalised words that are not names, or that pages write in other forms (Nov. vs November).
_NOT_NAMES = {
    "the", "this", "that", "these", "those", "there", "they", "their", "its", "his", "her", "and", "but", "for",
    "not", "nor", "with", "from", "after", "before", "since", "while", "when", "according", "official",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "president", "senator",
}


_MONTHS = "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"


def _fetch_text(url: str, timeout: float) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "text/html,*/*"}, stream=True)
        ctype = r.headers.get("Content-Type", "")
        text = None
        if r.status_code == 200 and ("html" in ctype or "text/plain" in ctype):
            raw = r.raw.read(MAX_BYTES, decode_content=True).decode(r.encoding or "utf-8", errors="replace")
            raw = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", raw)
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))).lower()
        r.close()
        return text
    except requests.RequestException:
        return None


def archived_copy(url: str, timeout: float = 15.0) -> Optional[str]:
    """Address of the Internet Archive's latest snapshot of a page, if it has one. Used to read pages
    that turn automated readers away; the archive serves public copies and asks for no workaround."""
    try:
        r = requests.get("https://archive.org/wayback/available", params={"url": url}, timeout=timeout)
        snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and str(snap.get("status")) == "200":
            return snap["url"].replace("http://", "https://", 1)
    except (requests.RequestException, ValueError):
        pass
    return None


def page_text(url: str, timeout: float = 15.0) -> Optional[str]:
    """Visible text of an HTML page, lower-cased; None if it can't be read directly or from the archive."""
    if url in _page_cache:
        return _page_cache[url]
    text = _fetch_text(url, timeout)
    if text is None or len(text) < 500:
        snap = archived_copy(url)
        archived = _fetch_text(snap, timeout) if snap else None
        if archived and len(archived) > len(text or ""):
            text = archived
    _page_cache[url] = text
    return text


def specifics(note_text: str) -> Tuple[List[str], List[str]]:
    """(numbers, names) a reader would expect to find in the sources."""
    body = URL_RE.sub(" ", note_text)
    # Day-of-month next to a month name: pages write dates in too many forms to match ("today", "Tuesday").
    body = re.sub(rf"\b({_MONTHS})\.?\s+\d{{1,2}}\b|\b\d{{1,2}}\s+({_MONTHS})\b", r"\1\2", body, flags=re.I)
    numbers = sorted({n.replace(",", "") for n in re.findall(r"\d[\d,]*\d", body)})
    names: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]+", sentence)
        for w in words[1:]:  # the first word of a sentence is capitalised anyway
            w = re.sub(r"['\u2019]s?$", "", w)  # possessives
            if w[0].isupper() and len(w) >= 4 and w.lower() not in _NOT_NAMES:
                names.append(w)
    return numbers, sorted(set(names))


def unsupported_specifics(note_text: str, pages: Optional[Dict[str, Optional[str]]] = None) -> List[str]:
    """Numbers and names in the note that are on none of the readable cited pages.
    `pages` (url -> text or None) is for tests; by default the cited URLs are fetched."""
    urls = extract_urls(note_text)
    if pages is None:
        pages = {u: (None if _host(u) in UNREADABLE_HOSTS else page_text(u)) for u in urls}
    readable = " ".join(t for t in pages.values() if t)
    if len(readable) < MIN_TEXT_CHARS:
        return []
    digits_only = readable.replace(",", "")
    numbers, names = specifics(note_text)
    missing = [n for n in numbers if n not in digits_only]
    missing += [w for w in names if w.lower().replace("’", "'") not in readable.replace("’", "'")]
    return missing
