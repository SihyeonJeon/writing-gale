"""
Reverse image search for a post's picture: where else on the web does this image appear?

Most helpful media notes say what the image really is and link the original ("the photo is from a 2019
flood in Venice"). A text search rarely finds that; a search by the image itself does. This calls Google
Cloud Vision's web detection once per post and hands the model the pages carrying the same image, as
leads to open and verify.

Needs GOOGLE_VISION_API_KEY; without it nothing happens. Calls are counted in the ledger and capped
(MONTHLY_CAP per 30 days, DAILY_CAP per 24 hours) to stay inside the API's free monthly allowance.
"""

import base64
import os
from typing import Dict, List, Optional

import requests

import ledger
from data_models import PostWithContext

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
MONTHLY_CAP = int(os.getenv("VISION_MONTHLY_CAP", "900"))
DAILY_CAP = int(os.getenv("VISION_DAILY_CAP", "30"))
MAX_PAGES = 8
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def first_image_url(pwc: PostWithContext) -> Optional[str]:
    """The post's own first photo, else its video thumbnail, else the quoted post's."""
    for post in [pwc.post, pwc.quoted_post]:
        for m in post.media if post else []:
            u = m.url if m.media_type == "photo" else m.preview_image_url
            if u:
                return u
    return None


def within_budget() -> bool:
    month = [r for r in ledger.records_since(24 * 30) if r.get("decision") == "vision"]
    day = [r for r in ledger.records_since(24) if r.get("decision") == "vision"]
    return len(month) < MONTHLY_CAP and len(day) < DAILY_CAP


def web_detect(image_bytes: bytes, key: str) -> Optional[Dict]:
    body = {
        "requests": [{
            "image": {"content": base64.b64encode(image_bytes).decode()},
            "features": [{"type": "WEB_DETECTION", "maxResults": 20}],
        }]
    }
    try:
        r = requests.post(ENDPOINT, headers={"x-goog-api-key": key}, json=body, timeout=30)
        if r.status_code != 200:
            return None
        return ((r.json().get("responses") or [{}])[0]).get("webDetection") or {}
    except (requests.RequestException, ValueError):
        return None


def summarise(web: Dict) -> Dict:
    pages = []
    for p in web.get("pagesWithMatchingImages") or []:
        # Pages carrying the whole image say more about its origin than pages with a crop of it.
        full = bool(p.get("fullMatchingImages"))
        if p.get("url"):
            pages.append({"url": p["url"], "title": (p.get("pageTitle") or "").strip()[:120], "full": full})
    pages.sort(key=lambda p: not p["full"])
    return {
        "best_guess": [g.get("label") for g in web.get("bestGuessLabels") or [] if g.get("label")],
        "entities": [e["description"] for e in (web.get("webEntities") or [])[:6] if e.get("description")],
        "pages": pages[:MAX_PAGES],
    }


def search(pwc: PostWithContext) -> Optional[Dict]:
    """Summary of where the post's image appears, or None (no image, no key, over budget, or failure)."""
    key = os.getenv("GOOGLE_VISION_API_KEY")
    url = first_image_url(pwc)
    if not key or not url or not within_budget():
        return None
    try:
        img = requests.get(url, timeout=20, headers={"User-Agent": UA})
    except requests.RequestException:
        return None
    if img.status_code != 200 or not img.headers.get("Content-Type", "").startswith("image/"):
        return None
    ledger.record(pwc.post.post_id, "vision")  # counted when the paid call is made, whatever it returns
    web = web_detect(img.content, key)
    return summarise(web) if web is not None else None


def describe_for_prompt(found: Optional[Dict]) -> str:
    if not found or not (found["pages"] or found["best_guess"]):
        return ""
    out = "\nReverse image search of the post's first image (leads only; open a page and check its date before relying on it):\n"
    if found["best_guess"]:
        out += f"- Search engine's best guess: {', '.join(found['best_guess'])}\n"
    if found["entities"]:
        out += f"- Related entities: {', '.join(found['entities'])}\n"
    for p in found["pages"]:
        out += f"- {'Same image' if p['full'] else 'Part of the image'} on: {p['url']}" + (f" ({p['title']})" if p["title"] else "") + "\n"
    return out


def page_urls(found: Optional[Dict]) -> List[str]:
    return [p["url"] for p in (found or {}).get("pages", [])]
