import base64
import json
import os
import threading
import time

import requests


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Per-process usage, printed by main and used for the per-run budget.
# xAI reports cost as usage.cost_in_usd_ticks (1 tick = 1e-10 USD).
USAGE = {"calls": 0, "usd": 0.0}
_usage_lock = threading.Lock()
USD_PER_TICK = 1e-10


def _record_usage(data: dict | None) -> None:
    ticks = ((data or {}).get("usage") or {}).get("cost_in_usd_ticks") or 0
    with _usage_lock:
        USAGE["calls"] += 1
        USAGE["usd"] += ticks * USD_PER_TICK


def _make_request(payload: dict, max_attempts: int = 3) -> tuple[str, list[str]]:
    """
    Call the xAI Responses API and return (output text, URLs the search tools cited).
    Retries timeouts, 429 and 5xx with exponential backoff.
    """
    responses_url = "https://api.x.ai/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    }
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(responses_url, headers=headers, json=payload, timeout=300)
        except requests.RequestException as e:
            if attempt == max_attempts:
                raise Exception(f"Error making request: {e}") from e
            time.sleep(2**attempt)
            continue
        _record_usage(response.json() if response.status_code == 200 else None)
        if response.status_code == 200:
            break
        if response.status_code in RETRYABLE_STATUS and attempt < max_attempts:
            time.sleep(2**attempt)
            continue
        raise Exception(f"Error making request: {response.status_code} {response.text}")
    data = response.json()
    blocks = [
        block
        for item in data.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
        if block.get("type") == "output_text"
    ]
    cited = [
        a["url"]
        for block in blocks
        for a in block.get("annotations") or []
        if a.get("type") == "url_citation" and a.get("url")
    ]
    text = data.get("output_text") or "".join(block.get("text", "") for block in blocks)
    return text, cited


IMAGE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def _inline_image(url: str) -> str | None:
    """Download an image and return a base64 data URL; None if it can't be fetched.
    Inlining avoids the model provider's own fetcher being rate-limited or blocked by the host."""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": IMAGE_UA})
    except requests.RequestException:
        return None
    ctype = r.headers.get("Content-Type", "").split(";")[0]
    if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) > 10_000_000:
        return None
    return f"data:{ctype};base64,{base64.b64encode(r.content).decode()}"


def _content(prompt: str, image_urls: list[str] | None):
    images = [d for d in (_inline_image(u) for u in (image_urls or [])[:4]) if d]
    if not images:
        return prompt
    return [{"type": "input_text", "text": prompt}] + [
        {"type": "input_image", "image_url": d, "detail": "high"} for d in images
    ]


def get_grok_search_json_response(
    prompt: str,
    schema: dict,
    model: str = "grok-4.3",
    history: list | None = None,
    image_urls: list[str] | None = None,
    search: bool = True,
) -> dict:
    """
    One call that lets Grok search the web and X, and answer with JSON
    matching `schema`. `history` is prior input/output turns for follow-up repair calls.
    The returned dict also has "_cited_urls": pages the search tools actually returned.
    """
    payload = {
        "model": model,
        "input": [*(history or []), {"role": "user", "content": _content(prompt, image_urls)}],
        "tools": [
            {"type": "web_search", "enable_image_understanding": True},
            {
                "type": "x_search",
                "enable_image_understanding": True,
                "enable_video_understanding": True,
            },
        ]
        if search
        else [],
        "text": {
            "format": {"type": "json_schema", "name": "note", "schema": schema, "strict": True}
        },
        "store": False,
    }
    text, cited = _make_request(payload)
    result = json.loads(text)
    result["_cited_urls"] = cited
    return result
