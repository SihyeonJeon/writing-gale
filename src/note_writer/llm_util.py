import json
import os
import time

import requests


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _make_request(payload: dict, max_attempts: int = 3) -> str:
    """
    Call the xAI Responses API and return the output text.
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
        if response.status_code == 200:
            break
        if response.status_code in RETRYABLE_STATUS and attempt < max_attempts:
            time.sleep(2**attempt)
            continue
        raise Exception(f"Error making request: {response.status_code} {response.text}")
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    return "".join(
        block.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for block in item.get("content", [])
        if block.get("type") == "output_text"
    )


def get_grok_search_json_response(
    prompt: str,
    schema: dict,
    model: str = "grok-4.3",
    history: list | None = None,
) -> dict:
    """
    One call that lets Grok search the web and X, and answer with JSON
    matching `schema`. `history` is prior input/output turns for follow-up repair calls.
    """
    payload = {
        "model": model,
        "input": [*(history or []), {"role": "user", "content": prompt}],
        "tools": [
            {"type": "web_search", "enable_image_understanding": True},
            {
                "type": "x_search",
                "enable_image_understanding": True,
                "enable_video_understanding": True,
            },
        ],
        "text": {
            "format": {"type": "json_schema", "name": "note", "schema": schema, "strict": True}
        },
        "store": False,
    }
    return json.loads(_make_request(payload))
