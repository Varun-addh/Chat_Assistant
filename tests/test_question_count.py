"""Quick sanity check: list Groq models.

Security note:
- Do NOT hardcode API keys in source files.
- Provide GROQ_API_KEY via environment (.env locally, or your secret manager in prod).
"""

import os

import requests


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Missing GROQ_API_KEY in environment")

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main()
