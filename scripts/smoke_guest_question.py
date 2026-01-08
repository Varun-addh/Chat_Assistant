import httpx

BASE = "http://localhost:7860"


def main() -> None:
    with httpx.Client(timeout=30) as client:
        s = client.post(f"{BASE}/api/session")
        print("session", s.status_code, s.text[:200])
        s.raise_for_status()
        sid = s.json()["session_id"]

        q = client.post(
            f"{BASE}/api/question",
            json={
                "session_id": sid,
                "question": "hello",
                "stream": False,
                "save_to_history": False,
            },
        )
        print("question", q.status_code)
        print(q.text[:300])


if __name__ == "__main__":
    main()
