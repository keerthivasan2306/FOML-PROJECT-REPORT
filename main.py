"""
main.py — ResilienceLens live event loop
Streams events → POSTs to ticket_api → prints CLI report
"""

import threading
import requests

from data.stream_simulator import stream_data, event_queue
from utils.helpers import print_report

TICKET_API = "http://localhost:8000"


def main():
    # Start streaming in background
    threading.Thread(target=stream_data, daemon=True).start()
    print("🚀 ResilienceLens started — streaming events...\n")

    while True:
        if not event_queue.empty():
            event = event_queue.get()

            try:
                resp = requests.post(f"{TICKET_API}/detect", json=event, timeout=5)
                new_tickets = resp.json() if resp.ok else []
            except requests.exceptions.ConnectionError:
                print("⚠  ticket_api not reachable — run: uvicorn ticket_api:app --reload")
                new_tickets = []

            # Derive a display score from ticket severity
            score = 20
            issues = []
            for t in new_tickets:
                issues.append(f"[{t['id']}] {t['title']}")
                if t["severity"] == "critical":
                    score = max(score, 90)
                elif t["severity"] == "high":
                    score = max(score, 70)
                elif t["severity"] == "medium":
                    score = max(score, 50)

            print("\033c", end="")
            print("Live Event:", event)
            if new_tickets:
                print(f"\n🎫 {len(new_tickets)} ticket(s) created")
            print_report(score, issues)


if __name__ == "__main__":
    main()
