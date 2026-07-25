"""CLI chat interface."""
from __future__ import annotations

import argparse

from rag import RagPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", help="One-shot question (omit for interactive mode)")
    args = p.parse_args()

    pipe = RagPipeline()

    def answer(q: str) -> None:
        resp = pipe.ask(q)
        print("\n" + resp.answer)
        if resp.sources:
            print("\nSources:")
            for s in resp.sources:
                print(f"  - {s['source']} (score {s['score']})")

    if args.query:
        answer(args.query)
        return

    print("RAG chat — type 'exit' to quit")
    while True:
        q = input("\nyou> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
