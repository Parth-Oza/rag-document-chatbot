"""Tiny evaluation harness: retrieval hit-rate over a labeled QA set."""
from __future__ import annotations

import json

from rag import RagPipeline


def main() -> None:
    with open("eval_set.json") as f:
        eval_set = json.load(f)

    pipe = RagPipeline()
    hits = 0
    for item in eval_set:
        retrieved = pipe.retrieve(item["question"])
        sources = {d.metadata.get("source", "") for d, _ in retrieved}
        ok = any(item["expected_source"] in s for s in sources)
        hits += ok
        print(f"{'HIT ' if ok else 'MISS'} {item['question']}")

    print(f"\nRetrieval hit-rate: {hits}/{len(eval_set)} "
          f"({hits/len(eval_set):.0%})")


if __name__ == "__main__":
    main()
