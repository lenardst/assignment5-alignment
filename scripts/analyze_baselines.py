"""Pretty-print category-2 / category-3 examples from the baselines run.

Part (a) of the prompting_baselines problem asks: of >=10 cat2 outputs, how many
are actually mathematically correct but just not parsed properly? Same for cat3?
This helper makes that read fast: it pulls a random sample per category and
prints them side-by-side with the ground truth so you can eyeball them.

    uv run python scripts/analyze_baselines.py \
        --results-dir experiments/prompting_baselines \
        --prompt r1_zero --category cat2_format_wrong --k 15
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True,
                        choices=["question_only", "r1_zero", "r1_zero_three_shot"])
    parser.add_argument("--category", required=True,
                        choices=["cat1_format_correct", "cat2_format_wrong", "cat3_unformatted"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-response-chars", type=int, default=1200)
    args = parser.parse_args()

    records = load_records(args.results_dir / f"{args.prompt}.jsonl")
    filtered = [r for r in records if r["category"] == args.category]
    print(f"{args.prompt} / {args.category}: {len(filtered)} / {len(records)} total\n")

    rng = random.Random(args.seed)
    picks = rng.sample(filtered, min(args.k, len(filtered)))
    for i, r in enumerate(picks, 1):
        print("=" * 80)
        print(f"[{i}/{len(picks)}]  finish_reason={r['finish_reason']}  reward={r['reward']}")
        print(f"GT: {r['ground_truth']}")
        print("Question:")
        print(textwrap.indent(textwrap.fill(r["question"], 100), "  "))
        resp = r["response"]
        if len(resp) > args.max_response_chars:
            resp = resp[: args.max_response_chars] + f"\n  ... [truncated {len(r['response']) - args.max_response_chars} chars]"
        print("Response:")
        print(textwrap.indent(resp, "  "))
    print("=" * 80)


if __name__ == "__main__":
    main()
