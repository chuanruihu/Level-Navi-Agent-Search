"""Search through SearchAction without a search API key or a running LLM."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.actions import SearchAction
from src.plugins import ParallelSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='Question or search keywords')
    args = parser.parse_args()
    results = SearchAction(topk=3, searcher_class=ParallelSearch)({'query': args.query})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
