"""Command-line interface for baseline Color Moments retrieval."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import retrieve_similar_images


def parse_args() -> argparse.Namespace:
    """Parse retrieval CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieve similar images using global RGB Color Moments."
    )
    parser.add_argument("--image", required=True, help="Path to the query image.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results.")
    return parser.parse_args()


def print_retrieval(result: dict) -> None:
    """Print a readable retrieval report."""
    query = result["query"]
    query_id = query["image_id"] if query["image_id"] is not None else "N/A"
    query_category = query["category"] if query["category"] is not None else "N/A"

    print("=" * 60)
    print("CONTENT-BASED IMAGE RETRIEVAL")
    print("=" * 60)
    print(f"Query          : {query['filepath']}")
    print(f"Query ID       : {query_id}")
    print(f"Query Category : {query_category}\n")
    print(f"Feature        : {result['method']}")
    print(f"Feature Dim    : {query['features'].shape[0]}")
    print(f"Metric         : {result['metric']}")
    print(f"Top-K          : {result['top_k']}\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Rank':<6}{'Image':<14}{'Category':<16}{'Distance':>12}")
    print("-" * 60)
    for item in result["results"]:
        print(
            f"{item['rank']:<6}{item['filename']:<14}"
            f"{item['category']:<16}{item['distance']:>12.6f}"
        )


def main() -> int:
    """Run retrieval once from command-line arguments."""
    args = parse_args()
    try:
        result = retrieve_similar_images(args.image, args.top_k)
    except Exception as exc:
        print(f"ERROR: Retrieval failed: {exc}")
        return 1
    print_retrieval(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
