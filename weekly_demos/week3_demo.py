"""
Week 3 Demo: Boolean Search & Vector Space (TF-IDF Cosine Similarity) Ranking.
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.indexing import IndexStore
from engine.boolean_search import boolean_search
from engine.vector_search import VectorSearchEngine


def run_week3_demo(mode: str = "both", query: str = ""):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)
    vector_engine = VectorSearchEngine(store)

    print("=" * 65)
    print(" Week 3 Demo: Boolean Search & Vector Space (TF-IDF) Retrieval")
    print("=" * 65)

    if mode in ("1", "boolean", "both"):
        bool_query = query if (query and mode in ("1", "boolean")) else "AI AND healthcare NOT imaging"
        print(f"\n--- [1] Boolean Search ---")
        print(f"Query: \"{bool_query}\"")
        bool_results = boolean_search(bool_query, store)
        print(f"Matched {len(bool_results)} document(s):")
        for res in bool_results[:10]:
            print(f"  - [Doc {res['doc_id']:2d}] {res['title']}")
        if len(bool_results) > 10:
            print(f"  ... and {len(bool_results) - 10} more.")

    if mode in ("2", "vector", "both"):
        vec_query = query if (query and mode in ("2", "vector")) else "artificial intelligence in medicine"
        print(f"\n--- [2] Vector Search (TF-IDF + Cosine Similarity) ---")
        print(f"Query: \"{vec_query}\"")
        vec_results = vector_engine.search(vec_query, top_k=5)
        print(f"Top {len(vec_results)} Ranked Results:")
        for rank, res in enumerate(vec_results, 1):
            print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Cosine: {res['score']:.4f})")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 3 IR Demo: Boolean and Vector Search")
    parser.add_argument("--mode", choices=["boolean", "vector", "both"], default="both", help="Search mode")
    parser.add_argument("--query", type=str, default="", help="Search query")
    args = parser.parse_args()

    run_week3_demo(mode=args.mode, query=args.query)
