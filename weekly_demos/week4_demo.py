"""
Week 4 Demo: Advanced Retrieval Techniques.
Showcases BM25 Probabilistic Ranking, Structured Field Search, Region Retrieval,
Proximity/Phrase Matching, and Non-Overlapped List Search.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.indexing import IndexStore
from engine.bm25_search import BM25SearchEngine
from engine.field_search import search_by_field
from engine.region_search import search_with_regions
from engine.proximity_search import exact_phrase_search, near_proximity_search
from engine.non_overlapped_search import non_overlapped_search


def run_week4_demo():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)

    print("=" * 70)
    print(" Week 4 Demo: Advanced Information Retrieval Suite")
    print("=" * 70)

    # 1. BM25 Probabilistic Ranking
    print("\n[1] Field-Aware Okapi BM25 Ranking")
    bm25_query = "machine learning healthcare"
    print(f"Query: \"{bm25_query}\"")
    bm25_engine = BM25SearchEngine(store)
    bm25_res = bm25_engine.search(bm25_query, top_k=3)
    for rank, res in enumerate(bm25_res, 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Score: {res['score']:.4f})")

    # 2. Structured Field Search
    print("\n[2] Structured Field Search (Title Only)")
    field_query = "neural network"
    print(f"Query: \"{field_query}\" (Scope: Title)")
    title_res = search_by_field(field_query, store, field="title", mode="bm25", top_k=3)
    for rank, res in enumerate(title_res, 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Title Score: {res['score']:.4f})")

    # 3. Region Retrieval (Passage Extraction)
    print("\n[3] Region Retrieval (Sentence Level Matching)")
    region_query = "artificial intelligence models clinicians"
    print(f"Query: \"{region_query}\"")
    region_res = search_with_regions(region_query, store, top_k=2)
    for rank, res in enumerate(region_res, 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']}")
        print(f"          Score: {res['score']:.4f} | Matched Region: {res['region']}")
        print(f"          Snippet: \"{res['snippet']}\"")

    # 4. Phrase & Proximity Search
    print("\n[4] Positional Search: Exact Phrase vs NEAR/3 Proximity")
    phrase = "artificial intelligence"
    print(f"Exact Phrase: \"{phrase}\"")
    phrase_res = exact_phrase_search(phrase, store, top_k=3)
    for rank, res in enumerate(phrase_res, 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Occurrences: {res['occurrences']})")

    near_t1, near_t2, dist = "artificial", "healthcare", 3
    print(f"Proximity: {near_t1} NEAR/{dist} {near_t2}")
    near_res = near_proximity_search(near_t1, near_t2, dist, store, top_k=3)
    for rank, res in enumerate(near_res, 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Pairs: {res['pair_count']}, Min Dist: {res['min_distance']})")

    # 5. Non-Overlapped List Search
    print("\n[5] Non-Overlapped List Search (Multi-Field Deduplication)")
    nol_query = "deep learning"
    print(f"Query: \"{nol_query}\"")
    nol_res = non_overlapped_search(nol_query, store, top_k=3)
    print("  Per-field match counts:")
    for f_name, doc_list in nol_res["per_field_matches"].items():
        print(f"    - {f_name.capitalize():11s}: {len(doc_list)} docs")
    print("  Top merged & deduplicated results:")
    for rank, res in enumerate(nol_res["results"], 1):
        print(f"  Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']}")
        print(f"          Combined Score: {res['score']:.4f} | Fields: {', '.join(res['contributing_fields'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_week4_demo()
