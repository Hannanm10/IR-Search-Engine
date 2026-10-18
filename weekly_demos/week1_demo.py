"""
Week 1 Demo: Basic Search Engine with Inverted Index and Term Frequency (TF) Ranking.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.indexing import IndexStore
from engine.preprocessing import preprocess


def tf_search(query: str, store: IndexStore, top_k: int = 10):
    """
    Search documents using simple Term Frequency (TF) summation.
    """
    terms = preprocess(query, remove_stop=True, do_stem=True)
    if not terms:
        return []

    scores = {}
    term_matches = {}

    for term in terms:
        postings = store.get_posting_list(term)
        for doc_id, tf in postings.items():
            scores[doc_id] = scores.get(doc_id, 0) + tf
            if doc_id not in term_matches:
                term_matches[doc_id] = []
            term_matches[doc_id].append(f"{term}:{tf}")

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_id, score in ranked_docs:
        doc = store.get_doc(doc_id)
        results.append({
            "doc_id": doc_id,
            "title": doc["title"] if doc else f"Doc {doc_id}",
            "score": score,
            "matches": term_matches.get(doc_id, [])
        })
    return results


def run_week1_demo(query: str = "artificial intelligence"):
    print("=" * 60)
    print(" Week 1 Demo: Basic Search Engine (Term Frequency Ranking)")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)
    print(f"Loaded {store.total_docs} documents into Inverted Index.")
    print(f"Total unique vocabulary terms: {len(store.inverted_index)}")
    print("-" * 60)
    print(f"Query: \"{query}\"")
    print(f"Processed Query Terms: {preprocess(query)}")
    print("-" * 60)

    results = tf_search(query, store, top_k=5)
    if not results:
        print("No matching documents found.")
        return

    for rank, res in enumerate(results, 1):
        print(f"Rank {rank}: [Doc {res['doc_id']}] {res['title']} -- Score: {res['score']}")
        print(f"         Matched terms: {', '.join(res['matches'])}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_input = " ".join(sys.argv[1:])
    else:
        query_input = "artificial intelligence"
    run_week1_demo(query_input)
