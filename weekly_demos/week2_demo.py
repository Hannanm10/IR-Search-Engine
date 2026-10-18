"""
Week 2 Demo: Document Ranking Pipeline Architecture Walkthrough.
Demonstrates the 5-stage retrieval pipeline with detailed step-by-step intermediate outputs.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.indexing import IndexStore
from engine.preprocessing import QueryProcessor


def search_pipeline(raw_query: str, store: IndexStore, top_k: int = 5):
    print("=" * 65)
    print(" Week 2 Demo: 5-Stage Document Ranking Pipeline Walkthrough")
    print("=" * 65)
    print(f"Input Query: \"{raw_query}\"\n")

    # STAGE 1: Query Processing
    print("[STAGE 1: Query Processing]")
    parsed = QueryProcessor.parse_query(raw_query)
    print(f"  - Raw Query    : {parsed.raw_query}")
    print(f"  - Detected Mode: {parsed.mode}")
    print(f"  - Raw Tokens   : {parsed.raw_tokens}")
    print(f"  - Stemmed Terms: {parsed.terms}")
    print(f"  - Operators    : {parsed.operators or 'None'}")
    print(f"  - Is Phrase    : {parsed.is_phrase}")
    print()

    # STAGE 2: Index Lookup
    print("[STAGE 2: Index Lookup]")
    term_postings = {}
    for term in parsed.terms:
        postings = store.get_posting_list(term)
        term_postings[term] = postings
        matching_docs = sorted(list(postings.keys()))
        print(f"  - Term '{term}' found in {len(matching_docs)} documents: {matching_docs[:10]}{'...' if len(matching_docs) > 10 else ''}")
    print()

    # STAGE 3: Candidate Retrieval
    print("[STAGE 3: Candidate Retrieval]")
    candidate_ids = set()
    for term, postings in term_postings.items():
        candidate_ids.update(postings.keys())
    print(f"  - Union Candidate Documents ({len(candidate_ids)} total): {sorted(list(candidate_ids))[:15]}{'...' if len(candidate_ids) > 15 else ''}")
    print()

    # STAGE 4: Scoring & Ranking
    print("[STAGE 4: Scoring & Ranking]")
    scored = []
    for doc_id in candidate_ids:
        score = sum(term_postings[t].get(doc_id, 0) for t in parsed.terms)
        scored.append((doc_id, score))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    for doc_id, score in ranked[:top_k]:
        print(f"  - Doc {doc_id:2d} -> Score: {score}")
    print()

    # STAGE 5: Result Formatting
    print("[STAGE 5: Result Formatter]")
    for rank, (doc_id, score) in enumerate(ranked[:top_k], 1):
        doc = store.get_doc(doc_id)
        title = doc["title"] if doc else f"Doc {doc_id}"
        abstract_snippet = doc["abstract"][:120] + "..." if doc else ""
        print(f"  Rank {rank}: [Doc {doc_id}] {title}")
        print(f"          Score: {score}")
        print(f"          Snippet: \"{abstract_snippet}\"")
        print()

    print("=" * 65)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)
    query_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "artificial intelligence"
    search_pipeline(query_input, store)
