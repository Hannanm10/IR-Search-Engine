"""
Week 6 Demo: Flexible Matching Module.
Demonstrates:
1. Normalized Term Relevance Scoring in [0.0, 1.0]
2. Partial Matching & Prefix Wildcards (e.g., 'comput*')
3. Typo-Tolerant Approximate Matching (e.g., 'artifcial' -> 'artificial')
4. Fuzzy Boolean Logic (Zadeh, Algebraic, and Salton's p-norm Extended Boolean)
5. Mandatory Include (+) and Exclude (-) Constraints
6. Ranked Results with Relevance Breakdown
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding for Windows command line compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from engine.indexing import IndexStore
from engine.flexible_matching import FlexibleSearchEngine, term_similarity


def run_week6_demo():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)
    engine = FlexibleSearchEngine(store)

    print("=" * 75)
    print(" Week 6 Demo: Flexible Matching & Extended Boolean Retrieval Suite")
    print("=" * 75)

    # 1. Normalized Term Relevance Score in [0, 1]
    print("\n[1] Continuous Term Relevance Scores in [0.0, 1.0]")
    sample_term = "healthcare"
    scores, matched_vocab = engine.calculator.compute_fuzzy_term_relevance(sample_term)
    print(f"  Target Term : \"{sample_term}\"")
    print(f"  Matched Vocabulary Stem: {[m['term'] for m in matched_vocab]}")
    print("  Sample Document Membership Degrees:")
    top_doc_indices = scores.argsort()[::-1][:4]
    for idx in top_doc_indices:
        doc_id = engine.calculator.doc_ids[idx]
        print(f"    - Doc #{doc_id:2d}: Relevance = {scores[idx]:.4f} ({scores[idx]*100:.1f}%)")

    # 2. Partial Matching: Wildcard Expansion & Typo Tolerance
    print("\n[2] Partial Matching: Wildcards & Typo Tolerance")
    # A. Wildcard Prefix
    wildcard_q = "comput*"
    res_wildcard = engine.search(wildcard_q, min_score=0.05, top_k=3)
    print(f"  A. Wildcard Prefix Query: \"{wildcard_q}\"")
    expanded = res_wildcard["term_explanations"].get(wildcard_q, {}).get("matched_vocab", [])
    print(f"     Expanded Vocabulary Stems: {[m['term'] for m in expanded]}")
    for rank, r in enumerate(res_wildcard["results"][:2], 1):
        print(f"     Rank {rank}: [Doc {r['doc_id']:2d}] {r['title']} (Score: {r['score']:.4f})")

    # B. Typo Tolerance
    typo_q = "artifcial"  # Missing 'i'
    print(f"\n  B. Typo-Tolerant Query: \"{typo_q}\" (Simulated Spelling Error)")
    sim = term_similarity("artifcial", "artificial")
    print(f"     Calculated String Similarity with 'artificial': {sim:.3f}")
    res_typo = engine.search(typo_q, partial_threshold=0.80, min_score=0.05, top_k=2)
    for rank, r in enumerate(res_typo["results"], 1):
        print(f"     Rank {rank}: [Doc {r['doc_id']:2d}] {r['title']} (Score: {r['score']:.4f})")

    # 3. Fuzzy AND / OR / NOT Logic Comparison
    print("\n[3] Fuzzy Boolean Logic Model Comparison")
    test_q = "neural AND vision"
    print(f"  Query: \"{test_q}\"")

    for model_name in ["zadeh", "algebraic", "p_norm"]:
        res_model = engine.search(test_q, fuzzy_model=model_name, p_norm=2.0, min_score=0.01, top_k=2)
        top_res = res_model["results"][0] if res_model["results"] else None
        top_str = f"[Doc {top_res['doc_id']}] Score: {top_res['score']:.4f}" if top_res else "No matches"
        print(f"    - {model_name.upper():11s} Model: Top Result -> {top_str}")

    # 4. Mandatory Include (+) and Exclude (-) Constraints
    print("\n[4] Mandatory Include (+) and Exclude (-) Filters")
    query_base = "AI AND healthcare"
    query_filtered = "AI AND healthcare +clinical -imaging"
    print(f"  Base Query    : \"{query_base}\"")
    print(f"  Filtered Query: \"{query_filtered}\" (MUST have 'clinical', MUST NOT have 'imaging')")

    res_base = engine.search(query_base, min_score=0.01, top_k=3)
    res_filtered = engine.search(query_filtered, min_score=0.01, top_k=3)

    print(f"  Base Matches    : {[r['doc_id'] for r in res_base['results']]}")
    print(f"  Filtered Matches: {[r['doc_id'] for r in res_filtered['results']]}")
    print("  Notice: Doc #1 is strictly excluded (contains 'imaging'), while Doc #9 is retained!")

    # 5. Full Extended Boolean Ranking & Explanation Breakdown
    print("\n[5] End-to-End Extended Boolean Query Execution")
    complex_q = "(artificial OR machine) AND (clinical OR health*)"
    print(f"  Complex Expression: \"{complex_q}\"")
    res_complex = engine.search(complex_q, fuzzy_model="p_norm", p_norm=2.0, min_score=0.05, top_k=3)
    print(f"  Total Candidates Matched: {res_complex['total_candidates']}")
    for rank, r in enumerate(res_complex["results"], 1):
        print(f"    Rank {rank}: [Doc {r['doc_id']:2d}] {r['title']}")
        print(f"            Fuzzy Score: {r['score']:.4f} ({r['relevance_pct']}%) | Domain: {r['category']}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_week6_demo()
