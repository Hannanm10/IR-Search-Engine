"""
Proximity and Exact Phrase Retrieval Engine using Positional Indexes.
Supports exact consecutive phrase matching and NEAR/N distance-constrained queries.
"""

import re
from typing import List, Dict, Any, Optional
from engine.indexing import IndexStore
from engine.preprocessing import tokenize, stem_single


def exact_phrase_search(
    phrase: str,
    store: IndexStore,
    field: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for exact consecutive phrase occurrence: pos[i+1] - pos[i] == 1.
    """
    cleaned = phrase.strip().strip('"').strip("'")
    tokens = tokenize(cleaned)
    if not tokens:
        return []

    stems = [stem_single(t) for t in tokens]
    if len(stems) == 1:
        # Single word: lookup positions/frequency
        postings = store.get_posting_list(stems[0], field=field)
        results = []
        for doc_id, freq in postings.items():
            doc = store.get_doc(doc_id)
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}") if doc else f"Doc {doc_id}",
                "score": float(freq),
                "occurrences": freq,
                "doc": doc
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # Find candidate documents that contain all terms
    candidate_doc_ids = None
    for term in stems:
        doc_ids = set()
        for doc in store.get_all_docs():
            if store.get_positions(term, doc["id"], field=field):
                doc_ids.add(doc["id"])
        if candidate_doc_ids is None:
            candidate_doc_ids = doc_ids
        else:
            candidate_doc_ids &= doc_ids

    if not candidate_doc_ids:
        return []

    matched_results = []
    for doc_id in sorted(list(candidate_doc_ids)):
        # Check positional adjacency
        first_positions = store.get_positions(stems[0], doc_id, field=field)
        exact_match_count = 0

        for start_pos in first_positions:
            match_found = True
            for offset in range(1, len(stems)):
                expected_pos = start_pos + offset
                other_positions = store.get_positions(stems[offset], doc_id, field=field)
                if expected_pos not in other_positions:
                    match_found = False
                    break
            if match_found:
                exact_match_count += 1

        if exact_match_count > 0:
            doc = store.get_doc(doc_id)
            matched_results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}") if doc else f"Doc {doc_id}",
                "score": float(exact_match_count),
                "occurrences": exact_match_count,
                "doc": doc
            })

    matched_results.sort(key=lambda x: x["score"], reverse=True)
    return matched_results[:top_k]


def near_proximity_search(
    term1: str,
    term2: str,
    max_distance: int,
    store: IndexStore,
    field: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for documents where term1 and term2 appear within max_distance tokens:
    1 <= abs(pos2 - pos1) <= max_distance.
    """
    stem1 = stem_single(term1)
    stem2 = stem_single(term2)

    all_docs = store.get_all_docs()
    results = []

    for doc in all_docs:
        doc_id = doc["id"]
        pos1 = store.get_positions(stem1, doc_id, field=field)
        pos2 = store.get_positions(stem2, doc_id, field=field)

        if not pos1 or not pos2:
            continue

        pair_count = 0
        min_dist = float("inf")
        score = 0.0

        for p1 in pos1:
            for p2 in pos2:
                if p1 == p2:
                    continue
                diff = abs(p2 - p1)
                if diff <= max_distance:
                    pair_count += 1
                    min_dist = min(min_dist, diff)
                    # Closer pairs receive higher score
                    score += 1.0 / diff

        if pair_count > 0:
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "score": round(score, 4),
                "pair_count": pair_count,
                "min_distance": min_dist,
                "doc": doc
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def proximity_search(
    query: str,
    store: IndexStore,
    field: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Unified entry point: auto-detects NEAR/N query or exact phrase query.
    """
    q = query.strip()
    # Check for NEAR/N
    near_match = re.search(r"(\b\w+\b)\s+NEAR/(\d+)\s+(\b\w+\b)", q, re.IGNORECASE)
    if near_match:
        t1, dist_str, t2 = near_match.group(1), near_match.group(2), near_match.group(3)
        return near_proximity_search(t1, t2, int(dist_str), store, field=field, top_k=top_k)

    # Quoted phrase or plain phrase
    phrase_text = q.replace('"', "").replace("'", "")
    return exact_phrase_search(phrase_text, store, field=field, top_k=top_k)
