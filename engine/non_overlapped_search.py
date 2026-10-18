"""
Non-Overlapped List Search: Retrieves independent per-field match lists and merges with deduplication.
"""

from typing import List, Dict, Any, Optional
from engine.indexing import IndexStore
from engine.bm25_search import BM25SearchEngine


def non_overlapped_search(
    query: str,
    store: IndexStore,
    top_k: int = 10,
    field_weights: Optional[Dict[str, float]] = None,
    combination_mode: str = "sum"
) -> Dict[str, Any]:
    """
    Search each field independently to obtain separate posting/match lists,
    then merge and deduplicate into a unified ranked result set.
    
    Returns:
        {
            "per_field_matches": {field_name: [doc_ids...]},
            "results": [
                {
                    "doc_id": id,
                    "title": title,
                    "score": combined_score,
                    "contributing_fields": ['Title', 'Abstract', ...],
                    "field_breakdown": {field: score, ...},
                    "doc": doc_dict
                }, ...
            ]
        }
    """
    weights = field_weights or BM25SearchEngine.DEFAULT_WEIGHTS
    engine = BM25SearchEngine(store)

    per_field_matches: Dict[str, List[int]] = {}
    per_field_scores: Dict[str, Dict[int, float]] = {}

    # 1. Retrieve independent lists per field
    for field in IndexStore.FIELDS:
        res = engine.search(query, field=field, top_k=store.total_docs)
        doc_ids = [r["doc_id"] for r in res]
        scores_map = {r["doc_id"]: r["score"] for r in res}
        per_field_matches[field] = doc_ids
        per_field_scores[field] = scores_map

    # 2. Merge and deduplicate across all fields
    all_matched_ids = set()
    for doc_ids in per_field_matches.values():
        all_matched_ids.update(doc_ids)

    # 3. Combine scores and track contributing fields
    merged_items = []
    for doc_id in all_matched_ids:
        doc = store.get_doc(doc_id)
        contributing_fields = []
        field_breakdown = {}
        total_score = 0.0
        max_score = 0.0

        for field in IndexStore.FIELDS:
            if doc_id in per_field_scores[field]:
                raw_score = per_field_scores[field][doc_id]
                w = weights.get(field, 1.0)
                weighted_score = raw_score * w
                field_breakdown[field] = round(weighted_score, 4)
                contributing_fields.append(field.capitalize())
                total_score += weighted_score
                max_score = max(max_score, weighted_score)

        final_score = total_score if combination_mode == "sum" else max_score

        merged_items.append({
            "doc_id": doc_id,
            "title": doc.get("title", f"Doc {doc_id}") if doc else f"Doc {doc_id}",
            "score": round(final_score, 4),
            "contributing_fields": contributing_fields,
            "field_breakdown": field_breakdown,
            "doc": doc
        })

    # 4. Sort descending by merged score
    merged_items.sort(key=lambda x: x["score"], reverse=True)

    return {
        "per_field_matches": per_field_matches,
        "results": merged_items[:top_k]
    }
