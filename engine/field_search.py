"""
Field Search Module: Structured search restricted to specific document fields (title, abstract, methodology, results).
"""

from typing import List, Dict, Any, Optional
from engine.indexing import IndexStore
from engine.bm25_search import BM25SearchEngine
from engine.vector_search import VectorSearchEngine
from engine.boolean_search import boolean_search
from engine.preprocessing import preprocess


def search_by_field(
    query: str,
    store: IndexStore,
    field: str = "all",
    mode: str = "bm25",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search within a specific document field ('title', 'abstract', 'methodology', 'results', or 'all').
    Retrieval modes supported: 'bm25', 'vector', 'boolean', 'tf'.
    """
    field_key = field.lower().strip()
    if field_key not in IndexStore.FIELDS and field_key != "all":
        field_key = "all"

    target_field = None if field_key == "all" else field_key

    if mode == "boolean":
        results = boolean_search(query, store, field=target_field)
        return results[:top_k]

    elif mode == "vector":
        engine = VectorSearchEngine(store)
        return engine.search(query, field=target_field, top_k=top_k)

    elif mode == "tf":
        # Pure TF ranking restricted to field
        terms = preprocess(query, remove_stop=True, do_stem=True)
        if not terms:
            return []
        scores = {}
        for term in terms:
            postings = store.get_posting_list(term, field=target_field)
            for doc_id, tf in postings.items():
                scores[doc_id] = scores.get(doc_id, 0) + tf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            doc = store.get_doc(doc_id)
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}") if doc else f"Doc {doc_id}",
                "score": float(score),
                "matched_field": field_key,
                "doc": doc
            })
        return results

    else:
        # Default: BM25
        engine = BM25SearchEngine(store)
        return engine.search(query, field=target_field, top_k=top_k)
