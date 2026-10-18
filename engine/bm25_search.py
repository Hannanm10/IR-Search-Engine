"""
BM25 Probabilistic Retrieval Engine with Field-Aware Scoring.
"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from engine.indexing import IndexStore
from engine.preprocessing import preprocess


class BM25SearchEngine:
    """
    Field-aware Okapi BM25 ranking engine.
    """

    DEFAULT_WEIGHTS = {
        "title": 2.0,
        "abstract": 1.0,
        "methodology": 0.5,
        "results": 0.5,
    }

    def __init__(self, store: IndexStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1 = k1
        self.b = b
        self.docs = store.get_all_docs()
        self.doc_ids = [d["id"] for d in self.docs]

        # Field-specific BM25 Okapi instances
        self.field_bm25: Dict[str, BM25Okapi] = {}
        self.global_bm25: Optional[BM25Okapi] = None
        self._build_bm25_models()

    def _build_bm25_models(self):
        if not self.docs:
            return

        # 1. Global corpus
        global_corpus = [
            preprocess(
                f"{d.get('title', '')} {d.get('abstract', '')} {d.get('methodology', '')} {d.get('results', '')}",
                remove_stop=True,
                do_stem=True
            )
            for d in self.docs
        ]
        self.global_bm25 = BM25Okapi(global_corpus, k1=self.k1, b=self.b)

        # 2. Per-field corpora
        for field in IndexStore.FIELDS:
            field_corpus = [
                preprocess(d.get(field, ""), remove_stop=True, do_stem=True)
                for d in self.docs
            ]
            self.field_bm25[field] = BM25Okapi(field_corpus, k1=self.k1, b=self.b)

    def search(
        self,
        query: str,
        field: Optional[str] = None,
        field_weights: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 search.
        If field is specified (e.g. 'title'), queries only that field.
        If field is None or 'all', computes weighted combination across all fields.
        """
        query_tokens = preprocess(query, remove_stop=True, do_stem=True)
        if not query_tokens:
            return []

        weights = field_weights or self.DEFAULT_WEIGHTS
        n_docs = len(self.doc_ids)

        if field and field in self.field_bm25:
            # Single field search
            scores = self.field_bm25[field].get_scores(query_tokens)
        else:
            # Weighted field-aware combination
            scores = [0.0] * n_docs
            for f_name, f_weight in weights.items():
                if f_name in self.field_bm25:
                    f_scores = self.field_bm25[f_name].get_scores(query_tokens)
                    for i in range(n_docs):
                        scores[i] += f_weight * f_scores[i]

        # Rank documents with positive scores
        scored_pairs = [(self.doc_ids[i], float(scores[i])) for i in range(n_docs) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored_pairs[:top_k]:
            doc = self.store.get_doc(doc_id)
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "score": round(score, 4),
                "doc": doc
            })

        return results


def bm25_search(
    query: str,
    store: IndexStore,
    field: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Helper functional interface for BM25 search."""
    engine = BM25SearchEngine(store)
    return engine.search(query, field=field, field_weights=weights, top_k=top_k)
