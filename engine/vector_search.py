"""
Vector Space Search Engine: TF-IDF matrix generation and Cosine Similarity ranking.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from engine.indexing import IndexStore
from engine.preprocessing import preprocess


class VectorSearchEngine:
    """
    Computes TF-IDF representations and cosine similarities across indexed documents.
    """

    def __init__(self, store: IndexStore):
        self.store = store
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, Any] = {}
        self.doc_ids: List[int] = [doc["id"] for doc in store.get_all_docs()]
        self._build_matrices()

    def _build_matrices(self):
        """Fit TF-IDF vectorizers on all fields and individual fields."""
        docs = self.store.get_all_docs()
        if not docs:
            return

        # 1. Global corpus (concatenated text)
        global_corpus = [
            f"{d.get('title', '')} {d.get('abstract', '')} {d.get('methodology', '')} {d.get('results', '')}"
            for d in docs
        ]
        
        # Preprocess tokenizer callable for sklearn
        def custom_tokenizer(text):
            return preprocess(text, remove_stop=True, do_stem=True)

        self.vectorizers["all"] = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
        self.tfidf_matrices["all"] = self.vectorizers["all"].fit_transform(global_corpus)

        # 2. Per-field corpora
        for field in IndexStore.FIELDS:
            field_corpus = [d.get(field, "") for d in docs]
            vec = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
            self.vectorizers[field] = vec
            self.tfidf_matrices[field] = vec.fit_transform(field_corpus)

    def search(self, query: str, field: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute TF-IDF vector search and return ranked results sorted by cosine similarity.
        """
        if not query.strip():
            return []

        key = field if (field and field in self.vectorizers) else "all"
        vectorizer = self.vectorizers[key]
        doc_matrix = self.tfidf_matrices[key]

        query_vec = vectorizer.transform([query])
        if query_vec.nnz == 0:
            return []

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, doc_matrix).flatten()

        # Sort indices descending
        matched_indices = np.where(similarities > 0.0)[0]
        if len(matched_indices) == 0:
            return []

        ranked_indices = matched_indices[np.argsort(-similarities[matched_indices])][:top_k]

        results = []
        for idx in ranked_indices:
            doc_id = self.doc_ids[idx]
            score = float(similarities[idx])
            doc = self.store.get_doc(doc_id)
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "score": round(score, 4),
                "doc": doc
            })

        return results


def vector_search(query: str, store: IndexStore, field: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
    """Helper functional interface for vector search."""
    engine = VectorSearchEngine(store)
    return engine.search(query, field=field, top_k=top_k)
