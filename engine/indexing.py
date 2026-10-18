"""
Indexing module: Inverted Index, Positional Index, Field Indexes, and IndexStore.
"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Optional
from engine.preprocessing import tokenize, stem, stem_single, preprocess


def build_inverted_index(documents: List[Dict[str, Any]], field_name: Optional[str] = None) -> Dict[str, Dict[int, int]]:
    """
    Build an inverted index mapping term -> {doc_id: term_frequency}.
    If field_name is specified, index only that field.
    Otherwise, index concatenated fields ('title', 'abstract', 'methodology', 'results').
    """
    index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for doc in documents:
        doc_id = doc["id"]
        if field_name:
            text = doc.get(field_name, "")
        else:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('methodology', '')} {doc.get('results', '')}"

        # Standard preprocessing (lowercase, tokenized, stopwords removed, stemmed)
        terms = preprocess(text, remove_stop=True, do_stem=True)
        for term in terms:
            index[term][doc_id] += 1

    # Convert defaultdict to standard dict
    return {term: dict(postings) for term, postings in index.items()}


def build_positional_index(documents: List[Dict[str, Any]], field_name: Optional[str] = None) -> Dict[str, Dict[int, List[int]]]:
    """
    Build a positional index mapping term -> {doc_id: [position_0, position_1, ...]}.
    Positions correspond to token positions in text.
    Tokens are stemmed so variations match, and positions reflect original token sequence.
    """
    pos_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    for doc in documents:
        doc_id = doc["id"]
        if field_name:
            text = doc.get(field_name, "")
        else:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('methodology', '')} {doc.get('results', '')}"

        # Tokenize (keep all tokens for accurate adjacency / proximity)
        tokens = tokenize(text)
        for pos, tok in enumerate(tokens):
            stemmed = stem_single(tok)
            pos_index[stemmed][doc_id].append(pos)

    return {term: dict(postings) for term, postings in pos_index.items()}


class IndexStore:
    """
    Encapsulates all search indexes, document metadata, and retrieval lookup helpers.
    """

    FIELDS = ["title", "abstract", "methodology", "results"]

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.documents: List[Dict[str, Any]] = []
        self.docs_by_id: Dict[int, Dict[str, Any]] = {}
        
        # Global indexes (all fields combined)
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.positional_index: Dict[str, Dict[int, List[int]]] = {}

        # Per-field indexes
        self.field_inverted_indexes: Dict[str, Dict[str, Dict[int, int]]] = {}
        self.field_positional_indexes: Dict[str, Dict[str, Dict[int, List[int]]]] = {}

        # Precomputed document lengths
        self.doc_lengths: Dict[int, int] = {}
        self.field_doc_lengths: Dict[str, Dict[int, int]] = {f: {} for f in self.FIELDS}
        self.avg_doc_len: float = 0.0
        self.field_avg_len: Dict[str, float] = {f: 0.0 for f in self.FIELDS}

        if data_path:
            self.load_documents(data_path)
            self.build_all_indexes()

    def load_documents(self, data_path: str):
        """Load documents from JSON file."""
        self.data_path = data_path
        with open(data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.docs_by_id = {doc["id"]: doc for doc in self.documents}

    def build_all_indexes(self):
        """Construct inverted, positional, and field-specific indexes."""
        if not self.documents:
            return

        # 1. Global indexes
        self.inverted_index = build_inverted_index(self.documents)
        self.positional_index = build_positional_index(self.documents)

        # 2. Field-specific indexes
        for field in self.FIELDS:
            self.field_inverted_indexes[field] = build_inverted_index(self.documents, field_name=field)
            self.field_positional_indexes[field] = build_positional_index(self.documents, field_name=field)

        # 3. Compute document lengths for ranking
        total_len = 0
        for doc in self.documents:
            d_id = doc["id"]
            full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('methodology', '')} {doc.get('results', '')}"
            full_tokens = tokenize(full_text)
            self.doc_lengths[d_id] = len(full_tokens)
            total_len += len(full_tokens)

            for field in self.FIELDS:
                f_tokens = tokenize(doc.get(field, ""))
                self.field_doc_lengths[field][d_id] = len(f_tokens)

        n_docs = len(self.documents)
        self.avg_doc_len = total_len / n_docs if n_docs > 0 else 0.0

        for field in self.FIELDS:
            f_total = sum(self.field_doc_lengths[field].values())
            self.field_avg_len[field] = f_total / n_docs if n_docs > 0 else 0.0

    def get_posting_list(self, term: str, field: Optional[str] = None) -> Dict[int, int]:
        """
        Get {doc_id: term_frequency} for a given stemmed term.
        If field is given (and valid), lookup in that field's inverted index.
        """
        if field and field in self.field_inverted_indexes:
            return self.field_inverted_indexes[field].get(term, {})
        return self.inverted_index.get(term, {})

    def get_positions(self, term: str, doc_id: int, field: Optional[str] = None) -> List[int]:
        """
        Get token positions for a term within a document.
        """
        if field and field in self.field_positional_indexes:
            return self.field_positional_indexes[field].get(term, {}).get(doc_id, [])
        return self.positional_index.get(term, {}).get(doc_id, [])

    def get_doc(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve document dictionary by id."""
        return self.docs_by_id.get(doc_id)

    def get_all_docs(self) -> List[Dict[str, Any]]:
        """Return all indexed documents."""
        return self.documents

    @property
    def total_docs(self) -> int:
        return len(self.documents)
