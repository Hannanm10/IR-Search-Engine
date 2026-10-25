"""
Flexible Matching Module:
- Term relevance scoring normalized between 0 and 1
- Partial, wildcard (*), and typo-tolerant matching
- Standard Zadeh's Fuzzy Logic (Classic Fuzzy Set Theory: min, max, complement)
- Extended Boolean AST query parser
- Mandatory Include (+) and Exclude (-) constraints
- Continuous relevance ranking and score breakdown
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from engine.indexing import IndexStore
from engine.preprocessing import tokenize, stem_single


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def term_similarity(query_token: str, vocab_token: str) -> float:
    """
    Computes normalized similarity in [0.0, 1.0] between query token and vocabulary token.
    Supports:
    - Exact match -> 1.0
    - Wildcard prefix e.g. 'comput*' -> 1.0 if vocab_token starts with 'comput'
    - Typo distance e.g. 'artifcial' vs 'artificial' -> ratio based on Levenshtein distance
    """
    q = query_token.lower().strip()
    v = vocab_token.lower().strip()

    if q == v:
        return 1.0

    # Wildcard prefix matching
    if q.endswith("*"):
        prefix = q[:-1]
        if prefix and v.startswith(prefix):
            return 1.0
        return 0.0

    # Levenshtein ratio
    max_len = max(len(q), len(v))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(q, v)
    sim = 1.0 - (dist / max_len)
    return max(0.0, sim)


class FuzzyScoreCalculator:
    """
    Computes normalized term relevance weights w(t, D) in [0.0, 1.0]
    for all documents and terms using normalized TF-IDF with length saturation.
    """

    def __init__(self, store: IndexStore):
        self.store = store
        self.doc_ids = [d["id"] for d in store.get_all_docs()]
        self.n_docs = len(self.doc_ids)
        self.doc_id_to_idx = {d_id: idx for idx, d_id in enumerate(self.doc_ids)}
        
        # Precompute vocabulary and IDFs
        self.vocabulary = list(store.inverted_index.keys())
        self.idfs: Dict[str, float] = {}
        max_idf = 0.0

        for term, postings in store.inverted_index.items():
            df = len(postings)
            # Standard smoothed Robertson-Sparck Jones IDF
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            idf = max(0.01, idf)
            self.idfs[term] = idf
            if idf > max_idf:
                max_idf = idf

        self.max_idf = max_idf if max_idf > 0 else 1.0

    def compute_exact_term_weights(self, stemmed_term: str) -> np.ndarray:
        """
        Computes normalized relevance vector w(t, D) in [0.0, 1.0] for all documents.
        """
        scores = np.zeros(self.n_docs, dtype=float)
        postings = self.store.get_posting_list(stemmed_term)
        if not postings:
            return scores

        idf_factor = self.idfs.get(stemmed_term, 0.01) / self.max_idf
        avgdl = self.store.avg_doc_len if self.store.avg_doc_len > 0 else 100.0

        for doc_id, tf in postings.items():
            idx = self.doc_id_to_idx.get(doc_id)
            if idx is None:
                continue
            doc_len = self.store.doc_lengths.get(doc_id, avgdl)
            # Saturated normalized TF component
            tf_norm = tf / (tf + 1.2 * (0.25 + 0.75 * (doc_len / avgdl)))
            # Bounded in [0.0, 1.0]
            score = float(tf_norm * idf_factor)
            scores[idx] = min(1.0, max(0.0, score))

        # Re-scale to ensure top document reaches healthy membership scale
        max_s = np.max(scores)
        if max_s > 0:
            scores = scores / max_s

        return scores

    def compute_fuzzy_term_relevance(
        self,
        query_token: str,
        similarity_threshold: float = 0.75
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Computes term relevance over all documents with partial matching,
        wildcard expansion, and typo tolerance.
        Returns:
            - scores: np.ndarray of shape (n_docs,) with values in [0.0, 1.0]
            - matched_vocab_details: list of matching vocabulary terms and their similarities
        """
        q_raw = query_token.lower().strip()
        q_stemmed = stem_single(q_raw.replace("*", ""))
        is_wildcard = q_raw.endswith("*")

        matching_terms = []

        # Find candidate terms from vocabulary
        for vocab_term in self.vocabulary:
            if is_wildcard:
                if vocab_term.startswith(q_stemmed) or vocab_term.startswith(q_raw.replace("*", "")):
                    matching_terms.append((vocab_term, 1.0))
            else:
                # Direct stemmed match
                if vocab_term == q_stemmed or vocab_term == q_raw:
                    matching_terms.append((vocab_term, 1.0))
                else:
                    # Typo check against stemmed and unstemmed versions
                    sim_raw = term_similarity(q_raw, vocab_term)
                    sim_stem = term_similarity(q_stemmed, vocab_term)
                    sim = max(sim_raw, sim_stem)
                    if sim >= similarity_threshold:
                        matching_terms.append((vocab_term, sim))

        if not matching_terms:
            return np.zeros(self.n_docs, dtype=float), []

        # Aggregate document relevance: r(q, D) = max [ sim(q, t') * w(t', D) ]
        aggregated_scores = np.zeros(self.n_docs, dtype=float)
        matched_details = []

        for v_term, sim in matching_terms:
            w = self.compute_exact_term_weights(v_term)
            effective_w = w * sim
            aggregated_scores = np.maximum(aggregated_scores, effective_w)
            matched_details.append({
                "term": v_term,
                "similarity": round(sim, 3),
                "max_weight": round(float(np.max(w)), 3)
            })

        matched_details.sort(key=lambda x: (x["similarity"], x["max_weight"]), reverse=True)
        return np.clip(aggregated_scores, 0.0, 1.0), matched_details


class ExtendedBooleanEvaluator:
    """
    Evaluates Boolean operations using classic Zadeh's Fuzzy Logic (Fuzzy Set Theory):
    - Fuzzy AND: min(v1, v2)
    - Fuzzy OR:  max(v1, v2)
    - Fuzzy NOT: 1.0 - v
    """

    @staticmethod
    def fuzzy_and(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Zadeh Fuzzy AND: intersection is minimum of membership values."""
        return np.minimum(v1, v2)

    @staticmethod
    def fuzzy_or(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Zadeh Fuzzy OR: union is maximum of membership values."""
        return np.maximum(v1, v2)

    @staticmethod
    def fuzzy_not(v: np.ndarray) -> np.ndarray:
        """Zadeh Fuzzy NOT: complement is 1.0 - membership value."""
        return np.clip(1.0 - v, 0.0, 1.0)


class ExtendedBooleanParser:
    """
    Recursive descent parser for Extended Boolean expressions:
    expr     := or_expr
    or_expr  := and_expr ( 'OR' and_expr )*
    and_expr := not_expr ( 'AND' not_expr )*
    not_expr := 'NOT' not_expr | primary
    primary  := '(' expr ')' | TERM
    """

    def __init__(
        self,
        tokens: List[str],
        calculator: FuzzyScoreCalculator,
        similarity_threshold: float = 0.75
    ):
        self.tokens = self._insert_implicit_operators(tokens)
        self.calc = calculator
        self.similarity_threshold = similarity_threshold
        self.pos = 0
        self.term_explanations: Dict[str, Any] = {}

    def _insert_implicit_operators(self, raw_tokens: List[str]) -> List[str]:
        result = []
        for i, tok in enumerate(raw_tokens):
            if i > 0:
                prev = result[-1]
                if prev not in {"AND", "OR", "NOT", "("}:
                    if tok == "NOT":
                        result.append("AND")
                    elif tok not in {"AND", "OR", ")"}:
                        result.append("OR")
            result.append(tok)
        return result

    def current_token(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self) -> np.ndarray:
        if not self.tokens:
            return np.zeros(self.calc.n_docs, dtype=float)
        return self.expr()

    def expr(self) -> np.ndarray:
        return self.or_expr()

    def or_expr(self) -> np.ndarray:
        res = self.and_expr()
        while self.current_token() == "OR":
            self.consume("OR")
            right = self.and_expr()
            res = ExtendedBooleanEvaluator.fuzzy_or(res, right)
        return res

    def and_expr(self) -> np.ndarray:
        res = self.not_expr()
        while self.current_token() == "AND":
            self.consume("AND")
            right = self.not_expr()
            res = ExtendedBooleanEvaluator.fuzzy_and(res, right)
        return res

    def not_expr(self) -> np.ndarray:
        if self.current_token() == "NOT":
            self.consume("NOT")
            sub = self.not_expr()
            return ExtendedBooleanEvaluator.fuzzy_not(sub)
        return self.primary()

    def primary(self) -> np.ndarray:
        tok = self.current_token()
        if tok == "(":
            self.consume("(")
            res = self.expr()
            if self.current_token() == ")":
                self.consume(")")
            return res
        elif tok and tok not in {"AND", "OR", "NOT", ")"}:
            term = self.consume()
            scores, matched_vocab = self.calc.compute_fuzzy_term_relevance(
                term,
                similarity_threshold=self.similarity_threshold
            )
            self.term_explanations[term] = {
                "matched_vocab": matched_vocab,
                "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0
            }
            return scores
        return np.zeros(self.calc.n_docs, dtype=float)


class FlexibleSearchEngine:
    """
    High-level Flexible Matching Engine combining:
    - Zadeh's Classic Fuzzy Logic (Min for AND, Max for OR, Complement for NOT)
    - Partial Matching & Typo Tolerance
    - Mandatory Include (+) & Exclude (-) Constraints
    - Score normalization into [0.0, 1.0] and ranking
    """

    def __init__(self, store: IndexStore):
        self.store = store
        self.calculator = FuzzyScoreCalculator(store)

    def search(
        self,
        query: str,
        partial_threshold: float = 0.75,
        include_terms: Optional[List[str]] = None,
        exclude_terms: Optional[List[str]] = None,
        min_score: float = 0.01,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute flexible search with continuous Zadeh fuzzy relevance scoring in [0.0, 1.0].
        """
        raw_query = query.strip()
        if not raw_query:
            return {"results": [], "term_explanations": {}, "total_candidates": 0}

        # 1. Parse inline +include and -exclude terms if present (e.g., 'clinical +ai -imaging')
        inline_includes = re.findall(r"\+(\w+)", raw_query)
        inline_excludes = re.findall(r"-(\w+)", raw_query)

        # Remove +/- prefixes for parser
        cleaned_query = re.sub(r"[\+\-](\w+)", r"\1", raw_query)

        all_includes = list(set((include_terms or []) + inline_includes))
        all_excludes = list(set((exclude_terms or []) + inline_excludes))

        # 2. Tokenize Boolean query
        token_str = cleaned_query.replace("(", " ( ").replace(")", " ) ")
        tokens = []
        for word in token_str.split():
            upper_w = word.upper()
            if upper_w in {"AND", "OR", "NOT", "(", ")"}:
                tokens.append(upper_w)
            else:
                tokens.append(word.lower())

        # 3. Parse and evaluate Extended Boolean AST using Zadeh's Fuzzy Logic
        parser = ExtendedBooleanParser(
            tokens=tokens,
            calculator=self.calculator,
            similarity_threshold=partial_threshold
        )
        base_scores = parser.parse()

        # 4. Apply Hard Include Constraints (+)
        # If document score for any mandatory include term is 0.0, disqualify document
        include_masks = np.ones(self.calculator.n_docs, dtype=bool)
        for inc_term in all_includes:
            inc_scores, _ = self.calculator.compute_fuzzy_term_relevance(inc_term, similarity_threshold=partial_threshold)
            include_masks = include_masks & (inc_scores > 0.0)

        # 5. Apply Hard Exclude Constraints (-)
        # If document score for any forbidden term > 0.0, disqualify document
        exclude_masks = np.ones(self.calculator.n_docs, dtype=bool)
        for exc_term in all_excludes:
            exc_scores, _ = self.calculator.compute_fuzzy_term_relevance(exc_term, similarity_threshold=partial_threshold)
            exclude_masks = exclude_masks & (exc_scores == 0.0)

        final_scores = base_scores * include_masks * exclude_masks
        final_scores = np.clip(final_scores, 0.0, 1.0)

        # 6. Rank Results
        candidate_indices = np.where(final_scores >= min_score)[0]
        sorted_indices = candidate_indices[np.argsort(-final_scores[candidate_indices])]

        results = []
        for idx in sorted_indices[:top_k]:
            doc_id = self.calculator.doc_ids[idx]
            doc = self.store.get_doc(doc_id)
            score = float(final_scores[idx])
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}") if doc else f"Doc {doc_id}",
                "score": round(score, 4),
                "relevance_pct": round(score * 100.0, 1),
                "category": doc.get("category", "General") if doc else "General",
                "subcategory": doc.get("subcategory", "General") if doc else "General",
                "doc": doc
            })

        return {
            "results": results,
            "term_explanations": parser.term_explanations,
            "total_candidates": len(candidate_indices),
            "fuzzy_model": "Zadeh's Classic Fuzzy Logic",
            "includes_applied": all_includes,
            "excludes_applied": all_excludes
        }
