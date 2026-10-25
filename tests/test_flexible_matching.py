"""
Automated unit and integration tests for Flexible Matching Module (engine/flexible_matching.py).
Tests:
- Normalized term relevance in [0, 1]
- Prefix wildcard matching (*)
- Typo tolerance via Levenshtein ratio
- Zadeh's Classic Fuzzy AND, OR, NOT operations (min, max, complement)
- Extended Boolean AST parsing
- Mandatory Include (+) and Exclude (-) constraints
- Descending relevance ranking
- Empty query validation
"""

import os
import unittest
import numpy as np

from engine.indexing import IndexStore
from engine.flexible_matching import (
    levenshtein_distance,
    term_similarity,
    FuzzyScoreCalculator,
    ExtendedBooleanEvaluator,
    ExtendedBooleanParser,
    FlexibleSearchEngine
)


class TestFlexibleMatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
        cls.store = IndexStore(data_path)
        cls.engine = FlexibleSearchEngine(cls.store)
        cls.calc = cls.engine.calculator

    def test_levenshtein_and_similarity(self):
        """Test edit distance and normalized string similarity."""
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(term_similarity("same", "same"), 1.0)
        
        # Typo similarity
        sim_typo = term_similarity("artifcial", "artificial")
        self.assertGreater(sim_typo, 0.85)

        # Wildcard prefix similarity
        self.assertEqual(term_similarity("comput*", "computer"), 1.0)
        self.assertEqual(term_similarity("comput*", "computing"), 1.0)
        self.assertEqual(term_similarity("comput*", "radiology"), 0.0)

    def test_term_relevance_normalized_range(self):
        """Test that computed term relevance weights fall strictly in [0.0, 1.0]."""
        scores, matched = self.calc.compute_fuzzy_term_relevance("intelligence")
        self.assertGreater(len(matched), 0)
        self.assertEqual(len(scores), self.store.total_docs)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        # Doc 1 has artificial intelligence, so its score should be > 0
        self.assertGreater(scores[0], 0.0)

    def test_wildcard_partial_matching(self):
        """Test prefix wildcard matching (e.g., comput*)."""
        res = self.engine.search("comput*", min_score=0.01, top_k=5)
        self.assertGreater(len(res["results"]), 0)
        # Should match vocabulary terms like computed, computer
        self.assertIn("comput*", res["term_explanations"])
        matched_terms = [t["term"] for t in res["term_explanations"]["comput*"]["matched_vocab"]]
        self.assertTrue(any("comput" in t for t in matched_terms))

    def test_typo_tolerant_matching(self):
        """Test search with typo (e.g. artifcial -> artificial)."""
        res = self.engine.search("artifcial", partial_threshold=0.80, min_score=0.01, top_k=5)
        self.assertGreater(len(res["results"]), 0)
        # Doc 1 or 9 should be top matches
        top_ids = [r["doc_id"] for r in res["results"][:3]]
        self.assertTrue(1 in top_ids or 9 in top_ids)

    def test_fuzzy_and_zadeh(self):
        """Test Zadeh Fuzzy AND: min(mu_A, mu_B)."""
        v1 = np.array([0.8, 0.2, 0.0, 0.9])
        v2 = np.array([0.5, 0.7, 0.4, 0.1])
        res = ExtendedBooleanEvaluator.fuzzy_and(v1, v2)
        np.testing.assert_array_almost_equal(res, [0.5, 0.2, 0.0, 0.1])

    def test_fuzzy_or_zadeh(self):
        """Test Zadeh Fuzzy OR: max(mu_A, mu_B)."""
        v1 = np.array([0.8, 0.2, 0.0, 0.9])
        v2 = np.array([0.5, 0.7, 0.4, 0.1])
        res = ExtendedBooleanEvaluator.fuzzy_or(v1, v2)
        np.testing.assert_array_almost_equal(res, [0.8, 0.7, 0.4, 0.9])

    def test_fuzzy_not(self):
        """Test Fuzzy NOT: 1 - mu."""
        v = np.array([0.8, 0.0, 1.0, 0.35])
        res = ExtendedBooleanEvaluator.fuzzy_not(v)
        np.testing.assert_array_almost_equal(res, [0.2, 1.0, 0.0, 0.65])

    def test_compound_zadeh_expression(self):
        """Test compound Zadeh fuzzy expression: (A OR B) AND NOT C."""
        va = np.array([0.8, 0.2, 0.5])
        vb = np.array([0.4, 0.9, 0.1])
        vc = np.array([0.1, 0.6, 0.0])
        # (va OR vb) = max(va, vb) = [0.8, 0.9, 0.5]
        # NOT vc = 1 - vc = [0.9, 0.4, 1.0]
        # AND = min(max(va, vb), 1 - vc) = min([0.8, 0.9, 0.5], [0.9, 0.4, 1.0]) = [0.8, 0.4, 0.5]
        expected = np.minimum(np.maximum(va, vb), 1.0 - vc)
        eval_or = ExtendedBooleanEvaluator.fuzzy_or(va, vb)
        eval_not = ExtendedBooleanEvaluator.fuzzy_not(vc)
        eval_res = ExtendedBooleanEvaluator.fuzzy_and(eval_or, eval_not)
        np.testing.assert_array_almost_equal(eval_res, expected)
        np.testing.assert_array_almost_equal(eval_res, [0.8, 0.4, 0.5])

    def test_mandatory_include_filter(self):
        """Test mandatory inclusion (+term): documents missing term must be filtered out."""
        # Query with mandatory include: 'healthcare +clinical'
        res = self.engine.search("healthcare +clinical", min_score=0.01, top_k=10)
        self.assertGreater(len(res["results"]), 0)
        for r in res["results"]:
            doc = self.store.get_doc(r["doc_id"])
            full_text = f"{doc['title']} {doc['abstract']} {doc['methodology']} {doc['results']}".lower()
            self.assertIn("clinic", full_text)

    def test_mandatory_exclude_filter(self):
        """Test mandatory exclusion (-term): documents containing term must be eliminated."""
        # Doc 1 has 'imaging', Doc 9 has NO 'imaging'
        res = self.engine.search("AI AND healthcare -imaging", min_score=0.01, top_k=10)
        self.assertGreater(len(res["results"]), 0)
        doc_ids = [r["doc_id"] for r in res["results"]]
        # Doc 1 must be strictly excluded
        self.assertNotIn(1, doc_ids)
        # Doc 9 must be retained
        self.assertIn(9, doc_ids)

    def test_result_ranking_order(self):
        """Test that returned documents are ranked in strictly descending score order."""
        res = self.engine.search("machine learning AND neural", min_score=0.01, top_k=10)
        self.assertGreater(len(res["results"]), 1)
        scores = [r["score"] for r in res["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query(self):
        """Test empty query handling."""
        res = self.engine.search("")
        self.assertEqual(res["results"], [])
        self.assertEqual(res["total_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
