"""
Automated unit and integration tests for all Search Engine modules using unittest.
Covers: Week 1 TF, Boolean (AND/OR/NOT), Vector Cosine, BM25, Field Search,
Exact Phrase, Proximity (NEAR/N), Region Retrieval, Non-overlapped list merge, and Empty query.
"""

import os
import unittest
from engine.indexing import IndexStore
from engine.preprocessing import preprocess, QueryProcessor
from engine.boolean_search import boolean_search
from engine.vector_search import VectorSearchEngine, vector_search
from engine.bm25_search import BM25SearchEngine, bm25_search
from engine.field_search import search_by_field
from engine.proximity_search import exact_phrase_search, near_proximity_search, proximity_search
from engine.region_search import search_with_regions
from engine.non_overlapped_search import non_overlapped_search
from weekly_demos.week1_demo import tf_search


class TestSearchEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
        cls.store = IndexStore(data_path)

    def test_week1_tf_search(self):
        """Test 1: artificial intelligence -> Relevant docs ranked by term freq"""
        results = tf_search("artificial intelligence", self.store, top_k=5)
        self.assertGreater(len(results), 0)
        top_ids = [r["doc_id"] for r in results[:2]]
        self.assertTrue(1 in top_ids or 9 in top_ids)
        self.assertGreater(results[0]["score"], 0)

    def test_boolean_and_search(self):
        """Test 2: AI AND healthcare -> Only docs containing both terms"""
        results = boolean_search("AI AND healthcare", self.store)
        doc_ids = {r["doc_id"] for r in results}
        self.assertIn(1, doc_ids)
        self.assertIn(9, doc_ids)
        self.assertNotIn(8, doc_ids)

    def test_boolean_or_search(self):
        """Test 3: AI OR healthcare -> Docs containing either term"""
        results = boolean_search("AI OR healthcare", self.store)
        doc_ids = {r["doc_id"] for r in results}
        self.assertIn(1, doc_ids)
        self.assertIn(8, doc_ids)
        self.assertIn(9, doc_ids)

    def test_boolean_not_search(self):
        """Test 4: AI NOT healthcare -> AI docs excluding healthcare"""
        results = boolean_search("AI NOT healthcare", self.store)
        doc_ids = {r["doc_id"] for r in results}
        self.assertIn(8, doc_ids)
        self.assertNotIn(1, doc_ids)
        self.assertNotIn(9, doc_ids)

    def test_vector_search(self):
        """Test 5: machine learning -> Ranked by cosine similarity"""
        engine = VectorSearchEngine(self.store)
        results = engine.search("machine learning", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["score"], 0.0)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bm25_search(self):
        """Test 6: machine learning -> Ranked by BM25 scores"""
        engine = BM25SearchEngine(self.store)
        results = engine.search("machine learning", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["score"], 0.0)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_field_search_title(self):
        """Test 7: neural network (Title only) -> Only title matches"""
        results = search_by_field("neural network", self.store, field="title", mode="bm25", top_k=5)
        self.assertGreater(len(results), 0)
        for r in results:
            doc = self.store.get_doc(r["doc_id"])
            title_lower = doc["title"].lower()
            self.assertTrue("neural" in title_lower or "network" in title_lower)

    def test_exact_phrase_search(self):
        """Test 8: 'artificial intelligence' -> Exact consecutive match only"""
        results = exact_phrase_search("artificial intelligence", self.store, top_k=5)
        self.assertGreater(len(results), 0)
        top_ids = [r["doc_id"] for r in results]
        self.assertTrue(1 in top_ids or 9 in top_ids)
        for r in results:
            self.assertGreaterEqual(r["occurrences"], 1)

    def test_proximity_near_search(self):
        """Test 9: artificial NEAR/3 healthcare -> Terms within 3 positions"""
        results = near_proximity_search("artificial", "healthcare", 3, self.store, top_k=5)
        self.assertGreater(len(results), 0)
        top_ids = [r["doc_id"] for r in results]
        self.assertIn(1, top_ids)
        for r in results:
            self.assertLessEqual(r["min_distance"], 3)

    def test_region_search(self):
        """Test 10: AI healthcare (All Fields) -> Shows matching snippet + field label"""
        results = search_with_regions("artificial intelligence healthcare", self.store, top_k=3)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("region", r)
            self.assertIn(r["region"], ["Title", "Abstract", "Methodology", "Results"])
            self.assertGreater(len(r["snippet"]), 0)

    def test_non_overlapped_search(self):
        """Test 11: deep learning (Non-overlapped) -> Deduplicated results from all fields"""
        merged = non_overlapped_search("deep learning", self.store, top_k=10)
        self.assertIn("per_field_matches", merged)
        self.assertIn("results", merged)
        self.assertGreater(len(merged["results"]), 0)
        doc_ids = [r["doc_id"] for r in merged["results"]]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))
        for r in merged["results"]:
            self.assertGreaterEqual(len(r["contributing_fields"]), 1)

    def test_empty_query(self):
        """Test 12: Empty query -> Validation handled gracefully with empty list"""
        self.assertEqual(tf_search("", self.store), [])
        self.assertEqual(boolean_search("", self.store), [])
        self.assertEqual(vector_search("", self.store), [])
        self.assertEqual(bm25_search("", self.store), [])
        self.assertEqual(search_by_field("", self.store), [])
        self.assertEqual(exact_phrase_search("", self.store), [])
        self.assertEqual(proximity_search("", self.store), [])
        self.assertEqual(search_with_regions("", self.store), [])
        self.assertEqual(non_overlapped_search("", self.store)["results"], [])


if __name__ == "__main__":
    unittest.main()
