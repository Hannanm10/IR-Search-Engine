"""
Unit and integration tests for Browsing & Navigation module (engine/navigation.py).
Tests taxonomy hierarchy, faceted filtering, related documents recommendation,
breadcrumbs, and preview generation.
"""

import os
import unittest
from engine.indexing import IndexStore
from engine.navigation import (
    get_taxonomy_tree,
    get_categories,
    get_subcategories,
    get_facet_options,
    filter_documents,
    get_related_documents,
    build_breadcrumb,
    get_document_preview
)


class TestNavigationModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
        cls.store = IndexStore(data_path)

    def test_taxonomy_tree(self):
        """Test hierarchical taxonomy tree structure and document counts."""
        tree = get_taxonomy_tree(self.store)
        self.assertGreaterEqual(len(tree), 5)
        self.assertIn("Healthcare & Medicine", tree)
        self.assertIn("Natural Language Processing", tree)
        self.assertIn("Computer Vision", tree)
        self.assertIn("Cybersecurity & ML Defense", tree)
        self.assertIn("Data Mining & Recommender Systems", tree)

        total_counted = sum(data["count"] for data in tree.values())
        self.assertEqual(total_counted, self.store.total_docs)

        # Verify subcategories exist
        hc_subs = tree["Healthcare & Medicine"]["subcategories"]
        self.assertGreater(len(hc_subs), 0)

    def test_categories_and_subcategories(self):
        """Test category and subcategory retrieval."""
        cats = get_categories(self.store)
        self.assertEqual(len(cats), 5)

        subs_all = get_subcategories(self.store)
        self.assertGreater(len(subs_all), 10)

        subs_cv = get_subcategories(self.store, category="Computer Vision")
        self.assertGreater(len(subs_cv), 0)
        self.assertIn("Vision Transformers & Detection", subs_cv)

    def test_faceted_filtering_by_category(self):
        """Test filtering documents by category."""
        results = filter_documents(self.store, category="Healthcare & Medicine")
        self.assertEqual(len(results), 10)
        for d in results:
            self.assertEqual(d["category"], "Healthcare & Medicine")

    def test_faceted_filtering_by_doc_type(self):
        """Test filtering by document type."""
        results = filter_documents(self.store, doc_types=["Review Paper"])
        self.assertGreater(len(results), 0)
        for d in results:
            self.assertEqual(d["doc_type"], "Review Paper")

    def test_faceted_filtering_by_year_range(self):
        """Test filtering by year range."""
        results = filter_documents(self.store, year_range=(2024, 2025))
        self.assertGreater(len(results), 0)
        for d in results:
            self.assertTrue(2024 <= d["year"] <= 2025)

    def test_faceted_filtering_by_author(self):
        """Test filtering by author."""
        results = filter_documents(self.store, authors=["Dr. Elena Rostova"])
        self.assertGreater(len(results), 0)
        for d in results:
            self.assertIn("Dr. Elena Rostova", d["authors"])

    def test_faceted_filtering_by_topic(self):
        """Test filtering by topic."""
        results = filter_documents(self.store, topics=["Deep Learning"])
        self.assertGreater(len(results), 0)
        for d in results:
            self.assertIn("Deep Learning", d["topics"])

    def test_combined_filters(self):
        """Test multi-criteria filtering simultaneously."""
        results = filter_documents(
            self.store,
            category="Healthcare & Medicine",
            doc_types=["Journal Article"],
            year_range=(2023, 2025)
        )
        self.assertGreater(len(results), 0)
        for d in results:
            self.assertEqual(d["category"], "Healthcare & Medicine")
            self.assertEqual(d["doc_type"], "Journal Article")
            self.assertTrue(2023 <= d["year"] <= 2025)

    def test_related_documents_recommendation(self):
        """Test content-based related documents."""
        doc_id = 1  # AI in Healthcare
        related = get_related_documents(doc_id, self.store, top_k=4)
        self.assertEqual(len(related), 4)

        # Check that source document itself is excluded
        rec_ids = [r["doc_id"] for r in related]
        self.assertNotIn(doc_id, rec_ids)

        # Doc 1 has internal links to [2, 3, 9], so they should be highly recommended
        self.assertTrue(2 in rec_ids or 3 in rec_ids or 9 in rec_ids)
        self.assertGreater(related[0]["score"], 0.0)

    def test_breadcrumbs(self):
        """Test breadcrumb construction across navigation states."""
        # Home
        b_home = build_breadcrumb(view="home")
        self.assertEqual(len(b_home), 1)

        # Category
        b_cat = build_breadcrumb(view="category", category="Computer Vision")
        self.assertEqual(len(b_cat), 2)
        self.assertIn("Computer Vision", b_cat[1]["label"])

        # Document
        b_doc = build_breadcrumb(
            view="doc",
            category="Computer Vision",
            subcategory="Neural Radiance Fields (NeRF)",
            doc_id=25,
            store=self.store
        )
        self.assertEqual(len(b_doc), 4)
        self.assertIn("Doc #25", b_doc[3]["label"])

    def test_document_preview(self):
        """Test document preview generation."""
        doc = self.store.get_doc(1)
        preview = get_document_preview(doc, self.store)
        self.assertEqual(preview["id"], 1)
        self.assertGreater(preview["word_count"], 50)
        self.assertGreaterEqual(preview["reading_time_mins"], 1)
        self.assertTrue(len(preview["abstract_preview"]) > 0)
        self.assertGreater(len(preview["cited_docs"]), 0)


if __name__ == "__main__":
    unittest.main()
