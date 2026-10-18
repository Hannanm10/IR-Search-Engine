"""
Week 5 Demo: Browsing, Taxonomy Navigation, Faceted Filtering, and Related Document Recommendation.
Demonstrates:
1. Category Menu & Subcategory Taxonomy
2. Search + Browse Integration (Collection-Scoped Search)
3. Breadcrumb Trail Generation
4. Related Document Recommendation Engine
5. Faceted Filtering across Multi-Dimensional Metadata
6. Document Preview Extraction
7. Internal Cross-Paper Links & Citation Navigation
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output encoding for Windows command line compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
from engine.bm25_search import BM25SearchEngine


def run_week5_demo():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")
    store = IndexStore(data_path)

    print("=" * 75)
    print(" Week 5 Demo: Browsing, Navigation & Recommendation Suite")
    print("=" * 75)

    # 1. Category Menu & Taxonomy Hierarchy
    print("\n[1] Category & Subcategory Taxonomy Tree")
    tree = get_taxonomy_tree(store)
    for cat_name, cat_data in tree.items():
        print(f"  {cat_data['icon']} {cat_name} ({cat_data['count']} Documents)")
        for subcat, doc_list in cat_data["subcategories"].items():
            print(f"      └── 📂 {subcat} [{len(doc_list)} doc(s): IDs {', '.join(str(d['id']) for d in doc_list)}]")

    # 2. Search + Browse Option (Scoped Search inside a Category)
    print("\n[2] Search + Browse Option: Category-Scoped Retrieval")
    target_category = "Computer Vision"
    search_query = "transformers"
    print(f"  Browsing Category : {target_category}")
    print(f"  Searching Query   : \"{search_query}\"")
    
    # Filter by category first, then run BM25 on the subset
    filtered_docs = filter_documents(store, category=target_category)
    bm25_engine = BM25SearchEngine(store)
    all_bm25 = bm25_engine.search(search_query, top_k=store.total_docs)
    category_doc_ids = {d["id"] for d in filtered_docs}
    scoped_results = [r for r in all_bm25 if r["doc_id"] in category_doc_ids][:3]

    print(f"  Top Matches within {target_category}:")
    for rank, res in enumerate(scoped_results, 1):
        print(f"    Rank {rank}: [Doc {res['doc_id']:2d}] {res['title']} (Score: {res['score']:.4f})")

    # 3. Breadcrumb Path Generation
    print("\n[3] Interactive Breadcrumb Trails")
    sample_doc_id = 25
    doc_25 = store.get_doc(sample_doc_id)
    trail = build_breadcrumb(
        view="doc",
        category=doc_25["category"],
        subcategory=doc_25["subcategory"],
        doc_id=sample_doc_id,
        store=store
    )
    print("  Navigation Trail for Document #25:")
    breadcrumb_str = "  " + "  →  ".join([crumb["label"] for crumb in trail])
    print(breadcrumb_str)

    # 4. Related Documents Recommendation
    print("\n[4] Content-Based Related Documents Recommendation")
    src_id = 1
    src_doc = store.get_doc(src_id)
    print(f"  Source Document: [Doc {src_id}] {src_doc['title']}")
    print(f"  Category: {src_doc['category']} | Subcategory: {src_doc['subcategory']}")
    print(f"  Topics: {', '.join(src_doc.get('topics', []))}")
    print(f"  Internal Citation Links: {src_doc.get('internal_links', [])}")
    print("  Recommended Similar Papers:")

    related = get_related_documents(src_id, store, top_k=3)
    for rank, rel in enumerate(related, 1):
        cited_badge = " [CITES THIS PAPER]" if rel["is_cited"] else ""
        print(f"    #{rank}: [Doc {rel['doc_id']:2d}] {rel['title']}{cited_badge}")
        print(f"         Relevance Score: {rel['score']:.4f} | Subcategory: {rel['subcategory']}")
        if rel["shared_topics"]:
            print(f"         Shared Topics: {', '.join(rel['shared_topics'])}")

    # 5. Faceted Filtering
    print("\n[5] Multi-Faceted Document Filtering")
    criteria = {
        "category": "Healthcare & Medicine",
        "doc_types": ["Journal Article"],
        "year_range": (2024, 2025)
    }
    print(f"  Filter Criteria: Category='{criteria['category']}', DocType={criteria['doc_types']}, Years={criteria['year_range']}")
    filtered = filter_documents(
        store,
        category=criteria["category"],
        doc_types=criteria["doc_types"],
        year_range=criteria["year_range"]
    )
    print(f"  Matching Documents ({len(filtered)} found):")
    for d in filtered[:4]:
        print(f"    - [Doc {d['id']:2d}] ({d.get('year')}) [{d.get('doc_type')}] {d['title']}")
    if len(filtered) > 4:
        print(f"      ... and {len(filtered) - 4} more")

    # 6. Document Preview & Metadata Summarization
    print("\n[6] Document Preview Highlights")
    preview = get_document_preview(src_doc, store)
    print(f"  Title        : {preview['title']}")
    print(f"  Authors      : {', '.join(preview['authors'])}")
    print(f"  Word Count   : {preview['word_count']} words (~{preview['reading_time_mins']} min read)")
    print(f"  Abstract Key : \"{preview['abstract_preview'][:110]}...\"")
    print(f"  Method Key   : \"{preview['methodology_preview'][:110]}...\"")
    print(f"  Results Key  : \"{preview['results_preview'][:110]}...\"")

    # 7. Internal Links & Citation Graph
    print("\n[7] Internal Cross-Document Exploration")
    print(f"  Document #{src_id} references {len(preview['cited_docs'])} peer papers in this collection:")
    for cited in preview["cited_docs"]:
        print(f"    └── 🔗 [Doc {cited['id']:2d}] {cited['title']} ({cited['category']} > {cited['subcategory']})")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_week5_demo()
