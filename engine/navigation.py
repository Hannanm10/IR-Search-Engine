"""
Navigation, Browsing, Taxonomy, Faceted Filtering, and Related Document Recommendation Engine.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from engine.indexing import IndexStore
from engine.preprocessing import tokenize, stem_single
from engine.region_search import split_sentences, extract_best_region

CATEGORY_ICONS: Dict[str, str] = {
    "Healthcare & Medicine": "🩺",
    "Natural Language Processing": "💬",
    "Computer Vision": "👁️",
    "Cybersecurity & ML Defense": "🛡️",
    "Data Mining & Recommender Systems": "📊",
}


def get_taxonomy_tree(store: IndexStore) -> Dict[str, Any]:
    """
    Build hierarchical taxonomy tree: Category -> Subcategory -> List of Documents.
    """
    taxonomy: Dict[str, Any] = {}
    docs = store.get_all_docs()

    for doc in docs:
        cat = doc.get("category", "General")
        subcat = doc.get("subcategory", "Uncategorized")

        if cat not in taxonomy:
            taxonomy[cat] = {
                "icon": CATEGORY_ICONS.get(cat, "📁"),
                "count": 0,
                "subcategories": defaultdict(list)
            }

        taxonomy[cat]["count"] += 1
        taxonomy[cat]["subcategories"][subcat].append({
            "id": doc["id"],
            "title": doc["title"],
            "year": doc.get("year", 2024),
            "doc_type": doc.get("doc_type", "Article"),
            "authors": doc.get("authors", []),
            "topics": doc.get("topics", [])
        })

    # Convert defaultdicts to regular dicts
    result = {}
    for cat, data in taxonomy.items():
        result[cat] = {
            "icon": data["icon"],
            "count": data["count"],
            "subcategories": dict(data["subcategories"])
        }
    return result


def get_categories(store: IndexStore) -> List[str]:
    """Return sorted unique category names."""
    cats = {doc.get("category", "General") for doc in store.get_all_docs()}
    return sorted(list(cats))


def get_subcategories(store: IndexStore, category: Optional[str] = None) -> List[str]:
    """Return sorted unique subcategories, optionally filtered by category."""
    docs = store.get_all_docs()
    if category and category != "All":
        docs = [d for d in docs if d.get("category") == category]
    subcats = {doc.get("subcategory", "General") for doc in docs}
    return sorted(list(subcats))


def get_facet_options(store: IndexStore, docs_subset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Compute available facet options and their frequencies from the corpus or subset.
    """
    docs = docs_subset if docs_subset is not None else store.get_all_docs()
    
    categories = defaultdict(int)
    subcategories = defaultdict(int)
    doc_types = defaultdict(int)
    years = defaultdict(int)
    authors = defaultdict(int)
    topics = defaultdict(int)
    languages = defaultdict(int)

    all_years = []

    for d in docs:
        categories[d.get("category", "General")] += 1
        subcategories[d.get("subcategory", "General")] += 1
        doc_types[d.get("doc_type", "Other")] += 1
        
        y = d.get("year", 2024)
        years[y] += 1
        all_years.append(y)

        for a in d.get("authors", []):
            authors[a] += 1

        for t in d.get("topics", []):
            topics[t] += 1

        languages[d.get("language", "English")] += 1

    min_year = min(all_years) if all_years else 2021
    max_year = max(all_years) if all_years else 2026

    return {
        "categories": dict(sorted(categories.items())),
        "subcategories": dict(sorted(subcategories.items())),
        "doc_types": dict(sorted(doc_types.items())),
        "years": dict(sorted(years.items())),
        "year_min": min_year,
        "year_max": max_year,
        "authors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)),
        "topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)),
        "languages": dict(sorted(languages.items()))
    }


def filter_documents(
    store: IndexStore,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    doc_types: Optional[List[str]] = None,
    year_range: Optional[Tuple[int, int]] = None,
    authors: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter documents based on multi-faceted criteria.
    """
    matched = []
    docs = store.get_all_docs()

    for doc in docs:
        # Category filter
        if category and category != "All" and doc.get("category") != category:
            continue

        # Subcategory filter
        if subcategory and subcategory != "All" and doc.get("subcategory") != subcategory:
            continue

        # Document type filter
        if doc_types and doc.get("doc_type") not in doc_types:
            continue

        # Year range filter
        if year_range:
            doc_year = doc.get("year", 2024)
            if not (year_range[0] <= doc_year <= year_range[1]):
                continue

        # Authors filter (matches if ANY selected author is on document)
        if authors:
            doc_authors = set(doc.get("authors", []))
            if not any(a in doc_authors for a in authors):
                continue

        # Topics filter (matches if ANY selected topic is on document)
        if topics:
            doc_topics = set(doc.get("topics", []))
            if not any(t in doc_topics for t in topics):
                continue

        # Language filter
        if language and language != "All" and doc.get("language") != language:
            continue

        matched.append(doc)

    return matched


def get_related_documents(
    doc_id: int,
    store: IndexStore,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Content-based document recommendation combining TF-IDF similarity,
    category/subcategory affinity, shared topics, and internal citation links.
    """
    source_doc = store.get_doc(doc_id)
    if not source_doc:
        return []

    from engine.vector_search import VectorSearchEngine
    vector_engine = VectorSearchEngine(store)

    # Search using source document's core terms (title + topics + abstract snippet)
    source_topics = " ".join(source_doc.get("topics", []))
    query_text = f"{source_doc.get('title', '')} {source_topics} {source_doc.get('abstract', '')[:180]}"
    
    # Retrieve top candidates across the corpus
    vector_results = vector_engine.search(query_text, top_k=store.total_docs)
    vector_scores = {r["doc_id"]: r["score"] for r in vector_results}

    source_cat = source_doc.get("category")
    source_subcat = source_doc.get("subcategory")
    source_topic_set = set(source_doc.get("topics", []))
    cited_ids = set(source_doc.get("internal_links", []))

    recommendations = []
    for other in store.get_all_docs():
        other_id = other["id"]
        if other_id == doc_id:
            continue

        base_sim = vector_scores.get(other_id, 0.0)
        boost = 0.0

        # Subcategory match boost (+0.25)
        if other.get("subcategory") == source_subcat:
            boost += 0.25
        elif other.get("category") == source_cat:
            boost += 0.12

        # Shared topics boost (+0.08 per shared tag)
        other_topics = set(other.get("topics", []))
        shared = source_topic_set.intersection(other_topics)
        boost += len(shared) * 0.08

        # Explicit internal citation boost (+0.30)
        is_cited = other_id in cited_ids
        if is_cited:
            boost += 0.30

        total_score = base_sim + boost

        recommendations.append({
            "doc_id": other_id,
            "title": other.get("title", f"Doc {other_id}"),
            "category": other.get("category", "General"),
            "subcategory": other.get("subcategory", "General"),
            "doc_type": other.get("doc_type", "Article"),
            "year": other.get("year", 2024),
            "authors": other.get("authors", []),
            "shared_topics": list(shared),
            "is_cited": is_cited,
            "raw_similarity": round(base_sim, 4),
            "score": round(total_score, 4),
            "doc": other
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_k]


def build_breadcrumb(
    view: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    doc_id: Optional[int] = None,
    store: Optional[IndexStore] = None
) -> List[Dict[str, Any]]:
    """
    Constructs an interactive breadcrumb trail reflecting current navigational state.
    """
    trail = [
        {"label": "🏠 All Collections", "view": "home", "category": None, "subcategory": None, "doc_id": None}
    ]

    cat_icon = CATEGORY_ICONS.get(category, "📁") if category else "📁"

    if category and category != "All":
        trail.append({
            "label": f"{cat_icon} {category}",
            "view": "category",
            "category": category,
            "subcategory": None,
            "doc_id": None
        })

    if subcategory and subcategory != "All":
        trail.append({
            "label": f"📂 {subcategory}",
            "view": "subcategory",
            "category": category,
            "subcategory": subcategory,
            "doc_id": None
        })

    if doc_id and store:
        doc = store.get_doc(doc_id)
        if doc:
            title_short = doc["title"][:40] + "..." if len(doc["title"]) > 40 else doc["title"]
            trail.append({
                "label": f"📄 Doc #{doc_id}: {title_short}",
                "view": "doc",
                "category": category,
                "subcategory": subcategory,
                "doc_id": doc_id
            })

    return trail


def get_document_preview(doc: Dict[str, Any], store: IndexStore) -> Dict[str, Any]:
    """
    Produce structured summary and quick preview highlights for a document.
    """
    abstract = doc.get("abstract", "")
    methodology = doc.get("methodology", "")
    results = doc.get("results", "")

    # Calculate word count & reading time estimate (~200 words/min)
    total_text = f"{doc.get('title', '')} {abstract} {methodology} {results}"
    words = len(tokenize(total_text))
    reading_time = max(1, round(words / 200))

    # Sentence extractions for quick preview
    abs_sentences = split_sentences(abstract)
    abstract_preview = " ".join(abs_sentences[:2]) if abs_sentences else abstract[:200]

    meth_sentences = split_sentences(methodology)
    method_preview = meth_sentences[0] if meth_sentences else methodology[:150]

    res_sentences = split_sentences(results)
    results_preview = res_sentences[0] if res_sentences else results[:150]

    # Resolve internal links into document summaries
    cited_docs = []
    for c_id in doc.get("internal_links", []):
        c_doc = store.get_doc(c_id)
        if c_doc:
            cited_docs.append({
                "id": c_id,
                "title": c_doc.get("title", f"Doc {c_id}"),
                "category": c_doc.get("category", "General"),
                "subcategory": c_doc.get("subcategory", "General")
            })

    return {
        "id": doc["id"],
        "title": doc.get("title", f"Doc {doc['id']}"),
        "category": doc.get("category", "General"),
        "subcategory": doc.get("subcategory", "General"),
        "doc_type": doc.get("doc_type", "Journal Article"),
        "year": doc.get("year", 2024),
        "authors": doc.get("authors", []),
        "topics": doc.get("topics", []),
        "language": doc.get("language", "English"),
        "word_count": words,
        "reading_time_mins": reading_time,
        "abstract_preview": abstract_preview,
        "methodology_preview": method_preview,
        "results_preview": results_preview,
        "cited_docs": cited_docs
    }
