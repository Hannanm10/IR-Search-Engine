"""
Region Retrieval Engine: Extracts best matching passage/sentence from document fields.
"""

import re
from typing import List, Dict, Any, Optional
from engine.indexing import IndexStore
from engine.preprocessing import preprocess, stem_single, tokenize
from engine.bm25_search import BM25SearchEngine


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries (. ! ?).
    """
    if not text:
        return []
    # Split on periods/exclamations/questions followed by space or end of string
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def highlight_terms(text: str, query_stems: List[str]) -> str:
    """
    Wraps matching terms in markdown bold syntax for UI display.
    """
    if not text or not query_stems:
        return text

    words = re.findall(r"\w+|\W+", text)
    highlighted = []
    for w in words:
        if re.match(r"^\w+$", w):
            stemmed = stem_single(w)
            if stemmed in query_stems:
                highlighted.append(f"**{w}**")
            else:
                highlighted.append(w)
        else:
            highlighted.append(w)
    return "".join(highlighted)


def extract_best_region(
    doc: Dict[str, Any],
    query_terms: List[str],
    preferred_field: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate all fields (or preferred_field) and find the single sentence with
    the highest query-term overlap.
    """
    fields_to_check = [preferred_field] if (preferred_field and preferred_field in IndexStore.FIELDS) else IndexStore.FIELDS

    best_field = "abstract"
    best_sentence = ""
    max_overlap = -1
    best_highlighted = ""

    for f in fields_to_check:
        text = doc.get(f, "")
        if not text:
            continue

        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        for s in sentences:
            s_tokens = tokenize(s)
            s_stems = set(stem_single(t) for t in s_tokens)
            overlap = sum(1 for q in query_terms if q in s_stems)

            # Preference: more overlap, or earlier fields on tie
            if overlap > max_overlap:
                max_overlap = overlap
                best_field = f
                best_sentence = s
                best_highlighted = highlight_terms(s, query_terms)

    if not best_sentence:
        best_sentence = doc.get("abstract", "")[:150]
        best_highlighted = best_sentence

    return {
        "field": best_field.capitalize(),
        "snippet": best_sentence,
        "highlighted": best_highlighted,
        "overlap_count": max(0, max_overlap)
    }


def search_with_regions(
    query: str,
    store: IndexStore,
    field: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Execute retrieval and enrich each result with its best matching region / snippet.
    """
    bm25_engine = BM25SearchEngine(store)
    results = bm25_engine.search(query, field=field, top_k=top_k)

    query_stems = preprocess(query, remove_stop=True, do_stem=True)

    for item in results:
        doc = item.get("doc", store.get_doc(item["doc_id"]))
        if doc:
            region_info = extract_best_region(doc, query_stems, preferred_field=field)
            item["region"] = region_info["field"]
            item["snippet"] = region_info["snippet"]
            item["highlighted_snippet"] = region_info["highlighted"]
            item["overlap_count"] = region_info["overlap_count"]

    return results
