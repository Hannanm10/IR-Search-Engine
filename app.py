"""
Mini IR Search Engine - Streamlit Web Application
A full-featured academic information retrieval interface integrating:
- Search Suite: BM25, TF-IDF Vector Space, Boolean Search, Positional Proximity & Phrase matching,
  Structured Field Scoping, and Region Passage Retrieval.
- Browsing & Navigation Suite: Category taxonomy, Search+Browse option, Breadcrumb trails,
  Related document recommendation, Recently viewed cache, Faceted filtering, Navigation history,
  Document previews, and Internal citation links.
"""

import os
import streamlit as st
import time

from engine.indexing import IndexStore
from engine.preprocessing import QueryProcessor, preprocess
from engine.bm25_search import BM25SearchEngine
from engine.vector_search import VectorSearchEngine
from engine.boolean_search import boolean_search
from engine.proximity_search import exact_phrase_search, near_proximity_search, proximity_search
from engine.region_search import extract_best_region
from engine.non_overlapped_search import non_overlapped_search
from engine.navigation import (
    CATEGORY_ICONS,
    get_taxonomy_tree,
    get_categories,
    get_subcategories,
    get_facet_options,
    filter_documents,
    get_related_documents,
    build_breadcrumb,
    get_document_preview
)
from weekly_demos.week1_demo import tf_search

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mini IR Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container background & spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Header Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 22px 30px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }

    /* Breadcrumb Bar */
    .breadcrumb-bar {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        color: #94A3B8;
    }

    /* Category Cards */
    .cat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .cat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 10px 28px -8px rgba(56, 189, 248, 0.2);
    }
    .cat-icon {
        font-size: 2.2rem;
        margin-bottom: 10px;
    }
    .cat-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .cat-meta {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }

    /* Result / Preview Cards */
    .result-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 8px 24px -6px rgba(56, 189, 248, 0.15);
    }

    /* Related Doc Card */
    .related-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(147, 197, 253, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    /* Badges */
    .badge-rank {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: #FFFFFF;
        padding: 3px 9px;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-score {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 9px;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-cat {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-topic {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.74rem;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .badge-cited {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-block;
    }

    .result-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .snippet-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38BDF8;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        margin-bottom: 10px;
        color: #CBD5E1;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_search_engine():
    data_path = os.path.join(os.path.dirname(__file__), "data", "documents.json")
    store = IndexStore(data_path)
    bm25_engine = BM25SearchEngine(store)
    vector_engine = VectorSearchEngine(store)
    return store, bm25_engine, vector_engine

store, bm25_engine, vector_engine = load_search_engine()


# ---------------------------------------------------------
# Session State Initialization (Navigation History & Cache)
# ---------------------------------------------------------
if "nav_mode" not in st.session_state:
    st.session_state.nav_mode = "🧭 Browse & Collections"

if "browse_view" not in st.session_state:
    st.session_state.browse_view = "catalog"  # 'catalog', 'collection', 'document'

if "browse_category" not in st.session_state:
    st.session_state.browse_category = None

if "browse_subcategory" not in st.session_state:
    st.session_state.browse_subcategory = None

if "browse_doc_id" not in st.session_state:
    st.session_state.browse_doc_id = None

if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []

if "nav_history" not in st.session_state:
    st.session_state.nav_history = [{"view": "catalog", "category": None, "subcategory": None, "doc_id": None}]

if "nav_history_idx" not in st.session_state:
    st.session_state.nav_history_idx = 0


def push_nav_state(view: str, category: str = None, subcategory: str = None, doc_id: int = None):
    """Pushes a new state to the navigation history stack and records recently viewed documents."""
    current = {
        "view": view,
        "category": category,
        "subcategory": subcategory,
        "doc_id": doc_id
    }
    # Truncate forward history if navigating from middle
    st.session_state.nav_history = st.session_state.nav_history[:st.session_state.nav_history_idx + 1]
    st.session_state.nav_history.append(current)
    st.session_state.nav_history_idx = len(st.session_state.nav_history) - 1

    # Update active state
    st.session_state.browse_view = view
    st.session_state.browse_category = category
    st.session_state.browse_subcategory = subcategory
    st.session_state.browse_doc_id = doc_id

    # Track recently viewed documents (max 8)
    if doc_id:
        recents = [d for d in st.session_state.recently_viewed if d != doc_id]
        recents.insert(0, doc_id)
        st.session_state.recently_viewed = recents[:8]


def go_nav_back():
    if st.session_state.nav_history_idx > 0:
        st.session_state.nav_history_idx -= 1
        state = st.session_state.nav_history[st.session_state.nav_history_idx]
        st.session_state.browse_view = state["view"]
        st.session_state.browse_category = state["category"]
        st.session_state.browse_subcategory = state["subcategory"]
        st.session_state.browse_doc_id = state["doc_id"]


def go_nav_forward():
    if st.session_state.nav_history_idx < len(st.session_state.nav_history) - 1:
        st.session_state.nav_history_idx += 1
        state = st.session_state.nav_history[st.session_state.nav_history_idx]
        st.session_state.browse_view = state["view"]
        st.session_state.browse_category = state["category"]
        st.session_state.browse_subcategory = state["subcategory"]
        st.session_state.browse_doc_id = state["doc_id"]


# ---------------------------------------------------------
# Hero Banner & Top Mode Selector
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔍 Mini IR Search Engine</div>
    <div class="hero-subtitle">
        A modular Information Retrieval system integrating different IR models and a full Hierarchical Browsing & Recommendation Suite.
    </div>
</div>
""", unsafe_allow_html=True)

mode_col1, mode_col2 = st.columns([4, 6])
with mode_col1:
    selected_mode = st.radio(
        "Navigation Mode",
        ["🧭 Browse & Collections", "🔍 Search Engine"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_mode_switch"
    )
st.session_state.nav_mode = selected_mode


# =========================================================
# MODE 1: BROWSING & NAVIGATION SUITE
# =========================================================
if st.session_state.nav_mode == "🧭 Browse & Collections":

    # --- TOP CONTROLS: History Buttons & Breadcrumb Trail ---
    top_nav_col1, top_nav_col2 = st.columns([2, 8])

    with top_nav_col1:
        btn_c1, btn_c2 = st.columns(2)
        can_back = st.session_state.nav_history_idx > 0
        can_forward = st.session_state.nav_history_idx < len(st.session_state.nav_history) - 1

        if btn_c1.button("⬅️ Back", disabled=not can_back, use_container_width=True, key="btn_nav_back"):
            go_nav_back()
            st.rerun()

        if btn_c2.button("➡️ Fwd", disabled=not can_forward, use_container_width=True, key="btn_nav_forward"):
            go_nav_forward()
            st.rerun()

    with top_nav_col2:
        # Generate breadcrumb items
        crumbs = build_breadcrumb(
            view=st.session_state.browse_view,
            category=st.session_state.browse_category,
            subcategory=st.session_state.browse_subcategory,
            doc_id=st.session_state.browse_doc_id,
            store=store
        )
        crumb_cols = st.columns(len(crumbs))
        for idx, crumb in enumerate(crumbs):
            with crumb_cols[idx]:
                if st.button(crumb["label"], key=f"crumb_{idx}_{crumb['view']}", use_container_width=True):
                    push_nav_state(crumb["view"], crumb["category"], crumb["subcategory"], crumb["doc_id"])
                    st.rerun()

    # --- RECENTLY VIEWED DOCUMENTS CAROUSEL ---
    if st.session_state.recently_viewed:
        with st.expander("🕒 Recently Viewed Documents", expanded=False):
            recent_cols = st.columns(min(len(st.session_state.recently_viewed), 4))
            for r_idx, r_id in enumerate(st.session_state.recently_viewed[:4]):
                r_doc = store.get_doc(r_id)
                if r_doc:
                    with recent_cols[r_idx]:
                        st.caption(f"{r_doc.get('category')} > {r_doc.get('subcategory')}")
                        if st.button(f"📄 #{r_id}: {r_doc['title'][:32]}...", key=f"recent_doc_{r_id}", use_container_width=True):
                            push_nav_state("document", r_doc.get("category"), r_doc.get("subcategory"), r_id)
                            st.rerun()

    # =====================================================
    # VIEW A: CATALOG VIEW (Category Menu Overview)
    # =====================================================
    if st.session_state.browse_view == "catalog":
        st.markdown("### 🗂️ Explore Document Collections by Category")
        st.markdown("Select a category below to browse its research subcategories, filter documents, or search within the collection.")

        taxonomy = get_taxonomy_tree(store)
        cat_grid = st.columns(3)
        cat_items = list(taxonomy.items())

        for idx, (cat_name, cat_data) in enumerate(cat_items):
            col = cat_grid[idx % 3]
            with col:
                st.markdown(f"""
                <div class="cat-card">
                    <div class="cat-icon">{cat_data['icon']}</div>
                    <div class="cat-title">{cat_name}</div>
                    <div class="cat-meta"><b>{cat_data['count']}</b> Research Documents &nbsp;|&nbsp; <b>{len(cat_data['subcategories'])}</b> Subcategories</div>
                    <p style="color:#94A3B8; font-size:0.86rem; line-height:1.4;">
                        Featuring peer-reviewed articles, conference proceedings, and survey papers in {cat_name.lower()}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Explore {cat_name} ➔", key=f"btn_explore_{idx}", use_container_width=True):
                    push_nav_state("collection", category=cat_name)
                    st.rerun()
                st.write("")

    # =====================================================
    # VIEW B: COLLECTION BROWSE VIEW (Subcategories & Filters)
    # =====================================================
    elif st.session_state.browse_view in ("collection", "subcategory"):
        cat_name = st.session_state.browse_category or "All"
        subcat_name = st.session_state.browse_subcategory

        # Sidebar: Faceted Filtering Controls
        with st.sidebar:
            st.markdown("### 🎛️ Faceted Filters")
            facet_meta = get_facet_options(store)

            # Subcategory Selector inside Category
            available_subcats = ["All"] + get_subcategories(store, category=cat_name)
            subcat_idx = 0
            if subcat_name and subcat_name in available_subcats:
                subcat_idx = available_subcats.index(subcat_name)

            selected_subcat = st.selectbox(
                "Subcategory",
                available_subcats,
                index=subcat_idx,
                help="Drill down to specific research subfield"
            )

            # Doc Type Multi-select
            doc_type_options = list(facet_meta["doc_types"].keys())
            selected_types = st.multiselect(
                "Document Type",
                doc_type_options,
                default=[],
                help="Filter by publication format"
            )

            # Publication Year Range Slider
            year_min, year_max = facet_meta["year_min"], facet_meta["year_max"]
            selected_years = st.slider(
                "Publication Year",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1
            )

            # Authors Filter
            top_authors = list(facet_meta["authors"].keys())[:15]
            selected_authors = st.multiselect(
                "Author",
                top_authors,
                default=[],
                help="Filter by author name"
            )

            # Topic Filter
            top_topics = list(facet_meta["topics"].keys())[:20]
            selected_topics = st.multiselect(
                "Topics & Tags",
                top_topics,
                default=[],
                help="Filter by research keyword"
            )

            if st.button("Reset All Filters", use_container_width=True):
                st.session_state.browse_subcategory = None
                st.rerun()

        # Update active subcategory from dropdown if changed
        active_sub = None if selected_subcat == "All" else selected_subcat

        # Filter the documents
        filtered_docs = filter_documents(
            store,
            category=None if cat_name == "All" else cat_name,
            subcategory=active_sub,
            doc_types=selected_types if selected_types else None,
            year_range=selected_years,
            authors=selected_authors if selected_authors else None,
            topics=selected_topics if selected_topics else None
        )

        # Header for Collection
        c_icon = CATEGORY_ICONS.get(cat_name, "📁")
        st.markdown(f"### {c_icon} {cat_name} {f'› {active_sub}' if active_sub else ''}")

        # Search + Browse Option: Scoped search within current collection
        search_col, toggle_col = st.columns([5, 2])
        with search_col:
            collection_query = st.text_input(
                "Search inside this collection",
                placeholder=f"Search within {cat_name}... (leave empty to browse all)",
                label_visibility="collapsed"
            ).strip()
        with toggle_col:
            st.caption(f"Showing **{len(filtered_docs)}** matching document(s)")

        # If scoped search is requested, rank filtered documents using BM25
        if collection_query:
            bm25_res = bm25_engine.search(collection_query, top_k=store.total_docs)
            filtered_ids = {d["id"] for d in filtered_docs}
            display_docs = [r["doc"] for r in bm25_res if r["doc_id"] in filtered_ids]
            st.info(f"🔎 Ranked {len(display_docs)} results for **\"{collection_query}\"** inside **{cat_name}**:")
        else:
            display_docs = filtered_docs

        if not display_docs:
            st.warning("No documents match the current filter criteria. Try broadening your selection or resetting filters.")
        else:
            # Document Preview Cards
            for doc in display_docs:
                preview = get_document_preview(doc, store)
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="badge-rank">Doc #{preview['id']}</span>
                                <span class="badge-cat" style="margin-left:6px;">{preview['category']}</span>
                                <span style="color:#94A3B8; font-size:0.82rem; margin-left:6px;">📂 {preview['subcategory']}</span>
                            </div>
                            <div>
                                <span style="color:#64748B; font-size:0.82rem;">{preview['doc_type']} &bull; {preview['year']} &bull; ⏱️ ~{preview['reading_time_mins']} min</span>
                            </div>
                        </div>
                        <div class="result-title">📄 {preview['title']}</div>
                        <div style="color:#94A3B8; font-size:0.84rem; margin-bottom:8px;">
                            <b>Authors:</b> {', '.join(preview['authors'])}
                        </div>
                        <div class="snippet-box">
                            <span style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; letter-spacing:0.5px; display:block; margin-bottom:3px;">Abstract Preview:</span>
                            "{preview['abstract_preview']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns([3, 7])
                    with btn_c1:
                        if st.button("📖 Open Full Document", key=f"btn_open_{preview['id']}", use_container_width=True):
                            push_nav_state("document", preview["category"], preview["subcategory"], preview["id"])
                            st.rerun()
                    with btn_c2:
                        with st.expander("👁️ Quick Highlights & Citations"):
                            st.markdown(f"**Methodology Snippet:** {preview['methodology_preview']}")
                            st.markdown(f"**Results Snippet:** {preview['results_preview']}")
                            if preview["topics"]:
                                tags_html = " ".join([f"<span class='badge-topic'>{t}</span>" for t in preview["topics"]])
                                st.markdown(f"**Topics:** {tags_html}", unsafe_allow_html=True)
                            if preview["cited_docs"]:
                                st.markdown("**Internal Citations in Paper:**")
                                for cd in preview["cited_docs"]:
                                    st.caption(f"🔗 Doc #{cd['id']}: {cd['title']} ({cd['category']})")

    # =====================================================
    # VIEW C: FULL DOCUMENT READING VIEW (Internal Links & Related Docs)
    # =====================================================
    elif st.session_state.browse_view == "document":
        doc_id = st.session_state.browse_doc_id
        doc = store.get_doc(doc_id)

        if not doc:
            st.error(f"Document #{doc_id} not found.")
        else:
            preview = get_document_preview(doc, store)

            st.markdown(f"""
            <div style="background:rgba(30, 41, 59, 0.7); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:24px 28px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span class="badge-rank">Document #{preview['id']}</span>
                    <span style="color:#94A3B8; font-size:0.85rem;">{preview['doc_type']} &bull; Published {preview['year']} &bull; Language: {preview['language']}</span>
                </div>
                <h2 style="color:#F8FAFC; margin-top:6px; margin-bottom:12px;">{preview['title']}</h2>
                <div style="color:#CBD5E1; font-size:0.92rem; margin-bottom:12px;">
                    <b>Authors:</b> {', '.join(preview['authors'])} &nbsp;|&nbsp; 
                    <b>Collection:</b> <span class="badge-cat">{preview['category']}</span> &nbsp;›&nbsp; <span>{preview['subcategory']}</span> &nbsp;|&nbsp;
                    <b>Length:</b> {preview['word_count']} words (~{preview['reading_time_mins']} min read)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Topic Tags
            if preview["topics"]:
                tag_cols = st.columns(len(preview["topics"]) + 2)
                st.markdown("**Tags & Topics:** " + " ".join([f"<span class='badge-topic'>{t}</span>" for t in preview["topics"]]), unsafe_allow_html=True)
                st.write("")

            # Main Paper Content Tabs
            t1, t2, t3 = st.tabs(["📄 Abstract", "🔬 Methodology", "📈 Results & Findings"])
            with t1:
                st.markdown(f"#### Abstract\n{doc.get('abstract')}")
            with t2:
                st.markdown(f"#### Research Methodology\n{doc.get('methodology')}")
            with t3:
                st.markdown(f"#### Experimental Results & Findings\n{doc.get('results')}")

            st.write("---")

            # Two Column Split: Internal Citation Links & Related Documents
            col_cited, col_related = st.columns(2)

            with col_cited:
                st.markdown("### 🔗 Internal Cross-References")
                st.caption("Peer papers in this collection directly referenced by this research:")
                if not preview["cited_docs"]:
                    st.info("No internal cross-citations recorded for this document.")
                else:
                    for cd in preview["cited_docs"]:
                        st.markdown(f"""
                        <div class="related-card">
                            <span class="badge-cited">Referenced Paper</span>
                            <span style="color:#94A3B8; font-size:0.8rem; margin-left:6px;">Doc #{cd['id']}</span>
                            <div style="font-weight:600; color:#F8FAFC; margin-top:4px;">{cd['title']}</div>
                            <div style="color:#94A3B8; font-size:0.78rem;">{cd['category']} &bull; {cd['subcategory']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Jump to Doc #{cd['id']} ➔", key=f"btn_jump_{cd['id']}", use_container_width=True):
                            push_nav_state("document", cd["category"], cd["subcategory"], cd["id"])
                            st.rerun()

            with col_related:
                st.markdown("### ✨ Related Research Recommendations")
                st.caption("Content-based similarity computed using Vector Cosine & Topic Affinity:")
                related_docs = get_related_documents(doc_id, store, top_k=3)
                for r_doc in related_docs:
                    cited_badge = "<span class='badge-cited' style='margin-left:6px;'>Internal Citation</span>" if r_doc["is_cited"] else ""
                    st.markdown(f"""
                    <div class="related-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="badge-score">Similarity: {r_doc['score']:.3f}</span>
                            <span>{cited_badge}</span>
                        </div>
                        <div style="font-weight:600; color:#F8FAFC; margin-top:6px;">{r_doc['title']}</div>
                        <div style="color:#94A3B8; font-size:0.78rem; margin-top:2px;">
                            {r_doc['category']} &bull; {r_doc['subcategory']} &bull; {r_doc['year']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Explore Doc #{r_doc['doc_id']} ➔", key=f"btn_rel_{r_doc['doc_id']}", use_container_width=True):
                        push_nav_state("document", r_doc["category"], r_doc["subcategory"], r_doc["doc_id"])
                        st.rerun()


# =========================================================
# MODE 2: CLASSICAL SEARCH ENGINE
# =========================================================
elif st.session_state.nav_mode == "🔍 Search Engine":

    # Sidebar: Search Engine Settings
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")

        retrieval_mode = st.selectbox(
            "Retrieval Algorithm",
            [
                "BM25 Probabilistic (Field-Aware)",
                "TF-IDF Vector Space (Cosine)",
                "Boolean Search (AND / OR / NOT)",
                "Exact Phrase / NEAR-N Proximity",
                "Non-Overlapped Multi-Field Merge",
                "Term Frequency (Week 1 Baseline)"
            ],
            index=0,
            help="Choose retrieval paradigm"
        )

        field_scope = st.selectbox(
            "Target Field Scope",
            ["All Fields", "Title", "Abstract", "Methodology", "Results"],
            index=0,
            help="Restricts retrieval to specific document section"
        )

        match_mode = st.radio(
            "Match Sensitivity",
            ["Broad Match", "Exact Match"],
            index=0,
            horizontal=True,
            help="Exact Match requires exact contiguous phrase matching; Broad Match uses ranking algorithm"
        )

        top_k = st.slider("Max Results to Display", min_value=1, max_value=25, value=10, step=1)

        st.markdown("---")
        st.markdown("### ⚖️ Field BM25 Weights")
        w_title = st.slider("Title Weight", 0.0, 5.0, 2.0, 0.5)
        w_abstract = st.slider("Abstract Weight", 0.0, 5.0, 1.0, 0.5)
        w_method = st.slider("Methodology Weight", 0.0, 5.0, 0.5, 0.1)
        w_results = st.slider("Results Weight", 0.0, 5.0, 0.5, 0.1)
        custom_weights = {
            "title": w_title,
            "abstract": w_abstract,
            "methodology": w_method,
            "results": w_results
        }

        st.markdown("---")
        st.markdown("### 📊 Corpus Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Documents", f"{store.total_docs}")
            st.metric("Avg Doc Len", f"{store.avg_doc_len:.0f} words")
        with c2:
            st.metric("Vocabulary", f"{len(store.inverted_index):,}")
            st.metric("Categories", f"{len(get_categories(store))}")

    # Query Presets / Chips
    st.markdown("##### 💡 Try Example Queries:")
    preset_cols = st.columns(6)

    presets = [
        ("🤖 AI Healthcare", "AI AND healthcare NOT imaging", "Boolean Search (AND / OR / NOT)"),
        ("📈 Machine Learning", "machine learning clinical", "BM25 Probabilistic (Field-Aware)"),
        ("💬 Exact Phrase", '"artificial intelligence"', "Exact Phrase / NEAR-N Proximity"),
        ("⚡ NEAR/3 Proximity", "artificial NEAR/3 healthcare", "Exact Phrase / NEAR-N Proximity"),
        ("🎯 Neural Network", "neural network", "BM25 Probabilistic (Field-Aware)"),
        ("🧩 Deep Learning", "deep learning", "Non-Overlapped Multi-Field Merge"),
    ]

    if "query_input" not in st.session_state:
        st.session_state.query_input = "artificial intelligence healthcare"

    for i, (label, sample_q, target_mode) in enumerate(presets):
        if preset_cols[i].button(label, use_container_width=True, key=f"chip_{i}"):
            st.session_state.query_input = sample_q

    # Search Form Input
    with st.form("search_form"):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            query_text = st.text_input(
                "Search Query",
                value=st.session_state.query_input,
                placeholder="Enter search terms, \"exact phrase\", boolean query, or term1 NEAR/N term2...",
                label_visibility="collapsed"
            )
        with btn_col:
            submitted = st.form_submit_button("Search 🔎", use_container_width=True, type="primary")

    active_query = query_text.strip()

    # Search Execution Logic
    if active_query:
        scope_arg = None if field_scope == "All Fields" else field_scope.lower()
        start_time = time.perf_counter()

        results = []
        status_label = ""
        extra_meta = {}

        # Honor Exact Match radio button
        if match_mode == "Exact Match" and not active_query.startswith('"'):
            results = exact_phrase_search(active_query, store, field=scope_arg, top_k=top_k)
            status_label = "Exact Consecutive Phrase Match"
        elif "Boolean" in retrieval_mode:
            results = boolean_search(active_query, store, field=scope_arg)
            status_label = f"Boolean Query Match ({len(results)} docs)"
        elif "Vector" in retrieval_mode:
            results = vector_engine.search(active_query, field=scope_arg, top_k=top_k)
            status_label = "TF-IDF Vector Space (Cosine Similarity)"
        elif "Phrase" in retrieval_mode or "Proximity" in retrieval_mode:
            results = proximity_search(active_query, store, field=scope_arg, top_k=top_k)
            status_label = "Positional Proximity / Phrase Retrieval"
        elif "Non-Overlapped" in retrieval_mode:
            nol_data = non_overlapped_search(active_query, store, top_k=top_k, field_weights=custom_weights)
            results = nol_data["results"]
            extra_meta["per_field"] = nol_data["per_field_matches"]
            status_label = "Non-Overlapped Multi-Field Deduplicated List"
        elif "Baseline" in retrieval_mode:
            results = tf_search(active_query, store, top_k=top_k)
            status_label = "Week 1 Term Frequency (TF) Ranking"
        else:
            results = bm25_engine.search(
                active_query,
                field=scope_arg,
                field_weights=custom_weights,
                top_k=top_k
            )
            status_label = "Field-Aware Okapi BM25 Ranking"

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Region Snippet Enrichment
        query_stems = preprocess(active_query, remove_stop=True, do_stem=True)
        for r in results:
            doc = r.get("doc") or store.get_doc(r["doc_id"])
            if doc and "snippet" not in r:
                reg_info = extract_best_region(doc, query_stems, preferred_field=scope_arg)
                r["region"] = reg_info["field"]
                r["snippet"] = reg_info["snippet"]
                r["highlighted_snippet"] = reg_info["highlighted"]

        # Render Search Results
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px; margin-bottom:18px;">
            <span style="color:#94A3B8; font-size:0.95rem;">
                Showing <b>{len(results)}</b> results for <code style="color:#38BDF8;">{active_query}</code>
                via <b>{status_label}</b>
            </span>
            <span style="color:#64748B; font-size:0.85rem;">
                Latency: <b>{duration_ms:.1f} ms</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        if extra_meta.get("per_field"):
            with st.expander("📊 Inspect Independent Per-Field Match Lists (Non-Overlapped)"):
                c_f1, c_f2, c_f3, c_f4 = st.columns(4)
                cols = [c_f1, c_f2, c_f3, c_f4]
                for idx, (f_name, doc_list) in enumerate(extra_meta["per_field"].items()):
                    with cols[idx]:
                        st.markdown(f"**{f_name.capitalize()} Matches** ({len(doc_list)})")
                        st.caption(", ".join(str(d) for d in doc_list) if doc_list else "None")

        if not results:
            st.warning("No matching documents found. Try broadening your query terms, checking spelling, or selecting 'All Fields'.")
        else:
            for rank, item in enumerate(results, 1):
                doc = item.get("doc") or store.get_doc(item["doc_id"])
                doc_id = item["doc_id"]
                title = doc.get("title", f"Document #{doc_id}")
                score = item.get("score", 1.0)
                region = item.get("region", "Abstract")
                snippet = item.get("snippet", doc.get("abstract", "")[:200])
                contrib = item.get("contributing_fields", [])
                category = doc.get("category", "General")
                subcategory = doc.get("subcategory", "General")

                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span class="badge-rank">#{rank}</span>
                                <span style="color:#94A3B8; font-size:0.85rem; margin-left:8px;">Doc ID: {doc_id}</span>
                                <span class="badge-cat" style="margin-left:8px;">{category}</span>
                            </div>
                            <div>
                                <span class="badge-score">Score: {score:.4f}</span>
                                <span style="background:rgba(245,158,11,0.15); color:#FBBF24; border:1px solid rgba(245,158,11,0.3); padding:3px 8px; border-radius:6px; font-size:0.8rem; margin-left:6px;">
                                    📍 Matched: {region}
                                </span>
                            </div>
                        </div>
                        <div class="result-title">📄 {title}</div>
                        <div class="snippet-box">
                            <span style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; letter-spacing:0.5px; display:block; margin-bottom:4px;">Best Matching Passage ({region}):</span>
                            "{snippet}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2 = st.columns([3, 7])
                    with btn_c1:
                        if st.button("🧭 View in Browse Mode", key=f"btn_search_view_{doc_id}", use_container_width=True):
                            st.session_state.nav_mode = "🧭 Browse & Collections"
                            push_nav_state("document", category, subcategory, doc_id)
                            st.rerun()
                    with btn_c2:
                        with st.expander(f"▶ Show Full Document (ID: {doc_id}) — {title}"):
                            t1, t2, t3 = st.tabs(["📄 Abstract", "🔬 Methodology", "📈 Results"])
                            with t1:
                                st.write(doc.get("abstract", "N/A"))
                            with t2:
                                st.write(doc.get("methodology", "N/A"))
                            with t3:
                                st.write(doc.get("results", "N/A"))
    else:
        st.info("👆 Enter a query above or click any of the preset query chips to begin searching.")
