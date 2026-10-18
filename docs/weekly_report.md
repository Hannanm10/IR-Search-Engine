# 📑 Weekly Progress & Technical Report
## Information Retrieval — 4-Week Modular Search Engine Project

---

## 🗓️ Week 1: Basic Search Engine & Inverted Indexing
### Objective
Establish foundational information retrieval components: document ingestion, standard NLP preprocessing, inverted index construction, and basic Term Frequency (TF) ranking.

### Deliverables & Key Implementations
1. **Dataset (`data/documents.json`)**:
   - Curated 50 structured academic research documents across 5 core topic clusters:
     - AI in Healthcare (Docs 1–10)
     - Natural Language Processing (Docs 11–20)
     - Computer Vision (Docs 21–30)
     - Cybersecurity & ML (Docs 31–40)
     - Data Mining & Recommender Systems (Docs 41–50)
   - Structured schema: `id`, `title`, `abstract`, `methodology`, `results`.

2. **Preprocessing Pipeline (`engine/preprocessing.py`)**:
   - Lowercase normalization and alphanumeric regex tokenization.
   - Comprehensive stopword removal combining standard IR stopword lists.
   - Morphological normalization using Porter Stemming (`nltk.stem.PorterStemmer`).

3. **Inverted Index (`engine/indexing.py`)**:
   - Built term-to-document posting lists storing document term frequencies:
     `term -> {doc_id: term_frequency}`.
   - Positional index mapping terms to exact token indices:
     `term -> {doc_id: [offsets...]}`.

4. **CLI Demonstration (`weekly_demos/week1_demo.py`)**:
   - Interactive CLI executing Term Frequency summation ranking:
     $$\text{Score}(D, Q) = \sum_{t \in Q} \text{TF}(t, D)$$

---

## 🗓️ Week 2: Modular Pipeline Architecture & Abstraction
### Objective
Refactor the initial codebase into a decoupled 5-stage retrieval pipeline. Document the architecture with formal diagrams and data flow specifications.

### Deliverables & Key Implementations
1. **`IndexStore` Abstraction (`engine/indexing.py`)**:
   - Encapsulated index data structures, vocabulary catalogs, document length statistics, and average document lengths across fields.
   - Added helper accessors `get_posting_list()`, `get_positions()`, and `get_doc()`.

2. **`QueryProcessor` & `ParsedQuery` (`engine/preprocessing.py`)**:
   - Syntactic query inspection to identify phrase quotes, Boolean conjunctions, and proximity keywords (`NEAR/N`).
   - Normalizes queries into structured `ParsedQuery` instances.

3. **5-Stage Pipeline Walkthrough (`weekly_demos/week2_demo.py`)**:
   - Stage 1: Query Processing
   - Stage 2: Index Lookup
   - Stage 3: Candidate Retrieval
   - Stage 4: Scoring & Ranking
   - Stage 5: Result Formatting & Snippet Extraction

4. **Architecture Documentation (`docs/architecture_diagram.md`)**:
   - Mermaid flowchart and end-to-end data lifecycle explanation.

---

## 🗓️ Week 3: Boolean Retrieval & Vector Space Model
### Objective
Implement exact set-theoretic Boolean retrieval alongside continuous algebraic Vector Space ranking using TF-IDF and Cosine Similarity.

### Deliverables & Key Implementations
1. **Boolean Search Engine (`engine/boolean_search.py`)**:
   - Implemented a Recursive Descent AST Parser with formal operator precedence:
     $$\text{NOT} > \text{AND} > \text{OR}$$
   - Full support for grouping parentheses `( ... )`, unary negation against document universes, and implicit conjunction insertion.
   - Set operations on posting lists:
     - $\text{Intersection } (A \cap B)$ for `AND`
     - $\text{Union } (A \cup B)$ for `OR`
     - $\text{Difference } (U \setminus A)$ for unary `NOT`

2. **Vector Space Engine (`engine/vector_search.py`)**:
   - Fitted `TfidfVectorizer` models across global documents and individual field corpora.
   - Computed Cosine Similarity against normalized document vectors:
     $$\text{Cosine}(Q, D) = \frac{\vec{V}(Q) \cdot \vec{V}(D)}{\|\vec{V}(Q)\| \|\vec{V}(D)\|}$$
   - Sorted candidates continuously by similarity scores in $[0, 1]$.

3. **CLI Demonstration (`weekly_demos/week3_demo.py`)**:
   - Interactive command-line test runner showcasing both Boolean set matches and Vector cosine rankings.

---

## 🗓️ Week 4: Advanced Retrieval Suite & Unified UI
### Objective
Expand the retrieval suite with probabilistic BM25 ranking, structured field scoping, passage region extraction, positional proximity/phrase matching, non-overlapped list merging, and a modern Streamlit user interface.

### Deliverables & Key Implementations
1. **Okapi BM25 Ranking (`engine/bm25_search.py`)**:
   - Implemented Okapi BM25 ($k_1=1.5, b=0.75$) with length normalization:
     $$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
   - Field-aware multi-field weighted scoring:
     $$\text{Score} = 2.0 \cdot \text{BM25}_{\text{title}} + 1.0 \cdot \text{BM25}_{\text{abstract}} + 0.5 \cdot \text{BM25}_{\text{methodology}} + 0.5 \cdot \text{BM25}_{\text{results}}$$

2. **Structured Field Search (`engine/field_search.py`)**:
   - Restricts queries to target metadata sections: `Title`, `Abstract`, `Methodology`, or `Results`.

3. **Region Retrieval (`engine/region_search.py`)**:
   - Sentence-level boundary segmentation using regex.
   - Dynamic maximum-overlap sentence extraction to display the exact matching passage and identify the originating field.

4. **Proximity & Phrase Search (`engine/proximity_search.py`)**:
   - Exact consecutive phrase verification using positional indexes:
     $$\text{pos}(t_{i+1}) - \text{pos}(t_i) = 1$$
   - `NEAR/N` proximity search:
     $$1 \le |\text{pos}(t_2) - \text{pos}(t_1)| \le N$$
   - Proximity scoring based on inverse token distance.

5. **Non-Overlapped List Search (`engine/non_overlapped_search.py`)**:
   - Evaluates independent match lists per document field.
   - Unions and deduplicates candidate IDs, recording contributing fields for each match.

6. **Unified Streamlit Web UI (`app.py`)**:
   - Dark-mode responsive design with typography, gradient badges, and interactive controls.
   - Live query chips, model selector, field scope dropdown, sensitivity toggle, and expandable document viewers.

7. **Test Suite (`tests/test_search.py`)**:
   - Automated `unittest` suite covering all 12 test specifications from the design document.
