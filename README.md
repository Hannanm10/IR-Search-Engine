# 🔍 Mini IR Search Engine

A modular, high-performance Information Retrieval system built in Python, showcasing foundational to advanced search techniques on structured academic research papers.

---

## 🌟 Key Features

| Capability | Module | Description |
|---|---|---|
| **Term Frequency (TF)** | `engine/indexing.py` | Classical term frequency summation (Week 1 baseline) |
| **Modular Pipeline** | `engine/preprocessing.py` | 5-stage architecture: Parse → Lookup → Retrieve → Score → Format |
| **Boolean Retrieval** | `engine/boolean_search.py` | Recursive descent AST parser supporting `AND`, `OR`, `NOT`, and `()` |
| **Vector Space Model** | `engine/vector_search.py` | TF-IDF term weighting and continuous Cosine Similarity ranking |
| **Okapi BM25** | `engine/bm25_search.py` | Probabilistic relevance ranking with tunable field weights |
| **Field-Scoped Search** | `engine/field_search.py` | Target searches to Title, Abstract, Methodology, or Results |
| **Region / Passage Retrieval** | `engine/region_search.py` | Automatic sentence extraction showing highest-overlap text |
| **Phrase & Proximity** | `engine/proximity_search.py` | Exact consecutive phrases (`"..."`) and `NEAR/N` distance matching |
| **Non-Overlapped Merging**| `engine/non_overlapped_search.py` | Multi-field independent retrieval with deduplicated ranking |
| **Browsing & Navigation** | `engine/navigation.py` | Taxonomy, breadcrumbs, related docs, recently viewed, and faceted filters |
| **Flexible Matching**     | `engine/flexible_matching.py` | Zadeh's Fuzzy Logic, typo tolerance, wildcards (*), and +/- constraints |
| **Modern Web Interface** | `app.py` | 3-Mode Streamlit UI (Search Engine, Browse & Collections, Flexible Matching) |

---

## 📁 Repository Structure

```
IR Modules/
├── app.py                          # Streamlit Web Application (Search, Browse, & Flexible modes)
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── implementation_plan.md          # Project plan & specifications
│
├── data/
│   └── documents.json              # 50 enriched academic research documents
│
├── engine/
│   ├── __init__.py
│   ├── preprocessing.py            # Tokenization, stopword removal, stemming, QueryProcessor
│   ├── indexing.py                 # Inverted index, positional index, IndexStore
│   ├── boolean_search.py           # Recursive descent Boolean AST search
│   ├── vector_search.py            # Scikit-learn TF-IDF + Cosine similarity
│   ├── bm25_search.py              # Rank-BM25 probabilistic ranking + field weights
│   ├── field_search.py             # Field-specific queries (Title, Abstract, etc.)
│   ├── region_search.py            # Sentence-level snippet & region extraction
│   ├── proximity_search.py         # Positional exact phrase & NEAR/N matching
│   ├── non_overlapped_search.py    # Independent field list retrieval & deduplicated merge
│   ├── navigation.py               # Browsing, taxonomy, faceted filtering, recommendations
│   └── flexible_matching.py        # Extended Boolean, Zadeh Fuzzy Logic, Partial/Typo matching
│
├── weekly_demos/
│   ├── week1_demo.py               # CLI: Inverted indexing + TF ranking
│   ├── week2_demo.py               # CLI: 5-stage modular pipeline walkthrough
│   ├── week3_demo.py               # CLI: Boolean search + Vector space model
│   ├── week4_demo.py               # CLI: BM25, field scope, regions, proximity, deduplication
│   ├── week5_demo.py               # CLI: Browsing, taxonomy, faceted filters, recommendations
│   └── week6_demo.py               # CLI: Flexible matching, Zadeh fuzzy logic, wildcards, include/exclude
│
├── tests/
│   ├── __init__.py
│   ├── test_search.py              # Automated 12-test search test suite
│   ├── test_navigation.py          # Automated 11-test navigation & browsing test suite
│   └── test_flexible_matching.py   # Automated 12-test flexible matching test suite
│
├── docs/
│   ├── architecture_diagram.md     # Pipeline diagrams & stage documentation
│   ├── weekly_report.md            # Comprehensive weekly progress report
│   └── screenshots/                # Visual deliverables for weekly milestones
│
└── scripts/
    └── generate_data.py            # Reproducible 50-document academic corpus generator
```

---

## ⚡ Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web UI
```bash
python -m streamlit run app.py
```
*Access the interface at `http://localhost:8501`.*

### 3. Run the Automated Test Suite
```bash
python -m unittest tests/test_search.py
```
*(All 12 unit and integration tests validate retrieval correctness).*

---

## 🖥️ Running Weekly CLI Demos

Each weekly demo is completely standalone and can be executed independently:

### Week 1 — Basic Search & Term Frequency Ranking
```bash
python weekly_demos/week1_demo.py "artificial intelligence"
```

### Week 2 — Modular 5-Stage Pipeline Walkthrough
```bash
python weekly_demos/week2_demo.py "artificial intelligence"
```

### Week 3 — Boolean & Vector Search
```bash
# Run both demonstrations:
python weekly_demos/week3_demo.py

# Or specify a custom Boolean query:
python weekly_demos/week3_demo.py --mode boolean --query "AI AND healthcare NOT imaging"

# Or specify a custom Vector query:
python weekly_demos/week3_demo.py --mode vector --query "machine learning clinical"
```

### Week 4 — Advanced Retrieval Suite
```bash
python weekly_demos/week4_demo.py
```

### Week 5 — Browsing, Navigation & Recommendation Suite
```bash
python weekly_demos/week5_demo.py
```

### Week 6 — Flexible Matching & Extended Boolean Suite
```bash
python weekly_demos/week6_demo.py
```

---

## 📊 Dataset Topic Clusters

The corpus consists of **50 structured research papers** (`data/documents.json`) across 5 domains:
1. **AI in Healthcare & Clinical Diagnostics** (Docs 1–10)
2. **Natural Language Processing & Transformers** (Docs 11–20)
3. **Computer Vision & Scene Understanding** (Docs 21–30)
4. **Cybersecurity & Machine Learning** (Docs 31–40)
5. **Data Mining & Recommender Systems** (Docs 41–50)

Each paper contains `id`, `title`, `abstract`, `methodology`, and `results`.

---

## 🔬 Retrieval Models & Theory

### 1. Boolean Retrieval (AST Parser)
Queries are converted into an abstract syntax tree using recursive descent parsing. Operator precedence:
$$\text{NOT} > \text{AND} > \text{OR}$$
Posting lists are combined using set intersections, unions, and complements with parentheses support:
- `AI AND healthcare NOT imaging`
- `(neural OR deep) AND vision`

### 2. Vector Space Model (TF-IDF + Cosine Similarity)
Documents and queries are vectorized into term-space matrices:
$$\text{Cosine}(Q, D) = \frac{\vec{V}_Q \cdot \vec{V}_D}{\|\vec{V}_Q\| \|\vec{V}_D\|}$$

### 3. Okapi BM25 Ranking
Probabilistic retrieval scoring incorporating saturation and length normalization:
$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Parameters: $k_1 = 1.5, b = 0.75$. Field weights default to $2.0 \times \text{Title} + 1.0 \times \text{Abstract} + 0.5 \times \text{Methodology} + 0.5 \times \text{Results}$.

### 4. Proximity & Phrase Search
- **Exact Phrase**: Adjacency condition $\text{pos}(t_{i+1}) - \text{pos}(t_i) = 1$ verified across positional indexes.
- **NEAR/N Proximity**: Documents with $1 \le |\text{pos}(t_2) - \text{pos}(t_1)| \le N$, scored by closeness.

### 5. Region Retrieval
Segments documents into sentences, measures token overlap, and highlights the single most informative sentence alongside the source field name (`Title`, `Abstract`, `Methodology`, or `Results`).

### 6. Flexible Matching & Extended Boolean Suite (Zadeh's Fuzzy Logic)
Provides continuous relevance degrees in $[0.0, 1.0]$ with typo and wildcard tolerance:
- **Normalized Term Weighting**:
  $$w(t, D) = \frac{\text{TF}(t, D)}{\text{TF}(t, D) + 1.2 \cdot \left(0.25 + 0.75 \cdot \frac{|D|}{\text{avgdl}}\right)} \times \frac{\text{IDF}(t)}{\max \text{IDF}}$$
- **Typo & Prefix Partial Matching**:
  $$\text{sim}(q, v) = \begin{cases} 1.0 & \text{if exact match or prefix wildcard } q^* \\ 1.0 - \frac{\text{Levenshtein}(q, v)}{\max(|q|, |v|)} & \text{if ratio } \ge \tau \end{cases}$$
- **Zadeh's Classic Fuzzy Set Logic**:
  - Fuzzy $\text{AND}$: $\mu_{A \land B}(D) = \min(\mu_A(D), \mu_B(D))$
  - Fuzzy $\text{OR}$: $\mu_{A \lor B}(D) = \max(\mu_A(D), \mu_B(D))$
  - Fuzzy $\text{NOT}$: $\mu_{\neg A}(D) = 1.0 - \mu_A(D)$
- **Mandatory Hard Filters**:
  - `+term` (MUST be present): Document score zeroed if $w(\text{term}, D) = 0$.
  - `-term` (MUST NOT be present): Document eliminated if $w(\text{term}, D) > 0$.

