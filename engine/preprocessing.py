"""
Preprocessing module for text tokenization, stopword removal, stemming, and query parsing.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from nltk.stem import PorterStemmer

# Comprehensive default English stop words set (ensures offline reliability)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}

# Try importing nltk stopwords if available, combine with default
try:
    from nltk.corpus import stopwords as nltk_stopwords
    STOPWORDS = set(nltk_stopwords.words("english")).union(DEFAULT_STOPWORDS)
except Exception:
    STOPWORDS = set(DEFAULT_STOPWORDS)

_stemmer = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """
    Convert text to lowercase and tokenize using alphanumeric word boundaries.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[set] = None) -> List[str]:
    """
    Filter out common stopwords from token list.
    """
    sw = custom_stopwords if custom_stopwords is not None else STOPWORDS
    return [t for t in tokens if t not in sw]


def stem(tokens: List[str]) -> List[str]:
    """
    Apply Porter Stemmer to a list of tokens.
    """
    return [_stemmer.stem(t) for t in tokens]


def stem_single(token: str) -> str:
    """
    Apply Porter Stemmer to a single token.
    """
    return _stemmer.stem(token.lower())


def preprocess(text: str, remove_stop: bool = True, do_stem: bool = True) -> List[str]:
    """
    Full pipeline: tokenize -> remove stopwords -> stem.
    """
    tokens = tokenize(text)
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if do_stem:
        tokens = stem(tokens)
    return tokens


@dataclass
class ParsedQuery:
    """
    Structured representation of a parsed user query.
    """
    raw_query: str
    terms: List[str] = field(default_factory=list)               # Preprocessed (stemmed) terms
    raw_tokens: List[str] = field(default_factory=list)          # Lowercase tokens before stopword/stem
    operators: List[str] = field(default_factory=list)           # 'AND', 'OR', 'NOT'
    is_phrase: bool = False                                      # Exact phrase query inside quotes
    phrase_terms: List[str] = field(default_factory=list)        # Ordered stemmed terms of phrase
    proximity_info: Optional[Dict[str, Any]] = None              # {'term1': t1, 'term2': t2, 'distance': n}
    mode: str = "basic"                                          # 'basic', 'boolean', 'phrase', 'proximity', 'vector', 'bm25'


class QueryProcessor:
    """
    Parses and categorizes queries for retrieval engines.
    """

    @staticmethod
    def parse_query(raw_query: str) -> ParsedQuery:
        query_str = raw_query.strip()
        if not query_str:
            return ParsedQuery(raw_query="")

        # 1. Check for NEAR/N proximity pattern: e.g. "artificial NEAR/3 healthcare"
        near_match = re.search(r"(\b\w+\b)\s+NEAR/(\d+)\s+(\b\w+\b)", query_str, re.IGNORECASE)
        if near_match:
            t1, dist_str, t2 = near_match.group(1), near_match.group(2), near_match.group(3)
            stem1, stem2 = stem_single(t1), stem_single(t2)
            distance = int(dist_str)
            return ParsedQuery(
                raw_query=query_str,
                terms=[stem1, stem2],
                raw_tokens=[t1.lower(), t2.lower()],
                proximity_info={"term1": stem1, "term2": stem2, "distance": distance, "raw_t1": t1, "raw_t2": t2},
                mode="proximity"
            )

        # 2. Check for exact phrase query: enclosed in quotes e.g. "artificial intelligence"
        phrase_match = re.findall(r'"([^"]+)"', query_str)
        if phrase_match:
            # We treat the first quoted group as the phrase
            phrase_text = phrase_match[0]
            raw_tokens = tokenize(phrase_text)
            phrase_stems = stem(raw_tokens)
            return ParsedQuery(
                raw_query=query_str,
                terms=phrase_stems,
                raw_tokens=raw_tokens,
                is_phrase=True,
                phrase_terms=phrase_stems,
                mode="phrase"
            )

        # 3. Check for Boolean operators: AND, OR, NOT (as distinct uppercase words or standalone keywords)
        bool_ops = re.findall(r"\b(AND|OR|NOT)\b", query_str)
        if bool_ops:
            raw_tokens = tokenize(query_str)
            terms = preprocess(query_str, remove_stop=False, do_stem=True)
            return ParsedQuery(
                raw_query=query_str,
                terms=terms,
                raw_tokens=raw_tokens,
                operators=bool_ops,
                mode="boolean"
            )

        # 4. Standard free-text query
        raw_tokens = tokenize(query_str)
        processed_terms = preprocess(query_str, remove_stop=True, do_stem=True)
        return ParsedQuery(
            raw_query=query_str,
            terms=processed_terms,
            raw_tokens=raw_tokens,
            mode="basic"
        )
