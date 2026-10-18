"""
Boolean Search Engine: Evaluates queries with AND, OR, NOT operators across inverted indexes.
Uses recursive descent parsing with operator precedence: NOT > AND > OR, and full parentheses support.
"""

from typing import Set, List, Dict, Any, Optional
from engine.indexing import IndexStore
from engine.preprocessing import stem_single, tokenize


def parse_boolean_query_tokens(raw_query: str) -> List[str]:
    """
    Tokenize query into terms, operators (AND, OR, NOT), and parentheses.
    """
    cleaned = raw_query.replace("(", " ( ").replace(")", " ) ")
    tokens = cleaned.split()
    normalized = []
    for tok in tokens:
        upper_tok = tok.upper()
        if upper_tok in {"AND", "OR", "NOT", "(", ")"}:
            normalized.append(upper_tok)
        else:
            words = tokenize(tok)
            for w in words:
                normalized.append(stem_single(w))
    return normalized


class BooleanQueryParser:
    """
    Recursive descent parser for Boolean expressions:
    expr     := or_expr
    or_expr  := and_expr ('OR' and_expr)*
    and_expr := not_expr ('AND' not_expr)*
    not_expr := 'NOT' not_expr | primary
    primary  := '(' expr ')' | TERM
    """

    def __init__(self, tokens: List[str], store: IndexStore, field: Optional[str] = None):
        self.raw_tokens = tokens
        self.store = store
        self.field = field
        self.all_doc_ids: Set[int] = {doc["id"] for doc in store.get_all_docs()}
        self.tokens: List[str] = self._normalize_operators(tokens)
        self.pos = 0

    def _normalize_operators(self, tokens: List[str]) -> List[str]:
        """
        Insert implicit operators:
        - If 'NOT' follows an operand or ')', insert 'AND' before 'NOT'
        - If an operand follows an operand or ')', insert 'OR'
        """
        normalized: List[str] = []
        for i, tok in enumerate(tokens):
            if i > 0:
                prev = normalized[-1]
                if prev not in {"AND", "OR", "NOT", "("}:
                    if tok == "NOT":
                        normalized.append("AND")
                    elif tok not in {"AND", "OR", ")"}:
                        normalized.append("OR")
            normalized.append(tok)
        return normalized

    def current_token(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of query")
        if expected and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Set[int]:
        if not self.tokens:
            return set()
        res = self.expr()
        return res

    def expr(self) -> Set[int]:
        return self.or_expr()

    def or_expr(self) -> Set[int]:
        res = self.and_expr()
        while self.current_token() == "OR":
            self.consume("OR")
            right = self.and_expr()
            res = res | right
        return res

    def and_expr(self) -> Set[int]:
        res = self.not_expr()
        while self.current_token() == "AND":
            self.consume("AND")
            right = self.not_expr()
            res = res & right
        return res

    def not_expr(self) -> Set[int]:
        if self.current_token() == "NOT":
            self.consume("NOT")
            sub = self.not_expr()
            return self.all_doc_ids - sub
        return self.primary()

    def primary(self) -> Set[int]:
        tok = self.current_token()
        if tok == "(":
            self.consume("(")
            res = self.expr()
            if self.current_token() == ")":
                self.consume(")")
            return res
        elif tok and tok not in {"AND", "OR", "NOT", ")"}:
            term = self.consume()
            postings = self.store.get_posting_list(term, field=self.field)
            return set(postings.keys())
        else:
            return set()


def boolean_search(query: str, store: IndexStore, field: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Execute a Boolean query on the IndexStore.
    Supports AND, OR, NOT operators and grouping parentheses.
    """
    if not query.strip():
        return []

    tokens = parse_boolean_query_tokens(query)
    if not tokens:
        return []

    try:
        parser = BooleanQueryParser(tokens, store, field=field)
        matched_ids = parser.parse()
    except Exception:
        # Fallback to simple union if malformed syntax
        matched_ids = set()
        for tok in tokens:
            if tok not in {"AND", "OR", "NOT", "(", ")"}:
                matched_ids.update(store.get_posting_list(tok, field=field).keys())

    results = []
    for doc_id in sorted(list(matched_ids)):
        doc = store.get_doc(doc_id)
        if doc:
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title", f"Doc {doc_id}"),
                "score": 1.0,
                "doc": doc
            })

    return results
