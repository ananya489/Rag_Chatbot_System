import re
import math
from collections import Counter
from dataclasses import dataclass, field


# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Common English stop words — removed before BM25 scoring because they
# appear in almost every document and carry no discriminating power.
# Keeping stop words would make "the return policy" score highly for
# chunks containing only "the" — the wrong behaviour.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "that", "this", "these", "those", "it", "its", "as", "if",
    "not", "no", "nor", "so", "yet", "both", "either", "each",
    "than", "then", "when", "where", "which", "who", "how", "what",
})


def tokenize(text: str) -> list[str]:
    """
    Convert a text string into a list of normalized tokens for BM25.

    Steps:
        1. Lowercase — "Python" and "python" are the same term
        2. Extract only alphanumeric tokens — strip punctuation, symbols
        3. Remove stop words — filter out low-information terms
        4. Require minimum length of 2 — removes single-char noise

    Args:
        text: Any raw string (query, chunk, document).

    Returns:
        List of lowercase, filtered tokens. May be empty for
        stop-word-only inputs — callers should handle this.

    Examples:
        "What is the return policy?" →
            ["what", "return", "policy"]

        "Article 21B subsection 4 defines liability" →
            ["article", "21b", "subsection", "4", "defines", "liability"]
    """
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]


# ── BM25 Index ────────────────────────────────────────────────────────────────

@dataclass
class BM25Index:
    """
    In-memory BM25 index over a corpus of text chunks.

    BM25 (Best Match 25) scores each document for a query using:
        - Term Frequency (TF): how often query terms appear in the document
        - Inverse Document Frequency (IDF): how rare the term is overall
        - Document length normalization: longer docs aren't unfairly rewarded

    Formula for each term t in query, document d:
        score(d, t) = IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1×(1-b+b×|d|/avgdl))

    Parameters:
        k1 = 1.5  — controls TF saturation (1.2–2.0 typical)
        b  = 0.75 — controls length normalization (0=none, 1=full)

    These are the standard BM25+ default values from the literature.
    """

    # BM25 hyperparameters
    k1: float = 1.5
    b:  float = 0.75

    # Internal state — populated by build()
    corpus_tokens: list[list[str]]        = field(default_factory=list)
    chunk_ids:     list[str]              = field(default_factory=list)
    chunk_texts:   list[str]              = field(default_factory=list)
    chunk_sources: list[str]             = field(default_factory=list)
    chunk_pages:   list[int | None]      = field(default_factory=list)
    doc_freqs:     dict[str, int]        = field(default_factory=dict)
    idf:           dict[str, float]      = field(default_factory=dict)
    doc_lengths:   list[int]             = field(default_factory=list)
    avgdl:         float                 = 0.0
    n_docs:        int                   = 0

    def build(self, chunks: list[dict]) -> "BM25Index":
        """
        Build the BM25 index from a list of chunk dicts.

        Must be called before search(). Rebuilding is safe — calling
        build() again replaces the existing index.

        Args:
            chunks: List of chunk dicts. Each must have:
                    - text (str)
                    - chunk_id (str)
                    - source (str)
                    Optional: page (int)

        Returns:
            self — enables chaining: index = BM25Index().build(chunks)
        """
        if not chunks:
            return self

        # Tokenize every chunk and record its metadata
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.chunk_ids     = [c["chunk_id"]        for c in chunks]
        self.chunk_texts   = [c["text"]            for c in chunks]
        self.chunk_sources = [c["source"]          for c in chunks]
        self.chunk_pages   = [c.get("page")        for c in chunks]
        self.n_docs        = len(chunks)

        # Document lengths and average document length
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl       = (sum(self.doc_lengths) / self.n_docs
                            if self.n_docs > 0 else 1.0)

        # Document frequency: how many docs contain each term
        # Used to compute IDF — rare terms get higher scores
        self.doc_freqs = {}
        for tokens in self.corpus_tokens:
            for term in set(tokens):   # set() ensures each term counted once per doc
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        # The +1 inside the log keeps IDF positive even for very common terms.
        # Terms appearing in every document still get a small positive IDF.
        self.idf = {
            term: math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )
            for term, df in self.doc_freqs.items()
        }

        print(f"[bm25] Index built: {self.n_docs} chunks, "
              f"{len(self.idf)} unique terms")
        return self

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Score all chunks for the query and return the top_k matches.

        Args:
            query: User's question or expanded query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts sorted by BM25 score descending:
                [{"chunk_id": ..., "text": ..., "source": ...,
                  "bm25_score": float, "page": int | None}, ...]

            Returns empty list if index is empty or query has no
            tokens after stop-word filtering.
        """
        if self.n_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            # All query terms were stop words — fall back gracefully
            print("[bm25] Query produced no tokens after filtering.")
            return []

        scores: list[float] = [0.0] * self.n_docs

        for term in query_tokens:
            if term not in self.idf:
                # Term never appears in any chunk — skip it
                continue

            term_idf = self.idf[term]

            for doc_idx, tokens in enumerate(self.corpus_tokens):
                # Term frequency in this document
                tf = tokens.count(term)
                if tf == 0:
                    continue

                # BM25 TF normalization
                dl      = self.doc_lengths[doc_idx]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc_idx] += term_idf * tf_norm

        # Rank and return top_k non-zero results
        ranked = sorted(
            [
                {
                    "chunk_id":   self.chunk_ids[i],
                    "text":       self.chunk_texts[i],
                    "source":     self.chunk_sources[i],
                    "bm25_score": scores[i],
                    "page":       self.chunk_pages[i],
                }
                for i in range(self.n_docs)
                if scores[i] > 0
            ],
            key=lambda x: x["bm25_score"],
            reverse=True,
        )
        return ranked[:top_k]

    def is_empty(self) -> bool:
        """Return True if the index has not been built or has no documents."""
        return self.n_docs == 0