"""
Phase 7: High-Performance BM25 Lexical Baseline for Prior Case Retrieval.
Supports text tokenization, legal stopword handling, inverted index construction,
and fast top-K retrieval.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Set


class LegalBM25:
    """
    BM25Okapi implementation tailored for legal document retrieval.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths = {}
        self.doc_ids = []
        self.inverted_index = {}  # term -> {doc_id: tf}
        self.idf = {}

    @staticmethod
    def tokenize(text_obj) -> List[str]:
        if isinstance(text_obj, list):
            text = " ".join(text_obj)
        else:
            text = str(text_obj)
        # Extract lowercase alpha-numeric tokens of length >= 2
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
        return tokens

    def fit(self, candidate_dict: Dict[str, Dict]):
        """
        Builds inverted index and computes IDF scores over the candidate pool.
        candidate_dict: {cand_id: {"text": [...]}}
        """
        self.corpus_size = len(candidate_dict)
        self.doc_ids = list(candidate_dict.keys())
        total_len = 0
        df = Counter()
        self.inverted_index = {}
        self.doc_lengths = {}

        for doc_id, doc_data in candidate_dict.items():
            tokens = self.tokenize(doc_data.get("text", []))
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_len += doc_len

            tf = Counter(tokens)
            for term, count in tf.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count
                df[term] += 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)

        # Compute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF: log((N - n + 0.5) / (n + 0.5) + 1)
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def query(self, query_text_obj, top_k: int = 7) -> List[Tuple[str, float]]:
        """
        Scores all candidates matching query tokens and returns top_k (doc_id, score).
        """
        q_tokens = self.tokenize(query_text_obj)
        q_tf = Counter(q_tokens)
        scores = Counter()

        for term, q_count in q_tf.items():
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            postings = self.inverted_index[term]

            for doc_id, tf_val in postings.items():
                doc_len = self.doc_lengths[doc_id]
                # BM25 term weighting
                numerator = tf_val * (self.k1 + 1.0)
                denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf_val * (numerator / denominator)

        if not scores:
            # Fallback to first top_k candidate IDs if no match
            return [(d, 0.0) for d in self.doc_ids[:top_k]]

        # If fewer than top_k matched, fill with arbitrary unseen candidates
        ranked = scores.most_common(top_k)
        if len(ranked) < top_k:
            ranked_set = {d for d, _ in ranked}
            for d in self.doc_ids:
                if d not in ranked_set:
                    ranked.append((d, 0.0))
                    if len(ranked) == top_k:
                        break

        return ranked[:top_k]
