import math
import numpy as np
from collections import Counter

class BM25():
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def bm25_score(self, query, document, corpus, k1=1.5, b=0.75):
        """
        Compute BM25 score for a document with respect to a query.

        :param query: List of terms in the query.
        :param document: List of terms in the document.
        :param corpus: List of all documents in the corpus.
        :param k1: Positive tuning parameter, default is 1.5.
        :param b: Between 0 and 1, controls the impact of document length, default is 0.75.
        :return: BM25 score for the document with respect to the query.
        """

        N = len(corpus)
        avgdl = sum(map(len, corpus)) / N
        doc_length = len(document)

        score = 0
        for term in query:
            f_i = document.count(term)
            q_i = sum(1 for doc in corpus if term in doc)
            idf = math.log((N - q_i + 0.5) / (q_i + 0.5) + 1.0)  # Added 1.0 to avoid division by zero
            term_score = (f_i * (k1 + 1)) / (f_i + k1 * (1 - b + b * (doc_length / avgdl)))
            score += idf * term_score

        return score
