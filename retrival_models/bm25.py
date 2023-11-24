import math
import numpy as np
from collections import OrderedDict

from .utils import extract_quote

def bm25_for_LaMP_1(task_input, profile):
    
    # extract the words from the non-template part of the sentence
    query, input_title = query_extract(task_input)
    
    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['title'] == input_title:
            profile.pop(i)
            break

    # extract the user profile
    user_profile_corpus = [item['abstract'].split()+item['title'].split() for item in profile]
    
    # compute the scores
    scores = [bm25_score(query, item, user_profile_corpus) for item in user_profile_corpus]
    retrieved_index = scores.index(max(scores))
    retrieved_profile = [profile[retrieved_index]]

    return retrieved_profile

def bm25_for_LaMP_2(task_input, profile):

    # extract the words from the non-template part of the sentence
    text = task_input.split('article:')[-1].strip()
    query = text.split()
    
    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    user_profile_corpus = [item['text'].split()+item['title'].split()+item['category'].split() for item in profile]
    # compute the scores
    scores = [bm25_score(query, item, user_profile_corpus) for item in user_profile_corpus]
    retrieved_index = scores.index(max(scores))
    retrieved_profile = [profile[retrieved_index]]

    return retrieved_profile

    

def bm25_for_LaMP_3(task_input, profile):
    
    # extract the words from the non-template part of the sentence
    text = task_input.split('just answer with 1, 2, 3, 4, or 5 without further explanation. review:')[-1].strip()
    query = text.split()

    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    user_profile_corpus = [item['text'].split()+item['score'].split() for item in profile]

    # compute the scores
    scores = [bm25_score(query, item, user_profile_corpus) for item in user_profile_corpus]
    retrieved_index = scores.index(max(scores))
    retrieved_profile = [profile[retrieved_index]]

    return retrieved_profile


def bm25_for_LaMP_4(task_input, profile):
    pass

def bm25_for_LaMP_5(task_input, profile):
    pass

def bm25_for_LaMP_6(task_input, profile):
    pass

def bm25_for_LaMP_7(task_input, profile):
    pass

BM25_MAPPING = OrderedDict(
    [
        ('LaMP_1', bm25_for_LaMP_1),
        ('LaMP_2', bm25_for_LaMP_2),
        ('LaMP_3', bm25_for_LaMP_3),
        ('LaMP_4', bm25_for_LaMP_4),
        ('LaMP_5', bm25_for_LaMP_5),
        ('LaMP_6', bm25_for_LaMP_6),
        ('LaMP_7', bm25_for_LaMP_7),
    ]
)

def build_bm25_fn(config:dict=None):

    return BM25_MAPPING[config['task_name']]


def bm25_score(query, document, corpus, k1=1.5, b=0.75):
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

def query_extract(input_text: str, task_name: str = None):
    """
    param: input_text: the original task input for query extraction
    """

    quote_sentences = extract_quote(input_text)
    query = []

    for sentence in quote_sentences:
        query.extend(sentence.split())

    return query, quote_sentences[0]

