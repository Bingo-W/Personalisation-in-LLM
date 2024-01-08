import math
import numpy as np
from collections import OrderedDict, Counter

from .utils import extract_quote

def bm25_for_LaMP_1(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):
    
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
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]

    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile

def bm25_for_LaMP_2(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):

    # extract the words from the non-template part of the sentence
    text = task_input.split('article:')[-1].strip()
    query = text.split()
    
    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    if retrieval_ablation == 'decouple':
        user_profile_corpus = [item['text'].split()+item['title'].split() for item in profile]
    else:
        user_profile_corpus = [item['text'].split()+item['title'].split()+item['category'].split() for item in profile]
    # compute the scores
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]

    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile

    

def bm25_for_LaMP_3(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):
    
    # extract the words from the non-template part of the sentence
    text = task_input.split('just answer with 1, 2, 3, 4, or 5 without further explanation. review:')[-1].strip()
    query = text.split()

    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    if retrieval_ablation == 'decouple':
        user_profile_corpus = [item['text'].split() for item in profile]
    else:
        user_profile_corpus = [item['text'].split()+item['score'].split() for item in profile]

    # compute the scores
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]
    
    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile


def bm25_for_LaMP_4(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):
    text = task_input.split('for the following article: ')[-1].strip()
    query = text.split()

    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    if retrieval_ablation == 'decouple':
        user_profile_corpus = [item['text'].split() for item in profile]
    else:
        user_profile_corpus = [item['text'].split()+item['title'].split() for item in profile]

    # compute the scores
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]

    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile


def bm25_for_LaMP_5(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):
    text = task_input.split('the following abstract of a paper: ')[-1].strip()
    query = text.split()

    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['abstract'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    if retrieval_ablation == 'decouple':
        user_profile_corpus = [item['abstract'].split() for item in profile]
    else:
        user_profile_corpus = [item['abstract'].split()+item['title'].split() for item in profile]

    # compute the scores
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]

    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile

def bm25_for_LaMP_6(task_input, profile, retrieve_num = 1):
    pass

def bm25_for_LaMP_7(task_input, profile, retrieve_num = 1, retrieval_ablation='both', target_index=0):
    text = task_input.split('the following tweet without any explanation before or after it: ')[-1].strip()
    query = text.split()

    # exclude the profile related to the input text
    for i, userprofile in enumerate(profile):
        if userprofile['text'] == text:
            profile.pop(i)
            break
    
    # extract the user profile
    user_profile_corpus = [item['text'].split() for item in profile]

    # compute the scores
    doc_freq = Counter()
    for doc in user_profile_corpus:
        doc_set = set(doc)
        doc_freq.update(doc_set)

    total_docs = len(user_profile_corpus)
    len_doc_corpus_mean = np.mean(np.array([len(doc) for doc in user_profile_corpus]))
    scores = [bm25_score_np(query, doc, total_docs, doc_freq, len_doc_corpus_mean) for doc in user_profile_corpus]

    sorted_score = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    retrieve_num = int(retrieve_num*len(profile)) if retrieve_num < 1 and retrieve_num > 0 else retrieve_num
    if target_index == 0:
        retrieved_index = [index for index, _ in sorted_score[:int(retrieve_num)]]
    else:
        target_index = target_index if target_index < len(profile) else len(profile)
        retrieve_num = retrieve_num if target_index+retrieve_num < len(profile) else 0
        retrieved_index = [index for index, _ in sorted_score[int(target_index):int(target_index+retrieve_num)]]
    
    # construct the list
    retrieved_profile = []
    for index in retrieved_index:
        retrieved_profile.append(profile[index])

    return retrieved_profile

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

def bm25_score_np(query, document, total_docs, doc_freq, len_doc_corpus_mean, k1=1.5, b=0.75):
    def idf(total_docs, doc_freq, term):
        idf = np.log((total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1.0)
        return idf
    
    query_terms, query_counts = np.unique(query, return_counts=True)
    document_terms, document_counts = np.unique(document, return_counts=True)
    

    common_terms = np.intersect1d(query_terms, document_terms)
    idf_terms = np.array([idf(total_docs, doc_freq, term) for term in common_terms])

    tf_query = query_counts[query_terms.searchsorted(common_terms)]
    tf_document = document_counts[document_terms.searchsorted(common_terms)]

    numerator = tf_document * (k1 + 1) *tf_query
    denominator = tf_document + k1 * (1 - b + b * len(document) / len_doc_corpus_mean)

    scores = idf_terms * numerator / denominator
    return np.sum(scores)

def query_extract(input_text: str, task_name: str = None):
    """
    param: input_text: the original task input for query extraction
    """

    quote_sentences = extract_quote(input_text)
    query = []

    for sentence in quote_sentences:
        query.extend(sentence.split())

    return query, quote_sentences[0]

