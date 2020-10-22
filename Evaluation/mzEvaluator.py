import math
import numpy as np
import interactions
import torch_utils as my_utils
from matchzoo.metrics import normalized_discounted_cumulative_gain
from setting_keywords import KeyWordSettings
from Models.base_model import BaseModel
import torch
from matchzoo.preprocessors.tfidf_preprocessor import TFIDF
from typing import List, Dict
from handlers.output_handler import FileHandler


def evaluate(model: BaseModel, testRatings: interactions.MatchInteraction, K: int, _use_cuda, output_ranking = False):
    """
    We could extend it to add more metrics in the future
    Parameters
    ----------
    model: a fitter (not wise)
    testRatings: the
    K: top k ranked documents
    output_ranking: output the ranked docs with respect to a query for error analysis

    Returns
    -------

    """
    ndcg_metric = normalized_discounted_cumulative_gain.NormalizedDiscountedCumulativeGain

    hits, ndcgs = [], []
    list_error_analysis = []
    for query, candidates in testRatings.unique_queries_test.items():
        docs, labels, doc_contents, _ = candidates
        query_content = testRatings.dict_query_contents[query]
        query_len = [testRatings.dict_query_lengths[query]] * len(labels)
        doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]

        query_idfs = None
        if len(TFIDF.get_term_idf()) > 0:
            query_idf_dict = TFIDF.get_term_idf()
            query_idfs = [query_idf_dict.get(int(word_idx), 0.0) for word_idx in query_content]
            query_idfs = np.tile(query_idfs, (len(labels), 1))
            query_idfs = my_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), _use_cuda)

        query_content = np.tile(query_content, (len(labels), 1))  # len(labels), query_contnt_leng)
        doc_contents = np.array(doc_contents)
        query_content = my_utils.gpu(query_content)
        doc_contents = my_utils.gpu(doc_contents)

        query_content = my_utils.gpu(my_utils.numpy2tensor(query_content, dtype = torch.int), _use_cuda)
        doc_contents = my_utils.gpu(my_utils.numpy2tensor(doc_contents, dtype = torch.int), _use_cuda)

        predictions = model.predict(query_content, doc_contents, query_lens=query_len, docs_lens=doc_lens,
                                    query_idf = query_idfs)
        ndcg_mz = ndcg_metric(K)(labels, predictions)
        ndcgs.append(ndcg_mz)
        positive_docs = set([d for d, lab in zip(docs, labels) if lab == 1])
        indices = np.argsort(-predictions)[:K]  # indices of items with highest scores
        docs = np.array(docs)
        ranked_docs = docs[indices]
        if output_ranking:
            labels = np.array(labels)
            ranked_labels = labels[indices]
            scores = predictions[indices]
            assert scores.shape == ranked_labels.shape
            ranked_doc_list = [{KeyWordSettings.Doc_cID: int(d),
                                KeyWordSettings.Doc_cLabel: int(lab),
                                KeyWordSettings.Doc_wImages: [],
                                KeyWordSettings.Doc_wContent: testRatings.dict_doc_raw_contents[d],
                                KeyWordSettings.Relevant_Score: float(score)}
                               for d, lab, score in zip(ranked_docs, ranked_labels, scores)]

            q_details = {KeyWordSettings.Query_id: int(query),
                         KeyWordSettings.Query_Images: [],
                         KeyWordSettings.Ranked_Docs: ranked_doc_list,
                         KeyWordSettings.Query_Content: testRatings.dict_query_raw_contents[query]}
            list_error_analysis.append(q_details)

        hit = getHitRatioForList(ranked_docs, positive_docs)
        hits.append(hit)
        # assert abs(ndcg_mine - ndcg_mz) < 1e-10, (ndcg_mine, ndcg_mz)

    results = {}
    results["ndcg"] = np.nanmean(ndcgs)
    results["ndcg_list"] = ndcgs
    results["hits"] = np.nanmean(hits)
    results["hits_list"] = hits

    if output_ranking: return results, sorted(list_error_analysis, key = lambda x: x["qid"])
    return results


def eval_bm25(testRatings: dict, K: int, output_ranking = False):
    """
    We could extend it to add more metrics in the future
    Parameters
    ----------
    rankings: `dict` (not wise)
    K: top k ranked documents
    output_ranking: output the ranked docs with respect to a query for error analysis

    Returns
    -------

    """
    ndcg_metric = normalized_discounted_cumulative_gain.NormalizedDiscountedCumulativeGain

    hits, ndcgs = [], []
    ndcg_at_1 = []
    list_error_analysis = []
    for query, candidates in testRatings.items():
        query_content, docs, labels, doc_contents, predictions = candidates
        predictions = np.array(predictions)
        doc_contents = np.array(doc_contents)
        # query_content = my_utils.gpu(my_utils.numpy2tensor(query_content, dtype = torch.int), _use_cuda)
        # doc_contents = my_utils.gpu(my_utils.numpy2tensor(doc_contents, dtype = torch.int), _use_cuda)

        # predictions = model.predict(query_content, doc_contents, query_lens=query_len, docs_lens=doc_lens)
        ndcg_mz = ndcg_metric(K)(labels, predictions)
        ndcg_at_1.append(ndcg_metric(1)(labels, predictions))
        ndcgs.append(ndcg_mz)
        positive_docs = set([d for d, lab in zip(docs, labels) if lab == 1])
        indices = np.argsort(-predictions)[:K]  # indices of items with highest scores
        docs = np.array(docs)
        ranked_docs = docs[indices]
        ranked_docs_contents = doc_contents[indices]
        if output_ranking:
            labels = np.array(labels)
            ranked_labels = labels[indices]
            scores = predictions[indices]
            assert scores.shape == ranked_labels.shape
            ranked_doc_list = [{KeyWordSettings.Doc_cID: int(d),
                                KeyWordSettings.Doc_cLabel: int(lab),
                                KeyWordSettings.Doc_wImages: [],
                                KeyWordSettings.Doc_wContent: doc_content,
                                KeyWordSettings.Relevant_Score: float(score)}
                               for d, lab, score, doc_content in zip(ranked_docs, ranked_labels, scores, ranked_docs_contents)]

            q_details = {KeyWordSettings.Query_id: int(query),
                         KeyWordSettings.Query_Images: [],
                         KeyWordSettings.Ranked_Docs: ranked_doc_list,
                         KeyWordSettings.Query_Content: query_content}
            list_error_analysis.append(q_details)

        hit = getHitRatioForList(ranked_docs, positive_docs)
        hits.append(hit)

    results = {}
    results["ndcg"] = np.nanmean(ndcgs)
    results["ndcg_list"] = ndcgs
    results["hits"] = np.nanmean(hits)
    results["hits_list"] = hits
    results["ndcg@1"] = np.nanmean(ndcg_at_1)

    if output_ranking: return results, sorted(list_error_analysis, key = lambda x: x["qid"])
    return results


def getHitRatioForList(ranklist, gtItems):
    for item in ranklist:
        if item in gtItems:
            return 1.0
    return 0.0

