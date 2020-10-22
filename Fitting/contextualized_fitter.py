import torch
import numpy as np
import torch_utils
from Models import base_model
import torch_utils as my_utils
import time
import os
from handlers import mz_sampler
from Evaluation import mzEvaluator as my_evaluator
import json
import matchzoo
import interactions
from handlers.output_handler import FileHandler
from handlers.tensorboard_writer import TensorboardWrapper
from matchzoo.preprocessors.tfidf_preprocessor import TFIDF
from setting_keywords import KeyWordSettings
from matchzoo.metrics import normalized_discounted_cumulative_gain
from Fitting.densebaseline_fit import DenseBaselineFitter
from tqdm import tqdm


class ContextualizedFitter(DenseBaselineFitter):
    def _get_multiple_negative_predictions_normal(self, query_ids: torch.Tensor,
                                                  query_contents: torch.Tensor,
                                                  doc_ids: torch.Tensor,
                                                  doc_contents: torch.Tensor,
                                                  negative_doc_ids: torch.Tensor,
                                                  negative_doc_contents: torch.Tensor,
                                                  query_lens: np.ndarray,
                                                  docs_lens: np.ndarray,
                                                  neg_docs_lens: np.ndarray, n: int, **kargs) -> torch.Tensor:
        """
        We compute prediction for every pair of (user, neg_item). Since shape of user_ids is (batch_size, )
        and neg_item_ids.shape = (batch_size, n), we need to reshape user_ids a little bit.
        Parameters
        ----------
        query_ids: shape (B, )
        query_contents: (B, L1) where L1 is the number of words of each query
        doc_ids: shape (B, )
        doc_contents: (B, L2) where L2 is the number of words of each doc
        negative_doc_ids (B, n)
        negative_doc_contents: shape (B, n, L2)
        query_lens: (B, )
        docs_lens: (B, )
        neg_docs_lens: (B, n)
        n: number of negs

        Note: Query and Doc have different lengths

        Returns
        -------

        """
        batch_size = query_ids.size(0)
        L2 = doc_contents.size(1)
        L1 = query_contents.size(1)
        assert negative_doc_contents.size() == (batch_size, n, L2)

        # duplicate content of positive query
        query_contents_tmp = query_contents.view(batch_size * L1, 1).expand(batch_size * L1, n).reshape(batch_size, L1, n)
        query_contents_tmp = query_contents_tmp.permute(0, 2, 1).reshape(batch_size * n, L1)  # (B, n, L1)
        query_ids_tmp = query_ids.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)

        assert query_contents_tmp.size() == (batch_size * n, L1)  # (B * n, L1)
        batch_negatives_tmp = negative_doc_contents.reshape(batch_size * n, L2)  # (B * n, L2)
        # get sorted indices for negative docs
        neg_docs_lens = neg_docs_lens.reshape(batch_size * n)
        positive_additional = {
            KeyWordSettings.QueryIDs: query_ids,
            KeyWordSettings.DocIDs: doc_ids,
            KeyWordSettings.UseCuda: self._use_cuda
        }
        positive_prediction = self._net(query_contents, doc_contents, **positive_additional)  # (batch_size)

        negative_additional = {
            KeyWordSettings.QueryIDs: query_ids_tmp,
            KeyWordSettings.DocIDs: negative_doc_ids.reshape(batch_size * n),  # (B * n, )
            KeyWordSettings.UseCuda: self._use_cuda
        }
        negative_prediction = self._net(query_contents_tmp, batch_negatives_tmp, **negative_additional)  # (B * n)

        if self._loss == "bpr" or self._loss == "hinge":
            positive_prediction = positive_prediction.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)
            assert positive_prediction.shape == negative_prediction.shape
            loss = self._loss_func(positive_prediction, negative_prediction)
        elif self._loss == "pce":
            negative_prediction = negative_prediction.view(batch_size, n)
            loss = self._loss_func(positive_prediction, negative_prediction)
        elif self._loss == "bce":
            # (B, ) vs. (B * n) shape.
            loss = self._loss_func(positive_prediction, negative_prediction)
        return loss

    def evaluate(self, testRatings: interactions.MatchInteraction, K: int, output_ranking = False, **kargs):
        """
        I decided to move this function into Fitter class since different models have different ways to evaluate (i.e.
        different data sources to use). Therefore, it is needed to have seperate evaluation methods in each Fitter
        class. Furthermore, I notice that this function uses _use_cuda which is a property of Fitter class.
        Parameters
        ----------
        testRatings
        K
        output_ranking
        kargs

        Returns
        -------

        """
        ndcg_metric = normalized_discounted_cumulative_gain.NormalizedDiscountedCumulativeGain

        hits, ndcgs = [], []
        ndcgs_at_1 = []
        list_error_analysis = []
        for query, candidates in tqdm(testRatings.unique_queries_test.items()):
            docs, labels, doc_contents, _ = candidates
            query_content = testRatings.dict_query_contents[query]
            query_len = [testRatings.dict_query_lengths[query]] * len(labels)
            doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]
            t1 = time.time()
            additional_data = {}
            if len(TFIDF.get_term_idf()) > 0:
                query_idf_dict = TFIDF.get_term_idf()
                query_idfs = [query_idf_dict.get(int(word_idx), 0.0) for word_idx in query_content]
                query_idfs = np.tile(query_idfs, (len(labels), 1))
                query_idfs = my_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), self._use_cuda)
                additional_data[KeyWordSettings.Query_Idf] = query_idfs

            query_content = np.tile(query_content, (len(labels), 1))  # len(labels), query_contnt_leng)
            doc_contents = np.array(doc_contents)
            query_content = my_utils.gpu(query_content)
            doc_contents = my_utils.gpu(doc_contents)

            query_content = my_utils.gpu(my_utils.numpy2tensor(query_content, dtype=torch.int), self._use_cuda)
            doc_contents = my_utils.gpu(my_utils.numpy2tensor(doc_contents, dtype=torch.int), self._use_cuda)
            additional_data[KeyWordSettings.Query_lens] = query_len
            additional_data[KeyWordSettings.Doc_lens] = doc_lens
            additional_data[KeyWordSettings.QueryIDs] = np.array([query] * len(labels))
            additional_data[KeyWordSettings.DocIDs] = np.array(docs)
            additional_data[KeyWordSettings.UseCuda] = self._use_cuda
            # additional_data[KeyWordSettings.QueryCharIndex] = query_chars
            # additional_data[KeyWordSettings.DocCharIndex] = doc_chars

            predictions = self._net.predict(query_content, doc_contents, **additional_data)
            ndcg_mz = ndcg_metric(K)(labels, predictions)
            ndcgs_at_1.append(ndcg_metric(1)(labels, predictions))
            ndcgs.append(ndcg_mz)
            positive_docs = set([d for d, lab in zip(docs, labels) if lab == 1])
            indices = np.argsort(-predictions)[:K]  # indices of items with highest scores
            docs = np.array(docs)
            ranked_docs = docs[indices]
            t2 = time.time()
            # print("Running time testing each query: ", (t2 - t1), "second")
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

            hit = my_evaluator.getHitRatioForList(ranked_docs, positive_docs)
            # ndcg_mine = getNDCGForList(ranklist, positive_docs)
            hits.append(hit)
            # assert abs(ndcg_mine - ndcg_mz) < 1e-10, (ndcg_mine, ndcg_mz)

        results = {}
        results["ndcg"] = np.nanmean(ndcgs)
        results["ndcg_list"] = ndcgs
        results["hits"] = np.nanmean(hits)
        results["hits_list"] = hits
        results["ndcg@1"] = np.nanmean(ndcgs_at_1)
        results["ndcg@1_list"] = ndcgs_at_1

        if output_ranking: return results, sorted(list_error_analysis, key=lambda x: x["qid"])
        return results
