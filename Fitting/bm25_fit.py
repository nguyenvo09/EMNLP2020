import os, sys
from Evaluation import mzEvaluator as my_evaluator
import json
from handlers.output_handler import FileHandler
from Models.BM25.src.query import QueryProcessor
from Models.BM25.src.parse import *
import operator
from setting_keywords import KeyWordSettings


class BM25Fitter:

    def __init__(self, params: dict):
        self.params = params

    def _fit(self, interactions: dict):
        rankings = {}
        # cnt = 0
        for query in interactions:
            content, labels, docIDs, docContents = interactions[query]
            # content = content[0]  # it should be an array
            qp = QueryParser2(content)
            cp = CorpusParser2(docContents, docIDs)
            qp.parse()
            queries = qp.get_queries()
            cp.parse()
            corpus = cp.get_corpus()

            proc = QueryProcessor(queries, corpus, params = self.params)
            results = proc.run()
            qid = query
            assert len(content) == 1, "Only one query please!!!!"
            assert len(results) == 1, "We only return queries for one query"
            result = results[-1]
            # for result in results:
            sorted_x = sorted(result.items(), key = operator.itemgetter(1))
            sorted_x.reverse()
            index = 0
            dict_scores = {}
            for i in sorted_x[:100]:
                tmp = (qid, i[0], index, i[1])
                docID, score = i[0], i[1]
                dict_scores[docID] = score
                # FileHandler.myprint('{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp))
                index += 1
                # qid += 1
            scores = []
            for doc in docIDs:
                r = dict_scores.get(str(doc), -sys.maxsize)
                scores.append(r)
            rankings[query] = (content[0], docIDs, labels, docContents, scores)
            # cnt += 1
            # if cnt == 3: break
        return rankings

    def fit(self, train_iteractions: dict, verbose = True,  # for printing out evaluation during training
            topN = 10, val_queries: dict = None, test_queries: dict = None, **kargs):
        """
        Fit the model.
        Parameters
        ----------
        train_iteractions: :class:`dict` None, since bm25 does not need to training.
        val_interactions: :class:`dict`
        test_interactions: :class:`dict`
        """
        val_scores = self._fit(val_queries)
        test_scores = self._fit(test_queries)

        if KeyWordSettings.Test2Hard in kargs and KeyWordSettings.Test3Hard in kargs:
            test2_scores = self._fit(kargs[KeyWordSettings.Test2Hard])
            test3_scores = self._fit(kargs[KeyWordSettings.Test3Hard])
            self._rank2(test2_scores, test3_scores, topN)

        self._rank(val_scores, test_scores, topN)

    def _rank(self, val_scores: dict, test_scores: dict, topN: int, **kargs):
        assert len(val_scores) in KeyWordSettings.QueryCountVal, len(val_scores)
        result_val, error_analysis_val = my_evaluator.eval_bm25(val_scores, topN, output_ranking = True)
        hits = result_val["hits"]
        ndcg = result_val["ndcg"]
        ndcg_at_1_val = result_val["ndcg@1"]

        assert len(test_scores) in KeyWordSettings.QueryCountTest, len(test_scores)
        result_test, error_analysis_test = my_evaluator.eval_bm25(test_scores, topN, output_ranking = True)
        hits_test = result_test["hits"]
        ndcg_test = result_test["ndcg"]
        ndcg_at_1_test = result_test["ndcg@1"]

        FileHandler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys = True, indent = 2))
        FileHandler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys = True, indent = 2))
        FileHandler.myprint('Best Vad hits@%d = %.5f | Best Vad ndcg@%d = %.5f '
                            '|Best Test hits@%d = %.5f |Best Test ndcg@%d = %.5f'
                            '|Best Test ndcg@1 = %.5f '
                            % (topN, hits, topN, ndcg, topN, hits_test, topN, ndcg_test, ndcg_at_1_test))
        return hits, ndcg, hits_test, ndcg_test

    def _rank2(self, test2: dict, test3: dict, topN: int, **kargs):
        assert len(test2) in KeyWordSettings.QueryCountTest, len(test2)
        result_test2, error_analysis_test2 = my_evaluator.eval_bm25(test2, topN, output_ranking = True)
        hits = result_test2["hits"]
        ndcg = result_test2["ndcg"]
        ndcg_at_1_val = result_test2["ndcg@1"]

        assert len(test3) in KeyWordSettings.QueryCountTest, len(test3)
        result_test3, error_analysis_test3 = my_evaluator.eval_bm25(test3, topN, output_ranking = True)
        hits_test = result_test3["hits"]
        ndcg_test = result_test3["ndcg"]
        ndcg_at_1_test = result_test3["ndcg@1"]

        FileHandler.save_error_analysis_test2(json.dumps(error_analysis_test2, sort_keys = True, indent = 2))
        FileHandler.save_error_analysis_test3(json.dumps(error_analysis_test3, sort_keys = True, indent = 2))
        FileHandler.myprint('Best Test2 hits@%d = %.5f | Best Test2 ndcg@%d = %.5f | Best Test2 ndcg@1 = %.5f '
                            '|Best Test3 hits@%d = %.5f |Best Test3 ndcg@%d = %.5f |Best Test3 ndcg@1 = %.5f '
                            % (topN, hits, topN, ndcg, ndcg_at_1_val,
                               topN, hits_test, topN, ndcg_test, ndcg_at_1_test))
        return hits, ndcg, hits_test, ndcg_test