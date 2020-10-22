import torch
import numpy as np
import torch_utils
from Models import base_model
import losses as my_losses
import torch_utils as my_utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from handlers import output_handler, mz_sampler
from Evaluation import mzEvaluator as my_evaluator
import datetime
import json
import matchzoo
import interactions
from handlers.output_handler import FileHandler
from handlers.tensorboard_writer import TensorboardWrapper
from matchzoo.preprocessors.tfidf_preprocessor import TFIDF
from setting_keywords import KeyWordSettings
from matchzoo.metrics import average_precision, discounted_cumulative_gain, \
    mean_average_precision, mean_reciprocal_rank, normalized_discounted_cumulative_gain, precision


class DenseBaselineFitter:

    def __init__(self, net: base_model.BaseModel,
                 loss = "bpr",
                 n_iter = 100,
                 testing_epochs = 5,
                 batch_size = 16,
                 reg_l2 = 1e-3,
                 learning_rate = 1e-4,
                 early_stopping = 0,  # means no early stopping
                 decay_step = None,
                 decay_weight = None,
                 optimizer_func = None,
                 use_cuda = False,
                 num_negative_samples = 4,
                 logfolder = None,
                 curr_date = None,
                 **kargs):

        """I put this fit function here for temporarily """
        assert loss in KeyWordSettings.LOSS_FUNCTIONS
        self._loss = loss
        # self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._testing_epochs = testing_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._reg_l2 = reg_l2
        # self._decay_step = decay_step
        # self._decay_weight = decay_weight

        self._optimizer_func = optimizer_func

        self._use_cuda = use_cuda
        self._num_negative_samples = num_negative_samples
        self._early_stopping_patience = early_stopping # for early stopping

        self._n_users, self._n_items = None, None
        self._net = net
        self._optimizer = None
        # self._lr_decay = None
        self._loss_func = None
        assert logfolder != ""
        self.logfolder = logfolder
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        self.saved_model = os.path.join(logfolder, "%s_saved_model" % int(curr_date))
        TensorboardWrapper.init_log_files(os.path.join(logfolder, "tensorboard_%s" % int(curr_date)))
        # for evaluation during training
        self._sampler = mz_sampler.Sampler()
        self._candidate = dict()

    def __repr__(self):
        """ Return a string of the model when you want to print"""
        # todo
        return "Vanilla matching Model"

    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions: interactions.MatchInteraction):
        """

        Parameters
        ----------
        interactions: :class:`interactions.MatchInteraction`
        Returns
        -------

        """
        # put the model into cuda if use cuda
        self._net = my_utils.gpu(self._net, self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay = self._reg_l2,
                lr = self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        # losses functions
        if self._loss == 'pointwise':
            self._loss_func = my_losses.pointwise_loss
        elif self._loss == "single_pointwise_square_loss":
            self._loss_func = my_losses.single_pointwise_square_loss
        elif self._loss == 'bpr':
            self._loss_func = my_losses.bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = my_losses.hinge_loss
        elif self._loss == 'bce':  # binary cross entropy
            self._loss_func = my_losses.pointwise_bceloss
        elif self._loss == "pce":
            self._loss_func = my_losses.positive_cross_entropy
        elif self._loss == "cosine_max_margin_loss_dvsh":
            self._loss_func = my_losses.cosine_max_margin_loss_dvsh
        elif self._loss == "cross_entropy":
            self._loss_func = my_losses.binary_cross_entropy_cls
        elif self._loss == "masked_cross_entropy":
            self._loss_func = my_losses.masked_binary_cross_entropy
        elif self._loss == "vanilla_cross_entropy":
            self._loss_func = my_losses.vanilla_cross_entropy
        elif self._loss == "regression_loss":
            self._loss_func = my_losses.regression_loss
        else:
            self._loss_func = my_losses.adaptive_hinge_loss
        FileHandler.myprint("Using: " + str(self._loss_func))

    def fit(self, train_iteractions: interactions.MatchInteraction,
            verbose = True,  # for printing out evaluation during training
            topN = 10,
            val_interactions: interactions.MatchInteraction = None,
            test_interactions: interactions.MatchInteraction = None):
        """
        Fit the model.
        Parameters
        ----------
        train_iteractions: :class:`matchzoo.DataPack` The input sequence dataset.
        val_interactions: :class:`matchzoo.DataPack`
        test_interactions: :class:`matchzoo.DataPack`
        """
        self._initialize(train_iteractions)
        best_hit, best_ndcg, best_epoch, test_ndcg, test_hit = 0, 0, 0, 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        for epoch_num in range(self._n_iter):

            # ------ Move to here ----------------------------------- #
            self._net.train(True)
            query_ids, left_contents, left_lengths, \
            doc_ids, right_contents, right_lengths, \
            neg_docs_ids, neg_docs_contents, neg_docs_lens = self._sampler.get_train_instances(train_iteractions, self._num_negative_samples)

            queries, query_content, query_lengths, \
            docs, doc_content, doc_lengths, \
            neg_docs, neg_docs_contents, neg_docs_lens = my_utils.shuffle(query_ids, left_contents, left_lengths,
                                                                doc_ids, right_contents, right_lengths,
                                                                neg_docs_ids, neg_docs_contents, neg_docs_lens)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                (batch_query, batch_query_content, batch_query_len,
                 batch_doc, batch_doc_content, batch_docs_lens,
                 batch_neg_docs, batch_neg_doc_content, batch_neg_docs_lens)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths,
                                                    docs, doc_content, doc_lengths,
                                                    neg_docs, neg_docs_contents, neg_docs_lens,
                                                    batch_size = self._batch_size)):
                # add idf here...
                query_idfs = None
                if len(TFIDF.get_term_idf()) != 0:
                    query_idf_dict = TFIDF.get_term_idf()
                    query_idfs = [[query_idf_dict.get(int(word_idx), 0.0) for word_idx in row] for row in batch_query_content]
                    query_idfs = torch_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), self._use_cuda)

                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                batch_doc = my_utils.gpu(torch.from_numpy(batch_doc), self._use_cuda)
                batch_doc_content = my_utils.gpu(torch.from_numpy(batch_doc_content), self._use_cuda)
                batch_neg_doc_content = my_utils.gpu(torch.from_numpy(batch_neg_doc_content), self._use_cuda)
                total_pairs += self._batch_size * self._num_negative_samples

                self._optimizer.zero_grad()
                if self._loss in ["bpr", "hinge", "pce", "bce"]:
                    loss = self._get_multiple_negative_predictions_normal(batch_query, batch_query_content,
                                batch_doc, batch_doc_content, batch_neg_docs, batch_neg_doc_content,
                                batch_query_len, batch_docs_lens, batch_neg_docs_lens, self._num_negative_samples,
                                                                          query_idf = query_idfs)
                epoch_loss += loss.item()
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()
                self._optimizer.step()
            epoch_loss /= float(total_pairs)
            TensorboardWrapper.mywriter().add_scalar("loss/epoch_loss_avg", epoch_loss, epoch_num)
            # print("Number of Minibatches: ", minibatch_num, "Avg. loss of epoch: ", epoch_loss)
            t2 = time.time()
            epoch_train_time = t2 - t1
            if verbose:  # validation after each epoch
                t1 = time.time()
                assert len(val_interactions.unique_queries_test) in KeyWordSettings.QueryCountVal, len(val_interactions.unique_queries_test)
                result_val = self.evaluate(val_interactions, topN)
                hits = result_val["hits"]
                ndcg = result_val["ndcg"]
                t2 = time.time()
                valiation_time = t2 - t1

                if epoch_num and epoch_num % self._testing_epochs == 0:
                    t1 = time.time()
                    assert len(test_interactions.unique_queries_test) in KeyWordSettings.QueryCountTest
                    result_test = self.evaluate(test_interactions, topN)
                    hits_test = result_test["hits"]
                    ndcg_test = result_test["ndcg"]
                    t2 = time.time()
                    testing_time = t2 - t1
                    TensorboardWrapper.mywriter().add_scalar("hit/hit_test", hits_test, epoch_num)
                    TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_test", ndcg_test, epoch_num)
                    FileHandler.myprint('|Epoch %03d | Test hits@%d = %.5f | Test ndcg@%d = %.5f | Testing time: %04.1f(s)'
                                        % (epoch_num, topN, hits_test, topN, ndcg_test, testing_time))

                TensorboardWrapper.mywriter().add_scalar("hit/hits_val", hits, epoch_num)
                TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_val", ndcg, epoch_num)
                FileHandler.myprint('|Epoch %03d | Train time: %04.1f(s) | Train loss: %.3f'
                                    '| Vad hits@%d = %.5f | Vad ndcg@%d = %.5f | Validation time: %04.1f(s)'
                                    % (epoch_num, epoch_train_time, epoch_loss, topN, hits, topN, ndcg, valiation_time))

                if hits > best_hit or (hits == best_hit and ndcg > best_ndcg):
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_hit, best_ndcg, best_epoch = hits, ndcg, epoch_num
                    # test_hit, test_ndcg = hits_test, ndcg_test
                else: count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    FileHandler.myprint("Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))
        FileHandler.myprint("Closing tensorboard")
        TensorboardWrapper.mywriter().close()
        FileHandler.myprint('Best result: | vad hits@%d = %.5f | vad ndcg@%d = %.5f | epoch = %d' % (
                            topN, best_hit, topN, best_ndcg, best_epoch))
        FileHandler.myprint_details(json.dumps(test_results_dict, sort_keys = True, indent = 2))

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
        # needs to check
        query_contents_tmp = query_contents.view(batch_size * L1, 1).expand(batch_size * L1, n).reshape(batch_size, L1, n)
        query_contents_tmp = query_contents_tmp.permute(0, 2, 1).reshape(batch_size * n, L1)  # (B, n, L1)

        assert query_contents_tmp.size() == (batch_size * n, L1)  # (B * n, L1)
        batch_negatives_tmp = negative_doc_contents.reshape(batch_size * n, L2)  # (B * n, L2)

        negative_prediction = self._net(query_contents_tmp, batch_negatives_tmp)  # (B * n)
        positive_prediction = self._net(query_contents, doc_contents) # (batch_size)

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

    def load_best_model_single(self, target_interactions: interactions.MatchInteraction, topN: int):
        """ Note: This function is used for Heat map visualization only. """
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        # assert len(val_interactions.unique_queries_test) in KeyWordSettings.QueryCountVal
        result_val, error_analysis_val = self.evaluate(target_interactions, topN, output_ranking = True)
        hits = result_val["hits"]
        ndcg = result_val["ndcg"]
        ndcg_at_1 = result_val["ndcg@1"]

        FileHandler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys = True, indent = 2))
        # FileHandler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys = True, indent = 2))
        FileHandler.myprint('Best Target hits@%d = %.5f | Best Target ndcg@%d = %.5f '
                            '|Best Target ndcg@1 = %.5f'
                            % (topN, hits, topN, ndcg, ndcg_at_1))

        return hits, ndcg

    def load_best_model(self, val_interactions: interactions.MatchInteraction,
                        test_interactions: interactions.MatchInteraction, topN: int):
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        assert len(val_interactions.unique_queries_test) in KeyWordSettings.QueryCountVal
        result_val, error_analysis_val = self.evaluate(val_interactions, topN, output_ranking = True)
        hits = result_val["hits"]
        ndcg = result_val["ndcg"]
        ndcg_at_1 = result_val["ndcg@1"]

        assert len(test_interactions.unique_queries_test) in KeyWordSettings.QueryCountTest
        result_test, error_analysis_test = self.evaluate(test_interactions, topN, output_ranking = True)
        hits_test = result_test["hits"]
        ndcg_test = result_test["ndcg"]
        ndcg_at_1_test = result_test["ndcg@1"]

        FileHandler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys = True, indent = 2))
        FileHandler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys = True, indent = 2))
        FileHandler.myprint('Best Vad hits@%d = %.5f | Best Vad ndcg@%d = %.5f '
                            '|Best Test hits@%d = %.5f |Best Test ndcg@%d = %.5f'
                            '|Best Vad ndcg@1 = %.5f |Best Test ndcg@1 = %.5f'
                            % (topN, hits, topN, ndcg, topN, hits_test, topN, ndcg_test, ndcg_at_1, ndcg_at_1_test))

        return hits, ndcg, hits_test, ndcg_test

    def load_best_model_test2_test3(self, test2: interactions.MatchInteraction,
                                    test3: interactions.MatchInteraction, topN: int):
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        assert len(test2.unique_queries_test) in KeyWordSettings.QueryCountTest
        result_test2, error_analysis_val = self.evaluate(test2, topN, output_ranking = True)
        hits_test2 = result_test2["hits"]
        ndcg_test2 = result_test2["ndcg"]
        ndcg_at_1_test2 = result_test2["ndcg@1"]

        FileHandler.save_error_analysis_test2(json.dumps(error_analysis_val, sort_keys = True, indent = 2))
        FileHandler.myprint('Best Test2_hard hits@%d = %.5f | Best Test2_hard ndcg@%d = %.5f '
                            '|Best Test2_hard ndcg@1 = %.5f '
                            % (topN, hits_test2, topN, ndcg_test2, ndcg_at_1_test2))

        return hits_test2, ndcg_test2

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
                query_idfs = my_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), self._use_cuda)

            query_content = np.tile(query_content, (len(labels), 1))  # len(labels), query_contnt_leng)
            doc_contents = np.array(doc_contents)
            query_content = my_utils.gpu(query_content)
            doc_contents = my_utils.gpu(doc_contents)

            query_content = my_utils.gpu(my_utils.numpy2tensor(query_content, dtype=torch.int), self._use_cuda)
            doc_contents = my_utils.gpu(my_utils.numpy2tensor(doc_contents, dtype=torch.int), self._use_cuda)

            predictions = self._net.predict(query_content, doc_contents, query_lens=query_len, docs_lens=doc_lens,
                                            query_idf=query_idfs)
            ndcg_mz = ndcg_metric(K)(labels, predictions)
            ndcgs_at_1.append(ndcg_metric(1)(labels, predictions))
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
