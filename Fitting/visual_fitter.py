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
from Fitting.densebaseline_fit import DenseBaselineFitter
import torch.nn.functional as F
from tqdm import tqdm


class VisualFitter(DenseBaselineFitter):

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
        super(VisualFitter, self).__init__(net, loss, n_iter, testing_epochs, batch_size, reg_l2, learning_rate,
                                           early_stopping, decay_step, decay_weight, optimizer_func,
                                           use_cuda, num_negative_samples, logfolder, curr_date, **kargs)
        self.use_visual = kargs[KeyWordSettings.UseVisual]
        self.image_loader = kargs[KeyWordSettings.ImageLoaderKey]
        self.index2queries = kargs[KeyWordSettings.Index2Query]
        self.index2docs = kargs[KeyWordSettings.Index2Doc]

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
            query_ids, left_contents, left_lengths, left_imgs_indices, \
            doc_ids, right_contents, right_lengths, right_imgs_indices, \
            neg_doc_ids, neg_docs_contents, neg_docs_lens, neg_docs_imgs_indices = \
                self._sampler.get_train_instances_visual(train_iteractions, self._num_negative_samples)

            queries, query_content, query_lengths, query_imgs, \
            docs, doc_content, doc_lengths, doc_imgs, \
            neg_doc_ids, neg_docs_contents, neg_docs_lens, neg_docs_imgs_indices = my_utils.shuffle(query_ids, left_contents, left_lengths, left_imgs_indices,
                                                                doc_ids, right_contents, right_lengths, right_imgs_indices,
                                                                neg_doc_ids, neg_docs_contents, neg_docs_lens, neg_docs_imgs_indices)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                (batch_query, batch_query_content, batch_query_len, batch_query_imgs_indices,
                 batch_doc, batch_doc_content, batch_docs_lens, batch_doc_imgs_indices,
                 batch_neg_doc, batch_neg_doc_content, batch_neg_docs_lens, batch_neg_docs_imgs_indices)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths, query_imgs,
                                                    docs, doc_content, doc_lengths, doc_imgs,
                                                    neg_doc_ids, neg_docs_contents, neg_docs_lens, neg_docs_imgs_indices,
                                                    batch_size = self._batch_size)):
                t10 = time.time()
                # add idf here...
                additional_data = {}
                if len(TFIDF.get_term_idf()) != 0:
                    query_idf_dict = TFIDF.get_term_idf()
                    query_idfs = [[query_idf_dict.get(int(word_idx), 0.0) for word_idx in row] for row in batch_query_content]
                    query_idfs = torch_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), self._use_cuda)
                    additional_data["query_idf"] = query_idfs

                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                batch_doc = my_utils.gpu(torch.from_numpy(batch_doc), self._use_cuda)
                batch_doc_content = my_utils.gpu(torch.from_numpy(batch_doc_content), self._use_cuda)
                batch_neg_doc_content = my_utils.gpu(torch.from_numpy(batch_neg_doc_content), self._use_cuda)

                if self.use_visual:  # load images tensors
                    batch_query_imgs_indices = my_utils.gpu(torch.from_numpy(batch_query_imgs_indices), self._use_cuda)
                    batch_doc_imgs_indices = my_utils.gpu(torch.from_numpy(batch_doc_imgs_indices), self._use_cuda)
                    batch_neg_docs_imgs_indices = my_utils.gpu(torch.from_numpy(batch_neg_docs_imgs_indices), self._use_cuda)
                    additional_data[KeyWordSettings.QueryImagesIndices] = batch_query_imgs_indices.unsqueeze(1)  # (B, 1, M1)
                    additional_data[KeyWordSettings.DocImagesIndices] = batch_doc_imgs_indices.unsqueeze(1)  # (B, 1, M2)
                    additional_data[KeyWordSettings.NegDocImagesIndices] = batch_neg_docs_imgs_indices  # (B, n, M2)

                total_pairs += self._batch_size * self._num_negative_samples
                self._optimizer.zero_grad()
                if self._loss in ["bpr", "hinge", "pce", "bce", "cosine_max_margin_loss_dvsh"]:
                    loss = self._get_multiple_negative_predictions_normal(batch_query, batch_query_content,
                                batch_doc, batch_doc_content, batch_neg_doc, batch_neg_doc_content,
                                batch_query_len, batch_docs_lens, batch_neg_docs_lens, self._num_negative_samples,
                                                                          **additional_data)
                epoch_loss += loss.item()
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()
                self._optimizer.step()
                t11 = time.time()
                # print("Running time for one mini-batch: ", t11 - t10, "seconds")
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
        additional_data_negative = {}
        if self.use_visual:
            query_images = kargs[KeyWordSettings.QueryImagesIndices]  # (B, 1, M, 3, 224, 224)
            neg_doc_images = kargs[KeyWordSettings.NegDocImagesIndices]  # (B, n, M, 3, 224, 224)
            additional_data_negative[KeyWordSettings.QueryImagesIndices] = query_images  # query_images_tmp
            additional_data_negative[KeyWordSettings.DocImagesIndices] = neg_doc_images

        # needs to check
        query_contents_tmp = query_contents.view(batch_size * L1, 1).expand(batch_size * L1, n).reshape(batch_size, L1, n)
        query_contents_tmp = query_contents_tmp.permute(0, 2, 1).reshape(batch_size * n, L1)  # (B, n, L1)
        query_ids_tmp = query_ids.view(batch_size, 1).expand(batch_size, n).reshape(batch_size * n)
        additional_data_negative[KeyWordSettings.QueryIDs] = query_ids_tmp
        additional_data_negative[KeyWordSettings.DocIDs] = negative_doc_ids.reshape(batch_size * n)
        additional_data_negative[KeyWordSettings.UseCuda] = self._use_cuda

        assert query_contents_tmp.size() == (batch_size * n, L1)  # (B * n, L1)
        batch_negatives_tmp = negative_doc_contents.reshape(batch_size * n, L2)  # (B * n, L2)
        kargs[KeyWordSettings.QueryIDs] = query_ids
        kargs[KeyWordSettings.DocIDs] = doc_ids
        kargs[KeyWordSettings.UseCuda] = self._use_cuda

        # why don't we combine all to 1???
        positive_prediction = self._net(query_contents, doc_contents, **kargs)  # (batch_size)
        negative_prediction = self._net(query_contents_tmp, batch_negatives_tmp, **additional_data_negative)  # (B * n)

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

    def evaluate(self, testRatings: interactions.MatchInteractionVisual, K: int, output_ranking = False, **kargs):
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
            t3 = time.time()
            docs, labels, doc_contents, _ = candidates
            query_content = testRatings.dict_query_contents[query]
            query_images_indices = testRatings.dict_query_imgages[query]
            query_len = [testRatings.dict_query_lengths[query]] * len(labels)
            doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]
            doc_images_indices = [testRatings.dict_doc_imgages[d] for d in docs]

            additional_data = {}
            additional_data[KeyWordSettings.Query_lens] = query_len
            additional_data[KeyWordSettings.Doc_lens] = doc_lens
            if len(TFIDF.get_term_idf()) > 0:
                query_idf_dict = TFIDF.get_term_idf()
                query_idfs = [query_idf_dict.get(int(word_idx), 0.0) for word_idx in query_content]
                query_idfs = np.tile(query_idfs, (len(labels), 1))
                query_idfs = my_utils.gpu(torch.from_numpy(np.array(query_idfs)).float(), self._use_cuda)
                additional_data[KeyWordSettings.Query_Idf] = query_idfs
            if self.use_visual:
                t1 = time.time()
                query_images_indices = np.array(query_images_indices)
                assert query_images_indices.shape == (len(query_images_indices), )
                query_images = query_images_indices.reshape(1, 1, len(query_images_indices))
                doc_images = np.array(doc_images_indices)
                query_images = torch_utils.gpu(torch.from_numpy(query_images), self._use_cuda)
                doc_images = torch_utils.gpu(torch.from_numpy(doc_images), self._use_cuda)
                additional_data[KeyWordSettings.QueryImagesIndices] = query_images  # (1, 1, M1)
                additional_data[KeyWordSettings.DocImagesIndices] = doc_images.unsqueeze(1)  # (B, 1, M2)
                t2 = time.time()
                # print("Loading time images to gpu of validation: ", t2 - t1, "seconds")
            if output_ranking: additional_data[KeyWordSettings.OutputRankingKey] = True  # for error analysis
            query_content = np.tile(query_content, (len(labels), 1))  # len(labels), query_contnt_leng)
            doc_contents = np.array(doc_contents)
            query_content = my_utils.gpu(query_content)
            doc_contents = my_utils.gpu(doc_contents)

            query_content = my_utils.gpu(my_utils.numpy2tensor(query_content, dtype=torch.int), self._use_cuda)
            doc_contents = my_utils.gpu(my_utils.numpy2tensor(doc_contents, dtype=torch.int), self._use_cuda)
            additional_data[KeyWordSettings.QueryIDs] = np.array([query] * len(labels))
            additional_data[KeyWordSettings.DocIDs] = np.array(docs)
            additional_data[KeyWordSettings.UseCuda] = self._use_cuda

            predictions = self._net.predict(query_content, doc_contents, **additional_data)
            if output_ranking:
                assert len(predictions.shape) == 1
                _, M2 = doc_images.shape
                predictions = predictions.reshape(len(doc_lens), 1 + (len(query_images_indices) * M2))
                predictions, visual_sims = predictions[:, 0], predictions[:, 1:]
                visual_sims = visual_sims.reshape(len(doc_lens), len(query_images_indices), M2)
                visual_sims = visual_sims.transpose(0, 2, 1)  # (B, M2, M1)

            t4 = time.time()
            # print("Computing time of each query: ", (t4 - t3), "seconds")
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
                visual_scores = visual_sims[indices]  # (B, M2 * M1)) due to transpose
                assert scores.shape == ranked_labels.shape
                ranked_doc_list = [{KeyWordSettings.Doc_cID: int(d),
                                    KeyWordSettings.Doc_URL: self.index2docs[int(d)],
                                    KeyWordSettings.Doc_cLabel: int(lab),
                                    KeyWordSettings.Doc_wImages: ["%s %s" % (x, str(y)) for x, y in
                                                                  zip(list(map(self.image_loader.right_img_index2path.get,
                                                                               testRatings.dict_doc_imgages[d])), visual_score.tolist())],
                                    KeyWordSettings.Doc_wContent: testRatings.dict_doc_raw_contents[d],
                                    KeyWordSettings.Relevant_Score: float(score)}
                                   for d, lab, score, visual_score in zip(ranked_docs, ranked_labels, scores, visual_scores)]

                q_details = {KeyWordSettings.Query_id: int(query),
                             KeyWordSettings.Query_TweetID: "http://twitter.com/user/status/" + self.index2queries[int(query)],
                             KeyWordSettings.Query_Images: list(map(self.image_loader.left_img_index2path.get, query_images_indices)),
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
