"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import torch_utils
import time
import interactions


class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()
        self._candidate = dict()  # negative candidates

    def random_sample_items(self, num_items, shape):
        """
        Randomly sample a number of items based on shape.
        (we need to improve this since it is likely to sample a positive instance)
        https://github.com/maciejkula/spotlight/issues/36
        https://github.com/graytowne/caser_pytorch/blob/master/train_caser.py
        Parameters
        ----------

        num_items: int
            Total number of items from which we should sample:
            the maximum value of a sampled item id will be smaller
            than this.
        shape: int or tuple of ints
            Shape of the sampled array.

        Returns
        -------

        items: np.array of shape [shape]
            Sampled item ids.
        """
        items = np.random.randint(0, num_items, shape, dtype = np.int64)

        return items

    # reuse from https://github.com/nguyenvo09/caser_pytorch/blob/master/train_caser.py#L203
    def get_train_instances(self, interactions: interactions.MatchInteraction,
                            num_negatives: int):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}
        Parameters
        ----------
        interactions: :class:`matchzoo.DataPack`
            training instances, used for generate candidates. Note that
            since I am using MatchZoo datapack, there are negative cases in left-right relation ship as
            well.
        num_negatives: int
            total number of negatives to sample for each sequence
        """

        query_ids = interactions.pos_queries.astype(np.int64)  # may not be unique
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)

        doc_ids = interactions.pos_docs.astype(np.int64)
        doc_contents = interactions.np_doc_contents.astype(np.int64)
        doc_lengths = interactions.np_doc_lengths.astype(np.int64)

        negative_samples = np.zeros((query_ids.shape[0], num_negatives, interactions.padded_doc_length), np.int64)
        negative_samples_lens = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        negative_docs_ids = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        self._candidate = interactions.negatives

        for i, u in enumerate(query_ids):
            for j in range(num_negatives):
                x = self._candidate[u]
                neg_item = x[np.random.randint(len(x))]  # int
                # print("Neg_item: ", neg_item)
                neg_item_content = interactions.dict_doc_contents[neg_item]  # np.array
                negative_samples[i, j] = neg_item_content
                negative_samples_lens[i, j] = interactions.dict_doc_lengths[neg_item]
                negative_docs_ids[i, j] = neg_item
            # if u <= 0:
            #     print("Negative samples: ", negative_samples[i])
        # print(negative_samples)
        return query_ids, query_contents, query_lengths, \
               doc_ids, doc_contents, doc_lengths, \
               negative_docs_ids, negative_samples, negative_samples_lens

    def get_train_instances_visual(self, interactions: interactions.MatchInteractionVisual, num_negatives: int):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}
        Parameters
        ----------
        interactions: :class:`matchzoo.DataPack`
            training instances, used for generate candidates. Note that
            since I am using MatchZoo datapack, there are negative cases in left-right relation ship as
            well.
        num_negatives: int
            total number of negatives to sample for each sequence
        """

        query_ids = interactions.pos_queries.astype(np.int64)  # may not be unique
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)
        query_images_paths = interactions.query_images

        doc_ids = interactions.pos_docs.astype(np.int64)
        doc_contents = interactions.np_doc_contents.astype(np.int64)
        doc_lengths = interactions.np_doc_lengths.astype(np.int64)
        doc_images_paths = interactions.doc_images

        negative_samples = np.zeros((query_ids.shape[0], num_negatives, interactions.padded_doc_length), np.int64)
        negative_samples_lens = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        negative_samples_imgs = np.zeros((query_ids.shape[0], num_negatives, interactions.padded_doc_images_len), np.int64)
        negative_docs_ids = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        self._candidate = interactions.negatives

        for i, u in enumerate(query_ids):
            for j in range(num_negatives):
                x = self._candidate[u]
                neg_item = x[np.random.randint(len(x))]  # int
                # print("Neg_item: ", neg_item)
                neg_item_content = interactions.dict_doc_contents[neg_item]  # np.array
                negative_samples[i, j] = neg_item_content
                negative_samples_lens[i, j] = interactions.dict_doc_lengths[neg_item]
                negative_samples_imgs[i, j] = interactions.dict_doc_imgages[neg_item]
                negative_docs_ids[i, j] = neg_item
            # if u <= 0:
            #     print("Negative samples: ", negative_samples[i])
        # print(negative_samples)
        return query_ids, query_contents, query_lengths, query_images_paths, \
               doc_ids, doc_contents, doc_lengths, doc_images_paths, \
               negative_docs_ids, negative_samples, negative_samples_lens, negative_samples_imgs
