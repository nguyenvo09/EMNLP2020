import typing
import csv
from pathlib import Path

import pandas as pd
import time
import matchzoo
import os
from typing import List, Set, Tuple, Dict
from tqdm import tqdm
import torch
import numpy as np
import itertools
from handlers.output_handler import FileHandler
import torchvision
import torch_utils


def load_data2(
    data_root: str,
    stage: str = 'train',
    prefix: str = "Snopes"
) -> pd.DataFrame:
    """
    Load WikiQA data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.
    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test', 'test2_hard', 'test3_hard'):
        raise ValueError("%s is not a valid stage. Must be one of `train`, `dev`, and `test`." % stage)

    # data_root = _download_data()
    data_root = data_root
    file_path = os.path.join(data_root, '%s.%s.tsv' % (prefix, stage))
    data_pack = _read_data2(file_path)
    return data_pack


def _read_data2(path):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['QueryText'],
        'raw_text_left': table['QueryText'].copy(),
        'images_left': table['QueryText'].copy(),  # temporary use

        'text_right': table['DocText'],
        'raw_text_right': table['DocText'].copy(),
        'images_right': table['DocText'].copy(),  # temporary use

        'id_left': table['QueryID'],
        'id_right': table['DocID'],
        'label': table['Label']
    })
    return df


class ElmoLoader(object):
    def __init__(self, queries_feats_path: str, docs_feats_path: str, fixed_len_left: int, fixed_len_right: int):
        """

        Parameters
        ----------
        queries_feats_path: `str` path to extracted features
        docs_feats_path: `str` path to extracted features
        """
        self.queries_content, self.left_tensor_feats = self.load_elmo_features(queries_feats_path, fixed_len_left)
        self.docs_content, self.right_tensor_feats = self.load_elmo_features(docs_feats_path, fixed_len_right)
        FileHandler.myprint("Left Elmo tensor feats: " + str(self.left_tensor_feats.size()))
        FileHandler.myprint("Right Elmo tensor feats: " + str(self.right_tensor_feats.size()))

    def load_elmo_features(self, pth_file: str, max_len: int):
        ids, text, feats = torch.load(pth_file)  # note that that q_feats needs to be padded
        dict_content = dict(list(zip(ids, text)))
        tensor_feats = self._pad_tensor(feats, max_len, ids, dict_content)
        return dict_content, tensor_feats

    def _pad_tensor(self, feats: List[np.ndarray], max_len: int, tids: List[int], dict_content: Dict[int, str], do_test=True):
        assert len(feats) == len(tids)
        tensors = []
        lens = []
        new_order = np.argsort(tids)  # ex: [2, 3, 1, 0] -> [3, 2, 0, 1]
        for idx in tqdm(new_order):  # based on index, we retriev the tensor and id
            tid = tids[idx]
            tsr = feats[idx]
            tsr = list(tsr.squeeze(0))  # only the first dimension
            assert len(tsr) <= max_len, (len(tsr), "vs. ", max_len)
            curr_len = len(tsr)
            assert len(dict_content[tid]) == curr_len
            lens.append(curr_len)
            tsr = tsr + [[0.0] * 1024 for _ in range(abs(max_len - curr_len))]
            tsr = torch.from_numpy(np.array(tsr))
            tensors.append(tsr.float())
        tensors = torch.stack(tensors, dim = 0)  # (|Q|, L, H)
        # change row of this tensor based on sorted ids
        assert len(tids) == len(set(tids))
        new_lens = np.array(lens)[new_order]
        if do_test:
            for tsr, _len in zip(tensors, new_lens):
                assert np.sum(torch_utils.cpu(tsr[_len:]).numpy()) < 1e-10  # ensure padded are all zero
        return tensors

    def elmo_load_data(self, data_root: str, stage: str = 'train', prefix: str = "Snopes") \
            -> typing.Union[matchzoo.DataPack, tuple]:
        """
        Replace the text content by using the content from `self.queries_content` and `self.docs_content`
        We don't need to pre-process data anymore.
        Parameters
        ----------
        path: path to tsv file.

        Returns
        -------

        """
        if stage not in ('train', 'dev', 'test', 'test2_hard', 'test3_hard', 'heat'):
            raise ValueError("%s is not a valid stage. Must be one of `train`, `dev`, and `test`." % stage)

        # data_root = _download_data()
        data_root = data_root
        file_path = os.path.join(data_root, '%s.%s.tsv' % (prefix, stage))
        data_pack = self._elmo_read_data(file_path)
        return data_pack

    def _elmo_read_data(self, path):
        def _replace_text_content(ids: List[int], new_text_dict: Dict[int, str]):
            ans = [" ".join(new_text_dict[v]) for v in ids]
            return ans

        table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
        df = pd.DataFrame({
            'text_left': _replace_text_content(table["QueryID"], self.queries_content),
            'raw_text_left': table['QueryText'].copy(),
            'images_left': table['QueryImages'].copy().progress_apply(str.split),  # indices

            'text_right': _replace_text_content(table["DocID"], self.docs_content),
            'raw_text_right': table['DocText'].copy(),
            'images_right': table['DocImages'].copy().progress_apply(str.split),  # indices

            'id_left': table['QueryID'],
            'id_right': table['DocID'],
            'label': table['Label']
        })
        return matchzoo.pack(df)


class ImagesLoader(object):

    def __init__(self, left_pth_file: str, max_num_left_images: int, right_pth_file: str, max_num_right_images: int,
                 use_cuda: bool, image_dim: int = 2048):
        """
        Parameters
        ----------
        left_pth_file:  `str` the giant fat file of all tensors of images of queries
        max_num_left_images: `int`
        right_pth_file: `str` the giant fat file of all tensors of images of docs
        max_num_right_images: `int`
        image_dim: `int` dimension of image
        """
        self.left_pth_file = left_pth_file
        self.max_num_left_images = max_num_left_images
        self.right_pth_file = right_pth_file
        self.max_num_right_images = max_num_right_images
        self._use_cuda = use_cuda
        self._image_dim = image_dim

    def fit_side(self, data_packs: List[matchzoo.DataPack], fat_pth_file, max_len_images, side):
        tensor, path2index, index2path = read_images(data_packs, fat_pth_file=fat_pth_file, max_len_images=max_len_images,
                                                     side=side, image_dim=self._image_dim)
        assert len(tensor.size()) == 2
        D = tensor.size(1)
        return tensor, path2index, index2path, D

    def fit(self, data_packs: List[matchzoo.DataPack]):
        """Need tested!!!!"""
        FileHandler.myprint("Loading images of queries.....")
        self.left_tensor, \
        self.left_img_path2index, \
        self.left_img_index2path, D1 = self.fit_side(data_packs, fat_pth_file = self.left_pth_file,
                                                     max_len_images = self.max_num_left_images, side="left")
        FileHandler.myprint("Loading images of docs.....")
        self.right_tensor, \
        self.right_img_path2index, \
        self.right_img_index2path, D2 = self.fit_side(data_packs, fat_pth_file = self.right_pth_file,
                                                      max_len_images = self.max_num_right_images, side="right")
        assert D1 == D2
        self.visual_features_size = D1
        FileHandler.myprint("Visual Feature Dimension: %s" % (self.visual_features_size))

    def transform(self, pack: matchzoo.DataPack):
        """ Converting the raw path to mapped indices """
        def left_to_indices(images: List[str]):
            images_indices = [self.left_img_path2index[p] for p in images[:self.max_num_left_images]]
            images_indices += [0] * (self.max_num_left_images - len(images_indices))  # padding
            return images_indices

        def right_to_indices(images: List[str]):
            images_indices = [self.right_img_path2index[p] for p in images[:self.max_num_right_images]]
            images_indices += [0] * (self.max_num_right_images - len(images_indices))  # padding
            return images_indices

        pack.left["images_left"] = pack.left["images_left"].apply(left_to_indices)
        pack.right["images_right"] = pack.right["images_right"].apply(right_to_indices)
        return pack


def read_images(data_packs: List[matchzoo.DataPack],
                fat_pth_file: str, max_len_images: int, side, image_dim: int = 2048) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    - Prune the the maximum number of images per query or per doc based on `max_len_images`
    - Read tensors from the giant pth file.
    Parameters
    ----------
    data_packs: List[matchzoo.DataPack] typically are datapacks from train, dev and valid
    fat_pth_file: `str` the giant fat file of all tensors of images
    max_len_images: `int` the maximum number of images per query or per doc
    side: `str` left or right

    Returns
    -------
    Returns:
        tsr: `torch.Tensor`  All tensors of left or right side (pruned based on argument `max_len_images`).
        Expected tensor is (x, 3, 224, 224)
        path_to_index: `Dict[str, int]` the path to index where key is the path of an image and value is
        the corresponding row index (dim = 0) of `tsr`
    """
    tic = time.time()
    assert side == "left" or side == "right"
    if side == "left":
        img_paths = [pack.left["images_left"].progress_apply(lambda x: x[:max_len_images]).tolist() for pack in data_packs]
        img_paths = list(itertools.chain.from_iterable(img_paths))
    elif side == "right":
        img_paths = [pack.right["images_right"].progress_apply(lambda x: x[:max_len_images]).tolist() for pack in data_packs]
        img_paths = list(itertools.chain.from_iterable(img_paths))

    # fat_tensor, paths = torch.load(os.path.join("..", fat_pth_file))
    fat_tensor, paths = torch.load(fat_pth_file)
    mapper = dict(zip(paths, range(len(paths))))  # indices here are rows of `fat_tensor`
    assert len(paths) == fat_tensor.size(0)
    tsr = [torch.zeros((image_dim, )).float()]  # for padding!!!
    path_to_index = {}
    index_to_path = {}
    counter = itertools.count()
    for row in tqdm(img_paths):
        for p in row:  # can we do this faster?????
            if p in path_to_index: continue
            path_to_index[p] = next(counter) + 1  # 0 index is for padding
            index_to_path[path_to_index[p]] = p  # index to path
            idx = mapper[p]  # find the index of a row in `fat_tensor`
            img = fat_tensor[idx]
            assert img.size() == (image_dim, )  # (3, 224, 224)
            tsr.append(img)
            assert len(tsr) - 1 == path_to_index[p], (len(tsr) - 1, path_to_index[p])  # must match

    tsr = torch.stack(tsr, dim = 0)  # (x, 2048)
    toc = time.time()
    # FileHandler.myprint('loading all images time: %d (seconds)' % (toc - tic))
    return tsr, path_to_index, index_to_path  # (max_len, 3, 224, 224)
