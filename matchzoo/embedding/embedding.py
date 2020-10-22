"""Matchzoo toolkit for token embedding."""

import csv
import typing

import numpy as np
import pandas as pd

import matchzoo as mz
from handlers.output_handler import FileHandler

class Embedding(object):
    """
    Embedding class.

    Examples::
        >>> import matchzoo as mz
        >>> train_raw = mz.datasets.toy.load_data()
        >>> pp = mz.preprocessors.NaivePreprocessor()
        >>> train = pp.fit_transform(train_raw, verbose=0)
        >>> vocab_unit = mz.build_vocab_unit(train, verbose=0)
        >>> term_index = vocab_unit.state['term_index']
        >>> embed_path = mz.datasets.embeddings.EMBED_RANK

    To load from a file:
        >>> embedding = mz.embedding.load_from_file(embed_path)
        >>> matrix = embedding.build_matrix(term_index)
        >>> matrix.shape[0] == len(term_index)
        True

    To build your own:
        >>> data = {'A':[0, 1], 'B':[2, 3]}
        >>> embedding = mz.Embedding(data, 2)
        >>> matrix = embedding.build_matrix({'A': 2, 'B': 1, '_PAD': 0})
        >>> matrix.shape == (3, 2)
        True

    """

    def __init__(self, data: dict, output_dim: int):
        """
        Embedding.

        :param data: Dictionary to use as term to vector mapping.
        :param output_dim: The dimension of embedding.
        """
        self._data = data
        self._output_dim = output_dim

    def build_matrix(
        self,
        term_index: typing.Union[
            dict, mz.preprocessors.units.Vocabulary.TermIndex],
        initializer=lambda: np.random.uniform(-0.2, 0.2)
    ) -> np.ndarray:
        """
        Build a matrix using `term_index`.

        :param term_index: A `dict` or `TermIndex` to build with.
        :param initializer: A callable that returns a default value for missing
            terms in data. (default: a random uniform distribution in range)
            `(-0.2, 0.2)`).
        :return: A matrix.
        """
        input_dim = len(term_index)
        matrix = np.empty((input_dim, self._output_dim))
        valid_keys = self._data.keys()
        for term, index in sorted(term_index.items(), key = lambda x: x[1]):  # Starting the smallest index to the largest
            if term in valid_keys:
                matrix[index] = self._data[term]
            else:
                matrix[index] = initializer()
        return matrix


def load_from_file(file_path: str, mode: str = 'word2vec', term_index: mz.preprocessors.units.Vocabulary.TermIndex = None) -> Embedding:
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec', 'fasttext'
        or 'glove'.(default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    embedding_data = {}
    output_dim = 0
    count_word_hit = 0
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r') as f:
            output_dim = int(f.readline().strip().split(' ')[-1])
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    elif mode == 'glove':
        with open(file_path, 'r', encoding = "utf-8") as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    else: raise TypeError("%s is not a supported embedding type. `word2vec`, `fasttext` or `glove` expected." % mode)

    FileHandler.myprint("Word hit: " + str((count_word_hit, len(term_index))) + " " + str(count_word_hit / len(term_index) * 100))

    return Embedding(embedding_data, output_dim)


def load_from_file_matching(file_path: str, mode: str = 'word2vec',
                            term_index: mz.preprocessors.units.Vocabulary.TermIndex = None, **kargs) -> Embedding:
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec', 'fasttext'
        or 'glove'.(default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    embedding_data = {}
    output_dim = 0
    count_word_hit = 0
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r') as f:
            output_dim = int(f.readline().strip().split(' ')[-1])
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    elif mode == 'glove':
        with open(file_path, 'r', encoding = "utf-8") as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    else: raise TypeError("%s is not a supported embedding type. `word2vec`, `fasttext` or `glove` expected." % mode)
    output_handler = kargs["output_handler_multiprocessing"]
    output_handler.myprint("Word hit: " + str((count_word_hit, len(term_index))) + " " + str(count_word_hit / len(term_index) * 100))

    return Embedding(embedding_data, output_dim)


def load_from_file_FC(file_path: str, mode: str = 'word2vec', term_index: mz.preprocessors.units.Vocabulary.TermIndex = None, **kargs) -> Embedding:
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec', 'fasttext'
        or 'glove'.(default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    embedding_data = {}
    output_dim = 0
    count_word_hit = 0
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r') as f:
            output_dim = int(f.readline().strip().split(' ')[-1])
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    elif mode == 'glove':
        with open(file_path, 'r', encoding = "utf-8") as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                if current_line[0] not in term_index: continue
                embedding_data[current_line[0]] = current_line[1:]
                count_word_hit += 1
    else: raise TypeError("%s is not a supported embedding type. `word2vec`, `fasttext` or `glove` expected." % mode)

    output_handler = kargs["output_handler_fact_checking"]
    output_handler.myprint("Word hit: " + str((count_word_hit, len(term_index))) + " " + str(count_word_hit / len(term_index) * 100))

    return Embedding(embedding_data, output_dim)