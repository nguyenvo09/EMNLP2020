"""Matchzoo toolkit for token embedding."""

import csv
import typing

import numpy as np
import pandas as pd

import matchzoo as mz
from handlers.output_handler import FileHandler


class EntityEmbedding(object):
    """
    Embedding class for entities

    Examples::

    """

    def __init__(self, output_dim: int):
        """
        Embedding.

        :param data: Dictionary to use as term to vector mapping.
        :param output_dim: The dimension of embedding.
        """
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
        # Starting the smallest index to the largest to ensure reproducibility
        for term, index in sorted(term_index.items(), key = lambda x: x[1]):
            matrix[index] = initializer()
        return matrix
