import matchzoo
from typing import List, Dict, Tuple
import torch_utils
import collections
import numpy as np

class TFIDF:

	idf_dict = {}
	idf_char_ngram = {}

	@classmethod
	def init(cls, corpus: List[List[str]], char_ngram_copus: List[List[str]] = []):
		"""docid, value: list of words """
		stats = cls._idf(corpus)
		cls.idf_dict = stats
		if char_ngram_copus:
			cls.idf_char_ngram = cls._idf(char_ngram_copus)

	@classmethod
	def get_term_idf(cls):
		return cls.idf_dict

	@classmethod
	def get_char_ngram_idf(cls):
		return cls.idf_char_ngram

	@classmethod
	def _df(cls, list_of_tokens: list) -> dict:
		stats = collections.Counter()
		for tokens in list_of_tokens:
			stats.update(set(tokens))
		return stats

	@classmethod
	def _idf(cls, list_of_tokens: list) -> dict:
		num_docs = len(list_of_tokens)
		stats = cls._df(list_of_tokens)
		for key, val in stats.most_common():
			stats[key] = np.log((1.0 + num_docs) / (1.0 + val)) + 1.0
		return stats