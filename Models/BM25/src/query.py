__author__ = 'Nick Hirakawa'

from .invdx import build_data_structures
from .rank import score_BM25
from handlers.output_handler import FileHandler
from typing import List, Dict, Tuple
import collections
import matchzoo
import torch_utils
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import os
import multiprocessing


class QueryProcessor:
	def __init__(self, queries: List[List[str]], corpus: Dict[str, List[str]], params: dict = {}):
		"""docid, value: list of words """
		self.queries = queries
		self.index, self.dlt = build_data_structures(corpus)
		self.params = params
		FileHandler.myprint(str(params))

	def run(self):
		results = []
		for query in self.queries:
			results.append(self.run_query(query))
		return results

	def run_base_retrieval(self, multi_threading = False):
		assert type(self.queries) == dict
		results = dict()
		cnt = 0
		if multi_threading:
			print("Running in parrallel......")
			def processInput(qid):
				local_results = {}
				qtext = self.queries[qid]
				local_results[qid] = self.run_query(qtext)
				return local_results

			num_cores = multiprocessing.cpu_count()
			mid_results = Parallel(n_jobs=num_cores)(delayed(processInput)(qid) for qid in tqdm(sorted(list(self.queries.keys()))) )
			for mid in mid_results:
				for q in mid: results[q] = mid[q]
			return results

		else:
			for qid in tqdm(sorted(self.queries.keys())):
				qtext = self.queries[qid]
				results[qid] = self.run_query(qtext)
				# cnt += 1
				# if cnt == 20: break
			return results

	def run_query(self, query):
		query_result = dict()
		for term in query:
			if term in self.index:
				doc_dict = self.index[term]  # retrieve index entry for each term (key: term, value: list of docs that have that term
				# for each document and its word frequency
				for docid, freq in doc_dict.items():
					score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
									   dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length(),
									   b = self.params["b"], k1 = self.params["k1"]) # calculate score
					# this document has already been scored once
					if docid in query_result: query_result[docid] += score
					else: query_result[docid] = score
		return query_result