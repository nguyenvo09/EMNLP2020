__author__ = 'Nick Hirakawa'

import re
from typing import List

class CorpusParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile('^#\s*\d+')
		self.corpus = dict()

	def parse(self):
		with open(self.filename) as f:
			s = ''.join(f.readlines())
		blobs = s.split('#')[1:]
		for x in blobs:
			text = x.split()
			docid = text.pop(0)
			self.corpus[docid] = text

	def get_corpus(self):
		return self.corpus

class CorpusParser2:

	def __init__(self, docs_contents, docsIDs):
		# self.filename = filename
		# self.regex = re.compile('^#\s*\d+')
		self.corpus = {}
		for did, dc in zip(docsIDs, docs_contents):
			self.corpus[str(did)] = dc.split()

	def parse(self):
		pass
		# with open(self.filename) as f:
		# 	s = ''.join(f.readlines())
		# blobs = s.split('#')[1:]
		# for x in blobs:
		# 	text = x.split()
		# 	docid = text.pop(0)
		# 	self.corpus[docid] = text

	def get_corpus(self):
		return self.corpus


class QueryParser:

	def __init__(self, filename):
		self.filename = filename
		self.queries = []

	def parse(self):
		with open(self.filename) as f:
			lines = ''.join(f.readlines())
		self.queries = [x.rstrip().split() for x in lines.split('\n')[:-1]]

	def get_queries(self):
		return self.queries


class QueryParser2:

	def __init__(self, queries: List[str]):
		self.queries = [e.split() for e in queries]
		# self.queries = []

	def parse(self):
		# self.queries = [x.rstrip().split() for x in self.queries]
		pass

	def get_queries(self):
		return self.queries

if __name__ == '__main__':
	qp = QueryParser('text/queries.txt')
	print(qp.get_queries())