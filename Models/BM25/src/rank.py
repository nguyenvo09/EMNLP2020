__author__ = 'Nick Hirakawa'


from math import log

# k1 = 1.2
k2 = 100
# b = 0.75
R = 0.0


def score_BM25(n, f, qf, r, N, dl: float, avdl: float, b: float = 0.75, k1: float = 1.2) -> float:
	"""

	Parameters
	----------
	n: int number of documents that contaian the term
	f: term frequency
	qf: int (always equal to 1)
	r
	N: `int` the number of documents of a corpus
	dl: the length of a document
	avdl: average lengths of all documents

	Returns
	-------

	"""
	K = compute_K(dl, avdl, b = b, k1 = k1)
	first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
	second = ((k1 + 1) * f) / (K + f)
	third = ((k2 + 1) * qf) / (k2 + qf)  # always equal to 1 since qf = 1
	return first * second * third


def compute_K(dl: float, avdl: float, b: float = 0.75, k1: float = 1.2) -> float:
	"""

	Parameters
	----------
	dl: length of a document
	avdl: average length of documents

	Returns
	-------

	"""
	return k1 * ((1 - b) + b * (float(dl) / float(avdl)) )