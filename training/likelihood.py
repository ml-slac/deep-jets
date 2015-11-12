import numpy as np
class Likelihood1D(object):
	"""docstring for Likelihood1D"""
	def __init__(self, bins = np.linspace(0, 500, 100)):
		super(Likelihood1D, self).__init__()
		self._bins = bins

	def fit(self, signal, background, weights=(None, None)):
		self._sig_p, _ = np.histogram(signal, bins=self._bins, normed=True, weights=weights[0])
		self._bkg_p, _ = np.histogram(background, bins=self._bins, normed=True, weights=weights[1])

		self._ratio = np.divide(self._sig_p, self._bkg_p)
		self._ratio[np.isinf(self._ratio)] = np.isfinite(self._ratio).max()
		self._ratio[np.isnan(self._ratio)] = 1
		return self

	def predict(self, X):
		try:
			return self._ratio[self._bins.searchsorted(X) - 1]
		except IndexError:
			if not hasattr(X, '__iter__'):
				return 1.0
			return np.array([1.0 for _ in X])

class Likelihood2D(object):
	"""docstring for Likelihood2D"""
	def __init__(self, binsx, binsy):
		super(Likelihood2D, self).__init__()
		self.binsx = binsx
		self.binsy = binsy
		self._bins = (binsx, binsy)

	def fit(self, signal, background, weights=(None, None)):
		self._sig_p, _, _ = np.histogram2d(*signal, bins=self._bins, normed=True, weights=weights[0])
		self._bkg_p, _, _ = np.histogram2d(*background, bins=self._bins, normed=True, weights=weights[1])

		self._ratio = np.divide(self._sig_p, self._bkg_p)
		self._ratio[np.isinf(self._ratio)] = np.isfinite(self._ratio).max()
		self._ratio[np.isnan(self._ratio)] = 1
		return self

	def predict(self, X):
		try:
			return self._ratio[self._bins[0].searchsorted(X[0]) - 1, self._bins[1].searchsorted(X[1]) - 1]
		except IndexError:
			if not hasattr(X, '__iter__'):
				return 1.0
			return np.array([1.0 for _ in X])
		