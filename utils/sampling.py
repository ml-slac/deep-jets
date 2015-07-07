import numpy as np

'''
Functionality to have weighted sampling for training 
and testing -- i.e., matching pT distributions.
'''

class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = np.cumsum(p)

    def sample(self, k=1):
        rs = np.random.random(k)
        # binary search to get indices
        return np.searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = np.zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p


def multinomial_sample(p, k=1):
    """
    Wrapper to generate a k samples,
    using the above class.
    """
    return MultinomialSampler(p).sample(k)


class WeightedDataset(object):
    """docstring for WeightedDataset"""
    def __init__(self, X, y=None, weights=None, copy = True):
        if copy:
            self._X = X.copy()
            if not y is None:
                self._y = y.copy()
            else:
                self._y = None
        else:
            self._X = X
            self._y = y

        if weights is None:
            weights = np.ones(X.shape[0])


        if not type(weights) in [np.array, list, np.ndarray]:
            raise TypeError('weights must be a numpy array or a list')

        if len(weights) != X.shape[0]:
            raise ValueError('weights must have the same length as the first axis of X')
        
        if type(weights) is list:
            self._weights = np.array(weights)
        else:
            self._weights = weights

        self._weights[self._weights < 0] = 0.0

        self._weights = self._weights / np.sum(self._weights)

    def sample_idx(self, n):
        return multinomial_sample(self._weights, n)

    def sample(self, n):
        ix = multinomial_sample(self._weights, n)
        if self._y is None:
            return self._X[ix]
        else:
            return self._X[ix], self._y[ix]









