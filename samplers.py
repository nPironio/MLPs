import numpy as np


class Sampler:
    def sample(self, n=1):
        raise NotImplementedError


class NormalSampler(Sampler):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, n=1):
        sample = np.random.normal(loc=self.mu, scale=self.sigma, size=n)
        return sample if n > 1 else sample[0]


class DiscreteSampler(Sampler):
    def __init__(self, options, p=None):
        self.options = options
        self.p = p

    def sample(self, n=1):
        sample = np.random.choice(self.options, size=n, replace=True, p=self.p)
        return sample if n > 1 else sample[0]


class DecreasingList(Sampler):
    def __init__(self, length: DiscreteSampler, max_value, min_value=1, beta=0.1):
        self.length = length
        self.max_value = max_value
        self.min_value = min_value
        self.beta = beta

    def sample(self, n=1):
        return [self._create_list() for _ in range(n)] if n > 1 else self._create_list()

    def _create_list(self):
        length = self.length.sample()
        list = np.empty(length, dtype=np.int32)
        last_val = self.max_value
        for i in range(length):
            if last_val == self.min_value:
                list[i] = self.min_value
            else:
                probs = (self.beta ** np.arange(start=last_val - self.min_value, stop=self.min_value - 1, step=-1))
                last_val = np.random.choice(np.arange(start=self.min_value, stop=last_val), p=probs / probs.sum())
                list[i] = last_val

        return list
