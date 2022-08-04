class Scaler:
    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class IdentityScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse(self, x):
        return x


class NormalScaler(Scaler):
    def __init__(self):
        self.mus = None
        self.stds = None

    def fit(self, x):
        self.mus = x.mean(axis=0)
        self.stds = x.std(axis=0)

    def transform(self, x):
        return (x - self.mus) / self.stds

    def inverse(self, x):
        return (x * self.stds) + self.mus


class ZeroOneScaler(Scaler):
    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
        self.range = self.max - self.min

    def transform(self, x):
        return (x - self.min) / self.range

    def inverse(self, x):
        return (x * self.range) + self.min
