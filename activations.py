import numpy as np


class Activation:

    def __init__(self):
        self.fn = None

    def __call__(self, x):
        return self.fn(x)

    def derivative(self, activation):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class SignActivation(Activation):
    def __init__(self):
        super().__init__()
        self.fn = np.sign

    def derivative(self, activation):
        return np.zeros_like(activation)

    def predict(self, x):
        return x


class SigmoidActivation(Activation):
    def __init__(self):
        super().__init__()
        self.fn = np.tanh

    def derivative(self, activation):
        return 1 - np.square(activation)

    def predict(self, x, threshold=0):
        return np.where((x > 0), 1, -1)


class RectifiedLinearActivation(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        mask = x >= 0
        return x * mask

    def derivative(self, activation):
        return (activation >= 0).astype(np.float32)

    def predict(self, x):
        return x


class LinearActivation(Activation):
    def __init__(self):
        super().__init__()
        self.fn = lambda x: x

    def derivative(self, activation):
        return np.ones_like(activation)

    def predict(self, x):
        return x
