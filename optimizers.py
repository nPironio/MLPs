import numpy as np
from collections import deque


class BackPropagation:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def compute_direction(self, activations, target):
        deltas = []
        error = (target - activations[-1]).mean(axis=0)
        derivative = self.model.activation.derivative(activations[-1])
        delta = derivative * error
        for w_idx, (W, activation) in enumerate(zip(self.model.weights[::-1], activations[::-1][1:])):
            deltas.append(self.weights_delta(w_idx, np.matmul(activation.T, delta)))
            error = np.matmul(delta, W.T)
            derivative = self.model.activation.derivative(activation)
            delta = self.model.remove_bias(derivative * error)

        return deltas[::-1]

    def compute_deltas(self, activations, target):
        return self.compute_direction(activations, target)

    def weights_delta(self, weight_index, error_variation_on_layer):
        return self.lr * error_variation_on_layer

    def lr_update(self, error):
        pass


class MomentumAdaptativeBackProp(BackPropagation):
    def __init__(self, momentum_alpha=0.9, lr_increase=0.001, lr_decrease=0.9, consistency_K=5, **kwargs):
        super().__init__(**kwargs)

        self.momentum_alpha = momentum_alpha
        self.last_delta = [np.zeros_like(W) for W in self.model.weights]

        self.lr_increase = lr_increase
        self.lr_decrease = lr_decrease
        self.error_window = deque([np.inf] * consistency_K, maxlen=int(consistency_K))
        self.last_error = 0

    def compute_deltas(self, activations, target):
        deltas = self.compute_direction(activations, target)
        self.last_delta = deltas
        return deltas

    def weights_delta(self, weight_index, error_variation_on_layer):
        return self.lr * error_variation_on_layer + self.momentum_alpha * self.last_delta[-(weight_index+1)]

    def lr_update(self, error):
        error_delta = error - self.last_error
        self.last_error = error
        self.error_window.append(error_delta)
        error_delta_mean = np.mean(self.error_window)
        if error_delta_mean < 0:
            self.lr += self.lr_increase
        elif error_delta_mean > 0:
            self.lr -= self.lr_decrease * self.lr
