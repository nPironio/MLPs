from matplotlib import pyplot as plt
import numpy as np

import copy
from collections import defaultdict

import activations
import scalers
import optimizers


def set_choice(options_dict, name):
    try:
        return options_dict[name]()
    except:
        raise NameError(f"No está implementada la opción {name}")


activation_constructors = {"linear": activations.LinearActivation,
                           "sign": activations.SignActivation,
                           "sigmoid": activations.SigmoidActivation,
                           "relu": activations.RectifiedLinearActivation}

scaler_constructors = {"identity": scalers.IdentityScaler,
                       "zero-one": scalers.ZeroOneScaler,
                       "normal": scalers.NormalScaler}


class MLP:
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 activation=None, input_scaler="identity", output_scaler="identity", bias=True):
        self.global_step = 0
        self.bias = bias
        self.weights = self._initialize_weights([input_size, *hidden_sizes, output_size])
        self.input_size = input_size

        self.history = defaultdict(list)

        self.activation = set_choice(activation_constructors, activation)

        self.input_scaler = set_choice(scaler_constructors, input_scaler)
        self.output_scaler = set_choice(scaler_constructors, output_scaler)


    def _initialize_weights(self, sizes):
        weights = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            weights.append(np.random.normal(0, in_size ** (-1 / 2), (in_size + (1 if self.bias else 0), out_size)))
        return weights

    def __call__(self, x, classify=True):
        res = self.input_scaler.transform(x) if x.ndim > 1 else self.input_scaler.transform(x.reshape(1, -1))
        for W in self.weights:
            input_b = self.add_bias(res)
            res = self.activation(np.matmul(input_b, W))

        res = self.output_scaler.inverse(res)
        return self.activation.predict(res) if classify else res

    def add_bias(self, x):
        return np.append(x, np.ones((x.shape[0], 1)), axis=-1) if self.bias else x

    def remove_bias(self, x):
        return x[:, :-1] if self.bias else x

    def get_activations(self, x):
        hidden_activations = []
        res = x
        for W in self.weights:
            res_b = self.add_bias(res)
            hidden_activations.append(res_b)
            res = self.activation(np.matmul(res_b, W))
        hidden_activations.append(res)
        return hidden_activations

    def weights_update(self, weights_deltas):
        for i, delta in enumerate(weights_deltas):
            self.weights[i] += delta

    def error(self, y, y_hat):
        return np.square(np.linalg.norm(y - y_hat, axis=1)).mean()

    def fit(self, train_xs, train_ys, validation_xs=None, validation_ys=None, optimizer=None, lr=0.001,
            shuffle=False, batch_size=1, max_epochs=100, early_stopping_patience=np.inf, error_threshold=0,
            verbose=True):

        epoch = 0
        train_error = 0
        val_error = error_threshold + 1
        best_val_epoch, best_val_error, best_model_weights = 0, np.infty, copy.deepcopy(self.weights)

        train_split_num = int(np.ceil(len(train_xs) / batch_size))
        val_split_num = int(np.ceil(len(validation_xs) / batch_size))
        error_threshold = 0 if not early_stopping_patience else error_threshold

        train_xs = self.input_scaler.fit_transform(train_xs)
        train_ys = self.output_scaler.fit_transform(train_ys)

        validation_xs = self.input_scaler.transform(validation_xs)
        validation_ys = self.output_scaler.transform(validation_ys)

        if not optimizer:
            optimizer = optimizers.BackPropagation(self, lr)

        stop = False
        while epoch < max_epochs and not stop:
            # training step
            train_error = 0
            order = np.random.permutation(len(train_xs)) if shuffle else np.arange(len(train_xs))
            x_order, y_order = np.array_split(train_xs[order], train_split_num), np.array_split(train_ys[order],
                                                                                                train_split_num)

            for (x, y) in zip(x_order, y_order):
                x = x.reshape(-1, 1) if x.ndim == 1 else x
                y = y.reshape(-1, 1) if y.ndim == 1 else y
                activations = self.get_activations(x)
                deltas = optimizer.compute_deltas(activations, y)
                self.weights_update(deltas)
                step_error = self.error(y, activations[-1])
                train_error += step_error
                optimizer.lr_update(step_error)
                self.log("train step error", (self.global_step, step_error))
                self.global_step += 1

            # Validation step
            if validation_xs is not None:
                val_error = 0
                order = np.random.permutation(len(validation_xs)) if shuffle else np.arange(len(validation_xs))
                x_order, y_order = np.array_split(validation_xs[order], val_split_num), np.array_split(
                    validation_ys[order], val_split_num)

                for (x, y) in zip(x_order, y_order):
                    x = x.reshape(-1, 1) if x.ndim == 1 else x
                    y = y.reshape(-1, 1) if y.ndim == 1 else y
                    activations = self.get_activations(x)
                    step_error = self.error(y, activations[-1])
                    val_error += step_error
                    self.log("validation step error", (self.global_step, step_error))
                    self.global_step += 1

                if val_error < best_val_error:
                    if verbose:
                        print(f"Mejor validación encontrada en {epoch=}")
                    best_model_weights = copy.deepcopy(self.weights)
                    best_val_epoch = epoch
                    best_val_error = val_error
                elif epoch - best_val_epoch > early_stopping_patience:
                    if verbose:
                        print(
                            f"Hace {early_stopping_patience} que no mejora validación, actualizando pesos a los de la epoca {best_val_epoch}")
                    self.weights = best_model_weights
                    stop = True

                if val_error < error_threshold:
                    stop = True

                self.log("validation error", (epoch, val_error))

            if epoch % 1 == 0 and verbose:
                print(
                    f"{epoch=}, {train_error=}{', val_error=' + str(val_error) if validation_xs is not None else ''}\n")

            self.log("train error", (epoch, train_error))
            epoch += 1

        return best_val_epoch

    def log(self, name, value):
        self.history[name].append(value)

    def plot_history(self, keys, xlabel):
        for key in keys:
            epochs, error = zip(*self.history[key])
            plt.plot(epochs, error, label=key)
        plt.xlabel(xlabel)
        plt.legend()
