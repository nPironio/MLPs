from typing import Dict

import numpy as np

from optimizers import MomentumAdaptativeBackProp
from samplers import Sampler

def create_folds(xs, ys, num_folds):
    random_order = np.random.permutation(xs.shape[0])
    folds_idxs = np.array_split(random_order, num_folds)
    folds = []
    for i in range(num_folds):
        train_idxs = np.concatenate([f for j, f in enumerate(folds_idxs) if j!=i])
        val_idxs = folds_idxs[i]
        folds.append((xs[train_idxs], ys[train_idxs], xs[val_idxs], ys[val_idxs]))

    return folds


def random_search(xs, ys,
                  model_class, model_hparams: Dict[str, Sampler], optimizer_hparams: Dict[str, Sampler], training_hparams: Dict[str, Sampler],
                  num_folds=5, seed=23, num_iterations=10):
    # TODO: check that the same combination is not tried more than once

    input_size = xs.shape[1]
    target_size = ys.shape[1]
    folds = create_folds(xs, ys, num_folds)
    trained_models = []
    for iteration in range(num_iterations):
        print(f"{iteration=}")
        model_params_sample = {key: sampler.sample() for key, sampler in model_hparams.items()}
        optimizer_params_sample = {key: sampler.sample() for key, sampler in optimizer_hparams.items()}
        training_params_sample = {key: sampler.sample() for key, sampler in training_hparams.items()}
        scores = []
        best_epochs = []
        best_model = None

        for x_train, y_train, x_val, y_val in folds:
            model = model_class(input_size, target_size, **model_params_sample)
            optimizer = MomentumAdaptativeBackProp(model=model, **optimizer_params_sample)
            best_epoch = model.fit(x_train, y_train, x_val, y_val, optimizer=optimizer, shuffle=True, verbose=False, **training_params_sample)
            score = model.error(y_val, model(x_val, classify=False))
            scores.append(score)
            best_model = model if score == min(scores) else best_model
            best_epochs.append(best_epoch)

        trained_models.append((np.mean(scores),
                               model_params_sample,
                               optimizer_params_sample,
                               training_params_sample,
                               np.mean(best_epochs),
                               best_model
                               ))

    return sorted(trained_models, key=lambda x: x[0])
