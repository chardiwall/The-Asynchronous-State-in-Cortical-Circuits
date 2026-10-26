"""The LSM's one trained part: a linear readout of the liquid state. Standardisation, the
choice of regularisation and the classifier all see the training split only: C is picked
on a held-out validation slice OF THE TRAINING DATA, then the model is refit on all of it."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fit(x, y, c, max_iter):
    scaler = StandardScaler().fit(x)
    return scaler, LogisticRegression(C=c, max_iter=max_iter).fit(scaler.transform(x), y)


def train_and_score(train_x, train_y, test_x, test_y, c, max_iter: int,
                    validation_fraction: float = 0.2, rng=None) -> dict:
    grid = list(np.atleast_1d(c))
    validation = {}
    if len(grid) > 1:
        order = (rng or np.random.default_rng(0)).permutation(len(train_y))
        cut = int(len(order) * validation_fraction)
        held, fit = order[:cut], order[cut:]
        for value in grid:
            scaler, model = _fit(train_x[fit], train_y[fit], value, max_iter)
            validation[float(value)] = float(model.score(scaler.transform(train_x[held]), train_y[held]))
    best = max(validation, key=validation.get) if validation else float(grid[0])
    scaler, model = _fit(train_x, train_y, best, max_iter)
    return {"C": best, "validation_accuracy": validation.get(best, float("nan")),
            "train_accuracy": float(model.score(scaler.transform(train_x), train_y)),
            "test_accuracy": float(model.score(scaler.transform(test_x), test_y)),
            "n_train": int(len(train_y)), "n_test": int(len(test_y)), "n_features": int(train_x.shape[1])}
