"""The LSM's one trained part: a linear readout of the liquid state. Standardisation and the
classifier are fit on the training split only (no leakage into the test score)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_and_score(train_x, train_y, test_x, test_y, c: float, max_iter: int) -> dict:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=c, max_iter=max_iter).fit(scaler.transform(train_x), train_y)
    return {"train_accuracy": float(model.score(scaler.transform(train_x), train_y)),
            "test_accuracy": float(model.score(scaler.transform(test_x), test_y)),
            "n_train": int(len(train_y)), "n_test": int(len(test_y)), "n_features": int(train_x.shape[1])}
