"""Classifiers (task 07): A = intercept-gradient baseline, B/C = windowed classifier on synthetic libraries."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier

from avo.features import intercept_gradient, window_features


class FullProba:
    """Wraps a fitted sklearn classifier so predict_proba always returns all n_classes columns."""

    def __init__(self, model, n_classes):
        self.model, self.n_classes = model, n_classes

    def predict_proba(self, X):
        p = np.zeros((len(X), self.n_classes))
        p[:, self.model.classes_.astype(int)] = self.model.predict_proba(X)
        return p


def _balanced_indices(y, seed):
    """Resample so each class contributes equally (fast alternative to class_weight='balanced')."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = len(y) // len(classes)
    return np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=True) for c in classes])


def train_window_model(X, y, n_classes, cfg, seed):
    """Fit the windowed classifier on class-balanced samples. Classes too rare to learn are dropped."""
    m = cfg["model"]
    counts = np.bincount(y, minlength=n_classes)
    keep = np.flatnonzero(np.isin(y, np.flatnonzero(counts >= m["min_class_samples"])))
    idx = keep[_balanced_indices(y[keep], seed)]
    clf = HistGradientBoostingClassifier(max_iter=m["max_iter"], learning_rate=m["learning_rate"],
                                         max_leaf_nodes=m["max_leaf_nodes"], early_stopping=False,
                                         random_state=seed)
    return FullProba(clf.fit(X[idx], y[idx]), n_classes)


def train_ig_baseline(I, G, y, n_classes):
    return FullProba(LinearDiscriminantAnalysis().fit(np.column_stack([I, G]), y), n_classes)


def predict_window(model, traces, half, ig_op, centers=None):
    centers = np.arange(traces.shape[1]) if centers is None else centers
    return model.predict_proba(window_features(traces, centers, half, ig_op))


def predict_ig(model, traces, ig_op, centers=None):
    ig = intercept_gradient(traces, ig_op)
    centers = np.arange(traces.shape[1]) if centers is None else centers
    return model.predict_proba(ig[:, centers].T)


def entropy(p):
    q = np.clip(p, 1e-12, 1)
    return -(q * np.log(q)).sum(axis=-1) / np.log(p.shape[-1])
