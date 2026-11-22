"""Source-only class means and regularized shared-covariance distance.

The minimum-distance expansion follows the previously audited CESNET geometry
controls. This is an established scoring baseline, not a proposed new method.
"""
import numpy as np


def fit_geometry(embedding, labels, classes, ridge_fraction):
    values = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels)
    if values.ndim != 2 or not np.isfinite(values).all() or len(labels) != len(values) or ridge_fraction <= 0:
        raise ValueError('Invalid source geometry inputs')
    if not np.isin(labels, np.arange(classes)).all() or any(not np.any(labels == c) for c in range(classes)):
        raise ValueError('Geometry fitting requires all source classes and no unknown labels')
    centers = np.stack([values[labels == c].mean(0) for c in range(classes)])
    residual = values - centers[labels]
    covariance = residual.T @ residual / len(residual)
    ridge = max(ridge_fraction * float(np.trace(covariance)) / len(covariance), 1e-8)
    precision = np.linalg.inv(covariance + ridge * np.eye(len(covariance)))
    return dict(centers=centers, precision=precision, covariance=covariance,
                ridge=np.asarray(ridge), fit_rows=np.asarray(len(values)))


def distance(emb, centers, precision):
    x = emb.astype(np.float64)
    values = (np.sum((x@precision)*x, 1)[:, None]
              + np.sum((centers@precision)*centers, 1)[None, :]
              - 2*(x@precision)@centers.T)
    return np.maximum(values.min(1), 0)
