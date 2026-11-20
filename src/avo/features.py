"""Feature extraction shared by the synthetic library and observed seismic."""

import numpy as np


def ig_operator(stack_defs, n_angles=5):
    """Least-squares operator (2, n_stacks) mapping partial-stack amplitudes to intercept and gradient."""
    x = [np.mean(np.sin(np.radians(np.linspace(s.angle_min, s.angle_max, n_angles))) ** 2) for s in stack_defs]
    A = np.column_stack([np.ones(len(x)), x])
    return np.linalg.pinv(A)


def intercept_gradient(traces, ig_op):
    return ig_op @ traces


def window_features(traces, centers, half, ig_op):
    """Windowed near/mid/far amplitudes plus intercept and gradient around each centre sample."""
    n = traces.shape[1]
    idx = np.clip(np.asarray(centers)[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
    ig = intercept_gradient(traces, ig_op)
    amp = traces[:, idx].transpose(1, 0, 2).reshape(len(centers), -1)
    igw = ig[:, idx].transpose(1, 0, 2).reshape(len(centers), -1)
    return np.hstack([amp, igw])
