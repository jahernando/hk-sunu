"""Counting statistics with Gaussian background uncertainties."""

import numpy as np
from scipy.special import gammaln


def sample_background(components, rel_errors, n_toys, rng):
    """Toy total backgrounds, varying each component independently."""
    components = np.asarray(components, dtype=float)
    rel_errors = np.asarray(rel_errors, dtype=float)
    shifts = rng.normal(size=(n_toys, components.size))
    toys = components * (1.0 + rel_errors * shifts)
    return np.clip(toys, 0.0, None).sum(axis=1)


def p_value(observed, components, rel_errors, n_toys=200_000, seed=1):
    """Probability of observing at most `observed` events from background alone."""
    rng = np.random.default_rng(seed)
    means = sample_background(components, rel_errors, n_toys, rng)
    return np.mean(rng.poisson(means) <= observed)


def upper_limit(observed, components, rel_errors, cl=0.90, n_toys=20_000, seed=1):
    """Bayesian upper limit on the signal, with a flat prior for signal >= 0.

    `observed` may be non-integer, which allows the expected sensitivity to be
    evaluated at the nominal background.
    """
    rng = np.random.default_rng(seed)
    means = sample_background(components, rel_errors, n_toys, rng)
    span = observed + np.sum(components) + 10.0 * np.sqrt(observed + np.sum(components) + 1.0)
    signal = np.linspace(0.0, span, 4001)
    mean = signal[:, np.newaxis] + means[np.newaxis, :]
    mean = np.clip(mean, 1e-12, None)
    log_likelihood = observed * np.log(mean) - mean - gammaln(observed + 1.0)
    posterior = np.exp(log_likelihood).mean(axis=1)
    cumulative = np.cumsum(posterior)
    cumulative /= cumulative[-1]
    return np.interp(cl, cumulative, signal)
