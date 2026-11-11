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


def asimov_significance(signal, background, sigma_b=0.0):
    """Median discovery significance for a counting experiment.

    Asimov approximation with an absolute background uncertainty `sigma_b`
    (Cowan, Cranmer, Gross and Vitells, arXiv:1007.1727, and Cowan's note on
    discovery significance with background uncertainty).
    """
    s, b = float(signal), float(background)
    if s <= 0.0 or b <= 0.0:
        return 0.0
    if sigma_b <= 0.0:
        return np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s))
    v = sigma_b**2
    term1 = (s + b) * np.log((s + b) * (b + v) / (b**2 + (s + b) * v))
    term2 = b**2 / v * np.log1p(v * s / (b * (b + v)))
    return np.sqrt(2.0 * (term1 - term2))


def binned_significance(signal, components, rel_errors):
    """Median discovery significance of a binned shape analysis (Asimov data).

    `signal` has one entry per bin and `components` one row per background.
    Each background normalisation is a nuisance parameter with a Gaussian
    constraint `rel_errors`, fully correlated across bins, so the spectral
    shapes can constrain the backgrounds as in a spectral fit.
    """
    from scipy.optimize import minimize

    signal = np.asarray(signal, dtype=float)
    components = np.atleast_2d(np.asarray(components, dtype=float))
    rel_errors = np.asarray(rel_errors, dtype=float)
    observed = signal + components.sum(axis=0)

    def q(theta):
        mean = np.clip(((1.0 + theta)[:, np.newaxis] * components).sum(axis=0), 1e-12, None)
        poisson = 2.0 * np.sum(mean - observed + observed * np.log(observed / mean))
        return poisson + np.sum((theta / rel_errors) ** 2)

    bounds = [(-0.99, None)] * len(rel_errors)
    best = minimize(q, np.zeros(len(rel_errors)), bounds=bounds, method="L-BFGS-B")
    return float(np.sqrt(max(best.fun, 0.0)))
