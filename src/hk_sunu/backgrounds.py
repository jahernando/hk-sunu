"""Analytic background shapes."""

import numpy as np

MICHEL_ENDPOINT_MEV = 52.8


def michel_spectrum(e):
    """Michel electron spectrum normalised to unit area, in MeV^-1."""
    e = np.asarray(e, dtype=float)
    x = e / MICHEL_ENDPOINT_MEV
    shape = 2.0 * x**2 * (3.0 - 2.0 * x) / MICHEL_ENDPOINT_MEV
    return np.where((x >= 0.0) & (x <= 1.0), shape, 0.0)
