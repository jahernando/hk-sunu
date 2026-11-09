"""Cherenkov emission of charged particles in water."""

import numpy as np
from scipy.integrate import trapezoid

REFRACTIVE_INDEX = 1.34  # blue light seen by the PMTs
MASS_MEV = {"e": 0.511, "mu": 105.658, "pi": 139.57, "p": 938.272}


def threshold_momentum(mass, n=REFRACTIVE_INDEX):
    """Momentum in MeV/c at which Cherenkov emission starts."""
    return mass / np.sqrt(n**2 - 1.0)


def threshold_kinetic_energy(mass, n=REFRACTIVE_INDEX):
    """Kinetic energy in MeV at which Cherenkov emission starts."""
    return np.hypot(threshold_momentum(mass, n), mass) - mass


def angle_deg(momentum, mass, n=REFRACTIVE_INDEX):
    """Cherenkov angle in degrees, NaN below threshold."""
    momentum = np.asarray(momentum, dtype=float)
    beta = momentum / np.hypot(momentum, mass)
    cosine = 1.0 / (n * beta)
    return np.degrees(np.arccos(np.where(cosine < 1.0, cosine, np.nan)))


def electron_equivalent_energy(momentum, mass, n=REFRACTIVE_INDEX):
    """Energy in MeV of an electron emitting the same Cherenkov light.

    Integrates sin^2(theta) over the track, relative to a fully relativistic
    particle, assuming the same constant dE/dx for every particle. Hadronic
    interactions and the decay electron of a stopping muon are ignored.
    """
    t_threshold = threshold_kinetic_energy(mass, n)
    t_final = np.hypot(momentum, mass) - mass
    if t_final <= t_threshold:
        return 0.0
    kinetic = np.linspace(t_threshold, t_final, 5001)
    beta_squared = 1.0 - (mass / (kinetic + mass)) ** 2
    light = np.clip(1.0 - 1.0 / (n**2 * beta_squared), 0.0, None)
    return trapezoid(light, kinetic) / (1.0 - 1.0 / n**2)
