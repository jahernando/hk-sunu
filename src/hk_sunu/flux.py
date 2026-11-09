"""Parametric DSNB anti-nu_e flux.

Implements Eq. (1) of the Super-Kamiokande DSNB analysis (arXiv:2109.11174) for
a single supernova class with a Fermi-Dirac emission spectrum. The temperature
is effective, so it absorbs oscillation effects, and pinching is neglected.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import trapezoid

C_KM_S = 299792.458
MPC_CM = 3.0857e24
YEAR_S = 3.15576e7
ERG_MEV = 6.2415e5
FD_NORM = 120.0 / (7.0 * np.pi**4)


@dataclass(frozen=True)
class Cosmology:
    """Flat Lambda-CDM background."""

    h0: float = 70.0  # km/s/Mpc
    omega_m: float = 0.3

    @property
    def hubble_distance_mpc(self):
        return C_KM_S / self.h0

    def e_of_z(self, z):
        return np.sqrt(self.omega_m * (1.0 + z) ** 3 + 1.0 - self.omega_m)


@dataclass(frozen=True)
class SupernovaRate:
    """Comoving core-collapse rate from a broken power-law star formation rate.

    Defaults follow the Yuksel et al. (2008) parametrisation, scaled by the
    number of core-collapse progenitors per unit of stellar mass formed.
    """

    sfr0: float = 0.0178  # Msun / yr / Mpc^3
    progenitors_per_msun: float = 0.0088  # Salpeter initial mass function
    alpha: float = 3.4
    beta: float = -0.3
    gamma: float = -3.5
    b: float = 5000.0
    c: float = 9.0
    eta: float = -10.0

    def __call__(self, z):
        """Rate in SN / yr / Mpc^3 at redshift `z`."""
        x = 1.0 + np.asarray(z, dtype=float)
        shape = (
            x ** (self.alpha * self.eta)
            + (x / self.b) ** (self.beta * self.eta)
            + (x / self.c) ** (self.gamma * self.eta)
        ) ** (1.0 / self.eta)
        return self.progenitors_per_msun * self.sfr0 * shape


@dataclass(frozen=True)
class FermiDirac:
    """Time-integrated emission of one flavour, in neutrinos per MeV."""

    temperature: float = 6.0  # MeV
    total_energy_erg: float = 5e52

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return (
            self.total_energy_erg
            * ERG_MEV
            * FD_NORM
            * e**2
            / self.temperature**4
            / (np.exp(e / self.temperature) + 1.0)
        )


@dataclass(frozen=True)
class DSNBFlux:
    """Differential DSNB flux of one flavour at Earth."""

    spectrum: FermiDirac = field(default_factory=FermiDirac)
    rate: SupernovaRate = field(default_factory=SupernovaRate)
    cosmology: Cosmology = field(default_factory=Cosmology)
    z_max: float = 5.0
    n_z: int = 2001

    def __call__(self, e_nu):
        """Flux in cm^-2 s^-1 MeV^-1 for neutrino energy `e_nu` in MeV."""
        e_nu = np.asarray(e_nu, dtype=float)
        z = np.linspace(0.0, self.z_max, self.n_z)
        weight = self.rate(z) / self.cosmology.e_of_z(z)
        emitted = self.spectrum(e_nu[..., np.newaxis] * (1.0 + z))
        integral = trapezoid(weight * emitted, z, axis=-1)
        return integral * self.cosmology.hubble_distance_mpc / MPC_CM**2 / YEAR_S
