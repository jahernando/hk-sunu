"""Raw IBD event rates: no detection efficiency, resolution or backgrounds."""

import numpy as np
from scipy.integrate import trapezoid

from . import ibd
from .flux import YEAR_S


def spectrum_per_year(flux, detector, e_nu):
    """IBD events per year per MeV of neutrino energy.

    `flux` is any callable returning cm^-2 s^-1 MeV^-1 for energies in MeV.
    """
    return flux(e_nu) * ibd.cross_section(e_nu) * detector.free_protons * YEAR_S


def integrated_flux(flux, e_min, e_max, n=4001):
    """Flux in cm^-2 s^-1 integrated between neutrino energies in MeV."""
    e_nu = np.linspace(e_min, e_max, n)
    return trapezoid(flux(e_nu), e_nu)


def events_per_year(flux, detector, e_min, e_max, n=4001):
    """IBD events per year between neutrino energies in MeV."""
    e_nu = np.linspace(e_min, e_max, n)
    return trapezoid(spectrum_per_year(flux, detector, e_nu), e_nu)
