"""Charged-current neutrino reactions on oxygen-16.

Shell-model cross sections of Nakazato, Suzuki and Sakuda, PTEP 2018, 123E02
(arXiv:1809.08398), using the fit of their Eq. (4) and Table 4. The final states
are collected in four groups, each with one representative excitation energy.
"""

import numpy as np
from scipy.integrate import trapezoid

from .flux import YEAR_S
from .ibd import M_E_MEV

E_NU_MAX_MEV = 100.0  # range of validity of the fit

# Each row: representative excitation energy (MeV), a, b, c
GROUPS = {
    "nu_e": np.array(
        [
            [15.21, -40.008, 4.918, 1.036],
            [22.47, -39.305, 4.343, 0.961],
            [25.51, -39.655, 5.263, 1.236],
            [29.35, -39.166, 3.947, 0.901],
        ]
    ),
    "nubar_e": np.array(
        [
            [11.23, -40.656, 4.528, 0.887],
            [18.50, -40.026, 4.117, 0.895],
            [21.54, -40.060, 3.743, 0.565],
            [25.38, -39.862, 3.636, 0.846],
        ]
    ),
}


def group_cross_section(e_nu, group):
    """Partial cross section in cm^2 for one row of `GROUPS`.

    The quadratic fit turns upwards very close to threshold, so the cross
    section is set to zero below the vertex of the parabola.
    """
    e_nu = np.asarray(e_nu, dtype=float)
    e_x, a, b, c = group
    sigma = np.zeros_like(e_nu)
    gap = e_nu**0.25 - e_x**0.25
    above = (gap > 0.0) & (e_nu <= E_NU_MAX_MEV)
    log_gap = np.log10(gap[above])
    physical = log_gap > -b / (2.0 * c)
    values = np.zeros_like(log_gap)
    values[physical] = 10.0 ** (a + b * log_gap[physical] + c * log_gap[physical] ** 2)
    sigma[above] = values
    return sigma


def cross_section(e_nu, flavour="nu_e"):
    """Total cross section per oxygen nucleus in cm^2."""
    return sum(group_cross_section(e_nu, group) for group in GROUPS[flavour])


def oxygen_nuclei(detector):
    """Number of oxygen nuclei in the fiducial mass of a water detector."""
    return detector.free_protons / 2.0


def spectrum_per_year(flux, detector, e_kin, flavour="nu_e"):
    """Events per year per MeV of lepton kinetic energy.

    Each group contributes at E_nu = e_kin + m_e + E_x, without energy resolution.
    """
    e_kin = np.asarray(e_kin, dtype=float)
    total = np.zeros_like(e_kin)
    for group in GROUPS[flavour]:
        e_nu = e_kin + M_E_MEV + group[0]
        total += flux(e_nu) * group_cross_section(e_nu, group)
    return total * oxygen_nuclei(detector) * YEAR_S


def events_per_year(flux, detector, e_min, e_max, flavour="nu_e", n=4001):
    """Events per year between lepton kinetic energies in MeV."""
    e_kin = np.linspace(e_min, e_max, n)
    return trapezoid(spectrum_per_year(flux, detector, e_kin, flavour), e_kin)
