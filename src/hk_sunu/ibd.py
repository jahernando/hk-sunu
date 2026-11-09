"""Inverse beta decay on free protons: anti-nu_e + p -> e+ + n.

Uses the Strumia-Vissani approximation (Phys. Lett. B 564 (2003) 42), with the
positron energy taken at zero nucleon recoil, E_e = E_nu - DELTA.
"""

import numpy as np

DELTA_MEV = 1.293  # m_n - m_p
M_E_MEV = 0.511
THRESHOLD_MEV = 1.806


def cross_section(e_nu):
    """IBD cross section in cm^2, for neutrino energy `e_nu` in MeV."""
    e_nu = np.asarray(e_nu, dtype=float)
    sigma = np.zeros_like(e_nu)
    above = e_nu > THRESHOLD_MEV
    e = e_nu[above]
    e_pos = e - DELTA_MEV
    p_pos = np.sqrt(e_pos**2 - M_E_MEV**2)
    log_e = np.log(e)
    exponent = -0.07056 + 0.02018 * log_e - 0.001953 * log_e**3
    sigma[above] = 1e-43 * p_pos * e_pos * e**exponent
    return sigma


def enu_from_erec(e_rec):
    """Neutrino energy in MeV for positron kinetic energy `e_rec` in MeV.

    This is the convention behind the SK windows: E_rec = 15.5 MeV corresponds
    to E_nu = 17.3 MeV.
    """
    return np.asarray(e_rec, dtype=float) + DELTA_MEV + M_E_MEV


def erec_from_enu(e_nu):
    """Positron kinetic energy in MeV for neutrino energy `e_nu` in MeV."""
    return np.asarray(e_nu, dtype=float) - DELTA_MEV - M_E_MEV
