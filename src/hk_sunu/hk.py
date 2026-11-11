"""Hyper-Kamiokande projections scaled from the SK-IV DSNB analysis.

The SK-IV expectations per 2-MeV bin (Fig. 24 of arXiv:2109.11174) are taken as
the reference and scaled to HK by exposure, by the neutron-tagging performance
and, for spallation, by the muon flux and energy at the HK site. Everything
else in the selection (cuts, energy resolution, positron identification) is
assumed to be the same as in SK-IV. This is a first-order projection, not a
detector simulation.
"""

from dataclasses import dataclass

import numpy as np

from . import sk4

# Detector and site parameters (one tank). Sources are listed in NB02.
FIDUCIAL_KTON = 187.0  # 1.5 m from the ID walls (arXiv:2101.05269); 188 in PoS(ICRC2025)1215
MUON_FLUX = {"HK": 7.55e-7, "SK": 1.54e-7}  # cm^-2 s^-1, Design Report Table XXIX (MUSIC)
MUON_MEAN_ENERGY_GEV = {"HK": 203.0, "SK": 258.0}  # Design Report Table XXIX

# Backgrounds whose events carry a real neutron: they scale with the tagging efficiency.
REAL_NEUTRON = ("non_ncqe", "ncqe", "reactor", "li9")


@dataclass(frozen=True)
class NtagScenario:
    """Neutron tagging in HK relative to SK-IV, at the SK-IV working points.

    `efficiency_scale` multiplies the per-bin probability of tagging exactly one
    true neutron. `mistag_scale` multiplies the per-event probability that a
    noise fluctuation is accepted as a neutron, so it scales the accidentals.
    """

    name: str
    efficiency_scale: float
    mistag_scale: float
    note: str = ""


def spallation_scale(muon_flux_ratio, muon_energy_ratio, energy_power=0.75):
    """Isotope production per unit volume in HK relative to SK.

    The yield per muon grows as <E_mu>^alpha with alpha ~ 0.75 (Li and Beacom,
    arXiv:1402.4687), so the production rate scales as flux x <E_mu>^alpha.
    The spallation cuts are assumed to keep the same survival fraction.
    """
    return muon_flux_ratio * muon_energy_ratio**energy_power


def sk4_exposure_kton_yr():
    """SK-IV exposure per 2-MeV bin of the 7.5-29.5 MeV window, in kton yr.

    The lowest bin only uses the period with the 58-hit trigger.
    """
    days = np.full(11, sk4.LIVETIME_DAYS)
    days[0] = sk4.LIVETIME_LOW_DAYS
    return sk4.FIDUCIAL_KTON * days / 365.25


def sk4_spallation_cut_efficiency():
    """Signal efficiency of the SK-IV spallation cut per bin of 7.5-29.5 MeV.

    Fig. 18 gives the cumulative efficiency after noise reduction and the
    spallation cut. Above 23.5 MeV there is no spallation cut and the curve sits
    on a 97.8% plateau, which is taken as the noise-reduction efficiency.
    """
    eff = sk4.efficiencies()[:11]["spallation"] / 100.0
    plateau = eff[-3:].mean()
    return np.clip(eff / plateau, 0.0, 1.0)


def spallation_cut_efficiency(spall_scale):
    """Signal efficiency of the same spallation cut with `spall_scale` x more muons.

    The cut vetoes regions around preceding muons, so its loss on uncorrelated
    events is Poissonian, eff = exp(-lambda), and lambda grows with the muon
    rate: eff_HK = eff_SK ** spall_scale. With a factor 5 this turns the SK
    efficiencies of 79% and 90% into 31% and 59%, close to the 29% and 54% of
    the HK Design Report (arXiv:1805.04163).
    """
    return sk4_spallation_cut_efficiency() ** spall_scale


def project(exposure_kton_yr, scenario, spall_scale, spallation_dominated_below=15.5,
            signal_key="dsnb_horiuchi09"):
    """Expected signal and backgrounds in HK, per 2-MeV bin of 7.5-29.5 MeV.

    - Everything scales with exposure.
    - The same spallation cut costs more efficiency, by eff_SK ** (spall_scale - 1),
      on signal and on every background.
    - The residual spallation (9Li) scales with `spall_scale`.
    - Events with a real neutron scale with the tagging efficiency.
    - Accidentals are proportional to the events reaching the tagging step.
      Below `spallation_dominated_below` (MeV) that sample is dominated by
      spallation and solar events, and it is conservatively scaled by
      `spall_scale`.

    Returns a dict of arrays with the same keys as the SK-IV spectrum.
    """
    window = sk4.spectrum()[:11]
    x = exposure_kton_yr / sk4_exposure_kton_yr()
    r, k = scenario.efficiency_scale, scenario.mistag_scale
    veto = sk4_spallation_cut_efficiency() ** (spall_scale - 1.0)
    pre_ntag = np.where(window["erec_hi"] <= spallation_dominated_below, spall_scale, 1.0)

    out = {"erec_lo": window["erec_lo"], "erec_hi": window["erec_hi"]}
    out["signal"] = x * veto * r * window[signal_key]
    for key in REAL_NEUTRON:
        out[key] = x * veto * r * window[key]
    out["li9"] = out["li9"] * spall_scale
    out["accidental"] = x * veto * k * pre_ntag * window["accidental"]
    out["total"] = sum(out[key] for key in sk4.BACKGROUNDS)
    return out
