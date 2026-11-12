"""Super-Kamiokande with gadolinium (SK-VI, SK-VII and later), scaled from SK-IV.

The SK-Gd DSNB analysis (arXiv:2511.02222) does not publish the expected
backgrounds per bin in numbers, only in a figure. The Gd phases are therefore
built from the SK-IV expectations (Fig. 24 of arXiv:2109.11174) with the same
rules as the HK projection (`hk.project`):

- the signal and the backgrounds with a real neutron scale with the ratio of
  the signal efficiency after all cuts, SK-Gd (Table 3, NN tagging) over SK-IV
  (Fig. 18, one tagged neutron);
- the accidentals scale with the ratio of the misidentification rate per
  prompt event, 0.02% for the NN tagging (Sec. V.3.3) over the SK-IV rate per
  bin, inferred as accidentals (Fig. 24) / events reaching the tagging step
  (Fig. 19).

The spallation cut, positron identification and energy resolution are kept as
in SK-IV. This is a first-order model of SK-Gd, good enough to compare
sensitivities, not to reproduce the SK-Gd limits.
"""

import numpy as np

from . import hk, sk4

FIDUCIAL_KTON = 22.5

# arXiv:2511.02222, Table 3: signal efficiency after all cuts, NN tagging, in %,
# for the 11 bins of E_rec = 7.5-29.5 MeV.
EFFICIENCY_NN = {
    "SK-VI": np.array([15.1, 15.9, 18.8, 26.0, 25.5, 30.0, 32.8, 32.3, 37.2, 36.9, 37.1]),
    "SK-VII": np.array([21.9, 20.8, 27.2, 33.4, 40.1, 45.2, 52.4, 52.7, 55.5, 56.1, 55.8]),
}
GD_FRACTION = {"SK-VI": 0.011, "SK-VII": 0.033}  # wt%, Table 1
MISTAG_PER_EVENT_NN = 2e-4  # Sec. V.3.3, "0.02% per prompt event"

# Table 5: relative background uncertainties (SK-VI / SK-VII)
BACKGROUND_REL_ERROR = {
    "SK-VI": {"non_ncqe": 0.36, "ncqe": 0.71, "li9": 0.55, "reactor": 1.00, "accidental": 0.05},
    "SK-VII": {"non_ncqe": 0.41, "ncqe": 0.71, "li9": 0.55, "reactor": 1.00, "accidental": 0.05},
}

# Livetimes in days. SK-VI and SK-VII from arXiv:2511.02222 (Table 1). The
# totals come from the SK press release of 25 June 2026 (Neutrino 2026):
# 3349 days of pure water (2008-2020) and 1653 days with Gd (2020 to June 2026).
LIVETIME_DAYS = {"SK-VI": 552.2, "SK-VII": 404.0}
PURE_WATER_DAYS_2008_2020 = 3349.0
GD_DAYS_TO_JUNE_2026 = 1653.0
GD_DAYS_PER_YEAR = GD_DAYS_TO_JUNE_2026 / 6.0  # mid-2020 to mid-2026, about 275


def sk4_mistag_per_event():
    """SK-IV probability per event of accepting a noise candidate, per bin.

    Accidentals expected in Fig. 24 divided by the events reaching the
    neutron-tagging step in Fig. 19. The SK-IV BDT working point changes from
    bin to bin, so this ranges from 6e-4 to 0.14.
    """
    window = sk4.spectrum()[:11]
    reaching = sk4.candidates()[:11]["after_positron_id"]
    return window["accidental"] / reaching


def scenario(phase):
    """`hk.NtagScenario` that turns SK-IV into an SK-Gd phase."""
    sk4_eff = sk4.efficiencies()[:11]["ntag_one"]
    r = EFFICIENCY_NN[phase] / sk4_eff
    k = MISTAG_PER_EVENT_NN / sk4_mistag_per_event()
    return hk.NtagScenario(f"{phase} ({GD_FRACTION[phase]}% Gd)", r, k,
                           "SK-IV scaled to the SK-Gd NN efficiency and mistag rate")


def project(livetime_days, phase):
    """Expected SK-Gd spectrum per 2-MeV bin of 7.5-29.5 MeV for a livetime."""
    exposure = FIDUCIAL_KTON * livetime_days / 365.25
    return hk.project(exposure, scenario(phase), spall_scale=1.0)


def rel_errors(phase):
    """Background uncertainties of a phase, in the order of `sk4.BACKGROUNDS`."""
    return np.array([BACKGROUND_REL_ERROR[phase][key] for key in sk4.BACKGROUNDS])


def livetimes_at(year):
    """SK livetime in days per phase type at a calendar `year` (>= 2026.5).

    Pure water is SK-IV and SK-V (frozen at 3349 days). SK-VI is complete. The
    0.03% Gd phases (SK-VII onward) grow at the 2020-2026 average rate of about
    275 livetime days per calendar year from June 2026. How long SK runs once
    HK starts has not been checked in the sources; it is a scenario here.
    """
    sk_vii = GD_DAYS_TO_JUNE_2026 - LIVETIME_DAYS["SK-VI"] + max(year - 2026.5, 0.0) * GD_DAYS_PER_YEAR
    return {"pure_water": PURE_WATER_DAYS_2008_2020, "SK-VI": LIVETIME_DAYS["SK-VI"], "SK-VII": sk_vii}
