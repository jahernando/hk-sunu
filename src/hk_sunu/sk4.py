"""Published inputs of the SK-IV DSNB analysis (arXiv:2109.11174v2).

Tables and quoted numbers are transcribed from the paper. The CSV files in
`data/` hold bin contents recovered from the vector graphics of Figs. 18, 19
and 24 (see `data/README.md`). Values marked as read by eye come from raster
figures and are only approximate.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FIDUCIAL_KTON = 22.5
LIVETIME_DAYS = 2970.1
LIVETIME_LOW_DAYS = 2075.3  # used below E_rec = 9.5 MeV
AFT_EFFICIENCY = 0.94

# Table V: 90% C.L. limits per neutrino energy bin, in cm^-2 s^-1 MeV^-1
TABLE_V = {
    "enu_lo": np.arange(9.3, 30.0, 2.0),
    "expected": np.array(
        [44.4, 11.4, 4.17, 1.87, 0.848, 0.464, 0.328, 0.211, 0.213, 0.198, 0.150]
    ),
    "observed": np.array(
        [37.1, 20.4, 9.34, 3.29, 0.508, 0.684, 0.127, 0.375, 0.0777, 0.242, 0.0709]
    ),
    "p_value": np.array(
        [0.346, 0.886, 0.938, 0.830, 0.243, 0.686, 0.073, 0.597, 0.051, 0.605, 0.126]
    ),
}

# Table VIII: predicted flux and combined SK-I to IV 90% C.L. limit (E_nu > 17.3 MeV)
TABLE_VIII = [
    ("Totani+95 Constant", 4.67, 2.6),
    ("Kaplinghat+00 HMA (max)", 3.00, 2.6),
    ("Horiuchi+09 6 MeV, max", 1.94, 2.6),
    ("Ando+03 (updated 05)", 1.74, 2.7),
    ("Kresse+21 (High, NO)", 1.57, 2.7),
    ("Galais+09 (NO)", 1.56, 2.6),
    ("Galais+09 (IO)", 1.50, 2.6),
    ("Horiuchi+18 xi2.5=0.1", 1.23, 2.7),
    ("Kresse+21 (High, IO)", 1.21, 2.7),
    ("Kresse+21 (Fid, NO)", 1.20, 2.7),
    ("Kresse+21 (Fid, IO)", 1.02, 2.7),
    ("Kresse+21 (Low, NO)", 0.96, 2.7),
    ("Tabrizi+21 (NO)", 0.92, 2.7),
    ("Kresse+21 (Low, IO)", 0.84, 2.7),
    ("Lunardini09 Failed SN", 0.73, 2.8),
    ("Hartmann+97 CE", 0.63, 2.6),
    ("Horiuchi+18 xi2.5=0.5", 0.55, 2.6),
    ("Nakazato+15 (max, IO)", 0.53, 2.7),
    ("Horiuchi+21", 0.28, 2.5),
    ("Malaney97 CGI", 0.26, 2.6),
    ("Nakazato+15 (min, NO)", 0.19, 2.7),
]

# Table IX: predicted rate for Horiuchi+09 6 MeV max, events per year
HORIUCHI09_RATE = 3.54
HORIUCHI09_FLUX = 1.94

# Table XIV: atmospheric events in the spectral fit, Horiuchi+09 hypothesis
TABLE_XIV = {
    "Decay-e": ("218.5", 169.5),
    "nu_e CC": ("121.4", 109.9),
    "NCQE": ("81.9-138.4", 88.2),
    "mu/pi": ("10.6-67.2", 27.9),
}

# Table IV: efficiency of the solar cut for events with != 1 tagged neutron
TABLE_IV = {
    "erec_lo": np.array([15.5, 16.5, 17.5, 18.5]),
    "efficiency": np.array([0.731, 0.822, 0.883, 0.966]),
}

# Relative background uncertainties in the model-independent analysis (Sec. VI A).
# NCQE quotes 60-80%; the accidental value is an assumption within the quoted
# "few to 30%" range.
BACKGROUND_REL_ERROR = {
    "non_ncqe": 0.19,
    "ncqe": 0.70,
    "li9": 0.60,
    "reactor": 1.00,
    "accidental": 0.20,
}

BACKGROUNDS = ["non_ncqe", "ncqe", "li9", "reactor", "accidental"]

# Spallation inputs (Secs. IV B and VI A 2)
LI9_RATE_PER_KTON_DAY = 0.86
LI9_BRANCHING = 0.508

# Fig. 4: reconstructed 9Li beta+n spectrum, events per year per 0.5 MeV, by eye
LI9_FIG4_BELOW_7P5 = 2434.0
LI9_FIG4 = {
    "erec_lo": np.arange(7.5, 14.1, 0.5),
    "per_year": np.array(
        [148, 124, 96, 75, 54, 38, 25, 15, 9, 5, 3, 2, 1, 1], dtype=float
    ),
}

# Spallation isotopes: (name, half-life s, end-point MeV, beta+n, total yield, yield above 3.5 MeV).
# Half-lives and yields from Li & Beacom, PRC 89, 045801 (arXiv:1402.4687), Table 1,
# in 10^-7 per muon per g cm^-2 at the SK depth. End-points and the beta+n flag from Fig. 3.
SPALLATION_ISOTOPES = [
    ("16N", 7.13, 10.40, False, 18.0, 18.0),
    ("15C", 2.449, 9.77, False, 0.82, 0.28),
    ("16C", 0.747, 8.01, True, 0.02, 0.003),
    ("14B", 0.0138, 20.60, False, 0.02, 0.02),
    ("13O", 0.0086, 17.80, False, 0.26, 0.24),
    ("13B", 0.0174, 13.44, False, 1.9, 1.6),
    ("12N", 0.0110, 17.30, False, 1.3, 1.1),
    ("12B", 0.0202, 13.37, False, 12.0, 9.8),
    ("12Be", 0.0236, 11.70, False, 0.10, 0.08),
    ("11Be", 13.8, 11.50, False, 0.81, 0.54),
    ("11Li", 0.0085, 20.60, True, 0.01, 0.01),
    ("9C", 0.127, 16.80, False, 0.89, 0.69),
    ("9Li", 0.178, 13.60, True, 1.9, 1.5),
    ("8B", 0.77, 15.00, False, 5.8, 5.0),
    ("8Li", 0.838, 16.00, False, 13.0, 11.0),
    ("8He", 0.119, 10.65, True, 0.23, 0.16),
]

# Neutron tagging (Fig. 15 caption)
NTAG_PRESELECTION_EFFICIENCY = 0.447
NTAG_CANDIDATES_PER_EVENT = 7.0

# Fig. 15: BDT signal efficiency vs mistag probability per candidate, by eye
BDT_ROC_BY_EYE = {
    "efficiency": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "mistag": np.array([8e-6, 4.5e-5, 1.5e-4, 4.5e-4, 1.3e-3, 3.5e-3, 1.0e-2, 3.0e-2]),
}


def _read_csv(name):
    return np.genfromtxt(DATA_DIR / name, delimiter=",", names=True)


def spectrum():
    """Fig. 24: expected backgrounds, data and DSNB, events per 2 MeV."""
    return _read_csv("sk4_fig24_spectrum.csv")


def efficiencies():
    """Fig. 18 (top): cumulative signal efficiency in percent, one tagged neutron."""
    return _read_csv("sk4_fig18_efficiency.csv")


def candidates():
    """Fig. 19: data events per 2 MeV after successive reduction steps."""
    return _read_csv("sk4_fig19_candidates.csv")
