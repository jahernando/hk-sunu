"""Binned version of the SK-IV DSNB spectral fit with sidebands (Sec. VII of arXiv:2109.11174).

SK fits signal and background spectra in six regions: three Cherenkov-angle
ranges times exactly one / not one tagged neutron (Table VI), over
E_rec = 15.5-79.5 MeV. The atmospheric normalisations are free in the fit, so
the sidebands constrain the backgrounds in the signal region. This module uses
the best-fit curves of Fig. 26, sampled at the centres of 2-MeV bins
(`data/sk4_fig26_bestfit.csv`, extracted by `scripts/extract_sk4_fig26.py`),
as templates.

Regions (Table VI):
    I, II, III: one tagged neutron, theta_C = 20-38, 38-50, 78-90 deg
    IV, V, VI : != 1 tagged neutron, same angles

Background categories: decay electrons from invisible mu/pi ("decaye"),
nu_e CC ("nuecc"), NCQE ("ncqe"), visible mu/pi ("mupi") and spallation
("spall"). Accidental coincidences are not a separate category: they are
inside the one-tag templates (the decay-electron spectrum comes from data).

Checks against the paper:
- the categories summed over the six regions give 158 (decay-e), 110 (nu_e CC),
  92 (NCQE) and 28 (mu/pi) events, against 169.5, 109.9, 88.2 and 27.9 best-fit
  events in Table XIV;
- the DSNB curve in region II is 1.56 times the Horiuchi+09 prediction of
  Fig. 24, bin by bin within 5%: the SK-IV best fit. Templates are divided by
  that factor so that mu = 1 is Horiuchi+09 (6 MeV, max).
"""

import numpy as np
from scipy.optimize import minimize

from . import hk, sk4

REGIONS = ["I", "II", "III", "IV", "V", "VI"]
ONE_TAG = slice(0, 3)
OTHER_TAG = slice(3, 6)
BACKGROUNDS = ["decaye", "nuecc", "ncqe", "mupi", "spall"]
ATMOSPHERIC = ["decaye", "nuecc", "ncqe", "mupi"]
CENTRES = np.arange(16.5, 79.0, 2.0)  # 32 bins of 15.5-79.5 MeV
WINDOW_BINS = 7  # 15.5-29.5 MeV, where Fig. 18 and Fig. 24 give per-bin information

# Sec. VII C: 1-sigma uncertainty on the fraction of events with one tagged
# neutron, per atmospheric category (SK-IV). Neutrino 2026 quotes about
# -30/+12-27% for the neutron multiplicity.
MIGRATION_SK4 = 0.40


def _read_fig26():
    path = sk4.DATA_DIR / "sk4_fig26_bestfit.csv"
    with open(path) as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    centres = np.array(header[2:], dtype=float)
    assert np.allclose(centres, CENTRES)
    table = {}
    for region, comp, *values in rows:
        table.setdefault(comp, np.zeros((6, CENTRES.size)))[REGIONS.index(region)] = np.array(values, float)
    return table


def horiuchi_scale(table=None):
    """Best-fit DSNB normalisation of Fig. 26 relative to Horiuchi+09 (Fig. 24)."""
    table = table or _read_fig26()
    fig24 = sk4.spectrum()
    window = (fig24["erec_lo"] >= 15.5) & (fig24["erec_lo"] < 29.5)
    return table["dsnb"][1, :WINDOW_BINS].sum() / fig24["dsnb_horiuchi09"][window].sum()


def templates():
    """SK-IV templates per category, shape (6 regions, 32 bins), DSNB at mu = 1."""
    table = _read_fig26()
    out = {key: table[key].copy() for key in BACKGROUNDS}
    out["dsnb"] = table["dsnb"] / horiuchi_scale(table)
    return out


def accidental_fraction():
    """Fraction of accidental coincidences in the one-tag background, per bin.

    Taken from Fig. 24 (region II, nominal expectation) and applied to the
    three one-tag regions.
    """
    fig24 = sk4.spectrum()
    rows = fig24[fig24["erec_lo"] >= 15.5][:CENTRES.size]
    return np.where(rows["total"] > 0, rows["accidental"] / rows["total"], 0.0)


def _extend(per_window_bin, tail=None):
    """Extend a 7-bin array of 15.5-29.5 MeV to the 32 bins (tail = last-3 mean)."""
    per_window_bin = np.broadcast_to(np.asarray(per_window_bin, float), (WINDOW_BINS,))
    tail = per_window_bin[-3:].mean() if tail is None else tail
    return np.concatenate([per_window_bin, np.full(CENTRES.size - WINDOW_BINS, tail)])


def project(exposure_ratio, efficiency_scale=1.0, mistag_scale=1.0, spall_scale=1.0):
    """Templates for another exposure, neutron tagging and spallation level.

    - Everything scales with `exposure_ratio` (exposure / SK-IV 2970 days).
    - One-tag events with a real neutron (and the signal) scale by
      `efficiency_scale` r; accidentals in the one-tag regions by
      `mistag_scale` k. Signal and real-neutron events that leave or enter the
      one-tag regions are moved to the != 1 regions; extra accidentals are
      only added to the one-tag regions (conservative).
    - Spallation scales by `spall_scale`, and the SK-IV spallation cut costs
      eff_SK ** (spall_scale - 1) on every component below 23.5 MeV, as in
      `hk.project`.

    `efficiency_scale` and `mistag_scale` may be scalars or 7-bin arrays for
    15.5-29.5 MeV; above 29.5 MeV the mean of the last three bins is used.
    """
    base = templates()
    r = _extend(efficiency_scale)
    k = _extend(mistag_scale)
    acc = accidental_fraction()
    veto = np.ones(CENTRES.size)
    veto[:WINDOW_BINS] = hk.sk4_spallation_cut_efficiency()[4:] ** (spall_scale - 1.0)

    out = {}
    for key, t in base.items():
        one, other = t[ONE_TAG], t[OTHER_TAG]
        if key == "dsnb":
            one_new = r * one
            other_new = other + (1.0 - r) * one
        else:
            real = (1.0 - acc) * one
            one_new = r * real + k * acc * one
            # Real-neutron events lost from (or gained by) the one-tag regions go to
            # the != 1 regions. Extra accidentals are only added: the 0-tag events
            # they come from are not removed from the != 1 regions (conservative).
            other_new = np.clip(other + (1.0 - r) * real, 0.0, None)
        new = np.vstack([one_new, other_new]) * exposure_ratio * veto
        if key == "spall":
            new *= spall_scale
        out[key] = new
    return out


def _expected(theta, mu, tmpl, migration):
    """Expected counts for normalisations `theta` (one per background) and migration pulls."""
    n_bkg = len(BACKGROUNDS)
    norms, pulls = theta[:n_bkg], theta[n_bkg:]
    total = mu * tmpl["dsnb"]
    for i, key in enumerate(BACKGROUNDS):
        t = tmpl[key] * norms[i]
        if key in ATMOSPHERIC and migration > 0:
            j = ATMOSPHERIC.index(key)
            one, other = t[ONE_TAG].sum(), t[OTHER_TAG].sum()
            shift = migration * pulls[j]
            t = t.copy()
            t[ONE_TAG] *= 1.0 + shift
            if other > 0:
                t[OTHER_TAG] *= 1.0 - shift * one / other
        total = total + t
    return np.clip(total, 1e-12, None)


def _nll(theta, mu, data, tmpl, migration, norm_priors):
    lam = _expected(theta, mu, tmpl, migration)
    keep = data > 0
    nll = np.sum(lam) - np.sum(data[keep] * np.log(lam[keep]))
    n_bkg = len(BACKGROUNDS)
    pulls = theta[n_bkg:]
    nll += 0.5 * np.sum(pulls**2)
    for i, key in enumerate(BACKGROUNDS):
        if norm_priors.get(key):
            nll += 0.5 * ((theta[i] - 1.0) / norm_priors[key]) ** 2
    return nll


def _profile(mu, data, tmpl, migration, norm_priors):
    n_par = len(BACKGROUNDS) + len(ATMOSPHERIC)
    x0 = np.concatenate([np.ones(len(BACKGROUNDS)), np.zeros(len(ATMOSPHERIC))])
    bounds = [(1e-6, None)] * len(BACKGROUNDS) + [(-3.0, 3.0)] * len(ATMOSPHERIC)
    best = minimize(_nll, x0, args=(mu, data, tmpl, migration, norm_priors),
                    bounds=bounds[:n_par], method="L-BFGS-B")
    return best.fun


def asimov_significance(tmpl, mu=1.0, migration=MIGRATION_SK4, norm_priors=None):
    """Median discovery significance for signal strength `mu` (Asimov data).

    Background normalisations are free unless `norm_priors` gives a relative
    Gaussian width for a category (e.g. {"spall": 1.0}).
    """
    norm_priors = norm_priors or {}
    data = mu * tmpl["dsnb"] + sum(tmpl[key] for key in BACKGROUNDS)
    q0 = 2.0 * (_profile(0.0, data, tmpl, migration, norm_priors)
                - _profile(mu, data, tmpl, migration, norm_priors))
    return float(np.sqrt(max(q0, 0.0)))


def asimov_upper_limit(tmpl, migration=MIGRATION_SK4, norm_priors=None, cl=0.90):
    """Median upper limit on mu for background-only Asimov data (profile likelihood).

    One-sided limit: -2 Delta ln L = z^2 with z = 1.28 for 90% C.L.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    norm_priors = norm_priors or {}
    data = sum(tmpl[key] for key in BACKGROUNDS)
    l0 = _profile(0.0, data, tmpl, migration, norm_priors)
    z2 = norm.ppf(cl) ** 2
    return brentq(lambda m: 2.0 * (_profile(m, data, tmpl, migration, norm_priors) - l0) - z2, 1e-3, 50.0)


def combine(*tmpls):
    """Concatenate independent datasets bin by bin (shared normalisations)."""
    return {key: np.hstack([t[key] for t in tmpls]) for key in tmpls[0]}
