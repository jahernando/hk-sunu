import numpy as np
import pytest
from scipy.integrate import quad

from hk_sunu import cherenkov, ibd, oxygen, sk4, stats
from hk_sunu.backgrounds import michel_spectrum
from hk_sunu.detector import HYPER_K, SUPER_K
from hk_sunu.flux import DSNBFlux, ERG_MEV, FermiDirac, SupernovaRate
from hk_sunu.rates import events_per_year


def test_cross_section_vanishes_below_threshold():
    assert ibd.cross_section(np.array([1.0, 1.8])).tolist() == [0.0, 0.0]


def test_cross_section_at_10_mev():
    assert ibd.cross_section(np.array([10.0]))[0] == pytest.approx(6.78e-42, rel=0.01)


def test_sk_energy_window_convention():
    assert ibd.enu_from_erec(15.5) == pytest.approx(17.3, abs=0.01)


def test_fermi_dirac_integrates_to_total_energy():
    spectrum = FermiDirac(temperature=6.0, total_energy_erg=5e52)
    energy, _ = quad(lambda e: e * spectrum(e), 0.0, 300.0)
    assert energy == pytest.approx(5e52 * ERG_MEV, rel=1e-3)


def test_rate_at_redshift_zero():
    assert SupernovaRate()(0.0) == pytest.approx(0.0178 * 0.0088, rel=1e-3)


def test_free_protons_per_kton():
    assert SUPER_K.free_protons / 22.5 == pytest.approx(6.69e31, rel=1e-3)


def test_flux_scales_with_emitted_energy():
    e_nu = np.array([20.0])
    low = DSNBFlux(spectrum=FermiDirac(total_energy_erg=1e52))(e_nu)
    high = DSNBFlux(spectrum=FermiDirac(total_energy_erg=2e52))(e_nu)
    assert high[0] / low[0] == pytest.approx(2.0)


def test_events_scale_with_fiducial_mass():
    flux = DSNBFlux()
    ratio = events_per_year(flux, HYPER_K, 17.3, 81.3) / events_per_year(
        flux, SUPER_K, 17.3, 81.3
    )
    assert ratio == pytest.approx(187.0 / 22.5)


def test_sk4_spectrum_matches_published_event_count():
    spectrum = sk4.spectrum()
    assert spectrum["data"].sum() == 102


def test_sk4_components_add_up_to_total():
    spectrum = sk4.spectrum()
    summed = sum(spectrum[name] for name in sk4.BACKGROUNDS)
    assert summed == pytest.approx(spectrum["total"], abs=0.01)


def test_michel_spectrum_is_normalised():
    area, _ = quad(michel_spectrum, 0.0, 60.0, points=[52.8])
    assert area == pytest.approx(1.0, rel=1e-6)


def test_p_value_without_uncertainty_is_poisson():
    assert stats.p_value(0, [2.0], [0.0]) == pytest.approx(np.exp(-2.0), abs=0.005)


def test_upper_limit_without_background():
    assert stats.upper_limit(0, [0.0], [0.0]) == pytest.approx(2.303, abs=0.01)


@pytest.mark.parametrize(
    "flavour, e_nu, table",
    [
        ("nu_e", 30.0, [1.01e-42, 7.31e-43, 1.73e-44, 3.62e-44]),
        ("nu_e", 50.0, [1.70e-41, 2.70e-41, 3.30e-42, 1.36e-41]),
        ("nu_e", 90.0, [1.57e-40, 3.09e-40, 8.29e-41, 2.34e-40]),
        ("nubar_e", 30.0, [1.23e-42, 7.64e-43, 3.15e-43, 2.09e-43]),
        ("nubar_e", 50.0, [1.09e-41, 1.32e-41, 7.45e-42, 7.55e-42]),
    ],
)
def test_oxygen_fit_matches_tabulated_values(flavour, e_nu, table):
    fitted = oxygen.cross_section(np.array([e_nu]), flavour)[0]
    assert fitted == pytest.approx(sum(table), rel=0.10)


def test_oxygen_cross_section_vanishes_below_threshold():
    assert oxygen.cross_section(np.array([14.0]), "nu_e")[0] == 0.0


def test_spallation_table_is_consistent_with_quoted_yields():
    table = {row[0]: row for row in sk4.SPALLATION_ISOTOPES}
    assert len(table) == 16
    assert {name for name, row in table.items() if row[3]} == {"9Li", "8He", "16C", "11Li"}
    assert table["9Li"][4] == 1.9
    # Li & Beacom: 58 in total, including 18N and 17N, which are absent from Fig. 3
    assert sum(row[4] for row in sk4.SPALLATION_ISOTOPES) + 0.02 + 0.59 == pytest.approx(58.0, abs=0.5)


def test_cherenkov_thresholds_scale_with_mass():
    assert cherenkov.threshold_momentum(cherenkov.MASS_MEV["mu"]) == pytest.approx(118.5, abs=0.2)
    assert cherenkov.threshold_kinetic_energy(cherenkov.MASS_MEV["mu"]) == pytest.approx(53.1, abs=0.2)


def test_cherenkov_angle_saturates_for_electrons():
    angle = cherenkov.angle_deg(10.0, cherenkov.MASS_MEV["e"])
    assert angle == pytest.approx(41.6, abs=0.1)


def test_no_cherenkov_light_below_threshold():
    mass = cherenkov.MASS_MEV["pi"]
    assert np.isnan(cherenkov.angle_deg(100.0, mass))
    assert cherenkov.electron_equivalent_energy(100.0, mass) == 0.0


def test_electron_equivalent_energy_is_below_kinetic_energy():
    mass = cherenkov.MASS_MEV["mu"]
    kinetic = np.hypot(300.0, mass) - mass
    assert 0.0 < cherenkov.electron_equivalent_energy(300.0, mass) < kinetic
