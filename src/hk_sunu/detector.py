"""Water Cherenkov detector targets."""

from dataclasses import dataclass

AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS_G = 18.015
FREE_PROTONS_PER_MOLECULE = 2
GRAMS_PER_KTON = 1e9


@dataclass(frozen=True)
class WaterDetector:
    name: str
    fiducial_kton: float

    @property
    def free_protons(self):
        molecules = self.fiducial_kton * GRAMS_PER_KTON / WATER_MOLAR_MASS_G * AVOGADRO
        return FREE_PROTONS_PER_MOLECULE * molecules


SUPER_K = WaterDetector("SK", 22.5)
HYPER_K = WaterDetector("HK", 187.0)  # one tank, Design Report fiducial mass
