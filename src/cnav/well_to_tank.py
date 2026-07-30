"""
Analisi "well-to-tank" nel modello: quanta elettricità
rinnovabile serve, a monte, per avere una certa quantità di energia
effettivamente disponibile a bordo dell'aeromobile
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyCarrier:
    """Vettore energetico utilizzato dai diversi sitemi propulsivi, che caratterizza
    la frazione di elettricità rinnovabile iniziale che sopravvive fino al 
    serbatoio dell'aeromobile"""

    name: str
    efficiency: float  # elettricita' 0.85; LH2 0.46; e-SAF 0.26

    def renewable_electricity_for_MJ(self, energy_at_tank_MJ: float) -> float:
        """Elettricità rinnovabile [MJ] necessaria per produrre
        energy_at_tank_MJ, l'energia disponibile a bordo"""
        return energy_at_tank_MJ / self.efficiency


def build_energy_carriers(wtt) -> dict:
    """Crea i tre EnergyCarrier a partire da un oggetto WellToTankEfficiencies,
    così che uno studio di incertezza possa perturbare 'wtt' e rigenerare
    automaticamente i vettori energetici coerenti"""
    return {
        "electricity": EnergyCarrier("Grid electricity (battery)", wtt.electricity),
        "liquid_hydrogen": EnergyCarrier("Liquid hydrogen", wtt.liquid_hydrogen),
        "e_saf": EnergyCarrier("e-SAF", wtt.e_saf),
    }
