import math
from dataclasses import dataclass
from typing import Optional

from .aircraft_sizer import AircraftSizer, SizingResult
from .constants import TechAssumptions
from .mission import Mission
from .propulsion_systems import PropulsionSystem

"""
Mette insieme well-to-tank (well_to_tank.py) e tank-to-wake
(aircraft_sizer.py) per calcolare l'Electricity Intensity [MJ/(pax*nmi)], 
output del modello deterministico
"""


@dataclass
class IntensityResult:
    system_name: str
    sizing: SizingResult
    energy_at_tank_MJ: float
    renewable_electricity_MJ: float
    n_pax: float
    intensity_MJ_per_pax_nmi: float


def compute_intensity(mission: Mission, system: PropulsionSystem, tech: TechAssumptions,
                       sizer: Optional[AircraftSizer] = None,
                       pax_weight_kg: Optional[float] = None) -> IntensityResult:
    """Dimensiona l'aeromobile e calcola l'electricity intensity.

    pax_weight_kg: se non specificato, usa tech.pax_weight_kg (vedi
    TechAssumptions in constants.py).
    """
    sizer = sizer or AircraftSizer(tech)
    sizing = sizer.size(mission, system)

    if pax_weight_kg is None:
        pax_weight_kg = tech.pax_weight_kg

    energy_at_tank_MJ = sizing.we_total_kg * system.specific_energy_J_per_kg(tech) / 1e6
    renewable_MJ = system.energy_carrier.renewable_electricity_for_MJ(energy_at_tank_MJ)

    n_pax = sizing.payload_kg / pax_weight_kg
    if n_pax > 0 and sizing.converged:
        intensity = renewable_MJ / (n_pax * mission.range_nmi)
    else:
        intensity = math.nan

    return IntensityResult(
        system_name=system.name, sizing=sizing, energy_at_tank_MJ=energy_at_tank_MJ,
        renewable_electricity_MJ=renewable_MJ, n_pax=n_pax, intensity_MJ_per_pax_nmi=intensity,
    )


def most_efficient_system(mission: Mission, systems: list, tech: TechAssumptions,
                           sizer: Optional[AircraftSizer] = None) -> dict:
    """Calcola l'electricity intensity per tutti i sistemi dati e restituisce sia tutti
    i risultati che il nome del più efficiente"""
    results = {s.name: compute_intensity(mission, s, tech, sizer) for s in systems}
    valid = {name: r for name, r in results.items() if not math.isnan(r.intensity_MJ_per_pax_nmi)}
    best_name = min(valid, key=lambda name: valid[name].intensity_MJ_per_pax_nmi) if valid else None
    return {"results": results, "best": best_name}
