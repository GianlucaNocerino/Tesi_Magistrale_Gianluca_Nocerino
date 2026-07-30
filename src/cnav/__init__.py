"""
cnav - modello deterministico well-to-tank / tank-to-wake per comparare
sistemi propulsivi aeronautici carbon-neutral.

Basato su: Adler, E.J., Martins, J.R.R.A. (2025), "Energy demand comparison
for carbon-neutral flight", Progress in Aerospace Sciences 152, 101051.

Esempio d'uso rapido:

    from cnav import (Mission, TechAssumptions, WellToTankEfficiencies,
                       build_energy_carriers, build_default_systems,
                       compute_intensity)

    tech = TechAssumptions()
    wtt = WellToTankEfficiencies()
    systems = build_default_systems(build_energy_carriers(wtt))

    mission = Mission(range_nmi=500, cruise_speed_kt=450, propulsor="fan")
    for system in systems:
        result = compute_intensity(mission, system, tech)
        print(system.name, result.intensity_MJ_per_pax_nmi)
"""
from .aircraft_sizer import AircraftSizer, SizingResult
from .constants import TechAssumptions, WellToTankEfficiencies
from .energy_intensity import IntensityResult, compute_intensity, most_efficient_system
from .mission import Mission
from .propulsion_systems import (
    BatteryElectric,
    ESAFCombustion,
    HydrogenCombustion,
    HydrogenFuelCell,
    PropulsionSystem,
    build_default_systems,
)
from .well_to_tank import EnergyCarrier, build_energy_carriers

__all__ = [
    "AircraftSizer", "SizingResult",
    "TechAssumptions", "WellToTankEfficiencies",
    "IntensityResult", "compute_intensity", "most_efficient_system",
    "Mission",
    "BatteryElectric", "ESAFCombustion", "HydrogenCombustion", "HydrogenFuelCell",
    "PropulsionSystem", "build_default_systems",
    "EnergyCarrier", "build_energy_carriers",
]
