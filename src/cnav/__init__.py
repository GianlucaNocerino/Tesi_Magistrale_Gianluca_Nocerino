"""
cnav - modello deterministico well-to-tank / tank-to-wake per comparare
sistemi propulsivi aeronautici carbon-neutral.

Basato su: Adler, E.J., Martins, J.R.R.A. (2025), "Energy demand comparison
for carbon-neutral flight", Progress in Aerospace Sciences 152, 101051.

Il pacchetto è organizzato in tre sottopacchetti:
- cnav.model         il modello deterministico vero e proprio
- cnav.calibration   calibrazione del modello rispetto al paper (funzioni di costo WLS)
- cnav.sensitivity   analisi di sensibilità locale (indici di elasticità)

Questo __init__.py è la facciata pubblica: ri-esporta da cnav.model e
cnav.calibration i nomi di uso più comune, così "from cnav import ..."
continua a funzionare esattamente come prima della riorganizzazione in
sottocartelle. cnav.sensitivity non viene ri-esportato qui (come già
prima): si accede con
    from cnav.sensitivity.sensitivity_analysis_tools import ...

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
from .model import (
    AircraftSizer,
    SizingResult,
    TechAssumptions,
    WellToTankEfficiencies,
    IntensityResult,
    compute_intensity,
    most_efficient_system,
    Mission,
    BatteryElectric,
    ESAFCombustion,
    HydrogenCombustion,
    HydrogenFuelCell,
    PropulsionSystem,
    build_default_systems,
    EnergyCarrier,
    build_energy_carriers,
)
# sottomoduli esposti per intero, per compatibilità con codice esistente
# che fa "from cnav import T_to_W_definitions" o "from cnav import propulsive_efficiency"
from .model import T_to_W_definitions, propulsive_efficiency, atmosphere, units

from .calibration import (
    PAPER_DATA,
    SYSTEM_PARAMETERS,
    SHARED_PARAMETERS,
    fan_cost,
    propeller_cost,
    combined_cost,
    global_cost,
    system_cost,
    max_feasible_range_nmi,
)

__all__ = [
    "AircraftSizer", "SizingResult",
    "TechAssumptions", "WellToTankEfficiencies",
    "IntensityResult", "compute_intensity", "most_efficient_system",
    "Mission",
    "BatteryElectric", "ESAFCombustion", "HydrogenCombustion", "HydrogenFuelCell",
    "PropulsionSystem", "build_default_systems",
    "EnergyCarrier", "build_energy_carriers",
    "T_to_W_definitions", "propulsive_efficiency", "atmosphere", "units",
    "PAPER_DATA", "SYSTEM_PARAMETERS", "SHARED_PARAMETERS",
    "fan_cost", "propeller_cost", "combined_cost", "global_cost", "system_cost", "max_feasible_range_nmi",
]
