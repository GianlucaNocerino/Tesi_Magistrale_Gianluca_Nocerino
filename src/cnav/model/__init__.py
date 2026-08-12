"""
cnav.model - il modello deterministico vero e proprio: well-to-tank +
tank-to-wake (stile Breguet) per i quattro sistemi propulsivi
carbon-neutral. Nessuna dipendenza da cnav.calibration o
cnav.sensitivity: questo sottopacchetto è "il modello", loro sono
strumenti che lo usano dall'esterno.
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

# sottomoduli esposti per intero (non solo le classi/funzioni sopra),
# per chi vuole es. "from cnav.model import propulsive_efficiency as pe"
from . import atmosphere
from . import propulsive_efficiency
from . import T_to_W_definitions
from . import units

__all__ = [
    "AircraftSizer", "SizingResult",
    "TechAssumptions", "WellToTankEfficiencies",
    "IntensityResult", "compute_intensity", "most_efficient_system",
    "Mission",
    "BatteryElectric", "ESAFCombustion", "HydrogenCombustion", "HydrogenFuelCell",
    "PropulsionSystem", "build_default_systems",
    "EnergyCarrier", "build_energy_carriers",
    "atmosphere", "propulsive_efficiency", "T_to_W_definitions", "units",
]
