from abc import ABC, abstractmethod

from . import T_to_W_definitions as TWdef
from .constants import LHV_ESAF_J_per_kg, LHV_HYDROGEN_J_per_kg, TechAssumptions
from .mission import Mission
from .units import G, WH_TO_J
from .well_to_tank import EnergyCarrier

"""
I quattro sistemi propulsivi carbon-neutral, ciascuno come sottoclasse di PropulsionSystem.

Ogni sottoclasse deve solo specificare "cosa la rende diversa":
- come si calcola l'OEW
- se ha un TMS (batteria e fuel cell si, combustione no)
- l'efficienza propulsiva complessiva e l'energia specifica del vettore
  energetico usato (per Breguet e per la salita)
- se la massa diminuisce in volo (si per i sistemi a combustione, no per la batteria)

Tutta la parte "in comune" (Breguet, energia di salita, TMS) è ereditata
dalla classe base e vive in T_to_W_definitions.py
"""


class PropulsionSystem(ABC):
    """Base comune a tutti i sistemi propulsivi"""

    name: str
    energy_carrier: EnergyCarrier
    has_tms: bool = False
    mass_decays_in_flight: bool = True   # False solo per la batteria
    ld_baseline_multiplier: float = 1.0  # 0.95 per i sistemi a idrogeno

    # ---- da implementare in ogni sottoclasse -----------------------------
    @abstractmethod
    def operating_empty_weight_kg(self, mtow_kg: float, power_req_kW: float, ld: float,
                                   mission: Mission, we_fuel_estimate_kg: float,
                                   tech: TechAssumptions) -> float:
        """Peso a vuoto operativo (OEW)"""

    @abstractmethod
    def overall_efficiency(self, mission: Mission, tech: TechAssumptions,
                            power_req_kW: float) -> float:
        """Efficienza complessiva usata per Breguet e per l'energia di
        salita (prodotto di tutte le efficienze rilevanti, propulsiva
        inclusa)"""

    @abstractmethod
    def specific_energy_J_per_kg(self, tech: TechAssumptions) -> float:
        """Energia specifica del vettore energetico: energia
        specifica della batteria, oppure LHV del combustibile"""

    # ---- comune a tutti, con eventuale override -------------------------
    def _heat_loss_fraction(self, tech: TechAssumptions) -> float:
        """Frazione della potenza elettrica richiesta che si trasforma in
        calore da smaltire (solo per sistemi con TMS)"""
        raise NotImplementedError

    def tms_weight_and_drag(self, mtow_kg: float, ld: float, mission: Mission,
                             tech: TechAssumptions) -> tuple[float, float]:
        if not self.has_tms:
            return 0.0, 0.0
        heat_kW = TWdef.heat_rejected_kW(mtow_kg, ld, mission, tech, self._heat_loss_fraction(tech))
        return TWdef.tms_weight_kg(heat_kW, tech), TWdef.tms_drag_N(heat_kW, tech)

    def lift_to_drag(self, ld_baseline: float, ld_prev: float, mtow_kg: float,
                      mission: Mission, tech: TechAssumptions) -> float:
        """L/D corretto per la resistenza del TMS (se presente)
        e per il multiplier legato allo stivaggio dell'idrogeno"""
        effective_baseline = ld_baseline * self.ld_baseline_multiplier
        if not self.has_tms:
            return effective_baseline
        _, extra_drag_N = self.tms_weight_and_drag(mtow_kg, ld_prev, mission, tech)
        return TWdef.ld_with_extra_drag(effective_baseline, mtow_kg, extra_drag_N)

    def climb_energy_kg(self, mtow_kg: float, mission: Mission, tech: TechAssumptions,
                         power_req_kW: float) -> float:
        eta = self.overall_efficiency(mission, tech, power_req_kW)
        e_spec = self.specific_energy_J_per_kg(tech)
        return TWdef.climb_energy_weight_kg(mtow_kg, mission, eta, e_spec, tech)

    def cruise_and_reserve_energy_kg(self, mtow_kg: float, ld: float, mission: Mission,
                                      tech: TechAssumptions, power_req_kW: float) -> tuple[float, float]:
        eta = self.overall_efficiency(mission, tech, power_req_kW)
        e_spec = self.specific_energy_J_per_kg(tech)
        r_reserve_m = TWdef.reserve_range_m(mission, tech)

        if self.mass_decays_in_flight:
            we_cruise = TWdef.breguet_fuel_weight_kg(mtow_kg, ld, eta, e_spec, mission.range_m)
            we_total = TWdef.breguet_fuel_weight_kg(mtow_kg, ld, eta, e_spec, mission.range_m + r_reserve_m)
            we_reserve = we_total - we_cruise
        else:
            # La batteria non perde massa in volo, quindi il
            # legame energia-range è lineare, non esponenziale.
            we_cruise = mtow_kg * G * mission.range_m / (eta * e_spec * ld)
            we_reserve = mtow_kg * G * r_reserve_m / (eta * e_spec * ld)
        return we_cruise, we_reserve


class BatteryElectric(PropulsionSystem):

    name = "Battery-electric"
    has_tms = True
    mass_decays_in_flight = False

    def __init__(self, energy_carrier: EnergyCarrier):
        self.energy_carrier = energy_carrier

    def _heat_loss_fraction(self, tech: TechAssumptions) -> float:
        return (1.0 - tech.eta_motor) + (1.0 - tech.eta_battery)

    def overall_efficiency(self, mission, tech, power_req_kW) -> float:
        return tech.eta_motor * TWdef.propulsive_efficiency(mission, tech)

    def specific_energy_J_per_kg(self, tech) -> float:
        return tech.e_battery_Wh_per_kg * WH_TO_J

    def operating_empty_weight_kg(self, mtow_kg, power_req_kW, ld, mission,
                                   we_fuel_estimate_kg, tech) -> float:
        oew_baseline = tech.battery_oew_fraction * mtow_kg
        w_motor = power_req_kW / tech.motor_specific_power_kW_per_kg
        w_tms, _ = self.tms_weight_and_drag(mtow_kg, ld, mission, tech)
        return oew_baseline + w_motor + w_tms


class HydrogenFuelCell(PropulsionSystem):

    name = "Hydrogen fuel cell"
    has_tms = True
    ld_baseline_multiplier = 0.95

    def __init__(self, energy_carrier: EnergyCarrier):
        self.energy_carrier = energy_carrier

    def _heat_loss_fraction(self, tech: TechAssumptions) -> float:
        return (1.0 - tech.eta_motor) + (1.0 - tech.eta_fuel_cell) / tech.eta_fuel_cell

    def overall_efficiency(self, mission, tech, power_req_kW) -> float:
        return tech.eta_fuel_cell * tech.eta_motor * TWdef.propulsive_efficiency(mission, tech)

    def specific_energy_J_per_kg(self, tech) -> float:
        return LHV_HYDROGEN_J_per_kg

    def operating_empty_weight_kg(self, mtow_kg, power_req_kW, ld, mission,
                                   we_fuel_estimate_kg, tech) -> float:
        oew_frac = TWdef.oew_fraction(mtow_kg, mission.propulsor,tech)
        oew_baseline = oew_frac * mtow_kg * tech.hydrogen_empty_weight_multiplier

        w_motor = power_req_kW / tech.motor_specific_power_kW_per_kg
        w_fuel_cell = (power_req_kW / tech.eta_motor) / tech.fuel_cell_specific_power_kW_per_kg
        w_tank = TWdef.hydrogen_tank_weight_kg(we_fuel_estimate_kg, tech.gamma_tank)
        w_tms, _ = self.tms_weight_and_drag(mtow_kg, ld, mission, tech)

        conventional_specific_power = (tech.turbofan_core_specific_power_kW_per_kg
                                        if mission.propulsor == "fan"
                                        else tech.turboprop_specific_power_kW_per_kg)
        w_conventional_engine = power_req_kW / conventional_specific_power

        return oew_baseline + w_motor + w_fuel_cell + w_tank + w_tms - w_conventional_engine


class HydrogenCombustion(PropulsionSystem):

    name = "Hydrogen combustion"
    ld_baseline_multiplier = 0.95

    def __init__(self, energy_carrier: EnergyCarrier):
        self.energy_carrier = energy_carrier

    def overall_efficiency(self, mission, tech, power_req_kW) -> float:
        return TWdef.combustion_overall_efficiency(mission, tech, power_req_kW)

    def specific_energy_J_per_kg(self, tech) -> float:
        return LHV_HYDROGEN_J_per_kg

    def operating_empty_weight_kg(self, mtow_kg, power_req_kW, ld, mission,
                                   we_fuel_estimate_kg, tech) -> float:
        oew_frac = TWdef.oew_fraction(mtow_kg, mission.propulsor,tech)
        oew_baseline = oew_frac * mtow_kg * tech.hydrogen_empty_weight_multiplier
        w_tank = TWdef.hydrogen_tank_weight_kg(we_fuel_estimate_kg, tech.gamma_tank)
        return oew_baseline + w_tank


class ESAFCombustion(PropulsionSystem):

    name = "e-SAF combustion"

    def __init__(self, energy_carrier: EnergyCarrier):
        self.energy_carrier = energy_carrier

    def overall_efficiency(self, mission, tech, power_req_kW) -> float:
        return TWdef.combustion_overall_efficiency(mission, tech, power_req_kW)

    def specific_energy_J_per_kg(self, tech) -> float:
        return LHV_ESAF_J_per_kg

    def operating_empty_weight_kg(self, mtow_kg, power_req_kW, ld, mission,
                                   we_fuel_estimate_kg, tech) -> float:
        # nessuna correzione, è il velivolo convenzionale.
        return TWdef.oew_fraction(mtow_kg, mission.propulsor,tech) * mtow_kg


def build_default_systems(energy_carriers: dict) -> list:
    """Crea le 4 istanze di default, già collegate ai rispettivi vettori
    energetici well-to-tank."""
    return [
        BatteryElectric(energy_carriers["electricity"]),
        HydrogenFuelCell(energy_carriers["liquid_hydrogen"]),
        HydrogenCombustion(energy_carriers["liquid_hydrogen"]),
        ESAFCombustion(energy_carriers["e_saf"]),
    ]
