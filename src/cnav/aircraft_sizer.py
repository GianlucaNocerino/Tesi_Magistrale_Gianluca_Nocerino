"""
Ciclo di dimensionamento iterativo dell'aeromobile: a partire da una stima di MTOW, calcola OEW, L/D e peso di
energia/combustibile, aggiorna il MTOW e ripete fino a convergenza.
"""
from dataclasses import dataclass
from .constants import TechAssumptions
from . import T_to_W_definitions as TWdef
from .mission import Mission
from .propulsion_systems import PropulsionSystem


@dataclass
class SizingResult:
    mtow_kg: float
    oew_kg: float
    we_climb_kg: float
    we_cruise_kg: float
    we_reserve_kg: float
    we_total_kg: float
    ld: float
    power_req_kW: float
    payload_kg: float
    converged: bool
    iterations: int


class AircraftSizer:
    """
    Esegue il ciclo per una missione e un sistema propulsivo dati.

    Alcune combinazioni missione/sistema propulsivo (tipicamente: batteria
    a lungo raggio) non hanno soluzione finita: ad ogni iterazione il MTOW
    cresce per portare più energia, producendo un aeromobile piu' pesante
    che richiede ancora più energia, e così via. Se il MTOW supera
    'max_mtow_kg' il ciclo si interrompe e il risultato viene marcato
    come non convergente, invece di proseguire fino a max_iter producendo
    numeri via via più grandi e privi di significato fisico
    """

    def __init__(self, tech: TechAssumptions, max_iter: int = 300, tol_kg: float = 1e-3,
                 max_mtow_kg: float = 1e7):
        self.tech = tech
        self.max_iter = max_iter
        self.tol_kg = tol_kg
        self.max_mtow_kg = max_mtow_kg

    def size(self, mission: Mission, system: PropulsionSystem) -> SizingResult:
        tech = self.tech

        ld_baseline = tech.ld_baseline_fan if mission.propulsor == "fan" else tech.ld_baseline_propeller
        payload_kg = TWdef.payload_weight_kg(mission.range_km, tech)

        mtow_kg = 100_000.0       # stima iniziale (Algorithm 1)
        we_total_kg = mtow_kg / 5.0
        ld = ld_baseline

        converged = False
        power_req_kW = 0.0
        we_climb_kg = we_cruise_kg = we_reserve_kg = 0.0
        oew_kg = 0.0
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            power_req_kW = TWdef.power_required_kW(mtow_kg, mission.cruise_speed_ms, ld, mission.propulsor)

            oew_kg = system.operating_empty_weight_kg(
                mtow_kg=mtow_kg, power_req_kW=power_req_kW, ld=ld,
                mission=mission, we_fuel_estimate_kg=we_total_kg, tech=tech,
            )

            ld = system.lift_to_drag(ld_baseline=ld_baseline, ld_prev=ld, mtow_kg=mtow_kg,
                                      mission=mission, tech=tech)

            we_climb_kg = system.climb_energy_kg(mtow_kg, mission, tech, power_req_kW)
            we_cruise_kg, we_reserve_kg = system.cruise_and_reserve_energy_kg(
                mtow_kg, ld, mission, tech, power_req_kW,
            )
            we_total_kg = we_climb_kg + we_cruise_kg + we_reserve_kg

            mtow_new_kg = oew_kg + we_total_kg + payload_kg

            if abs(mtow_new_kg - mtow_kg) < self.tol_kg:
                mtow_kg = mtow_new_kg
                converged = True
                break

            if mtow_new_kg > self.max_mtow_kg:
                # Missione non fattibile con questo sistema propulsivo
                # (vedi docstring della classe): interrompiamo qui.
                mtow_kg = mtow_new_kg
                converged = False
                break

            mtow_kg = mtow_new_kg

        return SizingResult(
            mtow_kg=mtow_kg, oew_kg=oew_kg, we_climb_kg=we_climb_kg,
            we_cruise_kg=we_cruise_kg, we_reserve_kg=we_reserve_kg, we_total_kg=we_total_kg,
            ld=ld, power_req_kW=power_req_kW, payload_kg=payload_kg,
            converged=converged, iterations=iteration,
        )
