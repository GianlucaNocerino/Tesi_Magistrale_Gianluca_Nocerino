"""
Calibrazione del modello deterministico rispetto ai risultati riportati in:
    Adler, E.J., Martins, J.R.R.A. (2025), "Energy demand comparison for
    carbon-neutral flight", Progress in Aerospace Sciences 152, 101051.
    (Fig. 4: Electricity Intensity vs Range, fan @450 kt ed elica @250 kt)

Costruisce due funzioni di costo globali (una per il fan, una per
l'elica), ciascuna somma pesata delle funzioni di costo dei singoli
sistemi propulsivi disponibili in quella configurazione. Ogni funzione di
costo di sistema è una Weighted Least Squares sugli scarti normalizzati
tra intensity del modello e del paper (10 valori di Range, velocità
fissata), con una penalità se il modello non converge dove il paper
riporta un valore. Solo per Battery-electric si aggiunge un terzo
termine: lo scarto, normalizzato e pesato, tra il range massimo fattibile
calcolato dal modello e quello desumibile dal paper.

NOTA: il modello non espone il range massimo fattibile per la batteria
come output diretto (compute_intensity/IntensityResult non lo calcola:
per un dato punto operativo si ha solo il flag booleano
sizing.converged). Va quindi ricavato qui con una ricerca (bisezione)
sul Range a velocità fissata, vedi max_feasible_range_nmi.
"""
import math
from typing import Optional

from ..model.aircraft_sizer import AircraftSizer
from ..model.constants import TechAssumptions, WellToTankEfficiencies
from ..model.energy_intensity import compute_intensity
from ..model.mission import Mission
from ..model.propulsion_systems import build_default_systems
from ..model.well_to_tank import build_energy_carriers

# ---------------------------------------------------------------------
# 1. Dati di riferimento letti dalle tabelle del paper (Fig. 4)
#    None = il paper segna "infeasible" in quel punto (nessun confronto
#    sull'intensity lì: il mismatch di fattibilità è già catturato dal
#    terzo termine, max_range_nmi, per Battery-electric)
# ---------------------------------------------------------------------

PAPER_DATA = {
    "fan": {
        "speed_kt": 450,
        "ranges_nmi": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        "systems": {
            "Battery-electric": {
                "intensity": [6.36, 6.63, None, None, None, None, None, None, None, None],
                "max_range_nmi": 31.0,  # "infeasible (>31 nmi)" al Range=50
            },
            "e-SAF combustion": {
                "intensity": [8.76, 7.16, 6.18, 5.87, 5.70, 5.15, 4.81, 4.68, 5.09, 6.03],
            },
            "Hydrogen combustion": {
                "intensity": [5.90, 4.82, 4.17, 3.95, 3.83, 3.44, 3.21, 3.10, 3.34, 4.06],
            },
        },
    },
    "propeller": {
        "speed_kt": 250,
        "ranges_nmi": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        "systems": {
            "Battery-electric": {
                "intensity": [1.42, 1.41, 2.29, None, None, None, None, None, None, None],
                "max_range_nmi": 79.0,  # "infeasible (>79 nmi)" al Range=100
            },
            "e-SAF combustion": {
                "intensity": [9.97, 9.39, 9.07, 8.96, 8.40, 7.49, 6.95, 7.17, 9.17, 12.12],
            },
            "Hydrogen combustion": {
                "intensity": [6.99, 6.59, 6.33, 6.21, 5.75, 5.04, 4.65, 4.78, 6.11, 8.81],
            },
            "Hydrogen fuel cell": {
                "intensity": [6.31, 5.71, 5.34, 5.16, 4.56, 3.93, 3.79, 4.14, 5.85, 9.83],
            },
        },
    },
}

# ---------------------------------------------------------------------
# 2. Parametri "di competenza" di ciascun sistema: quali campi di
#    TechAssumptions/WellToTankEfficiencies muovono principalmente
#    l'output di quel sistema. Non usati per calcolare il costo (che
#    valuta sempre il modello completo), ma predisposti per il passo
#    successivo: una sensitivity analysis / ottimizzazione mirata che
#    su ciascuna funzione di costo di sistema vari solo i parametri
#    di sua competenza, invece di tutti i parametri del modello insieme.
# ---------------------------------------------------------------------

SYSTEM_PARAMETERS = {
    "Battery-electric": [
        "e_battery_Wh_per_kg", "eta_battery", "eta_motor",
        "motor_specific_power_kW_per_kg", "battery_oew_fraction",
        "delta_TMS_N_per_kW", "kappa_TMS_kg_per_kW", "electricity",
    ],
    "e-SAF combustion": [
        "e_saf",
    ],
    "Hydrogen combustion": [
        "gamma_tank", "hydrogen_empty_weight_multiplier", "hydrogen_ld_multiplier",
        "liquid_hydrogen",
    ],
    "Hydrogen fuel cell": [
        "eta_fuel_cell", "fuel_cell_specific_power_kW_per_kg", "gamma_tank",
        "hydrogen_empty_weight_multiplier", "hydrogen_ld_multiplier",
        "delta_TMS_N_per_kW", "kappa_TMS_kg_per_kW", "liquid_hydrogen",
        "motor_specific_power_kW_per_kg", "eta_motor",
    ],
}

# parametri condivisi da (quasi) tutti i sistemi: dimensionamento
# generale del velivolo, non specifici di un vettore energetico
SHARED_PARAMETERS = [
    "oew_fan_a", "oew_fan_b", "oew_fan_c",
    "oew_prop_a", "oew_prop_b", "oew_prop_c",
    "payload_a", "payload_b", "payload_c", "payload_d", "payload_e",
    "ld_baseline_fan", "ld_baseline_propeller",
    "eta_p_fan", "eta_p_propeller", "fan_pressure_ratio", "fan_scaling_mach_ref",
    "propeller_curve_peak_mach", "propeller_curve_rise_rate", "propeller_curve_decay_width",
    "reserve_loiter_time_s", "reserve_alternate_range_nmi", "reserve_loiter_speed_kt",
    "pax_weight_kg",
    "turbofan_core_specific_power_kW_per_kg", "turboprop_specific_power_kW_per_kg",
    "climb_rate_ft_per_min", "climb_speed_fraction",
]

DEFAULT_NONCONVERGENCE_PENALTY = 10.0  # "grande" rispetto a un tipico scarto normalizzato (~0.01-1)


def _normalized_sq_residual(model_val: float, target_val: float,
                             norm_val: Optional[float] = None) -> float:
    """Scarto quadratico normalizzato: ((model - target) / norm)^2.

    Di default (norm_val=None) la normalizzazione usa il valore di
    riferimento del paper (target_val), cioè l'errore relativo al
    quadrato — rende confrontabili grandezze di scala diversa (es.
    intensity in MJ/(pax*nmi) vs range in nmi). norm_val è però un
    parametro esplicito e sovrascrivibile esattamente come i pesi e le
    penalità: si può passare un valore di normalizzazione diverso
    (es. una deviazione standard, un valore fisso per tutto un
    sistema, ...) senza toccare il target_val usato per calcolare lo
    scarto vero e proprio."""
    if norm_val is None:
        norm_val = target_val
    return ((model_val - target_val) / norm_val) ** 2


# ---------------------------------------------------------------------
# 3. Range massimo fattibile per la batteria (non è un output diretto
#    del modello: va ricavato per bisezione su sizing.converged)
# ---------------------------------------------------------------------

def max_feasible_range_nmi(tech: TechAssumptions, wtt: WellToTankEfficiencies,
                            propulsor: str, speed_kt: float,
                            low_nmi: float = 1.0, high_nmi: float = 10_000.0,
                            tol_nmi: float = 0.5, max_bisect_iter: int = 60) -> float:
    """Range massimo a cui il sistema Battery-electric converge ancora,
    a velocità fissata, per bisezione.

    Assume monotonia (fattibile sotto una soglia, infattibile sopra),
    coerente con il comportamento del sizer (vedi aircraft_sizer.py) e
    con quanto riportato nel paper. Casi limite:
    - già infattibile a low_nmi -> ritorna 0.0
    - ancora fattibile a high_nmi -> ritorna high_nmi (nessuna soglia
      trovata nell'intervallo testato)
    """
    carriers = build_energy_carriers(wtt)
    battery = next(s for s in build_default_systems(carriers) if s.name == "Battery-electric")
    sizer = AircraftSizer(tech)

    def feasible(r_nmi: float) -> bool:
        mission = Mission(range_nmi=r_nmi, cruise_speed_kt=speed_kt, propulsor=propulsor)
        return sizer.size(mission, battery).converged

    if not feasible(low_nmi):
        return 0.0
    if feasible(high_nmi):
        return high_nmi

    lo, hi = low_nmi, high_nmi
    for _ in range(max_bisect_iter):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol_nmi:
            break
    return lo


# ---------------------------------------------------------------------
# 4. Funzione di costo di un singolo sistema propulsivo
# ---------------------------------------------------------------------

def system_cost(system_name: str, propulsor: str, tech: TechAssumptions,
                 wtt: WellToTankEfficiencies,
                 range_weights: Optional[list] = None,
                 intensity_norm_values: Optional[list] = None,
                 nonconvergence_penalty: float = DEFAULT_NONCONVERGENCE_PENALTY,
                 range_term_weight: float = 1.0,
                 range_norm_value: Optional[float] = None) -> float:
    """WLS di un sistema in una configurazione (fan @450kt oppure
    propeller @250kt): somma sui 10 Range del paper degli scarti
    quadratici normalizzati e pesati sull'intensity, con penalità di
    non convergenza; per Battery-electric aggiunge il terzo termine sul
    range massimo fattibile.

    Normalizzazione (come i pesi/penalità, sovrascrivibile):
    - intensity_norm_values: lista di 10 valori (uno per Range) usati
      per normalizzare lo scarto sull'intensity in quel punto. Se None
      (default), si usa il valore del paper in quel punto — cioè
      l'errore relativo al quadrato.
    - range_norm_value: valore usato per normalizzare lo scarto sul
      range massimo fattibile (solo Battery-electric). Se None
      (default), si usa il max_range_nmi del paper.
    """
    data = PAPER_DATA[propulsor]
    speed_kt = data["speed_kt"]
    ranges = data["ranges_nmi"]
    sys_data = data["systems"][system_name]
    paper_intensities = sys_data["intensity"]

    if range_weights is None:
        range_weights = [1.0] * len(ranges)
    if len(range_weights) != len(ranges):
        raise ValueError("range_weights deve avere un peso per ciascuno dei 10 Range del paper")
    if intensity_norm_values is None:
        intensity_norm_values = [None] * len(ranges)
    if len(intensity_norm_values) != len(ranges):
        raise ValueError("intensity_norm_values deve avere un valore per ciascuno dei 10 Range del paper")

    carriers = build_energy_carriers(wtt)
    systems = build_default_systems(carriers)
    system = next((s for s in systems if s.name == system_name), None)
    if system is None:
        raise ValueError(f"Sistema sconosciuto: {system_name}")

    cost = 0.0
    for r_nmi, paper_val, w, norm_val in zip(ranges, paper_intensities, range_weights, intensity_norm_values):
        if paper_val is None:
            continue  # il paper stesso lo marca infeasible: niente da confrontare qui sull'intensity
        mission = Mission(range_nmi=r_nmi, cruise_speed_kt=speed_kt, propulsor=propulsor)
        model_val = compute_intensity(mission, system, tech).intensity_MJ_per_pax_nmi
        if math.isnan(model_val):
            cost += w * nonconvergence_penalty
        else:
            cost += w * _normalized_sq_residual(model_val, paper_val, norm_val)

    max_range_paper = sys_data.get("max_range_nmi")
    if system_name == "Battery-electric" and max_range_paper is not None:
        model_range = max_feasible_range_nmi(tech, wtt, propulsor, speed_kt)
        cost += range_term_weight * _normalized_sq_residual(model_range, max_range_paper, range_norm_value)

    return cost


# ---------------------------------------------------------------------
# 5. Funzioni di costo globali (fan / propeller): somma pesata dei
#    costi di sistema
# ---------------------------------------------------------------------

def global_cost(propulsor: str, tech: TechAssumptions, wtt: WellToTankEfficiencies,
                 system_weights: Optional[dict] = None,
                 range_weights: Optional[list] = None,
                 nonconvergence_penalty: float = DEFAULT_NONCONVERGENCE_PENALTY,
                 range_term_weight: float = 1.0,
                 norm_overrides: Optional[dict] = None) -> tuple:
    """Costo globale per una configurazione (fan o propeller): somma
    pesata di system_cost su tutti i sistemi disponibili nel paper per
    quella configurazione.

    norm_overrides: dict opzionale {nome_sistema: {"intensity_norm_values": [...],
    "range_norm_value": ...}} per sovrascrivere, sistema per sistema, la
    normalizzazione di default (il valore del paper). Un sistema non
    presente nel dict usa la normalizzazione di default.

    Ritorna (costo_totale, {nome_sistema: costo_di_sistema}).
    """
    if propulsor not in PAPER_DATA:
        raise ValueError("propulsor deve essere 'fan' o 'propeller'")

    system_names = list(PAPER_DATA[propulsor]["systems"].keys())
    if system_weights is None:
        system_weights = {name: 1.0 for name in system_names}
    if norm_overrides is None:
        norm_overrides = {}

    per_system = {}
    total = 0.0
    for name in system_names:
        overrides = norm_overrides.get(name, {})
        c = system_cost(name, propulsor, tech, wtt,
                         range_weights=range_weights,
                         nonconvergence_penalty=nonconvergence_penalty,
                         range_term_weight=range_term_weight,
                         intensity_norm_values=overrides.get("intensity_norm_values"),
                         range_norm_value=overrides.get("range_norm_value"))
        per_system[name] = c
        total += system_weights.get(name, 1.0) * c

    return total, per_system


def fan_cost(tech: TechAssumptions, wtt: WellToTankEfficiencies, **kwargs) -> tuple:
    """Funzione di costo globale per i sistemi con fan (@450 kt)."""
    return global_cost("fan", tech, wtt, **kwargs)


def propeller_cost(tech: TechAssumptions, wtt: WellToTankEfficiencies, **kwargs) -> tuple:
    """Funzione di costo globale per i sistemi con elica (@250 kt)."""
    return global_cost("propeller", tech, wtt, **kwargs)
