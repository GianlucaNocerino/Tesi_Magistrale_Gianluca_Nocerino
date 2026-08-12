"""
Parametri tecnologici e coefficienti di riferimento assunti.

I valori di default riproducono la Table 1 e la Fig. 1 di:
    Adler, E.J., Martins, J.R.R.A. (2025), "Energy demand comparison for
    carbon-neutral flight", Progress in Aerospace Sciences 152, 101051.
    https://doi.org/10.1016/j.paerosci.2024.101051

Tutti gli input del modello sono raccolti qui, in TechAssumptions e
WellToTankEfficiencies.

TechAssumptions raggruppa i campi in due categorie:

1) Parametri tecnologici 
2) Coefficienti per le stime empiriche

Nota su fan_pressure_ratio: Adler & Martins NON specificano quale valore
di rapporto di compressione del fan abbiano usato per costruire la curva
Fig. 9 (dicono solo "adottata da Michel [55]", che nel suo articolo usa
1.5 come valore di esempio in quasi tutte le figure). Il default qui
riprende quel valore, ma e' una scelta di questo modello, non
un'informazione dell'articolo di riferimento della tesi. Vedi anche
propulsive_efficiency.py.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TechAssumptions:

    # --- elettrico / batteria ---
    motor_specific_power_kW_per_kg: float = 8.0    # include inverter ed elettronica
    eta_motor: float = 0.90                        # efficienza motore elettrico
    e_battery_Wh_per_kg: float = 300.0             # energia specifica pacco batteria
    eta_battery: float = 0.95                      # efficienza di scarica batteria

    # --- fuel cell / idrogeno ---
    fuel_cell_specific_power_kW_per_kg: float = 2.0
    eta_fuel_cell: float = 0.60
    gamma_tank: float = 0.50                       # efficienza gravimetrica serbatoio LH2
    hydrogen_empty_weight_multiplier: float = 1.1  # extra-peso strutturale aeromobili a LH2

    # --- TMS, comune a batteria e fuel cell ---
    delta_TMS_N_per_kW: float = 0.2                # resistenza per kW di calore da smaltire
    kappa_TMS_kg_per_kW: float = 0.2               # peso per kW di calore da smaltire

    # --- aerodinamica / propulsione, valori di riferimento ---
    ld_baseline_propeller: float = 15.0
    ld_baseline_fan: float = 20.0
    eta_p_propeller: float = 0.85                  # efficienza propulsiva di riferimento (propeller)
    eta_p_fan: float = 0.75                        # efficienza propulsiva di riferimento (fan/getto)
    fan_pressure_ratio: float = 1.5                # rapporto di compressione fan

    # --- motori convenzionali (per il "sottraggo il motore a combustione" nella fuel cell) ---
    turbofan_core_specific_power_kW_per_kg: float = 15.0
    turboprop_specific_power_kW_per_kg: float = 4.0

    # --- salita ---
    climb_rate_ft_per_min: float = 1500.0          # rateo di salita
    climb_speed_fraction: float = 0.75             # frazione della velocità di crociera usata come velocità di avanzamento in salita

    # --- missione di riserva ---
    reserve_loiter_time_s: float = 45 * 60.0
    reserve_alternate_range_nmi: float = 200.0
    # velocità di loiter: NaN (default) = usa la velocità di crociera della
    # missione (assunzione dell'articolo di riferimento); un valore numerico
    # esplicito la tratta come parametro indipendente, perturbabile nella
    # sensitivity analysis (vedi reserve_range_m in T_to_W_definitions.py)
    reserve_loiter_speed_kt: float = float("nan")

    # --- per la normalizzazione per passeggero ---
    pax_weight_kg: float = 100.0                   # non specificato dall'articolo

    # frazione di peso a vuoto costante per la batteria (esclude motori e batteria)
    battery_oew_fraction: float = 0.37

    # moltiplicatore su L/D per i sistemi a idrogeno
    hydrogen_ld_multiplier: float = 0.95

    # OEW/MTOW fan, forma a*MTOW^b + c
    oew_fan_a: float = -4.549e-4
    oew_fan_b: float = 0.466
    oew_fan_c: float = 0.6553

    # OEW/MTOW propeller (correlazione "grezza"), stessa forma
    oew_prop_a: float = -2.678e-5
    oew_prop_b: float = 0.8564
    oew_prop_c: float = 0.7493

    # peso payload, forma (a*R_km^b + c) * (d*R_km + e)
    payload_a: float = 0.01245
    payload_b: float = 1.098
    payload_c: float = 20.0
    payload_d: float = 0.001094
    payload_e: float = 107.0

    # Parametri per l'andamento dello scaling factor del propeller 
    # (curva APPROSSIMATA - vedi propulsive_efficiency.py):
    # sale come 1-exp(-rise_rate*mach) fino al picco a peak_mach, poi cala
    # come una gaussiana di larghezza decay_width. Questi dati sono ottenuti
    # dall'osservazione della figura corrispondente dell'articolo
    propeller_curve_peak_mach: float = 0.628
    propeller_curve_rise_rate: float = 11.65
    propeller_curve_decay_width: float = 0.038

    def with_changes(self, **kwargs) -> "TechAssumptions":
        """restituisce una copia con solo alcuni parametri modificati, es.:
            tech2 = tech.with_changes(e_battery_Wh_per_kg=600.0)
        """
        return replace(self, **kwargs)


@dataclass(frozen=True)
class WellToTankEfficiencies:
    """Frazione dell'elettricità rinnovabile dalla "rete" che arriva a bordo
    come energia utilizzabile, stivata in qualche forma"""

    electricity: float = 0.85       # ricarica batteria
    liquid_hydrogen: float = 0.46   # elettrolisi + liquefazione + distribuzione
    e_saf: float = 0.26             # elettrolisi + DAC + sintesi

    def with_changes(self, **kwargs) -> "WellToTankEfficiencies":
        return replace(self, **kwargs)


"""Poteri calorifici inferiori (Lower Heating Value): sono proprietà
chimiche di sostanze pure (idrogeno) o quasi (e-SAF), non parametri
tecnologici incerti, per questo restano costanti fisse e NON dentro
TechAssumptions"""
LHV_HYDROGEN_J_per_kg = 120e6   # 120 MJ/kg
LHV_ESAF_J_per_kg = 43e6        # 43 MJ/kg
