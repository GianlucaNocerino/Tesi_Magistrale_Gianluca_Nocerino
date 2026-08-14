import dataclasses
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from cnav.model.constants import TechAssumptions, WellToTankEfficiencies
from cnav.model.mission import Mission
from cnav.model.propulsion_systems import build_default_systems
from cnav.model.well_to_tank import build_energy_carriers
from cnav.model.energy_intensity import compute_intensity

"""
Libreria degli strumenti per l'analisi di sensibilità.
"""

"""Registro dinamico dei parametri, da dataclass"""

CATEGORIES = {
    "Technological": [
        "eta_motor", "e_battery_Wh_per_kg", "eta_fuel_cell", 
        "fan_pressure_ratio", "fan_scaling_mach_ref", "ld_baseline_propeller", "ld_baseline_fan",
        "eta_p_propeller", "eta_p_fan", "turbofan_core_specific_power_kW_per_kg",
        "turboprop_specific_power_kW_per_kg", "gamma_tank", 
        "delta_TMS_N_per_kW", "kappa_TMS_kg_per_kW",
        "electricity", "liquid_hydrogen", "e_saf",
        "motor_specific_power_kW_per_kg", "eta_battery", "fuel_cell_specific_power_kW_per_kg",
        "climb_rate_ft_per_min", "climb_speed_fraction"
    ],
    "Empirical": [
        "oew_fan_a", "oew_fan_b", "oew_fan_c", 
        "oew_prop_a", "oew_prop_b", "oew_prop_c",
        "payload_a", "payload_b", "payload_c", "payload_d", "payload_e",
        "propeller_curve_peak_mach", "propeller_curve_rise_rate", "propeller_curve_decay_width",
        "hydrogen_empty_weight_multiplier", "battery_oew_fraction", "hydrogen_ld_multiplier"
    ],
    "Regulatory": [
        "reserve_loiter_time_s", "reserve_alternate_range_nmi", 
        "reserve_loiter_speed_kt", "pax_weight_kg"
    ]
}

KEY_TO_CATEGORY = {key: cat for cat, keys in CATEGORIES.items() for key in keys}

PARAMETER_REGISTRY = {}
for source_dataclass in [TechAssumptions(), WellToTankEfficiencies()]:
    for key, nominal_value in dataclasses.asdict(source_dataclass).items():
        category = KEY_TO_CATEGORY.get(key, "Uncategorized")
        PARAMETER_REGISTRY[key] = {
            "category": category,
            "nominal": nominal_value
        }

def get_flat_nominal_params() -> dict:
    return {k: v["nominal"] for k, v in PARAMETER_REGISTRY.items()}


"""Interfaccia del modello"""

def evaluate_model(flat_params: dict, mission: Mission, system_name: str) -> float:
    default_tech = TechAssumptions()
    default_wtt = WellToTankEfficiencies()
    
    tech_kwargs = {}
    wtt_kwargs = {}
    
    for key, value in flat_params.items():
        if hasattr(default_tech, key):
            tech_kwargs[key] = value
        elif hasattr(default_wtt, key):
            wtt_kwargs[key] = value
        else:
            raise ValueError(f"Parametro sconosciuto o non mappato: {key}")
            
    tech = default_tech.with_changes(**tech_kwargs)
    wtt = default_wtt.with_changes(**wtt_kwargs)
    
    carriers = build_energy_carriers(wtt)
    systems = build_default_systems(carriers)
    
    target_system = next((s for s in systems if s.name == system_name), None)
    if not target_system:
        raise ValueError(f"Sistema {system_name} non trovato nel modello.")
        
    result = compute_intensity(mission, target_system, tech)
    return result.intensity_MJ_per_pax_nmi


def _elasticity(y_nom: float, y_plus: float, y_minus: float, perturbation: float) -> float:
    """Indice di elasticità con differenza centrata; se un solo lato della
    perturbazione converge, ricade su una differenza laterale (forward o
    backward) invece di scartare il punto. Restituisce NaN solo se
    nessuno dei due lati converge"""
    plus_ok, minus_ok = not np.isnan(y_plus), not np.isnan(y_minus)
    if plus_ok and minus_ok:
        return (y_plus - y_minus) / (2 * perturbation * y_nom)
    if plus_ok:
        return (y_plus - y_nom) / (perturbation * y_nom)
    if minus_ok:
        return (y_nom - y_minus) / (perturbation * y_nom)
    return float("nan")


"""Strumenti di analisi"""

def _compute_param_elasticities(mission: Mission, system_name: str, perturbation: float = 0.01,
                                 param_subset=None):
    """Calcola l'elasticità di ogni parametro (o del solo sottoinsieme
    param_subset, se dato) per un singolo punto operativo e un singolo
    sistema propulsivo. Nessun plotting: è il "motore" condiviso da
    run_linear_sensitivity (singolo punto) e run_sensitivity_sweep
    (multi-punto).

    Ritorna (y_nom, results) dove results è {param_name: elasticità}
    (solo i parametri con nominal != 0 ed elasticità calcolabile).
    Se la configurazione nominale non converge, ritorna (nan, {}).
    """
    base_params = get_flat_nominal_params()
    y_nom = evaluate_model(base_params, mission, system_name)

    if np.isnan(y_nom):
        return y_nom, {}

    results = {}
    for param_name, param_data in PARAMETER_REGISTRY.items():
        if param_subset is not None and param_name not in param_subset:
            continue

        nominal_val = param_data["nominal"]
        if nominal_val == 0.0:
            continue

        delta_x = nominal_val * perturbation
        params_plus = base_params.copy(); params_plus[param_name] += delta_x
        params_minus = base_params.copy(); params_minus[param_name] -= delta_x

        y_plus = evaluate_model(params_plus, mission, system_name)
        y_minus = evaluate_model(params_minus, mission, system_name)

        s_index = _elasticity(y_nom, y_plus, y_minus, perturbation)
        if not np.isnan(s_index):
            results[param_name] = s_index

    return y_nom, results


def run_linear_sensitivity(mission: Mission, system_name: str, perturbation: float = 0.01):
    """Genera il Tornado Plot per un singolo punto operativo (Range e Velocità)"""

    y_nom, results = _compute_param_elasticities(mission, system_name, perturbation)
    if np.isnan(y_nom):
        print("La configurazione nominale non converge.")
        return

    plot_data = {k: v for k, v in results.items() if abs(v) > 1e-4}
    sorted_plot_data = dict(sorted(plot_data.items(), key=lambda item: abs(item[1])))

    if sorted_plot_data:
        labels = list(sorted_plot_data.keys())
        values = list(sorted_plot_data.values())
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]
        
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color=colors, edgecolor='black')
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel('Indice di Elasticità', fontsize=12)
        plt.title(f'Tornado Plot - {system_name}\n({mission.range_nmi} nmi, {mission.cruise_speed_kt} kt)', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show(block=False)


def run_sensitivity_sweep(missions, system_names, perturbation: float = 0.01, verbose: bool = True):
    """Screening multi-punto: esegue _compute_param_elasticities su ogni
    combinazione (mission, system_name) e accumula tutto in un DataFrame
    tidy, una riga per (range, velocità, propulsore, sistema, parametro).

    missions: lista di oggetti Mission — costruiscile tu, es. con
        itertools.product su range_nmi, cruise_speed_kt e propulsor,
        così la config fan/elica è già inclusa (è un campo di Mission).
    system_names: lista di stringhe, es. ["Battery-electric",
        "Hydrogen fuel cell", "Hydrogen combustion", "e-SAF combustion"].

    Nessun grafico qui: usa rank_dominant_parameters sul risultato per
    la classifica, e run_linear_sensitivity (o plot_tornado_from_sweep)
    solo sui punti che poi vuoi effettivamente visualizzare.
    """
    rows = []
    for mission in missions:
        for system_name in system_names:
            y_nom, results = _compute_param_elasticities(mission, system_name, perturbation)
            if np.isnan(y_nom):
                if verbose:
                    print(f"[SKIP] Config nominale non converge: {system_name} "
                          f"({mission.propulsor}) @ {mission.range_nmi} nmi, "
                          f"{mission.cruise_speed_kt} kt")
                continue
            for param_name, elasticity in results.items():
                rows.append({
                    "range_nmi": mission.range_nmi,
                    "cruise_speed_kt": mission.cruise_speed_kt,
                    "propulsor": mission.propulsor,
                    "system_name": system_name,
                    "category": PARAMETER_REGISTRY[param_name]["category"],
                    "param_name": param_name,
                    "elasticity": elasticity,
                })
    return pd.DataFrame(rows)


def plot_tornado_from_sweep(df: pd.DataFrame, range_nmi, cruise_speed_kt, propulsor, system_name,
                             top_n=None, threshold: float = 1e-4, title: str = None):
    """Disegna il tornado plot per un punto/sistema specifico leggendo dati
    già calcolati in df (output di run_sensitivity_sweep) — nessun ricalcolo
    del modello. Utile per rivedere un punto interessante emerso dal ranking
    senza rilanciare run_linear_sensitivity.

    threshold: tiene solo i parametri con |elasticità| > threshold (criterio
    principale di selezione). top_n, se dato, applica in più un tetto al
    numero di parametri mostrati (i top_n più influenti tra quelli che
    superano threshold).
    title: se dato, sostituisce il titolo di default (che mostra solo
    system_name/propulsor/range/velocità — non menziona quale grandezza
    è stata perturbata, utile quando lo stesso df contiene più output
    diversi, es. intensity e max_range_nmi, vedi plot_condition_tornado
    in local_sensitivity.py)"""
    
    subset = df[
        (df["range_nmi"] == range_nmi) &
        (df["cruise_speed_kt"] == cruise_speed_kt) &
        (df["propulsor"] == propulsor) &
        (df["system_name"] == system_name)
    ].copy()

    subset = subset[subset["elasticity"].abs() > threshold]
    subset = subset.sort_values("elasticity", key=lambda x: x.abs())
    if top_n is not None:
        subset = subset.tail(top_n)

    if subset.empty:
        print("Nessun dato da plottare per questo punto/sistema.")
        return

    labels = subset["param_name"].tolist()
    values = subset["elasticity"].tolist()
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]

    if title is None:
        title = f'Tornado Plot - {system_name} ({propulsor})\n({range_nmi} nmi, {cruise_speed_kt} kt)'

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color=colors, edgecolor='black')
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Indice di Elasticità', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)

def run_sensitivity_3d_surface(system_name: str, propulsor: str, target_param: str, ranges_nmi: np.ndarray, speeds_kt: np.ndarray, perturbation: float = 0.01):
    """Genera la superficie 3D dell'elasticità per un singolo parametro, al variare di Range e Velocità"""
    
    base_params = get_flat_nominal_params()
    nominal_val = base_params.get(target_param, 0.0)
    if nominal_val == 0.0: return
    
    R, V = np.meshgrid(ranges_nmi, speeds_kt)
    Z = np.zeros_like(R, dtype=float)
    
    delta_x = nominal_val * perturbation
    params_plus = base_params.copy(); params_plus[target_param] += delta_x
    params_minus = base_params.copy(); params_minus[target_param] -= delta_x

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            mission = Mission(range_nmi=R[i, j], cruise_speed_kt=V[i, j], propulsor=propulsor)
            y_nom = evaluate_model(base_params, mission, system_name)
            y_plus = evaluate_model(params_plus, mission, system_name)
            y_minus = evaluate_model(params_minus, mission, system_name)

            Z[i, j] = np.nan if np.isnan(y_nom) else _elasticity(y_nom, y_plus, y_minus, perturbation)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(R, V, Z, cmap=cm.viridis, edgecolor='k', alpha=0.9)
    ax.set_xlabel('Range [nmi]')
    ax.set_ylabel('Velocità [kt]')
    ax.set_zlabel('Elasticità')
    ax.set_title(f'Sensibilità di "{target_param}"\n{system_name} ({propulsor})')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Elasticità')
    ax.view_init(elev=25, azim=-45)
    plt.tight_layout()
    plt.show(block=False)

