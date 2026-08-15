"""
Esempio d'uso della calibrazione deterministica:

    theta* = argmin_theta J(theta)

con:
    - parametri da calibrare scelti (non selezionati in automatico
      dalla sensitivity analysis - vedi FAN_BOUNDS / PROPELLER_BOUNDS sotto)
    - bounds SCELTI A MANO parametro per parametro (percentuale attorno al
      nominale, o bound assoluto, a piacere - vedi manual_bounds())
    - due calibrazioni DISTINTE: una per la configurazione fan (@450 kt),
      una per la configurazione propeller (@250 kt), ciascuna con il
      proprio sottoinsieme di parametri e i propri bounds (non un'unica
      calibrazione "combinata" su tutte e due insieme)

ATTENZIONE - budget di calcolo: ogni valutazione di J(theta) costa
indicativamente 0.3-0.6s (chiama il modello completo). Qui sotto i budget
(N_DE_RUNS, maxiter, popsize) sono tenuti volutamente bassi per fare girare
l'esempio in pochi minuti. Per la calibrazione "vera" da riportare in tesi,
alza N_DE_RUNS (3-5) e DE_MAXITER/DE_POPSIZE (vedi i commenti sotto) e
mettilo a girare con calma, idealmente offline: può richiedere da qualche
minuto a oltre un'ora per ciascuna delle due calibrazioni.

Uso:
    python examples/deterministic_calibration_example.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import TechAssumptions, WellToTankEfficiencies, fan_cost, propeller_cost
from cnav.calibration import (
    manual_bounds,
    theta_to_tech_wtt,
    run_deterministic_calibration,
    summarize_multiple_minima,
)

# ---------------------------------------------------------------------
# Budget di calcolo (abbassali/alzali qui)
# ---------------------------------------------------------------------
N_DE_RUNS = 1    # differential_evolution ripetuta N_DE_RUNS volte (seed diversi):
                 # è il modo per avere "più punti iniziali diversi"
                 # con un metodo globale popolazione-based come DE.
                 # Per la calibrazione "vera": alza a 3-5.
DE_MAXITER = 15  # per la calibrazione "vera": alza a 40-60
DE_POPSIZE = 8   # per la calibrazione "vera": alza a 10-15

# ---------------------------------------------------------------------
#  Parametri scelti a mano per la configurazione FAN (@450 kt) e i
#  rispettivi bounds. 
#  Ogni voce di FAN_BOUNDS è:
#     - un float w  -> bound percentuale [n - w*|n|, n + w*|n|]
#     - una tupla (lo, hi) -> bound assoluto, usato così com'è
#     (vedi manual_bounds() in cnav.calibration.deterministic_calibration
#     per i dettagli)
# ---------------------------------------------------------------------
FAN_BOUNDS = {
    "fan_pressure_ratio": 0.20,             # nominale di 1.5
    "fan_scaling_mach_ref": (0.5, 0.9),     
    "ld_baseline_fan": (16.0, 24.0),
    "pax_weight_kg": (70, 130),
    "oew_fan_b": 0.20,                      # nominale di 0.466
    "oew_fan_c": 0.20,                      # nominale di 0.6553
}

# ---------------------------------------------------------------------
# Parametri scelti a mano per la configurazione PROPELLER (@250 kt)
# ---------------------------------------------------------------------
PROPELLER_BOUNDS = {
    "propeller_curve_peak_mach": 0.20,      # nominale di 0.628
    "propeller_curve_rise_rate": 0.15,      # nominale di 11.65
    "propeller_curve_decay_width": 0.20,      # nominale di 0.038
    "ld_baseline_propeller": (16.0, 24.0),
    "pax_weight_kg": (70, 130),
    "oew_fan_b": 0.20,                      # nominale di 0.466
    "oew_fan_c": 0.20,                      # nominale di 0.6553
    "oew_prop_b": 0.20,                     # nominale di 0.8564
    "oew_prop_c": 0.20,                     # nominale di 0.7493
}


def calibrate_configuration(label: str, bounds_spec: dict, objective: str):
    """Esegue una calibrazione deterministica indipendente su un
    sottoinsieme di parametri/bounds scelti a mano, minimizzando solo
    la funzione di costo di quella configurazione (objective="fan" o
    "propeller")"""
    param_names = list(bounds_spec.keys())  # l'ordine del dict è quello scelto sopra
    bounds = manual_bounds(bounds_spec)

    print(f"\n{'=' * 70}\nCALIBRAZIONE {label.upper()}  (objective='{objective}')\n{'=' * 70}")
    print(f"Parametri ({len(param_names)}): {param_names}")
    print("Bounds:")
    for p in param_names:
        lo, hi = bounds[p]
        print(f"  {p:35s} [{lo:10.4g}, {hi:10.4g}]")

    t0 = time.time()
    best_run, df_runs = run_deterministic_calibration(
        param_names, bounds=bounds, method="differential_evolution",
        n_de_runs=N_DE_RUNS, objective=objective,
        de_kwargs=dict(maxiter=DE_MAXITER, popsize=DE_POPSIZE),
        verbose=True,
    )
    print(f"  tempo totale: {time.time() - t0:.1f}s")

    print("\nTutti i tentativi (ordinati per costo):")
    print(df_runs[["method", "cost", "success", "n_eval"]].to_string(index=False))

    near_best, equifinality_suspected = summarize_multiple_minima(df_runs, rel_tol=0.05)
    if equifinality_suspected:
        print("\n  ATTENZIONE: tra i tentativi entro il 5% dal costo minimo, theta* varia "
              "in modo apprezzabile: possibile equifinality (da approfondire al par. 5.2).")
    else:
        print("\n  Nessun indizio preliminare di più minimi ben separati ")

    return param_names, best_run, df_runs


# ---------------------------------------------------------------------
# Due calibrazioni distinte, per fan e propeller
# ---------------------------------------------------------------------
fan_params, fan_best, fan_df_runs = calibrate_configuration("fan", FAN_BOUNDS, objective="fan")
prop_params, prop_best, prop_df_runs = calibrate_configuration(
    "propeller", PROPELLER_BOUNDS, objective="propeller")

# ---------------------------------------------------------------------
# Confronto costo "nominale" vs calibrato, per ciascuna configurazione
# ---------------------------------------------------------------------
tech_nominal, wtt_nominal = TechAssumptions(), WellToTankEfficiencies()
j_fan_nominal, per_system_fan_nominal = fan_cost(tech_nominal, wtt_nominal)
j_prop_nominal, per_system_prop_nominal = propeller_cost(tech_nominal, wtt_nominal)

for label, params, best_run, j_nominal, cost_fn in [
    ("FAN", fan_params, fan_best, j_fan_nominal, fan_cost),
    ("PROPELLER", prop_params, prop_best, j_prop_nominal, propeller_cost),
]:
    theta_star = [best_run.theta_star[p] for p in params]
    tech_star, wtt_star = theta_to_tech_wtt(theta_star, params)
    j_star, per_system_star = cost_fn(tech_star, wtt_star)

    print(f"\n{'=' * 70}\nRISULTATO {label}\n{'=' * 70}")
    print(f"  J(theta_nominale) = {j_nominal:8.4f}")
    print(f"  J(theta*)          = {j_star:8.4f}")
    print(f"  Riduzione: {100 * (1 - j_star / j_nominal):.1f}%")

    print("\n  theta* (valori calibrati):")
    for p in params:
        nominal_val = getattr(tech_nominal, p) if hasattr(tech_nominal, p) else getattr(wtt_nominal, p)
        print(f"    {p:35s} nominale={nominal_val:10.4g}   calibrato={best_run.theta_star[p]:10.4g}")

    print("\n  Dettaglio costo per sistema, con theta*:")
    for name, c in per_system_star.items():
        print(f"    {name:25s} {c:8.4f}")
