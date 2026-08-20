"""
Esempio d'uso della calibrazione deterministica:

    theta* = argmin_theta J(theta)

con:
    - parametri da calibrare scelti
    - bounds scelti parametro per parametro (percentuale attorno al
      nominale, o bound assoluto, a piacere, vedi manual_bounds())
    - la funzione di costo J = cnav.calibration.cost_functions.combined_cost,
      cioè fan_weight*J_fan + propeller_weight*J_propeller, in un'unica
      ottimizzazione su theta = SHARED + FAN_ONLY + PROPELLER_ONLY.

Perché una calibrazione congiunta invece di due separate: i parametri di
SHARED_BOUNDS (pax_weight_kg, oew_fan_b/c) sono fisicamente gli stessi
nelle due configurazioni. Mettendoli in un unico vettore theta invece che
calibrandoli due volte separatamente (una per fan, una per propeller), il
loro valore è per costruzione identico nei due termini della somma.
I parametri di FAN_ONLY/PROPELLER_ONLY restano liberi di calibrarsi solo  
rispetto al termine di costo a cui appartengono fisicamente (non compaiono nell'altro).

Nota sull'architettura (vedi anche i docstring dei moduli coinvolti):
    - cost_functions.py sa cosa sono fan/propeller/combined_cost - qui è
      dove vive la "fisica" del costo, compresi i pesi custom.
    - deterministic_calibration.py, invece, non sa nulla di fan/propeller: è un
      motore generico che prende una J qualsiasi (costruita qui sotto con
      make_objective(combined_cost, ...)) e trova theta che la minimizza.
    - questo script è l'unico posto che decide di usare l'engine proprio
      per combined_cost con questa particolare suddivisione di parametri
      e questi pesi. Si potrebbe scrivere un altro script che lo usa per
      fan_cost da solo, o per una quarta cost_fn, senza toccare
      cnav.calibration.

Uso:
    python examples/deterministic_calibration_execution.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import TechAssumptions, WellToTankEfficiencies
from cnav.calibration import (
    manual_bounds,
    make_objective,
    run_deterministic_calibration,
    summarize_multiple_minima,
    combined_cost,
    theta_to_tech_wtt,
)

# ---------------------------------------------------------------------
# Budget di calcolo (abbassali/alzali qui)
# ---------------------------------------------------------------------
N_DE_RUNS = 5    # differential_evolution ripetuta N_DE_RUNS volte (seed diversi):
                 # è il modo per avere "più punti iniziali diversi"
                 # con un metodo globale popolazione-based come DE.
                 # Per la calibrazione "vera": alza a 3-5.
DE_MAXITER = 60  # per la calibrazione "vera": alza a 40-60
DE_POPSIZE = 15   # per la calibrazione "vera": alza a 10-15

RANGE_WEIGHTS = [1, 5.0, 5.0, 5.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # per modificare 
# i pesi della funzione di costo ai diversi valori del Range

# Pesi globali fan/propeller dentro combined_cost: J = fan_weight*J_fan +
# propeller_weight*J_propeller. Invece di pesi fissi, uso il reciproco
# del costo nominale grezzo (non pesato) di ciascun ramo, così
# J_fan(theta_nominale)*fan_weight == 1 e
# J_propeller(theta_nominale)*propeller_weight == 1 
# I due rami partono sulla stessa scala, e nessuno dei due domina il compromesso
# sui parametri condivisi solo perchè parte "più grande" in valore
# assoluto (es. la config propeller ha un sistema in più nel paper,
# Hydrogen fuel cell, che a parità di pesi tenderebbe a pesare di più
# nella somma).
tech_nominal, wtt_nominal = TechAssumptions(), WellToTankEfficiencies()
_, nominal_detail_raw = combined_cost(tech_nominal, wtt_nominal,
                                       fan_weight=1.0, propeller_weight=1.0,fan_kwargs={"range_weights": RANGE_WEIGHTS}, 
                                       propeller_kwargs={"range_weights": RANGE_WEIGHTS})

j_fan_nominal = nominal_detail_raw["fan"]["total"]
j_propeller_nominal = nominal_detail_raw["propeller"]["total"]

if j_fan_nominal <= 0 or j_propeller_nominal <= 0:
    raise ValueError(
        f"Costo nominale non positivo (fan={j_fan_nominal}, "
        f"propeller={j_propeller_nominal}): impossibile usarne il "
        f"reciproco come peso."
    )

FAN_WEIGHT = 1.0 / j_fan_nominal
PROPELLER_WEIGHT = 1.0 / j_propeller_nominal

print(f"J_fan(nominale) grezzo        = {j_fan_nominal:.4f}  -> fan_weight = {FAN_WEIGHT:.4g}")
print(f"J_propeller(nominale) grezzo  = {j_propeller_nominal:.4f}  -> propeller_weight = {PROPELLER_WEIGHT:.4g}")

# ---------------------------------------------------------------------
# Parametri comuni a fan e propeller: un'unica voce, un unico
# valore calibrato per entrambe le configurazioni.
# Ogni voce è:
#     - un float w  -> bound percentuale [n - w*|n|, n + w*|n|]
#     - una tupla (lo, hi) -> bound assoluto, usato così com'è
#     (vedi manual_bounds() in cnav.calibration.deterministic_calibration
#     per i dettagli)
# ---------------------------------------------------------------------
SHARED_BOUNDS = {
    "pax_weight_kg": (75, 105)                   
}

# ---------------------------------------------------------------------
# Parametri specifici della configurazione FAN (@450 kt): non compaiono
# nel calcolo di J_propeller, quindi si calibrano solo rispetto a J_fan.
# ---------------------------------------------------------------------
FAN_ONLY_BOUNDS = {
    "fan_pressure_ratio": (1.3, 1.55),
    "fan_scaling_mach_ref": (0.78, 0.85)
}

# ---------------------------------------------------------------------
# Parametri specifici della configurazione PROPELLER (@250 kt): non
# compaiono nel calcolo di J_fan, si calibrano solo rispetto a J_propeller.
# ---------------------------------------------------------------------
PROPELLER_ONLY_BOUNDS = {
    "propeller_curve_peak_mach": (0.45, 0.60),      
    "propeller_curve_rise_rate": (5, 20), 
    "propeller_curve_decay_width": (0.02, 0.08)
}

ALL_BOUNDS_SPEC = {**SHARED_BOUNDS, **FAN_ONLY_BOUNDS, **PROPELLER_ONLY_BOUNDS}
param_names = list(ALL_BOUNDS_SPEC.keys())
bounds = manual_bounds(ALL_BOUNDS_SPEC)

print(f"\n{'=' * 70}\nCALIBRAZIONE CONGIUNTA fan + propeller (combined_cost)\n{'=' * 70}")
print(f"Parametri condivisi ({len(SHARED_BOUNDS)}): {list(SHARED_BOUNDS)}")
print(f"Parametri solo-fan ({len(FAN_ONLY_BOUNDS)}): {list(FAN_ONLY_BOUNDS)}")
print(f"Parametri solo-propeller ({len(PROPELLER_ONLY_BOUNDS)}): {list(PROPELLER_ONLY_BOUNDS)}")
print(f"Pesi: fan_weight={FAN_WEIGHT}, propeller_weight={PROPELLER_WEIGHT}")
print("\nBounds:")
for p in param_names:
    lo, hi = bounds[p]
    print(f"  {p:35s} [{lo:10.4g}, {hi:10.4g}]")

# J costruita qui, esplicitamente, a partire da combined_cost
J = make_objective(
    combined_cost, param_names,
    cost_kwargs=dict(fan_weight=FAN_WEIGHT, propeller_weight=PROPELLER_WEIGHT, fan_kwargs={"range_weights": RANGE_WEIGHTS}, 
    propeller_kwargs={"range_weights": RANGE_WEIGHTS})
)

t0 = time.time()
best_run, df_runs = run_deterministic_calibration(
    J, param_names, bounds=bounds, method="differential_evolution",
    n_de_runs=N_DE_RUNS,
    de_kwargs=dict(maxiter=DE_MAXITER, popsize=DE_POPSIZE),
    verbose=True,
)
print(f"  tempo totale: {time.time() - t0:.1f}s")

print("\nTutti i tentativi (ordinati per costo):")
print(df_runs[["method", "cost", "success", "n_eval"]].to_string(index=False))

near_best, equifinality_suspected = summarize_multiple_minima(df_runs, rel_tol=0.05)
if equifinality_suspected:
    print("\n  ATTENZIONE: tra i tentativi entro il 5% dal costo minimo, theta* varia "
          "in modo apprezzabile: possibile equifinality")
else:
    print("\n  Nessun indizio preliminare di più minimi ben separati")

# ---------------------------------------------------------------------
# Diagnostica: costo fan e propeller SEPARATI (grezzi, non pesati) nel
# punto theta* trovato - combined_cost li restituisce già scorporati nel
# dettaglio, non serve nessuna funzione ausiliaria.
# ---------------------------------------------------------------------

# nominal_detail_raw è già stato calcolato sopra (serviva per fissare
# i pesi della combined cost function), lo riuso qui invece di richiamare 
# combined_cost una seconda volta sul nominale
nominal_detail = nominal_detail_raw

df_runs_sorted = df_runs.sort_values("cost").reset_index(drop=True)

for i, row in df_runs_sorted.iterrows():
    is_best = (i == 0)
    run_theta_star = [row[f"theta_{p}"] for p in param_names]
    tech_star, wtt_star = theta_to_tech_wtt(run_theta_star, param_names)
    _, star_detail = combined_cost(tech_star, wtt_star,
                                    fan_weight=FAN_WEIGHT, propeller_weight=PROPELLER_WEIGHT, fan_kwargs={"range_weights": RANGE_WEIGHTS},
                                    propeller_kwargs={"range_weights": RANGE_WEIGHTS})

    header = f"RUN #{i + 1}  [{row['method']}]  costo J = {row['cost']:.6g}"
    if is_best:
        header += "   <-- MIGLIORE"
    print(f"\n\n{'#' * 70}\n{header}\n{'#' * 70}")

    for label, key in [("FAN", "fan"), ("PROPELLER", "propeller")]:
        j_nominal = nominal_detail[key]["total"]
        j_star = star_detail[key]["total"]
        print(f"\n{'=' * 70}\nRISULTATO {label}\n{'=' * 70}")
        print(f"  J(theta_nominale) = {j_nominal:8.4f}")
        print(f"  J(theta*)          = {j_star:8.4f}")
        print(f"  Riduzione: {100 * (1 - j_star / j_nominal):.1f}%")
        print("\n  Dettaglio costo per sistema, con theta*:")
        for name, c in star_detail[key]["per_system"].items():
            print(f"    {name:25s} {c:8.4f}")

    print(f"\n{'=' * 70}\nTHETA* (valori calibrati, condivisi + specifici)\n{'=' * 70}")
    for group_label, group in [("condiviso", SHARED_BOUNDS), ("solo-fan", FAN_ONLY_BOUNDS),
                                ("solo-propeller", PROPELLER_ONLY_BOUNDS)]:
        for p in group:
            nominal_val = getattr(tech_nominal, p) if hasattr(tech_nominal, p) else getattr(wtt_nominal, p)
            print(f"  [{group_label:14s}] {p:35s} nominale={nominal_val:10.4g}   "
                  f"calibrato={row[f'theta_{p}']:10.4g}")
