"""
Esempio d'uso di cnav.calibration.calibration_uncertainty

Riusa esattamente gli stessi param_names/bounds/J della calibrazione
deterministica in deterministic_calibration_execution.py (SHARED_BOUNDS +
FAN_ONLY_BOUNDS + PROPELLER_ONLY_BOUNDS, combined_cost con fan_weight/
propeller_weight ricavati dal costo nominale), è la stessa "scatola"
theta, cambia solo cosa si fa con J(theta): prima si cercava il minimo,
ora si campiona l'intera regione sotto una soglia.

IMPORTANTE: imposta J_STAR qui sotto con il costo minimo realmente
TROVATO dalla tua ultima calibrazione.

Uso (fai prima un giro di prova con N piccolo!):
    python examples/calibration_uncertainty_execution.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import TechAssumptions, WellToTankEfficiencies
from cnav.calibration import (
    manual_bounds,
    make_objective,
    combined_cost,
    j_thr_from_margin,
    run_calibration_uncertainty,
    accepted_correlation_matrix,
)

# ---------------------------------------------------------------------
# Il tuo J* trovato in calibrazione deterministica, sostituisci con il 
# numero vero della tua run
# ---------------------------------------------------------------------
J_STAR = 0.806  # <-- SOSTITUISCI con best_run.cost della tua ultima calibrazione
MARGIN = 0.15  # tolleranza di riproduzione accettabile
J_THR = j_thr_from_margin(J_STAR, margin=MARGIN)

# ---------------------------------------------------------------------
# Budget del campionamento LHS (par. 7: N ~ 10**4 indicativamente).
# ATTENZIONE al costo: vedi il docstring di
# cnav.calibration.calibration_uncertainty per la stima del tempo.
# Comincia con N_SAMPLES piccolo per verificare che tutto funzioni,
# POI alza a 10_000 per il risultato da tesi (fallo girare offline).
# ---------------------------------------------------------------------
N_SAMPLES = 10000  # <-- prova con questo; per la run "vera" 10_000
SEED = 0
CHECKPOINT_PATH = "calibration_uncertainty_checkpoint.csv"  # salvato nella cwd
CHECKPOINT_EVERY = 100
RESUME = True  # se CHECKPOINT_PATH esiste già, riprende da lì invece di
               # ripetere le valutazioni già fatte

RANGE_WEIGHTS = [1, 5.0, 5.0, 5.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]

# ---------------------------------------------------------------------
# Stessi bounds e stessa J della calibrazione deterministica
# ---------------------------------------------------------------------
tech_nominal, wtt_nominal = TechAssumptions(), WellToTankEfficiencies()
_, nominal_detail_raw = combined_cost(tech_nominal, wtt_nominal,
                                       fan_weight=1.0, propeller_weight=1.0,
                                       fan_kwargs={"range_weights": RANGE_WEIGHTS},
                                       propeller_kwargs={"range_weights": RANGE_WEIGHTS})
FAN_WEIGHT = 1.0 / nominal_detail_raw["fan"]["total"]
PROPELLER_WEIGHT = 1.0 / nominal_detail_raw["propeller"]["total"]

SHARED_BOUNDS = {
    "pax_weight_kg": (75, 105),
}
FAN_ONLY_BOUNDS = {
    "fan_pressure_ratio": (1.3, 1.55),
    "fan_scaling_mach_ref": (0.78, 0.85),
}
PROPELLER_ONLY_BOUNDS = {
    "propeller_curve_peak_mach": (0.45, 0.60),
    "propeller_curve_rise_rate": (5, 20),
    "propeller_curve_decay_width": (0.02, 0.08),
}
ALL_BOUNDS_SPEC = {**SHARED_BOUNDS, **FAN_ONLY_BOUNDS, **PROPELLER_ONLY_BOUNDS}
param_names = list(ALL_BOUNDS_SPEC.keys())
bounds = manual_bounds(ALL_BOUNDS_SPEC)

J = make_objective(
    combined_cost, param_names,
    cost_kwargs=dict(fan_weight=FAN_WEIGHT, propeller_weight=PROPELLER_WEIGHT,
                      fan_kwargs={"range_weights": RANGE_WEIGHTS},
                      propeller_kwargs={"range_weights": RANGE_WEIGHTS}),
)

print(f"J_star = {J_STAR:.6g}  ->  J_thr = {J_THR:.6g}  (margine {100 * MARGIN:.0f}%)")
print(f"Parametri ({len(param_names)}): {param_names}")

# ---------------------------------------------------------------------
# Campionamento + valutazione + selezione di Theta_acc
# ---------------------------------------------------------------------
t0 = time.time()
result = run_calibration_uncertainty(
    J, param_names, bounds, j_thr=J_THR,
    n_samples=N_SAMPLES, seed=SEED,
    checkpoint_path=CHECKPOINT_PATH, checkpoint_every=CHECKPOINT_EVERY,
    resume_from_checkpoint=RESUME, verbose=True,
)
print(f"\nTempo totale: {time.time() - t0:.0f}s")

# ---------------------------------------------------------------------
# Diagnostica: range accettato per parametro + correlazioni (par. 7.1)
# ---------------------------------------------------------------------
print(f"\n{'=' * 70}\nRANGE ACCETTATO PER PARAMETRO (Theta_acc)\n{'=' * 70}")
for p in param_names:
    col = result.df_accepted[f"theta_{p}"]
    nominal_val = getattr(tech_nominal, p) if hasattr(tech_nominal, p) else getattr(wtt_nominal, p)
    print(f"  {p:30s} nominale={nominal_val:10.4g}   "
          f"accettato=[{col.min():10.4g}, {col.max():10.4g}]   "
          f"media={col.mean():10.4g}  std={col.std():10.4g}")

print(f"\n{'=' * 70}\nCORRELAZIONI TRA PARAMETRI ACCETTATI\n{'=' * 70}")
print(accepted_correlation_matrix(result).round(3).to_string())
print("\n(correlazioni |r| elevate tra due parametri sono un indizio di "
      "equifinality")

# Theta_acc va salvato per essere riusato tal quale nella Fase 7
# (uncertainty propagation): result.df_accepted contiene già tutto.
result.df_accepted.to_csv("theta_acc.csv", index=False)
print(f"\nTheta_acc salvato in theta_acc.csv "
      f"({result.n_accepted} righe, {100 * result.acceptance_fraction:.1f}% del campione)")