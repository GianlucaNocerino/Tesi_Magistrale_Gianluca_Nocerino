import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav.sensitivity.local_sensitivity import (
    run_local_sensitivity_report, plot_all_condition_tornados,
    average_relevant_elasticities, plot_all_average_elasticities,
    average_relevant_elasticities_by_propulsor, plot_all_average_elasticities_by_propulsor,
)

"""
Analisi di sensibilità locale:

Esegue la procedura completa (condizioni operative standard, punti di
confine tra tecnologie, elasticità sia dell'intensity sia, per
Battery-electric, del range massimo fattibile) e stampa un riepilogo:
per ciascuna condizione/output, i parametri dominanti (|S_loc| più
alto). Mostra anche come riattivare la verifica di robustezza rispetto
alla dimensione della perturbazione (disattivata di default sul
termine range perché costosa, vedi il docstring di
run_local_sensitivity_report) e come riusare plot_tornado_from_sweep
(già presente in sensitivity_analysis_tools.py) per visualizzare un
punto specifico.

Uso:
    python examples/local_sensitivity_report.py
"""

# Passata veloce: elasticità di riferimento su intensity + range massimo
# (Battery-electric), robustezza solo sull'intensity (compute_range_robustness
# resta False di default: il termine range è costoso, vedi il docstring).
df_elasticities, df_robustness = run_local_sensitivity_report(
    system_names=["Battery-electric", "Hydrogen fuel cell", "Hydrogen combustion", "e-SAF combustion"],
    reference_perturbation=0.03,
    robustness_perturbations=(0.01, 0.02, 0.03, 0.05),
    stability_tol=0.10,
)

print(f"Righe totali (elasticità): {len(df_elasticities)}")
print(f"Righe totali (robustezza, solo intensity): {len(df_robustness)}\n")

print("=== Top 5 parametri dominanti per condizione/sistema/output ===")
top = (df_elasticities
       .assign(abs_elasticity=df_elasticities["elasticity"].abs())
       .sort_values("abs_elasticity", ascending=False)
       .groupby(["condition", "system_name", "output"])
       .head(5))
print(top[["condition", "system_name", "output", "param_name", "elasticity"]].to_string(index=False))

print("\n=== Parametri con elasticità (intensity) instabile rispetto alla perturbazione (soglia 10%) ===")
unstable = df_robustness[~df_robustness["stable"]]
if unstable.empty:
    print("Nessuno: tutte le elasticità calcolate sono stabili sull'intervallo 1%-5% testato.")
else:
    print(unstable[["condition", "system_name", "param_name", "rel_spread"]]
          .sort_values("rel_spread", ascending=False).to_string(index=False))

# Tornado plot per ogni condizione/sistema/output, riusando
# plot_tornado_from_sweep senza ricalcolare nulla: mostra tutti (e
# soli) i parametri con |elasticità| > threshold, nessun tetto sul
# numero (top_n=None, il default).
#print("\n(genero un tornado plot per ogni condizione/sistema/output)")
#plot_all_condition_tornados(df_elasticities, threshold=0.05)

# Indice di elasticità medio (dei valori assoluti) dei parametri rilevanti,
# per sistema/propulsore/output: media calcolata solo sulle condizioni in
# cui il parametro supera la stessa soglia usata sopra per i tornado plot
# (0.05, non a caso)
df_avg = average_relevant_elasticities(df_elasticities, threshold=0.05)
print("\n=== Elasticità media assoluta dei parametri rilevanti, per sistema/propulsore/output ===")
print(df_avg.to_string(index=False))

#print("\n(genero un tornado plot dell'elasticità media assoluta per ogni sistema/propulsore/output)")
#plot_all_average_elasticities(df_avg, reference_line=0.5)

df_avg_propulsor = average_relevant_elasticities_by_propulsor(df_elasticities, threshold=0.05)
print("\n=== Elasticità media assoluta dei parametri rilevanti, per propulsore (tutti i sistemi) ===")
print(df_avg_propulsor.to_string(index=False))

print("\n(genero un tornado plot aggregato per fan e uno per propeller, tutti i sistemi combinati)")
plot_all_average_elasticities_by_propulsor(df_avg_propulsor, reference_line=0.5)

plt.show()

# Esempio di come riattivare la verifica di robustezza anche sul range
# massimo, limitandola a un solo punto e a pochi parametri per tenerla
# veloce (vedi range_param_subset e range_robustness_perturbations):
#
# df_e2, df_r2 = run_local_sensitivity_report(
#     system_names=["Battery-electric"],
#     include_boundaries=False,
#     compute_range_robustness=True,
#     range_param_subset={"e_battery_Wh_per_kg", "eta_motor", "battery_oew_fraction"},
#     range_robustness_perturbations=(0.01, 0.05),
# )