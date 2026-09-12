"""
Esempio d'uso del modulo di calibrazione (src/cnav/calibration/):
calcola le funzioni di costo globali - fan, propeller, e la loro
combinazione pesata (combined_cost) - con i parametri nominali di
TechAssumptions/WellToTankEfficiencies, e mostra il dettaglio per sistema.

Nota: qui si calcola J in un punto (i parametri nominali, o pesi
custom), non si cerca ancora theta* che la minimizza. Per la
minimizzazione vera e propria (differential_evolution/Nelder-Mead su
J(theta)) vedi execution/04_calibrazione/deterministic_calibration_execution.py.

Uso:
    python execution/04_calibrazione/cost_function_computation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cnav import (TechAssumptions, WellToTankEfficiencies,
                   fan_cost, propeller_cost, combined_cost,
                   max_feasible_range_nmi, PAPER_DATA)

tech = TechAssumptions()
wtt = WellToTankEfficiencies()

print("=== Range massimo fattibile (Battery-electric) ===")
for propulsor, key in [("fan", "fan"), ("propeller", "propeller")]:
    speed_kt = PAPER_DATA[key]["speed_kt"]
    model_range = max_feasible_range_nmi(tech, wtt, propulsor, speed_kt)
    paper_range = PAPER_DATA[key]["systems"]["Battery-electric"]["max_range_nmi"]
    print(f"  {propulsor:10s} @ {speed_kt} kt: modello = {model_range:6.1f} nmi, "
          f"paper = {paper_range:6.1f} nmi")

print("\n=== Costo globale FAN (@450 kt) ===")
total_fan, per_system_fan = fan_cost(tech, wtt)
for name, c in per_system_fan.items():
    print(f"  {name:25s} {c:8.4f}")
print(f"  {'TOTALE':25s} {total_fan:8.4f}")

print("\n=== Costo globale PROPELLER (@250 kt) ===")
total_prop, per_system_prop = propeller_cost(tech, wtt)
for name, c in per_system_prop.items():
    print(f"  {name:25s} {c:8.4f}")
print(f"  {'TOTALE':25s} {total_prop:8.4f}")

# Esempio: pesare di più il sistema Battery-electric nel costo globale
# fan, e pesare i 20 Range del paper "a U": molto i Range brevi (dove il
# paper ha effettivamente dati per la batteria), poco la zona centrale,
# di nuovo di più i Range lunghi
print("\n=== Esempio con pesi personalizzati (fan) ===")
custom_weights = {"Battery-electric": 3.0, "e-SAF combustion": 1.0, "Hydrogen combustion": 1.0}
custom_range_weights = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 2.0, 2.0,
                        2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
total_custom, per_system_custom = fan_cost(
    tech, wtt, system_weights=custom_weights, range_weights=custom_range_weights,
)
for name, c in per_system_custom.items():
    print(f"  {name:25s} {c:8.4f}")
print(f"  {'TOTALE':25s} {total_custom:8.4f}")

# ---------------------------------------------------------------------
# Costo combinato fan+propeller (combined_cost): costo_totale =
# fan_weight*J_fan + propeller_weight*J_propeller. E' la funzione di
# costo pensata per calibrare in un'unica ottimizzazione i parametri
# fisicamente comuni a fan e propeller (vedi
# execution/04_calibrazione/deterministic_calibration_execution.py per l'uso con
# run_deterministic_calibration): qui la calcoliamo soltanto, nel punto
# nominale, con pesi 1:1 e poi 2:1, per farsi un'idea di come i pesi
# spostano il totale prima ancora di lanciare l'ottimizzazione.
# ---------------------------------------------------------------------
print("\n=== Costo combinato fan+propeller (pesi 1:1) ===")
total_combined, detail_combined = combined_cost(tech, wtt)
print(f"  J_fan (grezzo)        {detail_combined['fan']['total']:8.4f}")
print(f"  J_propeller (grezzo)  {detail_combined['propeller']['total']:8.4f}")
print(f"  {'TOTALE (pesato)':25s} {total_combined:8.4f}")

print("\n=== Costo combinato fan+propeller (pesi 2:1, fan:propeller) ===")
total_combined_2to1, detail_2to1 = combined_cost(tech, wtt, fan_weight=2.0, propeller_weight=1.0)
print(f"  J_fan (grezzo)        {detail_2to1['fan']['total']:8.4f}")
print(f"  J_propeller (grezzo)  {detail_2to1['propeller']['total']:8.4f}")
print(f"  {'TOTALE (pesato)':25s} {total_combined_2to1:8.4f}")
