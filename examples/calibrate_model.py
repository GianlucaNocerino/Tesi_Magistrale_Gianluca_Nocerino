"""
Esempio d'uso del modulo di calibrazione (src/cnav/calibration.py):
calcola le due funzioni di costo globali (fan, propeller) con i
parametri nominali di TechAssumptions/WellToTankEfficiencies, e mostra
il dettaglio per sistema.

Uso:
    python examples/calibrate_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnav import (TechAssumptions, WellToTankEfficiencies,
                   fan_cost, propeller_cost, max_feasible_range_nmi, PAPER_DATA)

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
# fan, e dare più peso ai Range brevi (dove il paper ha effettivamente
# dati per la batteria)
print("\n=== Esempio con pesi personalizzati (fan) ===")
custom_weights = {"Battery-electric": 3.0, "e-SAF combustion": 1.0, "Hydrogen combustion": 1.0}
custom_range_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
total_custom, per_system_custom = fan_cost(
    tech, wtt, system_weights=custom_weights, range_weights=custom_range_weights,
)
for name, c in per_system_custom.items():
    print(f"  {name:25s} {c:8.4f}")
print(f"  {'TOTALE':25s} {total_custom:8.4f}")
